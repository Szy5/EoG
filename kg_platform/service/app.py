from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kg_platform.domain.errors import ErrorCode, KGPlatformError
from kg_platform.domain.models import (
    Direction,
    EntityKey,
    EntityQuery,
    GraphRef,
    QueryContext,
    RelationChoice,
    RelationKey,
    RelationQueryOptions,
    RelationRef,
    TripleQueryOptions,
)
from kg_platform.service.api_models import RelationsRequest, ResolveRequest, TriplesRequest


HTTP_STATUS = {
    ErrorCode.INVALID_ARGUMENT: 400,
    ErrorCode.ENTITY_NOT_FOUND: 404,
    ErrorCode.RELATION_NOT_FOUND: 404,
    ErrorCode.ENTITY_AMBIGUOUS: 409,
    ErrorCode.RELATION_NOT_APPLICABLE: 409,
    ErrorCode.CONTEXT_REQUIRED: 412,
    ErrorCode.SNAPSHOT_MISMATCH: 412,
    ErrorCode.RESULT_LIMIT_EXCEEDED: 422,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.BACKEND_UNAVAILABLE: 503,
    ErrorCode.DEADLINE_EXCEEDED: 504,
}


def _graph(dto) -> GraphRef:
    return GraphRef(dto.graph_id, dto.snapshot_id, dto.schema_version)


def _entity(entity) -> dict:
    return {
        "key": {
            "graph_id": entity.key.graph_id,
            "native_id": entity.key.native_id,
            "scope_id": entity.key.scope_id,
        },
        "label": entity.label,
        "types": list(entity.types),
        "description": entity.description,
        "external_ids": [list(item) for item in entity.external_ids],
    }


def _relation(relation) -> dict:
    return {
        "graph_id": relation.key.graph_id,
        "native_id": relation.key.native_id,
        "label": relation.label,
        "source_types": list(relation.source_types),
        "target_types": list(relation.target_types),
    }


def _node(node) -> dict:
    if hasattr(node, "key"):
        return {"kind": "entity", **_entity(node)}
    return {"kind": "literal", "lexical": node.lexical, "datatype": node.datatype, "language": node.language}


def create_app(backend) -> FastAPI:
    """Create the stateless HTTP projection of one configured backend.

    Each request opens a short backend session. Rollout-level relation grants
    stay in the training-side application service, keeping this API suitable
    for multiple service replicas.
    """
    app = FastAPI(title="KG Platform", version="1.0.0")
    app.state.backend = backend
    app.state.ready = True

    @app.exception_handler(KGPlatformError)
    async def kg_error_handler(_request: Request, exc: KGPlatformError):
        return JSONResponse(
            status_code=HTTP_STATUS.get(exc.code, 500),
            content={"error": exc.to_dict()},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError):
        error = KGPlatformError(ErrorCode.INVALID_ARGUMENT, "Request body validation failed.", details={"errors": exc.errors()})
        return JSONResponse(status_code=400, content={"error": error.to_dict()})

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        # metadata() may query SPARQL or validate an mmap snapshot, making this
        # a stronger signal than the process-only health endpoint.
        if not app.state.ready:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        metadata = await backend.metadata()
        return {"status": "ready", "graph_id": metadata.graph.graph_id, "snapshot_id": metadata.graph.snapshot_id}

    @app.get("/v1/metadata")
    async def metadata():
        meta = await backend.metadata()
        return {
            "graph": {
                "graph_id": meta.graph.graph_id,
                "snapshot_id": meta.graph.snapshot_id,
                "schema_version": meta.graph.schema_version,
            },
            "capabilities": {
                "scope": meta.capabilities.scope,
                "directions": sorted(value.value for value in meta.capabilities.directions),
                "reverse_edges_materialized": meta.capabilities.reverse_edges_materialized,
                "supports_literals": meta.capabilities.supports_literals,
                "supports_entity_text_lookup": meta.capabilities.supports_entity_text_lookup,
                "supports_relation_counts": meta.capabilities.supports_relation_counts,
                "supports_virtual_relations": meta.capabilities.supports_virtual_relations,
                "supports_cursor": meta.capabilities.supports_cursor,
                "stable_snapshot": meta.capabilities.stable_snapshot,
            },
            "counts": dict(meta.counts),
        }

    async def open_request_session(graph: GraphRef, request_id: str):
        """Bind one HTTP request to the snapshot stated by the client."""
        return await backend.open_session(QueryContext(request_id, request_id, graph, request_id))

    @app.post("/v1/entities:resolve")
    async def resolve(request: ResolveRequest):
        session = await open_request_session(_graph(request.graph), request.request_id)
        try:
            result = await session.resolve_entity(EntityQuery(request.query))
            if result.entity:
                return {"request_id": request.request_id, "entity": _entity(result.entity), "candidates": []}
            if result.candidates:
                raise KGPlatformError(
                    ErrorCode.ENTITY_AMBIGUOUS,
                    "Entity label is ambiguous in this graph snapshot.",
                    details={"candidates": [_entity(item) for item in result.candidates[:20]], "candidate_count": len(result.candidates)},
                )
            raise KGPlatformError(ErrorCode.ENTITY_NOT_FOUND, "Entity was not found.")
        finally:
            await session.aclose()

    @app.post("/v1/relations:list")
    async def relations(request: RelationsRequest):
        graph = _graph(request.graph)
        session = await open_request_session(graph, request.request_id)
        try:
            key = EntityKey(graph.graph_id, request.entity_id)
            page = await session.list_relations(
                key,
                RelationQueryOptions(frozenset(Direction(value) for value in request.directions), request.limit),
            )
            return {
                "request_id": request.request_id,
                "entity": _entity(page.entity),
                "choices": [
                    {"relation": _relation(item.relation), "direction": item.direction.value, "edge_count": item.edge_count}
                    for item in page.choices
                ],
                "truncated": page.truncated,
                "available_count": page.available_count,
            }
        finally:
            await session.aclose()

    @app.post("/v1/triples:list")
    async def triples(request: TriplesRequest):
        graph = _graph(request.graph)
        session = await open_request_session(graph, request.request_id)
        try:
            choices = []
            for item in request.choices:
                if item.relation:
                    rel = RelationRef(
                        RelationKey(item.relation.graph_id, item.relation.native_id),
                        item.relation.label,
                        tuple(item.relation.source_types),
                        tuple(item.relation.target_types),
                    )
                else:
                    rel = RelationRef(RelationKey(graph.graph_id, item.relation_id or ""), item.relation_id or "")
                choices.append(RelationChoice(rel, Direction(item.direction), item.edge_count))
            page = await session.list_triples(
                EntityKey(graph.graph_id, request.entity_id),
                choices,
                TripleQueryOptions(request.limit_per_choice, request.total_limit),
            )
            return {
                "request_id": request.request_id,
                "entity": _entity(page.entity),
                "triples": [
                    {
                        "head": _entity(item.head),
                        "relation": _relation(item.relation),
                        "tail": _node(item.tail),
                        "matched_direction": item.matched_direction.value if item.matched_direction else None,
                        "provenance": {
                            "source_edge_id": item.provenance.source_edge_id,
                            "virtual": item.provenance.virtual,
                            "source_path": list(item.provenance.source_path),
                        },
                    }
                    for item in page.triples
                ],
                "truncated": page.truncated,
                "available_count": page.available_count,
            }
        finally:
            await session.aclose()

    return app
