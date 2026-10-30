from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence

import httpx

from kg_platform.domain.capabilities import BackendCapabilities, BackendMetadata
from kg_platform.domain.errors import ErrorCode, KGPlatformError
from kg_platform.domain.models import (
    Direction,
    EdgeProvenance,
    EntityKey,
    EntityQuery,
    EntityRef,
    GraphRef,
    LiteralRef,
    QueryContext,
    RelationChoice,
    RelationKey,
    RelationPage,
    RelationQueryOptions,
    RelationRef,
    ResolutionResult,
    TriplePage,
    TripleQueryOptions,
    TripleRef,
)


STATUS_CODES = {
    400: ErrorCode.INVALID_ARGUMENT,
    404: ErrorCode.ENTITY_NOT_FOUND,
    409: ErrorCode.RELATION_NOT_APPLICABLE,
    412: ErrorCode.SNAPSHOT_MISMATCH,
    422: ErrorCode.INVALID_ARGUMENT,
    429: ErrorCode.RATE_LIMITED,
    503: ErrorCode.BACKEND_UNAVAILABLE,
    504: ErrorCode.DEADLINE_EXCEEDED,
}


def _entity(data: dict) -> EntityRef:
    key = data["key"]
    return EntityRef(
        EntityKey(key["graph_id"], key["native_id"], key.get("scope_id")),
        data["label"],
        tuple(data.get("types", [])),
        data.get("description"),
        tuple(tuple(item) for item in data.get("external_ids", [])),
    )


def _relation(data: dict) -> RelationRef:
    return RelationRef(
        RelationKey(data["graph_id"], data["native_id"]),
        data["label"],
        tuple(data.get("source_types", [])),
        tuple(data.get("target_types", [])),
    )


class RemoteKGBackend:
    """Training-side GraphBackend proxy over the typed HTTP API."""
    def __init__(
        self,
        endpoint: str,
        graph: GraphRef,
        *,
        timeout: float = 5.0,
        retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.graph = graph
        self.retries = retries
        self.client = httpx.AsyncClient(base_url=self.endpoint, timeout=timeout, transport=transport, trust_env=False)
        self._metadata: BackendMetadata | None = None

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Retry bounded transient failures while preserving typed KG errors."""
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.request(method, path, **kwargs)
                if response.is_success:
                    return response.json()
                payload = response.json() if response.content else {}
                error = payload.get("error", {})
                code_text = error.get("code")
                code = ErrorCode(code_text) if code_text in ErrorCode._value2member_map_ else STATUS_CODES.get(response.status_code, ErrorCode.BACKEND_UNAVAILABLE)
                exc = KGPlatformError(code, error.get("message", "Remote KG request failed."), details=error.get("details"))
                if not exc.retryable or attempt == self.retries:
                    raise exc
                last_error = exc
            except KGPlatformError:
                raise
            except httpx.TimeoutException as exc:
                last_error = KGPlatformError(ErrorCode.DEADLINE_EXCEEDED, "Remote KG request timed out.")
            except httpx.HTTPError as exc:
                last_error = KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, "Remote KG service is unavailable.")
            if attempt < self.retries:
                await asyncio.sleep(0.05 * (2**attempt))
        assert last_error is not None
        raise last_error

    async def metadata(self) -> BackendMetadata:
        data = await self._request("GET", "/v1/metadata")
        graph_data = data["graph"]
        remote_graph = GraphRef(graph_data["graph_id"], graph_data["snapshot_id"], graph_data["schema_version"])
        if remote_graph != self.graph:
            raise KGPlatformError(ErrorCode.SNAPSHOT_MISMATCH, "Remote KG metadata does not match configured snapshot.")
        caps = data["capabilities"]
        metadata = BackendMetadata(
            remote_graph,
            BackendCapabilities(
                scope=caps["scope"],
                directions=frozenset(Direction(value) for value in caps["directions"]),
                reverse_edges_materialized=caps["reverse_edges_materialized"],
                supports_literals=caps["supports_literals"],
                supports_entity_text_lookup=caps["supports_entity_text_lookup"],
                supports_relation_counts=caps["supports_relation_counts"],
                supports_virtual_relations=caps["supports_virtual_relations"],
                supports_cursor=caps["supports_cursor"],
                stable_snapshot=caps["stable_snapshot"],
            ),
            tuple((key, int(value)) for key, value in data.get("counts", {}).items()),
        )
        self._metadata = metadata
        return metadata

    async def open_session(self, context: QueryContext):
        if context.graph != self.graph:
            raise KGPlatformError(ErrorCode.SNAPSHOT_MISMATCH, "Configured remote graph does not match sample context.")
        return RemoteKGSession(self, context)

    async def aclose(self) -> None:
        await self.client.aclose()


class RemoteKGSession:
    """Context-bound request builder; the remote service itself is stateless."""
    def __init__(self, backend: RemoteKGBackend, context: QueryContext):
        self.backend = backend
        self.context = context

    def _graph(self) -> dict:
        graph = self.context.graph
        return {"graph_id": graph.graph_id, "snapshot_id": graph.snapshot_id, "schema_version": graph.schema_version}

    async def resolve_entity(self, query: EntityQuery) -> ResolutionResult:
        data = await self.backend._request(
            "POST", "/v1/entities:resolve", json={"request_id": str(uuid.uuid4()), "graph": self._graph(), "query": query.text}
        )
        return ResolutionResult(entity=_entity(data["entity"]))

    async def list_relations(self, entity: EntityKey, options: RelationQueryOptions) -> RelationPage:
        data = await self.backend._request(
            "POST",
            "/v1/relations:list",
            json={
                "request_id": str(uuid.uuid4()),
                "graph": self._graph(),
                "entity_id": entity.native_id,
                "directions": sorted(value.value for value in options.directions),
                "limit": options.limit,
            },
        )
        choices = tuple(
            RelationChoice(_relation(item["relation"]), Direction(item["direction"]), item.get("edge_count"))
            for item in data["choices"]
        )
        return RelationPage(_entity(data["entity"]), choices, data["truncated"], data.get("available_count"))

    async def list_triples(
        self, entity: EntityKey, choices: Sequence[RelationChoice], options: TripleQueryOptions
    ) -> TriplePage:
        wire_choices = [
            {
                "relation": {
                    "graph_id": item.relation.key.graph_id,
                    "native_id": item.relation.key.native_id,
                    "label": item.relation.label,
                    "source_types": list(item.relation.source_types),
                    "target_types": list(item.relation.target_types),
                },
                "direction": item.direction.value,
                "edge_count": item.edge_count,
            }
            for item in choices
        ]
        data = await self.backend._request(
            "POST",
            "/v1/triples:list",
            json={
                "request_id": str(uuid.uuid4()),
                "graph": self._graph(),
                "entity_id": entity.native_id,
                "choices": wire_choices,
                "limit_per_choice": options.limit_per_choice,
                "total_limit": options.total_limit,
            },
        )
        triples = []
        for item in data["triples"]:
            tail_data = item["tail"]
            tail = _entity(tail_data) if tail_data["kind"] == "entity" else LiteralRef(tail_data["lexical"], tail_data.get("datatype"), tail_data.get("language"))
            provenance = item["provenance"]
            triples.append(
                TripleRef(
                    _entity(item["head"]),
                    _relation(item["relation"]),
                    tail,
                    EdgeProvenance(self.context.graph, provenance.get("source_edge_id"), provenance.get("virtual", False), tuple(provenance.get("source_path", []))),
                    Direction(item["matched_direction"]) if item.get("matched_direction") else None,
                )
            )
        return TriplePage(_entity(data["entity"]), tuple(triples), data["truncated"], data.get("available_count"))

    async def aclose(self) -> None:
        return None
