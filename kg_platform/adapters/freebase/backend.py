from __future__ import annotations

import re
import time
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


NS = "http://rdf.freebase.com/ns/"
# IDs and predicates are interpolated into SPARQL identifiers, so allowlists
# are mandatory even though user labels are escaped as string literals below.
ID_RE = re.compile(r"^(?:m|g|en)\.[A-Za-z0-9_]+$")
PREDICATE_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+$")


def _native(uri: str) -> str:
    return uri[len(NS):].replace("/", ".") if uri.startswith(NS) else uri


def _escape_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")


class FreebaseSparqlBackend:
    """Async Freebase adapter for a read-only SPARQL 1.1 endpoint."""
    def __init__(self, endpoint: str, *, snapshot_id: str = "freebase@configured-endpoint", timeout: float = 15.0):
        self.endpoint = endpoint
        self.graph = GraphRef("freebase", snapshot_id)
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False)
        self.query_log: list[dict] = []

    async def query(self, sparql: str, *, purpose: str = "") -> list[dict]:
        """Execute one SELECT query and translate transport failures."""
        started = time.perf_counter()
        try:
            response = await self.client.get(
                self.endpoint,
                params={"query": sparql, "format": "application/sparql-results+json"},
                headers={"Accept": "application/sparql-results+json"},
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("results", {}).get("bindings", [])
            self.query_log.append(
                {
                    "purpose": purpose,
                    "sparql": sparql.strip(),
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "row_count": len(rows),
                }
            )
            return rows
        except httpx.TimeoutException as exc:
            self.query_log.append(
                {
                    "purpose": purpose,
                    "sparql": sparql.strip(),
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "row_count": 0,
                    "error": "timeout",
                }
            )
            raise KGPlatformError(ErrorCode.DEADLINE_EXCEEDED, "Freebase SPARQL query timed out.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            self.query_log.append(
                {
                    "purpose": purpose,
                    "sparql": sparql.strip(),
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "row_count": 0,
                    "error": str(exc),
                }
            )
            raise KGPlatformError(
                ErrorCode.BACKEND_UNAVAILABLE,
                f"Freebase SPARQL endpoint is unavailable. endpoint={self.endpoint}; cause={type(exc).__name__}: {exc}",
            ) from exc

    def drain_query_log(self) -> list[dict]:
        rows = list(self.query_log)
        self.query_log.clear()
        return rows

    async def metadata(self) -> BackendMetadata:
        # Readiness deliberately executes SPARQL; a listening HTTP socket does
        # not prove that Virtuoso can parse and answer queries.
        rows = await self.query("SELECT (1 AS ?ok) WHERE {}", purpose="readyz")
        if not rows or rows[0].get("ok", {}).get("value") != "1":
            raise KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, "Freebase SPARQL readiness query failed.")
        return BackendMetadata(
            self.graph,
            BackendCapabilities(
                scope="global",
                directions=frozenset({Direction.OUT, Direction.IN}),
                supports_literals=True,
                supports_virtual_relations=True,
            ),
        )

    async def open_session(self, context: QueryContext):
        if context.graph != self.graph:
            raise KGPlatformError(ErrorCode.SNAPSHOT_MISMATCH, "Freebase snapshot mismatch.")
        return FreebaseSparqlSession(self)

    async def aclose(self) -> None:
        await self.client.aclose()


class FreebaseSparqlSession:
    """Per-rollout Freebase view with a small entity-name cache."""
    def __init__(self, backend: FreebaseSparqlBackend):
        self.backend = backend
        self.names: dict[str, str] = {}

    def key(self, native_id: str) -> EntityKey:
        if not ID_RE.fullmatch(native_id):
            raise KGPlatformError(ErrorCode.INVALID_ARGUMENT, "Invalid Freebase entity ID.")
        return EntityKey("freebase", native_id)

    async def name(self, native_id: str) -> str:
        if native_id in self.names:
            return self.names[native_id]
        rows = await self.backend.query(
            f'''PREFIX ns: <{NS}> SELECT ?name WHERE {{ ns:{native_id} ns:type.object.name ?name . FILTER(LANG(?name) = "en" || LANG(?name) = "") }} LIMIT 1''',
            purpose="entity_name",
        )
        label = rows[0]["name"]["value"] if rows else native_id
        self.names[native_id] = label
        return label

    async def entity(self, native_id: str, label: str | None = None) -> EntityRef:
        return EntityRef(self.key(native_id), label or await self.name(native_id))

    async def _pick_by_type_count(self, candidates: list[EntityRef], label: str) -> EntityRef:
        """
        Mirror Direct-SPARQL: among exact-name hits, prefer the entity with most types.
        """
        if len(candidates) == 1:
            return candidates[0]
        values = " ".join(f"ns:{item.key.native_id}" for item in candidates)
        rows = await self.backend.query(
            f'''PREFIX ns: <{NS}> SELECT ?entity (COUNT(?type) AS ?count) WHERE {{
              VALUES ?entity {{ {values} }}
              ?entity ns:type.object.type ?type .
            }} GROUP BY ?entity''',
            purpose="resolve_rank_by_types",
        )
        counts = {
            _native(row["entity"]["value"]): int(row.get("count", {}).get("value", 0))
            for row in rows
            if row.get("entity", {}).get("value")
        }
        ranked = sorted(candidates, key=lambda item: (-counts.get(item.key.native_id, 0), item.key.native_id))
        best = ranked[0]
        self.names[best.key.native_id] = label
        return best

    async def resolve_entity(self, query: EntityQuery) -> ResolutionResult:
        raw = query.text.strip()
        # 如果输入是 MID 则直接查询
        if ID_RE.fullmatch(raw):
            rows = await self.backend.query(
                f'''PREFIX ns: <{NS}> SELECT ?predicate WHERE {{ ns:{raw} ?predicate ?object }} LIMIT 1''',
                purpose="resolve_by_mid",
            )
            return ResolutionResult(entity=await self.entity(raw)) if rows else ResolutionResult()
        if not raw or len(raw) > 200:
            return ResolutionResult()
        escaped = _escape_literal(raw)

        # 如果输入是name，则查询type.object.name
        rows = await self.backend.query(
            f'''PREFIX ns: <{NS}> SELECT DISTINCT ?entity WHERE {{
              ?entity ns:type.object.name "{escaped}"@en .
              FILTER(STRSTARTS(STR(?entity), "{NS}"))
            }} LIMIT 21''',
            purpose="resolve_by_name_en",
        )
        if not rows:
            rows = await self.backend.query(
                f'''PREFIX ns: <{NS}> SELECT DISTINCT ?entity WHERE {{
                  ?entity ns:type.object.name "{escaped}" .
                  FILTER(STRSTARTS(STR(?entity), "{NS}"))
                }} LIMIT 21''',
                purpose="resolve_by_name_plain",
            )
        candidates = []
        for row in rows:
            native_id = _native(row["entity"]["value"])
            if ID_RE.fullmatch(native_id):
                self.names[native_id] = raw
                candidates.append(await self.entity(native_id, raw))
        candidates = sorted({item.key.native_id: item for item in candidates}.values(), key=lambda item: item.key.native_id)
        if not candidates:
            return ResolutionResult()
        if len(candidates) == 1:
            return ResolutionResult(entity=candidates[0])

        # 如果有多个Mid选择到，自动选择type.object.type最多的那个Mid
        chosen = await self._pick_by_type_count(candidates, raw)
        return ResolutionResult(entity=chosen, candidates=tuple(candidates))

    async def list_relations(self, entity: EntityKey, options: RelationQueryOptions) -> RelationPage:
        native_id = self.key(entity.native_id).native_id
        entity_ref = await self.entity(native_id)
        choices: list[RelationChoice] = []
        available = 0
        for direction in sorted(options.directions, key=lambda value: value.value):
            pattern = f"ns:{native_id} ?relation ?other" if direction == Direction.OUT else f"?other ?relation ns:{native_id}"
            rows = await self.backend.query(
                f'''PREFIX ns: <{NS}> SELECT ?relation (COUNT(?other) AS ?count) WHERE {{
                  {pattern} . FILTER(STRSTARTS(STR(?relation), "{NS}"))
                }} GROUP BY ?relation ORDER BY ?relation LIMIT {options.limit + 1}''',
                purpose=f"list_relations_{direction.value}",
            )
            available += len(rows)
            for row in rows[:options.limit]:
                predicate = _native(row["relation"]["value"])
                if not PREDICATE_RE.fullmatch(predicate):
                    continue
                count = int(row.get("count", {}).get("value", 0))
                choices.append(RelationChoice(RelationRef(RelationKey("freebase", predicate), predicate), direction, count))
            remaining = max(0, options.limit - len(choices))
            if remaining:
                # Freebase often stores n-ary facts through an unnamed CVT
                # node. Expose the two-hop path as one relation so the model
                # does not have to reason about implementation-only nodes.
                if direction == Direction.OUT:
                    virtual_pattern = f"ns:{native_id} ?first ?cvt . ?cvt ?second ?other"
                else:
                    virtual_pattern = f"?other ?second ?cvt . ?cvt ?first ns:{native_id}"
                virtual_rows = await self.backend.query(
                    f'''PREFIX ns: <{NS}> SELECT ?first ?second (COUNT(?other) AS ?count) WHERE {{
                      {virtual_pattern} .
                      FILTER(isIRI(?cvt)) FILTER(isIRI(?other))
                      FILTER(STRSTARTS(STR(?first), "{NS}")) FILTER(STRSTARTS(STR(?second), "{NS}"))
                      FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?cvt_name }}
                      FILTER(?second != ns:type.object.type)
                    }} GROUP BY ?first ?second ORDER BY ?first ?second LIMIT {min(remaining, 50)}''',
                    purpose=f"list_relations_virtual_{direction.value}",
                )
                available += len(virtual_rows)
                for row in virtual_rows:
                    first = _native(row["first"]["value"])
                    second = _native(row["second"]["value"])
                    if not PREDICATE_RE.fullmatch(first) or not PREDICATE_RE.fullmatch(second):
                        continue
                    virtual_id = f"virtual:{first}|{second}"
                    label = f"{first} → {second}"
                    count = int(row.get("count", {}).get("value", 0))
                    choices.append(RelationChoice(RelationRef(RelationKey("freebase", virtual_id), label), direction, count))
        choices.sort(key=lambda item: (item.relation.key.native_id, item.direction.value))
        truncated = available > options.limit or len(choices) > options.limit
        return RelationPage(entity_ref, tuple(choices[:options.limit]), truncated, available)

    async def list_triples(self, entity: EntityKey, choices: Sequence[RelationChoice], options: TripleQueryOptions) -> TriplePage:
        native_id = self.key(entity.native_id).native_id
        entity_ref = await self.entity(native_id)
        triples: list[TripleRef] = []
        available = 0
        for choice in choices:
            predicate = choice.relation.key.native_id
            if predicate.startswith("virtual:"):
                # Virtual IDs are generated by list_relations, but validate
                # again because this is also a public backend boundary.
                path = predicate.removeprefix("virtual:").split("|")
                if len(path) != 2 or not all(PREDICATE_RE.fullmatch(value) for value in path):
                    raise KGPlatformError(ErrorCode.INVALID_ARGUMENT, "Invalid Freebase virtual relation.")
                first, second = path
                if choice.direction == Direction.OUT:
                    pattern = f"ns:{native_id} ns:{first} ?cvt . ?cvt ns:{second} ?other"
                else:
                    pattern = f"?other ns:{second} ?cvt . ?cvt ns:{first} ns:{native_id}"
                rows = await self.backend.query(
                    f'''PREFIX ns: <{NS}> SELECT ?other ?name WHERE {{
                      {pattern} . OPTIONAL {{ ?other ns:type.object.name ?name . FILTER(LANG(?name) = "en" || LANG(?name) = "") }}
                    }} LIMIT {options.limit_per_choice + 1}''',
                    purpose=f"list_triples_virtual_{choice.direction.value}",
                )
                if not rows:
                    raise KGPlatformError(ErrorCode.RELATION_NOT_APPLICABLE, "Virtual relation does not apply to this entity/direction.")
                available += len(rows)
                for row in rows[:options.limit_per_choice]:
                    value = row["other"]
                    other_native = _native(value["value"])
                    if value.get("type") != "uri" or not ID_RE.fullmatch(other_native):
                        continue
                    other = await self.entity(other_native, row.get("name", {}).get("value") or None)
                    relation = RelationRef(RelationKey("freebase", predicate), choice.relation.label)
                    head, tail = (entity_ref, other) if choice.direction == Direction.OUT else (other, entity_ref)
                    triples.append(
                        TripleRef(
                            head,
                            relation,
                            tail,
                            # Retain both physical predicates for audit/debug
                            # even though the model sees one virtual relation.
                            EdgeProvenance(self.backend.graph, virtual=True, source_path=(first, second)),
                            choice.direction,
                        )
                    )
                continue
            if not PREDICATE_RE.fullmatch(predicate):
                raise KGPlatformError(ErrorCode.INVALID_ARGUMENT, "Invalid Freebase predicate.")
            if choice.direction == Direction.OUT:
                pattern = f"ns:{native_id} ns:{predicate} ?other"
            else:
                pattern = f"?other ns:{predicate} ns:{native_id}"
            rows = await self.backend.query(
                f'''PREFIX ns: <{NS}> SELECT ?other ?name WHERE {{
                  {pattern} . OPTIONAL {{ ?other ns:type.object.name ?name . FILTER(LANG(?name) = "en" || LANG(?name) = "") }}
                }} LIMIT {options.limit_per_choice + 1}''',
                purpose=f"list_triples_{choice.direction.value}",
            )
            if not rows:
                raise KGPlatformError(ErrorCode.RELATION_NOT_APPLICABLE, "Relation does not apply to this entity/direction.")
            available += len(rows)
            for row in rows[:options.limit_per_choice]:
                value = row["other"]
                other_native = _native(value["value"])
                if value.get("type") == "uri" and ID_RE.fullmatch(other_native):
                    other = await self.entity(other_native, row.get("name", {}).get("value") or None)
                else:
                    other = LiteralRef(value.get("value", ""), value.get("datatype"), value.get("xml:lang"))
                relation = RelationRef(RelationKey("freebase", predicate), choice.relation.label)
                if choice.direction == Direction.OUT:
                    head, tail = entity_ref, other
                else:
                    # A literal cannot be the canonical head of an RDF edge.
                    if not isinstance(other, EntityRef):
                        continue
                    head, tail = other, entity_ref
                triples.append(TripleRef(head, relation, tail, EdgeProvenance(self.backend.graph), choice.direction))
        triples = triples[:options.total_limit]
        return TriplePage(entity_ref, tuple(triples), len(triples) < available, available)

    async def aclose(self) -> None:
        return None
