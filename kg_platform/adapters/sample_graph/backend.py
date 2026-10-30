from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from kg_platform.domain.capabilities import BackendCapabilities, BackendMetadata
from kg_platform.domain.errors import ErrorCode, KGPlatformError
from kg_platform.domain.models import (
    Direction,
    EdgeProvenance,
    EntityKey,
    EntityQuery,
    EntityRef,
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


class EmbeddedSampleGraphBackend:
    """Backend factory for graph fragments carried inside dataset samples."""
    def __init__(self, graph):
        self.graph = graph
        self._closed = False

    async def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            self.graph,
            BackendCapabilities(
                scope="sample",
                directions=frozenset({Direction.OUT, Direction.IN}),
                supports_literals=True,
            ),
        )

    async def open_session(self, context: QueryContext):
        # The graph payload is session-owned by design; it must never be cached
        # at backend level where another sample could observe it.
        if context.graph != self.graph:
            raise KGPlatformError(ErrorCode.SNAPSHOT_MISMATCH, "Embedded graph snapshot mismatch.")
        if context.embedded_graph is None or not context.sample_id:
            raise KGPlatformError(ErrorCode.CONTEXT_REQUIRED, "Embedded graph context is required.")
        return EmbeddedSampleGraphSession(context)

    async def aclose(self) -> None:
        self._closed = True


class EmbeddedSampleGraphSession:
    """Isolated in-memory index for one sample-local graph fragment."""
    def __init__(self, context: QueryContext):
        self.context = context
        graph = context.embedded_graph
        assert graph is not None
        self.entities = {
            item.native_id: EntityRef(
                EntityKey(context.graph.graph_id, item.native_id, context.sample_id),
                item.label,
                item.types,
                item.description,
            )
            for item in graph.entities
        }
        self.relations = {
            item.native_id: RelationRef(RelationKey(context.graph.graph_id, item.native_id), item.label)
            for item in graph.relations
        }
        self.by_label: dict[str, list[EntityRef]] = defaultdict(list)
        for entity in self.entities.values():
            self.by_label[entity.label.strip().casefold()].append(entity)
        self.out_edges: dict[str, list[tuple[int, object]]] = defaultdict(list)
        self.in_edges: dict[str, list[tuple[int, object]]] = defaultdict(list)
        for position, triple in enumerate(graph.triples):
            if triple.head_id not in self.entities or triple.relation_id not in self.relations:
                raise KGPlatformError(ErrorCode.INVALID_ARGUMENT, "Embedded triple references an unknown head/relation.")
            if triple.tail_id is not None and triple.tail_id not in self.entities:
                raise KGPlatformError(ErrorCode.INVALID_ARGUMENT, "Embedded triple references an unknown tail.")
            self.out_edges[triple.head_id].append((position, triple))
            if triple.tail_id is not None:
                self.in_edges[triple.tail_id].append((position, triple))
        self.closed = False

    def _check_key(self, key: EntityKey) -> EntityRef:
        # scope_id is part of identity for embedded graphs. Equal native IDs in
        # two dataset rows are intentionally different entities.
        if key.graph_id != self.context.graph.graph_id or key.scope_id != self.context.sample_id:
            raise KGPlatformError(ErrorCode.CONTEXT_REQUIRED, "Entity belongs to another graph scope.")
        try:
            return self.entities[key.native_id]
        except KeyError as exc:
            raise KGPlatformError(ErrorCode.ENTITY_NOT_FOUND, "Entity is absent from this sample graph.") from exc

    async def resolve_entity(self, query: EntityQuery) -> ResolutionResult:
        if query.text in self.entities:
            return ResolutionResult(entity=self.entities[query.text])
        candidates = tuple(sorted(self.by_label.get(query.text.strip().casefold(), []), key=lambda e: e.key.native_id))
        if len(candidates) == 1:
            return ResolutionResult(entity=candidates[0])
        return ResolutionResult(candidates=candidates)

    async def list_relations(self, entity: EntityKey, options: RelationQueryOptions) -> RelationPage:
        entity_ref = self._check_key(entity)
        counts: dict[tuple[str, Direction], int] = defaultdict(int)
        if Direction.OUT in options.directions:
            for _, edge in self.out_edges.get(entity.native_id, []):
                counts[(edge.relation_id, Direction.OUT)] += 1
        if Direction.IN in options.directions:
            for _, edge in self.in_edges.get(entity.native_id, []):
                counts[(edge.relation_id, Direction.IN)] += 1
        choices = [RelationChoice(self.relations[rid], direction, count) for (rid, direction), count in counts.items()]
        choices.sort(key=lambda c: (c.relation.key.native_id, c.direction.value))
        return RelationPage(entity_ref, tuple(choices[: options.limit]), len(choices) > options.limit, len(choices))

    async def list_triples(
        self, entity: EntityKey, choices: Sequence[RelationChoice], options: TripleQueryOptions
    ) -> TriplePage:
        entity_ref = self._check_key(entity)
        records: list[tuple[int, TripleRef]] = []
        available = 0
        for choice in choices:
            if choice.relation.key.native_id not in self.relations:
                raise KGPlatformError(ErrorCode.RELATION_NOT_FOUND, "Relation is unknown in this sample.")
            edges = self.out_edges.get(entity.native_id, []) if choice.direction == Direction.OUT else self.in_edges.get(entity.native_id, [])
            matched = [(position, edge) for position, edge in edges if edge.relation_id == choice.relation.key.native_id]
            if not matched:
                raise KGPlatformError(ErrorCode.RELATION_NOT_APPLICABLE, "Relation does not apply to this entity/direction.")
            available += len(matched)
            for position, edge in matched[: options.limit_per_choice]:
                # Incoming traversal selects by tail but preserves the source
                # graph's canonical head/tail order in the returned triple.
                tail = self.entities[edge.tail_id] if edge.tail_id is not None else edge.literal
                records.append(
                    (
                        position,
                        TripleRef(
                            self.entities[edge.head_id],
                            self.relations[edge.relation_id],
                            tail,
                            EdgeProvenance(self.context.graph, f"{self.context.sample_id}:{position}"),
                            choice.direction,
                        ),
                    )
                )
        records.sort(key=lambda item: item[0])
        selected = records[: options.total_limit]
        return TriplePage(entity_ref, tuple(item[1] for item in selected), len(selected) < available, available)

    async def aclose(self) -> None:
        self.closed = True
