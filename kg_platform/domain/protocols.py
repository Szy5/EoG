from __future__ import annotations

from typing import Protocol, Sequence

from .capabilities import BackendMetadata
from .models import (
    EntityKey,
    EntityQuery,
    QueryContext,
    RelationChoice,
    RelationPage,
    RelationQueryOptions,
    ResolutionResult,
    TriplePage,
    TripleQueryOptions,
)


class GraphSession(Protocol):
    """A query view bound to exactly one :class:`QueryContext`.

    Implementations return triples in source-canonical head/tail order;
    traversal direction belongs in ``TripleRef.matched_direction``.
    """
    async def resolve_entity(self, query: EntityQuery) -> ResolutionResult: ...

    async def list_relations(self, entity: EntityKey, options: RelationQueryOptions) -> RelationPage: ...

    async def list_triples(
        self, entity: EntityKey, choices: Sequence[RelationChoice], options: TripleQueryOptions
    ) -> TriplePage: ...

    async def aclose(self) -> None: ...


class GraphBackend(Protocol):
    """Factory and lifecycle contract implemented by every KG adapter."""
    async def metadata(self) -> BackendMetadata: ...

    async def open_session(self, context: QueryContext) -> GraphSession: ...

    async def aclose(self) -> None: ...
