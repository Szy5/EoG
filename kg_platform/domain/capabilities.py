from dataclasses import dataclass
from typing import Literal

from .models import Direction, GraphRef


@dataclass(frozen=True)
class BackendCapabilities:
    scope: Literal["global", "sample"]
    directions: frozenset[Direction]
    reverse_edges_materialized: bool = False
    supports_literals: bool = False
    supports_entity_text_lookup: bool = True
    supports_relation_counts: bool = True
    supports_virtual_relations: bool = False
    supports_cursor: bool = False
    stable_snapshot: bool = True


@dataclass(frozen=True)
class BackendMetadata:
    graph: GraphRef
    capabilities: BackendCapabilities
    counts: tuple[tuple[str, int], ...] = ()
