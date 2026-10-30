from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TypeAlias


class Direction(str, Enum):
    OUT = "out"
    IN = "in"


@dataclass(frozen=True)
class GraphRef:
    """Stable identity of a graph snapshot, never just a backend name."""
    graph_id: str
    snapshot_id: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.graph_id.strip() or not self.snapshot_id.strip():
            raise ValueError("graph_id and snapshot_id must be non-empty")
        if self.schema_version != 1:
            raise ValueError(f"unsupported graph schema version: {self.schema_version}")


@dataclass(frozen=True)
class EntityKey:
    """Backend-native entity identity; scope_id isolates sample-local graphs."""
    graph_id: str
    native_id: str
    scope_id: str | None = None


@dataclass(frozen=True)
class EntityRef:
    key: EntityKey
    label: str
    types: tuple[str, ...] = ()
    description: str | None = None
    external_ids: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LiteralRef:
    lexical: str
    datatype: str | None = None
    language: str | None = None


NodeRef: TypeAlias = EntityRef | LiteralRef


@dataclass(frozen=True)
class RelationKey:
    graph_id: str
    native_id: str


@dataclass(frozen=True)
class RelationRef:
    key: RelationKey
    label: str
    source_types: tuple[str, ...] = ()
    target_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationChoice:
    """A relation plus traversal direction offered for one focus entity."""
    relation: RelationRef
    direction: Direction
    edge_count: int | None = None


@dataclass(frozen=True)
class EdgeProvenance:
    graph: GraphRef
    source_edge_id: str | None = None
    virtual: bool = False
    source_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class TripleRef:
    """Canonical source triple and the direction used to match the focus."""
    head: EntityRef
    relation: RelationRef
    tail: NodeRef
    provenance: EdgeProvenance
    matched_direction: Direction | None = None


@dataclass(frozen=True)
class EntityQuery:
    text: str
    explicit_native_id: bool = False


@dataclass(frozen=True)
class ResolutionResult:
    entity: EntityRef | None = None
    candidates: tuple[EntityRef, ...] = ()


@dataclass(frozen=True)
class RelationQueryOptions:
    directions: frozenset[Direction] = frozenset({Direction.OUT, Direction.IN})
    limit: int = 200


@dataclass(frozen=True)
class TripleQueryOptions:
    limit_per_choice: int = 20
    total_limit: int = 60


@dataclass(frozen=True)
class RelationPage:
    entity: EntityRef
    choices: tuple[RelationChoice, ...]
    truncated: bool = False
    available_count: int | None = None


@dataclass(frozen=True)
class TriplePage:
    entity: EntityRef
    triples: tuple[TripleRef, ...]
    truncated: bool = False
    available_count: int | None = None


@dataclass(frozen=True)
class EmbeddedEntity:
    native_id: str
    label: str
    types: tuple[str, ...] = ()
    description: str | None = None


@dataclass(frozen=True)
class EmbeddedRelation:
    native_id: str
    label: str


@dataclass(frozen=True)
class EmbeddedTriple:
    head_id: str
    relation_id: str
    tail_id: str | None = None
    literal: LiteralRef | None = None

    def __post_init__(self) -> None:
        if (self.tail_id is None) == (self.literal is None):
            raise ValueError("embedded triple requires exactly one of tail_id or literal")


@dataclass(frozen=True)
class EmbeddedGraph:
    entities: tuple[EmbeddedEntity, ...]
    relations: tuple[EmbeddedRelation, ...]
    triples: tuple[EmbeddedTriple, ...]


@dataclass(frozen=True)
class QueryContext:
    """Immutable graph and dataset context bound to one rollout session."""
    session_id: str
    request_id: str
    graph: GraphRef
    sample_id: str
    question: str = ""
    anchors: tuple[EntityRef, ...] = ()
    entity_hints: tuple[EntityRef, ...] = ()
    embedded_graph: EmbeddedGraph | None = None
    deterministic_seed: int = 0


@dataclass(frozen=True)
class QueryPolicy:
    relation_candidate_limit: int = 200
    max_relations: int = 20
    max_requested_relations: int = 4
    max_triples_per_relation: int = 20
    max_triples_total: int = 60
    preserve_backend_order: bool = False


@dataclass(frozen=True)
class ToolResult:
    operation: Literal["get_relations", "get_triples"]
    text: str
    entity: EntityRef | None = None
    returned_count: int = 0
    available_count: int | None = None
    truncated: bool = False
    error_code: str | None = None
    debug: dict[str, object] = field(default_factory=dict, compare=False)
