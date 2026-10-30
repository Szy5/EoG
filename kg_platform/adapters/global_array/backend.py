from __future__ import annotations

import bisect
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from kg_platform.domain.capabilities import BackendCapabilities, BackendMetadata
from kg_platform.domain.errors import ErrorCode, KGPlatformError
from kg_platform.domain.models import (
    Direction,
    EdgeProvenance,
    EntityKey,
    EntityQuery,
    EntityRef,
    GraphRef,
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

REQUIRED_TEXT_COLUMNS = (
    "node_ids",
    "node_labels",
    "node_descriptions",
    "relation_ids",
    "relation_labels",
)
OPTIONAL_TEXT_COLUMNS = ("node_types", "edge_ids")
ARRAYS = (
    "edge_heads.npy",
    "edge_tails.npy",
    "edge_relation_types.npy",
    "out_order.npy",
    "out_offsets.npy",
    "in_order.npy",
    "in_offsets.npy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PackedTextColumn:
    """Read-only string sequence backed by mmap UTF-8 bytes and offsets."""

    def __init__(self, payload: np.ndarray, offsets: np.ndarray):
        self.payload = payload
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> str:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.payload[start:stop].tobytes().decode("utf-8")

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class GlobalArrayBackend:
    """Global directed graph over a mmap-friendly bidirectional array index."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        verify_checksums: bool = False,
        expected_schemas: frozenset[str] = frozenset({"kg-platform.global-array-index"}),
        expected_graph_id: str | None = None,
        backend_name: str = "Global array",
    ):
        self.index_path = Path(index_path).expanduser().resolve()
        self.backend_name = backend_name
        try:
            self.manifest = json.loads((self.index_path / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, f"{backend_name} manifest cannot be read.") from exc
        if self.manifest.get("schema") not in expected_schemas or self.manifest.get("schema_version") != 1:
            raise KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, f"Unsupported {backend_name} index manifest.")
        try:
            graph = self.manifest["graph"]
            self.graph = GraphRef(graph["graph_id"], graph["snapshot_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, f"Invalid {backend_name} graph identity.") from exc
        if expected_graph_id is not None and self.graph.graph_id != expected_graph_id:
            raise KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, f"Unexpected {backend_name} graph identity.")

        for stem in REQUIRED_TEXT_COLUMNS:
            self._load_text_column(stem, verify_checksums, required=True)
        for stem in OPTIONAL_TEXT_COLUMNS:
            self._load_text_column(stem, verify_checksums, required=False)
        for name in ARRAYS:
            spec = self.manifest.get("arrays", {}).get(name)
            path = self.index_path / name
            if not spec or not path.is_file() or path.stat().st_size != spec["bytes"]:
                raise KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, f"{backend_name} index files do not match manifest.")
            if verify_checksums and _sha256(path) != spec["sha256"]:
                raise KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, f"{backend_name} index checksum mismatch.")
            setattr(self, name.removesuffix(".npy"), np.load(path, mmap_mode="r", allow_pickle=False))
        self._validate_lengths()
        self._label_lookup: dict[str, list[int]] | None = None

    def _load_text_column(self, stem: str, verify_checksums: bool, *, required: bool) -> None:
        byte_name = f"{stem}.bytes.npy"
        offset_name = f"{stem}.offsets.npy"
        arrays = self.manifest.get("arrays", {})
        byte_spec = arrays.get(byte_name)
        offset_spec = arrays.get(offset_name)
        if not byte_spec and not offset_spec and not required:
            return
        for name, spec in ((byte_name, byte_spec), (offset_name, offset_spec)):
            path = self.index_path / name
            if not spec or not path.is_file() or path.stat().st_size != spec["bytes"]:
                raise KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, f"{self.backend_name} text columns do not match manifest.")
            if verify_checksums and _sha256(path) != spec["sha256"]:
                raise KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, f"{self.backend_name} text column checksum mismatch.")
        setattr(
            self,
            stem,
            PackedTextColumn(
                np.load(self.index_path / byte_name, mmap_mode="r", allow_pickle=False),
                np.load(self.index_path / offset_name, mmap_mode="r", allow_pickle=False),
            ),
        )

    def _validate_lengths(self) -> None:
        counts = self.manifest.get("counts", {})
        expected_nodes = counts.get("nodes")
        expected_edges = counts.get("edges")
        expected_relations = counts.get("relations")
        valid = (
            expected_nodes == len(self.node_ids) == len(self.node_labels) == len(self.node_descriptions)
            and expected_relations == len(self.relation_ids) == len(self.relation_labels)
            and expected_edges == len(self.edge_heads) == len(self.edge_tails) == len(self.edge_relation_types)
            and len(self.out_offsets) == len(self.node_ids) + 1
            and len(self.in_offsets) == len(self.node_ids) + 1
            and len(self.out_order) == len(self.edge_heads)
            and len(self.in_order) == len(self.edge_heads)
        )
        if hasattr(self, "node_types"):
            valid = valid and len(self.node_types) == len(self.node_ids)
        if hasattr(self, "edge_ids"):
            valid = valid and len(self.edge_ids) == len(self.edge_heads)
        if not valid:
            raise KGPlatformError(ErrorCode.BACKEND_UNAVAILABLE, f"{self.backend_name} index array lengths are invalid.")

    @staticmethod
    def _sorted_lookup(column: PackedTextColumn, value: str) -> int | None:
        index = bisect.bisect_left(column, value)
        return index if index < len(column) and column[index] == value else None

    def entity(self, index: int) -> EntityRef:
        if index < 0 or index >= len(self.node_ids):
            raise KGPlatformError(ErrorCode.ENTITY_NOT_FOUND, f"{self.backend_name} entity index is out of range.")
        types = (str(self.node_types[index]),) if hasattr(self, "node_types") else ()
        return EntityRef(
            EntityKey(self.graph.graph_id, str(self.node_ids[index])),
            str(self.node_labels[index]),
            types=types,
            description=str(self.node_descriptions[index]),
        )

    def relation(self, index: int) -> RelationRef:
        if index < 0 or index >= len(self.relation_ids):
            raise KGPlatformError(ErrorCode.RELATION_NOT_FOUND, f"{self.backend_name} relation index is out of range.")
        return RelationRef(RelationKey(self.graph.graph_id, str(self.relation_ids[index])), str(self.relation_labels[index]))

    def node_index(self, key: EntityKey) -> int:
        index = self._sorted_lookup(self.node_ids, key.native_id)
        if key.graph_id != self.graph.graph_id or key.scope_id is not None or index is None:
            raise KGPlatformError(ErrorCode.ENTITY_NOT_FOUND, f"Unknown {self.backend_name} entity ID.")
        return index

    async def metadata(self) -> BackendMetadata:
        counts = self.manifest["counts"]
        return BackendMetadata(
            self.graph,
            BackendCapabilities(
                scope="global",
                directions=frozenset({Direction.OUT, Direction.IN}),
                reverse_edges_materialized=False,
            ),
            tuple((key, int(value)) for key, value in counts.items()),
        )

    async def open_session(self, context: QueryContext):
        if context.graph != self.graph:
            raise KGPlatformError(ErrorCode.SNAPSHOT_MISMATCH, f"{self.backend_name} snapshot mismatch.")
        return GlobalArraySession(self)

    async def aclose(self) -> None:
        return None

    def label_lookup(self) -> dict[str, list[int]]:
        if self._label_lookup is None:
            lookup: dict[str, list[int]] = defaultdict(list)
            for index, label in enumerate(self.node_labels):
                lookup[str(label).strip().casefold()].append(index)
            self._label_lookup = dict(lookup)
        return self._label_lookup

    def edge_provenance_id(self, position: int) -> str:
        return str(self.edge_ids[position]) if hasattr(self, "edge_ids") else str(position)


class GlobalArraySession:
    def __init__(self, backend: GlobalArrayBackend):
        self.backend = backend

    async def resolve_entity(self, query: EntityQuery) -> ResolutionResult:
        raw = query.text.strip()
        node_index = self.backend._sorted_lookup(self.backend.node_ids, raw)
        if node_index is not None:
            return ResolutionResult(entity=self.backend.entity(node_index))
        candidates = tuple(self.backend.entity(index) for index in self.backend.label_lookup().get(raw.casefold(), []))
        if len(candidates) == 1:
            return ResolutionResult(entity=candidates[0])
        return ResolutionResult(candidates=candidates)

    def positions(self, node: int, direction: Direction) -> np.ndarray:
        if direction == Direction.OUT:
            return self.backend.out_order[int(self.backend.out_offsets[node]) : int(self.backend.out_offsets[node + 1])]
        return self.backend.in_order[int(self.backend.in_offsets[node]) : int(self.backend.in_offsets[node + 1])]

    async def list_relations(self, entity: EntityKey, options: RelationQueryOptions) -> RelationPage:
        node = self.backend.node_index(entity)
        counts: dict[tuple[int, Direction], int] = {}
        for direction in sorted(options.directions, key=lambda value: value.value):
            positions = self.positions(node, direction)
            values, value_counts = np.unique(self.backend.edge_relation_types[positions], return_counts=True)
            counts.update({(int(value), direction): int(count) for value, count in zip(values, value_counts, strict=True)})
        choices = [RelationChoice(self.backend.relation(rel), direction, count) for (rel, direction), count in counts.items()]
        choices.sort(key=lambda item: (item.relation.key.native_id, item.direction.value))
        return RelationPage(self.backend.entity(node), tuple(choices[: options.limit]), len(choices) > options.limit, len(choices))

    async def list_triples(
        self,
        entity: EntityKey,
        choices: Sequence[RelationChoice],
        options: TripleQueryOptions,
    ) -> TriplePage:
        node = self.backend.node_index(entity)
        records: list[tuple[int, Direction]] = []
        available = 0
        for choice in choices:
            if choice.relation.key.graph_id != self.backend.graph.graph_id:
                raise KGPlatformError(ErrorCode.RELATION_NOT_FOUND, f"Unknown {self.backend.backend_name} relation.")
            rel = self.backend._sorted_lookup(self.backend.relation_ids, choice.relation.key.native_id)
            if rel is None:
                raise KGPlatformError(ErrorCode.RELATION_NOT_FOUND, f"Unknown {self.backend.backend_name} relation.")
            positions = self.positions(node, choice.direction)
            matched = positions[self.backend.edge_relation_types[positions] == rel]
            if len(matched) == 0:
                raise KGPlatformError(ErrorCode.RELATION_NOT_APPLICABLE, "Relation does not apply to this entity/direction.")
            available += len(matched)
            records.extend((int(position), choice.direction) for position in matched[: options.limit_per_choice])
        records.sort(key=lambda item: item[0])
        records = records[: options.total_limit]
        triples = tuple(
            TripleRef(
                self.backend.entity(int(self.backend.edge_heads[position])),
                self.backend.relation(int(self.backend.edge_relation_types[position])),
                self.backend.entity(int(self.backend.edge_tails[position])),
                EdgeProvenance(self.backend.graph, self.backend.edge_provenance_id(position)),
                direction,
            )
            for position, direction in records
        )
        return TriplePage(self.backend.entity(node), triples, len(triples) < available, int(available))

    async def aclose(self) -> None:
        return None
