import importlib.util
import json
from pathlib import Path

import pytest

from kg_platform.adapters.global_array import GlobalArrayBackend
from kg_platform.application import KGApplicationService, KGSessionManager
from kg_platform.domain.errors import ErrorCode, KGPlatformError
from kg_platform.domain.models import (
    Direction,
    EntityKey,
    EntityQuery,
    EntityRef,
    QueryContext,
    QueryPolicy,
    RelationQueryOptions,
    TripleQueryOptions,
)

SCRIPT = Path(__file__).parents[2] / "scripts" / "build_car_kg_index.py"
SPEC = importlib.util.spec_from_file_location("build_car_kg_index", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build = MODULE.build


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def make_source(path: Path) -> tuple[Path, Path]:
    nodes = path / "nodes.jsonl"
    relationships = path / "relationships.jsonl"
    write_jsonl(
        nodes,
        [
            {"id": "n3", "name": "Shared", "type": "Feature"},
            {"id": "n1", "name": "Seed", "type": "Car"},
            {"id": "n2", "name": "Shared", "type": "Parameter"},
        ],
    )
    write_jsonl(
        relationships,
        [
            {"id": "edge-guides", "source": "n1", "relation": "guides", "target": "n2"},
            {"id": "edge-constrains", "source": "n3", "relation": "constrains", "target": "n2"},
        ],
    )
    return nodes, relationships


@pytest.mark.asyncio
async def test_car_builder_and_backend_preserve_types_direction_provenance_and_tool_access(tmp_path: Path) -> None:
    nodes, relationships = make_source(tmp_path)
    output = tmp_path / "index"
    snapshot = "car-test@v1"
    manifest = build(nodes, relationships, output, snapshot_id=snapshot)

    assert manifest["counts"] == {"nodes": 3, "edges": 2, "relations": 2}
    assert manifest["graph"] == {"graph_id": "car_kg", "snapshot_id": snapshot}
    assert "node_types.bytes.npy" in manifest["arrays"]
    assert "edge_ids.bytes.npy" in manifest["arrays"]

    backend = GlobalArrayBackend(output, verify_checksums=True)
    assert list(backend.node_ids) == ["n1", "n2", "n3"]
    assert backend.entity(0).types == ("Car",)
    assert backend.entity(0).description == "Entity type: Car"
    assert backend.node_index(EntityKey("car_kg", "n2")) == 1

    context = QueryContext(
        "car-test",
        "car-test",
        backend.graph,
        "car-test",
        anchors=(EntityRef(EntityKey("car_kg", "n1"), "Seed"),),
    )
    session = await backend.open_session(context)
    ambiguous = await session.resolve_entity(EntityQuery("Shared"))
    assert {item.key.native_id for item in ambiguous.candidates} == {"n2", "n3"}

    incoming = await session.list_relations(
        EntityKey("car_kg", "n2"),
        RelationQueryOptions(directions=frozenset({Direction.IN})),
    )
    assert {(item.relation.key.native_id, item.direction) for item in incoming.choices} == {
        ("guides", Direction.IN),
        ("constrains", Direction.IN),
    }
    constrains = next(item for item in incoming.choices if item.relation.key.native_id == "constrains")
    triples = await session.list_triples(EntityKey("car_kg", "n2"), [constrains], TripleQueryOptions())
    assert len(triples.triples) == 1
    triple = triples.triples[0]
    assert (triple.head.key.native_id, triple.tail.key.native_id) == ("n3", "n2")
    assert triple.matched_direction == Direction.IN
    assert triple.provenance.source_edge_id == "edge-constrains"

    manager = KGSessionManager(backend)
    explorer = KGApplicationService(manager, QueryPolicy(max_relations=20))
    await explorer.acquire(context)
    relation_result = await explorer.list_relations("car-test", "Seed")
    assert relation_result.error_code is None
    assert "guides" in relation_result.text
    alias = relation_result.text.splitlines()[0].split()[0]
    triple_result = await explorer.fetch_triples("car-test", "Seed", [alias])
    assert triple_result.error_code is None
    assert "Shared" in triple_result.text
    await manager.aclose()

    with pytest.raises(FileExistsError):
        build(nodes, relationships, output, snapshot_id=snapshot)


def test_car_builder_rejects_dangling_relationship_without_publishing(tmp_path: Path) -> None:
    nodes, relationships = make_source(tmp_path)
    records = [json.loads(line) for line in relationships.read_text().splitlines()]
    records[0]["target"] = "missing"
    write_jsonl(relationships, records)
    output = tmp_path / "index"

    with pytest.raises(ValueError, match="unknown target"):
        build(nodes, relationships, output)
    assert not output.exists()


def test_car_backend_checksum_verification_detects_tampering(tmp_path: Path) -> None:
    nodes, relationships = make_source(tmp_path)
    output = tmp_path / "index"
    build(nodes, relationships, output)
    payload = output / "node_labels.bytes.npy"
    original = payload.read_bytes()
    payload.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    with pytest.raises(KGPlatformError) as error:
        GlobalArrayBackend(output, verify_checksums=True)
    assert error.value.code == ErrorCode.BACKEND_UNAVAILABLE
