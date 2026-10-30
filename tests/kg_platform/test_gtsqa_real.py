import json
import os
from pathlib import Path

import pytest

from kg_platform.adapters.gtsqa import GTSQAArrayBackend
from kg_platform.application import KGApplicationService, KGSessionManager
from kg_platform.domain.models import EntityKey, EntityRef, QueryContext, QueryPolicy


INDEX = Path(os.getenv("GTSQA_INDEX_PATH", "/tmp/kg-platform-gtsqa-index"))
RAW_TEST = Path("data/raw_datasets/GTSQA/synkgqa_test.json")


@pytest.mark.asyncio
@pytest.mark.skipif(not (INDEX / "manifest.json").exists(), reason="built GTSQA index is not available")
async def test_real_gtsqa_raw_anchor_roundtrip_and_canonical_direction() -> None:
    backend = GTSQAArrayBackend(INDEX)
    metadata = await backend.metadata()
    assert dict(metadata.counts) == {"nodes": 2240723, "edges": 15235288, "relations": 523}
    sample = json.loads(RAW_TEST.read_text())[0]
    anchors = tuple(
        EntityRef(EntityKey("gtsqa", native_id), label)
        for label, native_id in zip(sample["initial_entities"], sample["initial_entity_ids"], strict=True)
    )
    context = QueryContext(sample["sample_id"], sample["sample_id"], metadata.graph, sample["sample_id"], sample["question"], anchors)
    manager = KGSessionManager(backend)
    app = KGApplicationService(manager, QueryPolicy(max_relations=50))
    await app.acquire(context)

    entity_label = sample["initial_entities"][0]
    relations = await app.list_relations(sample["sample_id"], entity_label)
    assert relations.error_code is None
    assert relations.returned_count > 0
    selected_alias = relations.text.splitlines()[0]
    triples = await app.fetch_triples(sample["sample_id"], entity_label, [selected_alias])
    assert triples.error_code is None
    assert triples.returned_count > 0
    for triple in triples.debug["triples"]:
        focus = sample["initial_entity_ids"][0]
        assert focus in (triple.head.key.native_id, triple.tail.key.native_id)
        if triple.matched_direction.value == "out":
            assert triple.head.key.native_id == focus
        else:
            assert triple.tail.key.native_id == focus
    await manager.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(not (INDEX / "manifest.json").exists(), reason="built GTSQA index is not available")
async def test_real_gtsqa_q100_has_both_directions_and_is_deterministic() -> None:
    backend = GTSQAArrayBackend(INDEX)
    manager = KGSessionManager(backend)
    app = KGApplicationService(manager, QueryPolicy(max_relations=200))
    anchor = EntityRef(EntityKey("gtsqa", "Q100"), "Boston")
    await app.acquire(QueryContext("boston", "boston", backend.graph, "boston", anchors=(anchor,)))
    first = await app.list_relations("boston", "Boston")
    second = await app.list_relations("boston", "Boston")
    assert first.text == second.text
    choices = second.debug["choices"]
    assert {choice.direction.value for choice in choices} == {"out", "in"}
    await manager.aclose()
