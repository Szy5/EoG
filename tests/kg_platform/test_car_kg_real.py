import os
from pathlib import Path

import pytest

from kg_platform.adapters.global_array import GlobalArrayBackend
from kg_platform.domain.models import (
    Direction,
    EntityKey,
    QueryContext,
    RelationQueryOptions,
    TripleQueryOptions,
)

INDEX = Path(os.getenv("CAR_KG_INDEX_PATH", "data/database/kg-platform-car-kg-index"))


@pytest.mark.asyncio
@pytest.mark.skipif(not (INDEX / "manifest.json").exists(), reason="built Car KG index is unavailable")
async def test_real_car_kg_first_benchmark_path_without_neo4j() -> None:
    backend = GlobalArrayBackend(INDEX, verify_checksums=True)
    metadata = await backend.metadata()
    assert dict(metadata.counts) == {"nodes": 10562, "edges": 22134, "relations": 11}
    anchor = EntityKey("car_kg", "meixue_5434")
    assert backend.entity(backend.node_index(anchor)).label == "Citroën Traction Avant DNA"

    session = await backend.open_session(QueryContext("car-real", "car-real", backend.graph, "CarKGQA-000001"))
    first_relations = await session.list_relations(
        anchor,
        RelationQueryOptions(directions=frozenset({Direction.OUT})),
    )
    guides = next(item for item in first_relations.choices if item.relation.key.native_id == "guides")
    first_hop = await session.list_triples(anchor, [guides], TripleQueryOptions(limit_per_choice=200, total_limit=200))

    answers: set[str] = set()
    for triple in first_hop.triples:
        middle = triple.tail.key
        relations = await session.list_relations(
            middle,
            RelationQueryOptions(directions=frozenset({Direction.IN})),
        )
        constrains = [item for item in relations.choices if item.relation.key.native_id == "constrains"]
        if not constrains:
            continue
        second_hop = await session.list_triples(
            middle,
            constrains,
            TripleQueryOptions(limit_per_choice=200, total_limit=200),
        )
        answers.update(triple.head.key.native_id for triple in second_hop.triples)

    assert {"meixue_8027", "meixue_8156", "meixue_8232"} <= answers
