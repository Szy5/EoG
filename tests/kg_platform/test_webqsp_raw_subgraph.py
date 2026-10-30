from pathlib import Path

import pyarrow.parquet as pq
import pytest

from kg_platform.adapters.sample_graph import EmbeddedSampleGraphBackend
from kg_platform.application import KGApplicationService, KGSessionManager
from kg_platform.domain.models import (
    EmbeddedEntity, EmbeddedGraph, EmbeddedRelation, EmbeddedTriple, EntityKey, EntityRef, GraphRef, QueryContext,
    QueryPolicy,
)
from kg_platform.datasets.webqsp import webqsp_row_to_kg_context


RAW = Path("data/raw_datasets/WebQSP/RoG_webqsp_train_top800_selected.all_answers_in_subgraph.parquet")


def parse_context(raw: dict, session_id: str) -> QueryContext:
    graph_data = raw["graph"]
    graph = GraphRef(graph_data["graph_id"], graph_data["snapshot_id"])
    sample_id = raw["sample_id"]
    embedded = raw["embedded_graph"]
    return QueryContext(
        session_id,
        session_id,
        graph,
        sample_id,
        raw["question"],
        tuple(EntityRef(EntityKey(graph.graph_id, item["native_id"], sample_id), item["label"]) for item in raw["anchors"]),
        embedded_graph=EmbeddedGraph(
            tuple(EmbeddedEntity(item["native_id"], item["label"]) for item in embedded["entities"]),
            tuple(EmbeddedRelation(item["native_id"], item["label"]) for item in embedded["relations"]),
            tuple(EmbeddedTriple(item["head_id"], item["relation_id"], tail_id=item["tail_id"]) for item in embedded["triples"]),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not RAW.exists(), reason="raw RoG WebQSP subgraph parquet is unavailable")
async def test_real_raw_webqsp_subgraph_converter_and_roundtrip() -> None:
    row = pq.ParquetFile(RAW).read_row_group(0).slice(0, 1).to_pylist()[0]
    assert row["id"] == "WebQTrn-0"
    raw_context = webqsp_row_to_kg_context(row)
    assert len(raw_context["embedded_graph"]["triples"]) == 800
    serialized = str(raw_context)
    assert "answers" not in raw_context and "gold_paths" not in raw_context
    assert "Jaxon Bieber" in serialized  # graph fact, not a seeded resolver hint
    assert raw_context["anchors"] == [
        {"native_id": raw_context["embedded_graph"]["entities"][next(
            index for index, item in enumerate(raw_context["embedded_graph"]["entities"]) if item["label"] == "Justin Bieber"
        )]["native_id"], "label": "Justin Bieber"}
    ]

    context = parse_context(raw_context, "real-webqsp")
    manager = KGSessionManager(EmbeddedSampleGraphBackend(context.graph))
    app = KGApplicationService(manager, QueryPolicy(max_relations=200))
    await app.acquire(context)
    relations = await app.list_relations("real-webqsp", "Justin Bieber")
    assert "people.person.sibling_s" in relations.text.splitlines()
    triples = await app.fetch_triples("real-webqsp", "Justin Bieber", ["people.person.sibling_s"])
    assert triples.returned_count > 0
    assert "Justin Bieber" in triples.text
    await manager.aclose()
