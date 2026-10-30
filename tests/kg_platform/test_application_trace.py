from __future__ import annotations

import pytest

from kg_platform.adapters.sample_graph import EmbeddedSampleGraphBackend
from kg_platform.application import KGApplicationService, KGSessionManager
from kg_platform.domain.models import (
    EmbeddedEntity,
    EmbeddedGraph,
    EmbeddedRelation,
    EmbeddedTriple,
    EntityKey,
    EntityRef,
    GraphRef,
    LiteralRef,
    QueryContext,
    QueryPolicy,
)


GRAPH = GraphRef("debug_sample", "debug_sample@v1")


def _context(session_id: str = "trace-1") -> QueryContext:
    embedded = EmbeddedGraph(
        entities=(
            EmbeddedEntity("Q100", "Boston", ("city",)),
            EmbeddedEntity("Q30", "United States of America", ("country",)),
            EmbeddedEntity("Q60", "New York", ("city",)),
            EmbeddedEntity("Q1", "Alpha", ("city",)),
            EmbeddedEntity("Q2", "Beta", ("city",)),
            EmbeddedEntity("Q3", "Gamma", ("city",)),
        ),
        relations=(
            EmbeddedRelation("P17", "country"),
            EmbeddedRelation("P47", "shares border with"),
            EmbeddedRelation("P571", "inception"),
            EmbeddedRelation("P1", "rel_one"),
            EmbeddedRelation("P2", "rel_two"),
            EmbeddedRelation("P3", "rel_three"),
        ),
        triples=(
            EmbeddedTriple("Q100", "P17", tail_id="Q30"),
            EmbeddedTriple("Q60", "P47", tail_id="Q100"),
            EmbeddedTriple("Q100", "P571", literal=LiteralRef("1630", "xsd:gYear")),
            EmbeddedTriple("Q100", "P1", tail_id="Q1"),
            EmbeddedTriple("Q100", "P2", tail_id="Q2"),
            EmbeddedTriple("Q100", "P3", tail_id="Q3"),
        ),
    )
    anchor = EntityRef(EntityKey("debug_sample", "Q100", "sample-1"), "Boston", ("city",))
    return QueryContext(session_id, session_id, GRAPH, "sample-1", "Tell me about Boston", (anchor,), embedded_graph=embedded)


@pytest.mark.asyncio
async def test_list_relations_trace_shows_policy_drop() -> None:
    manager = KGSessionManager(EmbeddedSampleGraphBackend(GRAPH))
    app = KGApplicationService(manager, QueryPolicy(max_relations=2))
    await app.acquire(_context())

    result = await app.list_relations("trace-1", "Boston")
    assert result.error_code is None
    assert result.returned_count == 2
    assert result.available_count is not None and result.available_count >= 2
    assert result.text.splitlines()  # agent text still present

    trace = result.debug["trace"]
    stage_ids = [stage["id"] for stage in trace["stages"]]
    assert stage_ids == [
        "S0_resolve",
        "S1_backend_raw",
        "S2_platform_policy",
        "S3_optional_rank_filter",
        "S4_final",
    ]
    by_id = {stage["id"]: stage for stage in trace["stages"]}
    assert by_id["S1_backend_raw"]["counts"]["available"] > by_id["S4_final"]["counts"]["returned"]
    assert by_id["S3_optional_rank_filter"]["status"] == "skipped"
    assert trace["summary"]["bottleneck_hint"] == "S2_platform_policy"
    assert trace["summary"]["drop_counts"]["S1_to_S2"] >= 1

    await manager.aclose()


@pytest.mark.asyncio
async def test_list_relations_text_contract_unchanged_with_trace() -> None:
    manager = KGSessionManager(EmbeddedSampleGraphBackend(GRAPH))
    app = KGApplicationService(manager, QueryPolicy(max_relations=20))
    await app.acquire(_context())
    result = await app.list_relations("trace-1", "Boston")
    assert "country" in result.text
    assert "trace" in result.debug
    await manager.aclose()
