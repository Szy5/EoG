from __future__ import annotations

import pytest

from kg_platform.adapters.sample_graph import EmbeddedSampleGraphBackend
from kg_platform.application import KGApplicationService, KGSessionManager
from kg_platform.domain.errors import ErrorCode, KGPlatformError
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


GRAPH = GraphRef("wikiwebqsp", "wikiwebqsp@test")


def context(session_id: str = "session-1", sample_id: str = "sample-1") -> QueryContext:
    embedded = EmbeddedGraph(
        entities=(
            EmbeddedEntity("Q100", "Boston", ("city",)),
            EmbeddedEntity("Q30", "United States of America", ("country",)),
            EmbeddedEntity("Q60", "New York", ("city",)),
            EmbeddedEntity("Q999", "Isolated", ("concept",)),
        ),
        relations=(
            EmbeddedRelation("P17", "country"),
            EmbeddedRelation("P47", "shares border with"),
            EmbeddedRelation("P571", "inception"),
        ),
        triples=(
            EmbeddedTriple("Q100", "P17", tail_id="Q30"),
            EmbeddedTriple("Q60", "P47", tail_id="Q100"),
            EmbeddedTriple("Q100", "P571", literal=LiteralRef("1630", "xsd:gYear")),
        ),
    )
    anchor = EntityRef(EntityKey("wikiwebqsp", "Q100", sample_id), "Boston", ("city",))
    return QueryContext(session_id, session_id, GRAPH, sample_id, "Tell me about Boston", (anchor,), embedded_graph=embedded)


@pytest.mark.asyncio
async def test_label_first_multihop_directions_literals_and_generation() -> None:
    manager = KGSessionManager(EmbeddedSampleGraphBackend(GRAPH))
    app = KGApplicationService(manager, QueryPolicy(max_relations=20))
    await app.acquire(context())

    relations = await app.list_relations("session-1", "bOSton")
    assert relations.error_code is None
    assert relations.text.splitlines() == ["country", "inception", "shares border with"]
    assert relations.returned_count == 3

    triples = await app.fetch_triples("session-1", "Boston", ["country", "shares border with", "inception"])
    assert triples.error_code is None
    assert triples.text.splitlines() == [
        "[Boston, country, United States of America]",
        "[New York, shares border with, Boston]",
        "[Boston, inception, 1630 [literal:xsd:gYear]]",
    ]

    discovered = await app.list_relations("session-1", "United States of America")
    assert discovered.text == "country"
    incoming = await app.fetch_triples("session-1", "United States of America", ["country"])
    assert incoming.text == "[Boston, country, United States of America]"

    stale = await app.fetch_triples("session-1", "Boston", ["country"])
    assert stale.error_code is None  # another entity's generation does not invalidate Boston
    await app.list_relations("session-1", "Boston")
    invalid = await app.fetch_triples("session-1", "Boston", ["not issued"])
    assert invalid.error_code == ErrorCode.STALE_RELATION_CHOICE.value

    await app.release("session-1")
    assert manager.active_count == 0
    await app.release("session-1")
    await manager.aclose()


@pytest.mark.asyncio
async def test_scope_isolation_context_digest_and_snapshot_mismatch() -> None:
    backend = EmbeddedSampleGraphBackend(GRAPH)
    manager = KGSessionManager(backend)
    await manager.acquire(context())
    with pytest.raises(KGPlatformError, match="Session context changed") as caught:
        await manager.acquire(context(sample_id="sample-2"))
    assert caught.value.code == ErrorCode.SESSION_CONTEXT_MISMATCH

    wrong_graph = QueryContext("wrong", "wrong", GraphRef("wikiwebqsp", "other"), "sample", embedded_graph=context().embedded_graph)
    with pytest.raises(KGPlatformError) as mismatch:
        await manager.acquire(wrong_graph)
    assert mismatch.value.code == ErrorCode.SNAPSHOT_MISMATCH
    await manager.aclose()


@pytest.mark.asyncio
async def test_ambiguous_entity_never_selects_first_candidate() -> None:
    graph = EmbeddedGraph(
        entities=(
            EmbeddedEntity("Q1", "Springfield", ("city",), "Illinois"),
            EmbeddedEntity("Q2", "Springfield", ("city",), "Massachusetts"),
        ),
        relations=(),
        triples=(),
    )
    manager = KGSessionManager(EmbeddedSampleGraphBackend(GRAPH))
    app = KGApplicationService(manager)
    await app.acquire(QueryContext("amb", "amb", GRAPH, "amb-sample", embedded_graph=graph))
    result = await app.list_relations("amb", "Springfield")
    assert result.error_code == ErrorCode.ENTITY_AMBIGUOUS.value
    managed = manager.get("amb")
    assert managed.resolve_registered_entity("Springfield [Illinois]").key.native_id == "Q1"
    assert managed.resolve_registered_entity("Springfield [Massachusetts]").key.native_id == "Q2"
    await manager.aclose()


@pytest.mark.asyncio
async def test_existing_entity_with_empty_adjacency_differs_from_missing() -> None:
    manager = KGSessionManager(EmbeddedSampleGraphBackend(GRAPH))
    app = KGApplicationService(manager)
    await app.acquire(context())
    empty = await app.list_relations("session-1", "Q999")
    assert empty.error_code is None
    assert empty.text == "No relations found."

    missing = await app.list_relations("session-1", "does-not-exist")
    assert missing.error_code == ErrorCode.ENTITY_NOT_FOUND.value
    await manager.aclose()
