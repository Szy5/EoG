import os

import pytest

from kg_platform.adapters.freebase import FreebaseSparqlBackend
from kg_platform.application import KGApplicationService, KGSessionManager
from kg_platform.domain.models import EntityKey, EntityRef, QueryContext, QueryPolicy, RelationQueryOptions, TripleQueryOptions


ENDPOINT = os.getenv("KGQA_SPARQL_ENDPOINT", "http://127.0.0.1:3001/sparql")


@pytest.mark.asyncio
@pytest.mark.network
async def test_real_freebase_jamaica_tool_roundtrip() -> None:
    backend = FreebaseSparqlBackend(ENDPOINT, timeout=20)
    anchor = EntityRef(EntityKey("freebase", "m.03_r3"), "Jamaica")
    manager = KGSessionManager(backend)
    app = KGApplicationService(manager, QueryPolicy(max_relations=200, max_triples_per_relation=20))
    await app.acquire(QueryContext("freebase-real", "freebase-real", backend.graph, "WebQTest-0", "what does jamaican people speak?", (anchor,)))
    relations = await app.list_relations("freebase-real", "Jamaica")
    assert relations.error_code is None
    assert "location.country.languages_spoken" in relations.text.splitlines()
    triples = await app.fetch_triples("freebase-real", "Jamaica", ["location.country.languages_spoken"])
    assert triples.error_code is None
    assert any("Jamaican" in line for line in triples.text.splitlines())
    await manager.aclose()


@pytest.mark.asyncio
@pytest.mark.network
async def test_real_freebase_cvt_is_backend_private_virtual_relation() -> None:
    backend = FreebaseSparqlBackend(ENDPOINT, timeout=20)
    session = await backend.open_session(QueryContext("cvt", "cvt", backend.graph, "WebQTrn-0"))
    page = await session.list_relations(EntityKey("freebase", "m.06w2sn5"), RelationQueryOptions(limit=200))
    choices = [
        choice for choice in page.choices
        if choice.relation.key.native_id.startswith("virtual:") and "sibling" in choice.relation.key.native_id
    ]
    assert choices
    triples = await session.list_triples(EntityKey("freebase", "m.06w2sn5"), choices[:1], TripleQueryOptions())
    assert any(getattr(item.head, "label", "") in {"Jazmyn Bieber", "Jaxon Bieber"} for item in triples.triples)
    assert all(item.provenance.virtual and len(item.provenance.source_path) == 2 for item in triples.triples)
    await backend.aclose()
