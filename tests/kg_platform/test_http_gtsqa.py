import os
from pathlib import Path

import httpx
import pytest

from kg_platform.adapters.gtsqa import GTSQAArrayBackend
from kg_platform.adapters.http_client import RemoteKGBackend
from kg_platform.application import KGApplicationService, KGSessionManager
from kg_platform.domain.models import EntityKey, EntityRef, QueryContext, QueryPolicy
from kg_platform.service import create_app


INDEX = Path(os.getenv("GTSQA_INDEX_PATH", "/tmp/kg-platform-gtsqa-index"))


@pytest.mark.asyncio
@pytest.mark.skipif(not (INDEX / "manifest.json").exists(), reason="built GTSQA index is unavailable")
async def test_real_gtsqa_over_http_bidirectional_roundtrip() -> None:
    store = GTSQAArrayBackend(INDEX)
    remote = RemoteKGBackend("http://gtsqa.test", store.graph, transport=httpx.ASGITransport(app=create_app(store)))
    manager = KGSessionManager(remote)
    explorer = KGApplicationService(manager, QueryPolicy(max_relations=200))
    anchor = EntityRef(EntityKey("gtsqa", "Q100"), "Boston")
    await explorer.acquire(QueryContext("gtsqa-http", "gtsqa-http", store.graph, "GTSQA-http", anchors=(anchor,)))
    relations = await explorer.list_relations("gtsqa-http", "Boston")
    assert relations.returned_count > 0
    assert {choice.direction.value for choice in relations.debug["choices"]} == {"out", "in"}
    alias = relations.text.splitlines()[0]
    triples = await explorer.fetch_triples("gtsqa-http", "Boston", [alias])
    assert triples.returned_count > 0
    await manager.aclose()
