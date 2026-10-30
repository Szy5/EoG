import asyncio
import json
import os
from pathlib import Path

import pytest

from kg_platform.config.loader import materialize_profile

try:
    from quarl.tool.kg_explorer_tool import RuntimeRegistry
    from verl.tools.utils.tool_registry import initialize_tools_from_config
except ModuleNotFoundError as exc:
    RuntimeRegistry = None
    initialize_tools_from_config = None
    TOOL_IMPORT_ERROR = exc
else:
    TOOL_IMPORT_ERROR = None


@pytest.mark.integration
@pytest.mark.skipif(TOOL_IMPORT_ERROR is not None, reason="optional veRL/Ray dependencies are unavailable")
def test_real_verl_two_tools_share_embedded_session_and_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resolved, _ = materialize_profile("configs/kg/profiles/wikiwebqsp.yaml", tmp_path)
    monkeypatch.setenv("KGQA_RESOLVED_CONFIG", str(resolved))
    tools = initialize_tools_from_config(
        "training/verl/examples/sglang_multiturn/config/tool_config/kg_explorer_tool_config.yaml"
    )
    tool_map = {tool.name: tool for tool in tools}
    kwargs = {
        "kg_context": {
            "schema_version": 1,
            "graph": {"graph_id": "wikiwebqsp", "snapshot_id": "wikiwebqsp@rog-selected-subgraph-v1", "schema_version": 1},
            "sample_id": "WebQTest-fixture",
            "question": "What country is Boston in?",
            "anchors": [{"native_id": "Q100", "label": "Boston", "types": ["city"]}],
            "entity_hints": [],
            "embedded_graph": {
                "entities": [
                    {"native_id": "Q100", "label": "Boston"},
                    {"native_id": "Q30", "label": "United States of America"},
                ],
                "relations": [{"native_id": "P17", "label": "country"}],
                "triples": [{"head_id": "Q100", "relation_id": "P17", "tail_id": "Q30"}],
            },
        }
    }

    async def scenario():
        instance = "verl-embedded"
        await tool_map["get_relations"].create(instance, **kwargs)
        await tool_map["get_triples"].create(instance, **kwargs)
        relation_raw, _, relation_metrics = await tool_map["get_relations"].execute(instance, {"entity": "Boston"})
        triple_raw, _, triple_metrics = await tool_map["get_triples"].execute(instance, {"entity": "Boston", "relations": ["country"]})
        runtime = tool_map["get_relations"].runtime
        assert runtime is tool_map["get_triples"].runtime
        assert runtime.manager.active_count == 1
        await tool_map["get_relations"].release(instance)
        assert runtime.manager.active_count == 1
        await tool_map["get_triples"].release(instance)
        assert runtime.manager.active_count == 0
        await RuntimeRegistry.aclose()
        return json.loads(relation_raw), relation_metrics, json.loads(triple_raw), triple_metrics

    relation, relation_metrics, triple, triple_metrics = asyncio.run(scenario())
    assert relation == {"result": "country"}
    assert relation_metrics["returned_count"] == 1
    assert triple == {"result": "[Boston, country, United States of America]"}
    assert triple_metrics["error_code"] is None
