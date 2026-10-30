from __future__ import annotations

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


INDEX = Path(os.getenv("GTSQA_INDEX_PATH", "/tmp/kg-platform-gtsqa-index"))


@pytest.mark.integration
@pytest.mark.skipif(TOOL_IMPORT_ERROR is not None, reason="optional veRL/Ray dependencies are unavailable")
@pytest.mark.skipif(not (INDEX / "manifest.json").exists(), reason="built GTSQA index is unavailable")
def test_verl_tools_query_embedded_gtsqa_without_freebase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved, profile = materialize_profile("configs/kg/profiles/gtsqa_training.yaml", tmp_path)
    assert profile.graphs["gtsqa"].transport.kind == "embedded"
    monkeypatch.setenv("KGQA_RESOLVED_CONFIG", str(resolved))

    tools = initialize_tools_from_config(
        "training/verl/examples/sglang_multiturn/config/tool_config/kg_explorer_tool_config.yaml"
    )
    tool_map = {tool.name: tool for tool in tools}
    kwargs = {
        "qid": "gtsqa-tool-smoke",
        "question": "Which relations connect to Boston?",
        "question_entities": [{"id": "Q100", "name": "Boston"}],
    }

    async def scenario():
        instance = "gtsqa-tool-smoke"
        await tool_map["get_relations"].create(instance, **kwargs)
        await tool_map["get_triples"].create(instance, **kwargs)
        relation_raw, _, relation_metrics = await tool_map["get_relations"].execute(
            instance, {"entity": "Boston"}
        )
        relation_result = json.loads(relation_raw)["result"]
        selected_relation = relation_result.splitlines()[0]
        triple_raw, _, triple_metrics = await tool_map["get_triples"].execute(
            instance,
            {"entity": "Boston", "relations": [selected_relation]},
        )
        await tool_map["get_relations"].release(instance)
        await tool_map["get_triples"].release(instance)
        await RuntimeRegistry.aclose()
        return relation_result, relation_metrics, json.loads(triple_raw)["result"], triple_metrics

    relation_result, relation_metrics, triple_result, triple_metrics = asyncio.run(scenario())
    assert relation_result != "No relations found."
    assert relation_metrics["graph_id"] == "gtsqa"
    assert relation_metrics["returned_count"] <= 50
    assert triple_result.startswith("[Boston,") or ", Boston]" in triple_result
    assert triple_metrics["graph_id"] == "gtsqa"
    assert triple_metrics["error_code"] is None
