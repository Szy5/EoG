from __future__ import annotations

from collections.abc import Iterable

from kg_platform.domain.models import GraphRef


def validate_kg_contexts(contexts: Iterable[dict], expected_graph: GraphRef) -> dict:
    seen: set[str] = set()
    count = 0
    for index, context in enumerate(contexts):
        if not isinstance(context, dict) or context.get("schema_version") != 1:
            raise ValueError(f"kg_context[{index}] has unsupported schema_version")
        graph = context.get("graph") or {}
        actual = GraphRef(str(graph.get("graph_id", "")), str(graph.get("snapshot_id", "")), int(graph.get("schema_version", 1)))
        if actual != expected_graph:
            raise ValueError(f"kg_context[{index}] graph/snapshot mismatch")
        sample_id = str(context.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"kg_context[{index}] sample_id is empty or duplicated")
        seen.add(sample_id)
        for anchor in context.get("anchors") or []:
            if not str(anchor.get("native_id", "")).strip() or not str(anchor.get("label", "")).strip():
                raise ValueError(f"kg_context[{index}] anchor requires native_id and label")
        if "answer_entities" in context or "target_kb_id" in context:
            raise ValueError(f"kg_context[{index}] contains forbidden answer resolver hints")
        count += 1
    return {"status": "valid", "sample_count": count, "graph_id": expected_graph.graph_id, "snapshot_id": expected_graph.snapshot_id}
