from __future__ import annotations

from kg_platform.domain.models import GraphRef


def gtsqa_row_to_kg_context(row: dict, graph: GraphRef) -> dict:
    if graph.graph_id != "gtsqa":
        raise ValueError("GTSQA rows require the gtsqa graph")
    labels = row.get("initial_entities")
    native_ids = row.get("initial_entity_ids")
    if not isinstance(labels, list) or not isinstance(native_ids, list) or len(labels) != len(native_ids):
        raise ValueError("GTSQA initial_entities and initial_entity_ids must be equal-length lists")
    anchors = []
    for label, native_id in zip(labels, native_ids, strict=True):
        label = str(label).strip()
        native_id = str(native_id).strip()
        if not label or not native_id.startswith("Q"):
            raise ValueError("GTSQA anchors require a label and QID")
        anchors.append({"native_id": native_id, "label": label})
    return {
        "schema_version": 1,
        "graph": {"graph_id": graph.graph_id, "snapshot_id": graph.snapshot_id, "schema_version": graph.schema_version},
        "sample_id": str(row["sample_id"]),
        "question": str(row.get("question", "")),
        "anchors": anchors,
        "entity_hints": [],
        "embedded_graph": None,
    }
