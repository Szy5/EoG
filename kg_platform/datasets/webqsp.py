from __future__ import annotations

import hashlib


WEBQSP_SUBGRAPH_SNAPSHOT = "wikiwebqsp@rog-selected-subgraph-v1"


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def webqsp_row_to_kg_context(row: dict, *, snapshot_id: str = WEBQSP_SUBGRAPH_SNAPSHOT) -> dict:
    sample_id = str(row["id"])
    triples = row.get("selected_subgraph")
    if not isinstance(triples, list):
        raise ValueError(f"sample {sample_id} has no selected_subgraph list")
    entity_values: dict[str, str] = {}
    relation_values: dict[str, str] = {}
    embedded_triples = []
    for position, triple in enumerate(triples):
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError(f"sample {sample_id} triple {position} is not [head, relation, tail]")
        head, relation, tail = (str(value).strip() for value in triple)
        if not head or not relation or not tail:
            raise ValueError(f"sample {sample_id} triple {position} contains an empty field")
        head_id = entity_values.setdefault(head, _stable_id("e", head))
        tail_id = entity_values.setdefault(tail, _stable_id("e", tail))
        relation_id = relation_values.setdefault(relation, _stable_id("r", relation))
        embedded_triples.append({"head_id": head_id, "relation_id": relation_id, "tail_id": tail_id})
    anchors = []
    for label in row.get("topic_entities") or []:
        label = str(label).strip()
        if not label:
            continue
        native_id = entity_values.get(label)
        if native_id is None:
            raise ValueError(f"sample {sample_id} topic entity is absent from selected_subgraph: {label}")
        anchors.append({"native_id": native_id, "label": label})
    return {
        "schema_version": 1,
        "graph": {"graph_id": "wikiwebqsp", "snapshot_id": snapshot_id, "schema_version": 1},
        "sample_id": sample_id,
        "question": str(row.get("question", "")),
        "anchors": anchors,
        "entity_hints": [],
        "embedded_graph": {
            "entities": [{"native_id": native_id, "label": label} for label, native_id in sorted(entity_values.items())],
            "relations": [{"native_id": native_id, "label": label} for label, native_id in sorted(relation_values.items())],
            "triples": embedded_triples,
        },
    }
