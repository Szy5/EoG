import json
from pathlib import Path

import pytest

from kg_platform.datasets.gtsqa import gtsqa_row_to_kg_context
from kg_platform.datasets.validation import validate_kg_contexts
from kg_platform.domain.models import GraphRef


RAW = Path("data/raw_datasets/GTSQA/synkgqa_test.json")
GRAPH = GraphRef("gtsqa", "gtsqa-ogbl-wikikg2@release-v1+sha256:eb2fe2638848deee60daf7c67aa3b0ff6bc4336198fc2b2b93c46fdc9bc653cb")


def test_all_real_gtsqa_test_rows_convert_without_answer_leakage() -> None:
    rows = json.loads(RAW.read_text())
    contexts = [gtsqa_row_to_kg_context(row, GRAPH) for row in rows]
    report = validate_kg_contexts(contexts, GRAPH)
    assert report["sample_count"] == 1622
    assert all("target_kb_id" not in context and "answer_entities" not in context for context in contexts)
    assert contexts[0]["anchors"] == [
        {"native_id": "Q2248695", "label": "The Little Colonel"},
        {"native_id": "Q919460", "label": "John Peverell Marley"},
    ]


def test_dataset_snapshot_mismatch_fails_preflight() -> None:
    context = gtsqa_row_to_kg_context(json.loads(RAW.read_text())[0], GRAPH)
    with pytest.raises(ValueError, match="mismatch"):
        validate_kg_contexts([context], GraphRef("gtsqa", "wrong"))
