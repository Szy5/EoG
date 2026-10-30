from __future__ import annotations

from kg_platform.application.tool_options import ToolHyperparams, augmented_question, merge_hyperparams, parse_question_entities


def test_augmented_question_matches_kgqatool() -> None:
    assert augmented_question("Where was X born?", ["Portugal", "Egypt"]) == "Where was X born? Portugal Egypt"
    assert augmented_question("Q", []) == "Q"
    assert augmented_question("", ["A"]) == "A"


def test_merge_hyperparams_resolves_bm25_top_k() -> None:
    base = ToolHyperparams(max_relations=20)
    assert base.resolved_bm25_top_k() == 60
    merged = merge_hyperparams(base, {"max_relations": 10, "bm25_top_k": 15})
    assert merged.max_relations == 10
    assert merged.resolved_bm25_top_k() == 15


def test_parse_question_entities_accepts_id_name() -> None:
    rows = parse_question_entities([{"id": "m.05r4w", "name": "Portugal"}, {"native_id": "", "label": "x"}])
    assert rows == [{"native_id": "m.05r4w", "label": "Portugal"}]
