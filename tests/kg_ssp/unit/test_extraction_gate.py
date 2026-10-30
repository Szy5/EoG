"""Gate applied to synthesized KGQA problems before they reach the solver.

The synthesized problem inherits the seed KG path's gold answers verbatim, so the proposer's own
answer list has to actually agree with them. Under the legacy gate (>= 1 overlapping entity) a
proposer could list the gold answer alongside anything else and pass.
"""

import pytest

from quarl.utils.problem_extraction import ProblemExtractor

GOLD = ["White people"]
EXTRA_INFO = {
    "question_entities": [
        {"id": "m.01", "name": "Fiona Coyne"},
        {"id": "m.02", "name": "Sherlock Holmes"},
    ]
}


def _legacy_extractor():
    return ProblemExtractor(answer_pattern="result", synthesis_mode="kg_path")


def _tightened_extractor():
    return ProblemExtractor(
        answer_pattern="result",
        synthesis_mode="kg_path",
        gate_entity_f1_min=0.5,
        gate_reject_question_entity_answers=True,
        gate_reject_question_leaking_gold=True,
    )


@pytest.mark.unit
def test_question_leaks_gold_answer_detects_substring() -> None:
    assert ProblemExtractor.question_leaks_gold_answer("Are the characters White people?", GOLD) is True
    assert ProblemExtractor.question_leaks_gold_answer("Which ethnic groups are represented?", GOLD) is False


@pytest.mark.unit
def test_validate_question_only_payload_accepts_question_only_json() -> None:
    question, failure_reason = ProblemExtractor._validate_question_only_payload(
        {"question": "Which ethnic groups are represented?"}
    )
    assert failure_reason is None
    assert question == "Which ethnic groups are represented?"


@pytest.mark.unit
def test_validate_question_only_payload_rejects_answers_field() -> None:
    question, failure_reason = ProblemExtractor._validate_question_only_payload(
        {"question": "Which ethnic groups are represented?", "answers": ["White people"]}
    )
    assert question is None
    assert failure_reason == "format_rule_failed"


@pytest.mark.unit
def test_legacy_gate_accepts_a_single_overlapping_entity() -> None:
    padded = ["White people", "English people", "Fiona Coyne", "Sherlock Holmes"]

    assert _legacy_extractor()._check_kgqa_gate("Which ethnic groups?", padded, GOLD, EXTRA_INFO) is None


@pytest.mark.unit
def test_tightened_gate_rejects_question_entities_in_the_answer() -> None:
    padded = ["White people", "English people", "Fiona Coyne", "Sherlock Holmes"]

    reason = _tightened_extractor()._check_kgqa_gate("Which ethnic groups?", padded, GOLD, EXTRA_INFO)

    assert reason == "answer_contains_question_entity"


@pytest.mark.unit
def test_tightened_gate_rejects_low_entity_f1() -> None:
    # One of four answers matches the single gold answer: F1 = 2*(0.25*1)/1.25 = 0.4 < 0.5.
    diluted = ["White people", "English people", "Irish people", "Scottish people"]

    reason = _tightened_extractor()._check_kgqa_gate("Which ethnic groups?", diluted, GOLD, EXTRA_INFO)

    assert reason == "answer_entity_f1_below_threshold"


@pytest.mark.unit
def test_tightened_gate_rejects_a_question_that_spells_out_the_gold_answer() -> None:
    leaking = "Are the characters White people or not?"

    reason = _tightened_extractor()._check_kgqa_gate(leaking, ["White people"], GOLD, EXTRA_INFO)

    assert reason == "question_leaks_gold_answer"


@pytest.mark.unit
def test_tightened_gate_accepts_a_well_formed_problem() -> None:
    assert (
        _tightened_extractor()._check_kgqa_gate(
            "What ethnic groups are represented by the characters of that world?",
            ["White people"],
            GOLD,
            EXTRA_INFO,
        )
        is None
    )


@pytest.mark.unit
def test_gate_rejects_empty_answers() -> None:
    assert _tightened_extractor()._check_kgqa_gate("Which ethnic groups?", [], GOLD, EXTRA_INFO) == (
        "no_proposer_answers"
    )
