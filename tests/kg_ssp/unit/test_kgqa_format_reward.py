import pytest

from quarl.reward.score.kgqa_format_reward import (
    analyze_protocol_critical_errors,
    classify_turn_action,
    compute_format_score,
    compute_score_batch,
    is_valid_answer_turn,
    is_valid_kg_query_turn,
    split_model_turns,
)
from quarl.reward.score.kgqa_protocol_gate import compute_score_batch as compute_protocol_gate_batch


@pytest.mark.unit
def test_valid_kg_query_turn() -> None:
    turn = "<think>Find the relation.</think><kg-query>{\"entity\":\"m.01\"}</kg-query>"

    assert classify_turn_action(turn) == "kg-query"
    assert is_valid_kg_query_turn(turn)


@pytest.mark.unit
def test_valid_answer_turn() -> None:
    turn = "<think>I have enough evidence.</think><answer>[\"Paris\"]</answer>"

    assert classify_turn_action(turn) == "answer"
    assert is_valid_answer_turn(turn)


@pytest.mark.unit
def test_split_model_turns_ignores_environment_information_block() -> None:
    response = (
        "<think>Find candidates.</think><kg-query>{\"entity\":\"m.01\"}</kg-query>"
        "<information>{\"rows\":[]}</information>"
        "<think>Answer from rows.</think><answer>[\"Paris\"]</answer>"
    )

    turns = split_model_turns(response)
    score = compute_format_score(response, extra_info={"split": "train"}, max_score=0.2)

    assert len(turns) == 2
    assert score["checked_turn_count"] == 2
    assert score["turn_format_valid_rate"] == pytest.approx(1.0)
    assert score["score"] == pytest.approx(0.2)


@pytest.mark.unit
def test_direct_answer_without_tool_call_is_valid() -> None:
    response = "<think>No query needed.</think><answer>[\"Paris\"]</answer>"

    score = compute_format_score(response, extra_info={"split": "train"}, max_score=0.2)

    assert score["checked_turn_count"] == 1
    assert score["score"] == pytest.approx(0.2)


@pytest.mark.unit
@pytest.mark.parametrize(
    "turn",
    [
        "<think>Missing close.</think><kg-query>{}",
        "<think>Mismatch.</think><kg-query>{}</answer>",
        "Let me think.\n<think>Extra before.</think><kg-query>{}</kg-query>",
        "<think>Extra after.</think><answer>[\"A\"]</answer> done",
        "<think>Two answers.</think><answer>A</answer><answer>B</answer>",
        "<think>Two queries.</think><kg-query>A</kg-query><kg-query>B</kg-query>",
        "<think>Old path.</think><search>Paris</search>",
        "<think>Forged info.</think><information>fake</information><answer>[\"A\"]</answer>",
    ],
)
def test_invalid_turns_get_no_format_credit(turn: str) -> None:
    score = compute_format_score(turn, extra_info={"split": "train"}, max_score=0.2)

    assert score["checked_turn_count"] >= 1
    assert score["score"] == 0.0
    assert score["turn_format_valid_rate"] == 0.0


@pytest.mark.unit
def test_zero_one_and_many_information_blocks_split_reasonably() -> None:
    zero = "<think>Done.</think><answer>[\"A\"]</answer>"
    one = (
        "<think>Q1.</think><kg-query>q1</kg-query>"
        "<information>i1</information>"
        "<think>Done.</think><answer>[\"A\"]</answer>"
    )
    many = (
        "<think>Q1.</think><kg-query>q1</kg-query>"
        "<information>i1</information>"
        "<think>Q2.</think><kg-query>q2</kg-query>"
        "<information>i2</information>"
        "<think>Done.</think><answer>[\"A\"]</answer>"
    )

    assert len(split_model_turns(zero)) == 1
    assert len(split_model_turns(one)) == 2
    assert len(split_model_turns(many)) == 3


@pytest.mark.unit
def test_information_after_answer_marks_sequence_invalid() -> None:
    response = (
        "<think>Done.</think><answer>[\"A\"]</answer>"
        "<information>unexpected</information>"
    )

    score = compute_format_score(response, extra_info={"split": "train"}, max_score=0.2)

    assert not score["information_sequence_valid"]
    assert score["score"] == 0.0


@pytest.mark.unit
def test_val_split_records_metrics_without_adding_reward() -> None:
    response = "<think>Done.</think><answer>[\"A\"]</answer>"

    score = compute_format_score(response, extra_info={"split": "val"}, max_score=0.2)

    assert score["score"] == 0.0
    assert score["format_score"] == 0.0
    assert score["measured_format_score"] == pytest.approx(0.2)
    assert not score["format_reward_applied"]


@pytest.mark.unit
def test_compute_score_batch_returns_reward_manager_fields() -> None:
    batch = [
        {
            "idx": 7,
            "response_str": "<think>Done.</think><answer>[\"A\"]</answer>",
            "extra_info": {"split": "train"},
        }
    ]

    result = compute_score_batch(batch, max_score=0.2)

    assert result == [
        {
            "idx": 7,
            "score": 0.2,
            "format_score": 0.2,
            "measured_format_score": 0.2,
            "checked_turn_count": 1,
            "valid_turn_count": 1,
            "turn_format_valid_rate": 1.0,
            "format_reward_applied": True,
            "information_sequence_valid": True,
        }
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("<think>a</think><think>b</think><answer>A</answer>", "adjacent_think_tags"),
        ("<think>a</think><answer>A</answer><answer>B</answer>", "duplicate_protocol_tags"),
        ("<think>a</think><answer>A", "unclosed_protocol_tags"),
        ("<answer>A</answer><think>a</think>", "invalid_tag_order"),
        (
            "<think>a</think><kg-query>q</kg-query><answer>A</answer>",
            "query_and_answer_same_turn",
        ),
        ("<information>x</information><think>a</think><answer>A</answer>", "invalid_information_sequence"),
    ],
)
def test_protocol_gate_rejects_critical_structure_errors(response: str, reason: str) -> None:
    diagnostics = analyze_protocol_critical_errors(response)
    result = compute_protocol_gate_batch([{"idx": 0, "response_str": response}])[0]

    assert diagnostics["critical_error"]
    assert reason in diagnostics["critical_error_reasons"]
    assert result["score"] == 0.0


@pytest.mark.unit
def test_protocol_gate_accepts_valid_multi_turn_response() -> None:
    response = (
        "<think>query</think><kg-query>q</kg-query>"
        "<information>facts</information>"
        "<think>answer</think><answer>A</answer>"
    )

    result = compute_protocol_gate_batch([{"idx": 0, "response_str": response}])[0]

    assert result["score"] == 1.0
    assert not result["critical_error"]
