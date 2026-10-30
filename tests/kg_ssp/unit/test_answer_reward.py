import pytest

from quarl.reward.score.kgqa_answer_em import compute_score_batch, parse_predicted_answers


@pytest.mark.unit
def test_kgqa_answer_reward_accepts_exact_multi_answer_set() -> None:
    results = compute_score_batch(
        [
            {
                "idx": 0,
                "response_str": '<answer>["Oxybutynin Chloride Oral", "Oxybutynin Oral"]</answer>',
                "ground_truth": {"target": ["Oxybutynin Oral", "Oxybutynin Chloride Oral"]},
            }
        ]
    )

    assert results[0]["score"] == 1.0
    assert results[0]["answer_correct"] is True
    assert results[0]["any_overlap"] == 1.0
    assert results[0]["hit_at_1"] == 1.0
    assert results[0]["token_f1"] == 1.0
    assert results[0]["entity_f1"] == 1.0


@pytest.mark.unit
def test_kgqa_answer_reward_gives_partial_credit_for_partial_overlap() -> None:
    results = compute_score_batch(
        [
            {
                "idx": 0,
                "response_str": '<answer>["Oxybutynin Oral"]</answer>',
                "ground_truth": {"target": ["Oxybutynin Oral", "Oxybutynin Chloride Oral"]},
            }
        ]
    )

    # The score is entity F1, so recovering one of two gold answers is worth 2/3, not full marks.
    assert results[0]["score"] == pytest.approx(2 / 3)
    assert results[0]["answer_correct"] is True
    assert results[0]["any_overlap"] == 1.0
    assert results[0]["hit_at_1"] == 1.0
    assert results[0]["token_f1"] == pytest.approx(0.8)
    assert results[0]["entity_f1"] == pytest.approx(2 / 3)


@pytest.mark.unit
def test_kgqa_answer_reward_penalises_padding_the_answer_list() -> None:
    """Padding the prediction set must cost score — that is the whole point of moving off overlap.

    Under the previous "at least one overlapping entity" score both rows below scored 1.0, so the
    policy could pad its answer list for free.
    """
    results = compute_score_batch(
        [
            {
                "idx": 0,
                "response_str": '<answer>["Paris"]</answer>',
                "ground_truth": {"target": ["Paris"]},
            },
            {
                "idx": 1,
                "response_str": '<answer>["Paris", "London", "Berlin", "Rome"]</answer>',
                "ground_truth": {"target": ["Paris"]},
            },
        ]
    )

    assert results[0]["score"] == 1.0
    assert results[1]["score"] == pytest.approx(0.4)
    # The legacy signal cannot tell them apart, which is exactly why it was replaced.
    assert results[0]["any_overlap"] == results[1]["any_overlap"] == 1.0


@pytest.mark.unit
def test_kgqa_answer_reward_accepts_duplicate_predictions_when_gold_has_single_answer() -> None:
    results = compute_score_batch(
        [
            {
                "idx": 0,
                "response_str": '<answer>["Paris", "Paris"]</answer>',
                "ground_truth": {"target": ["Paris"]},
            }
        ]
    )

    assert results[0]["score"] == 1.0
    assert results[0]["answer_correct"] is True


@pytest.mark.unit
def test_kgqa_answer_metrics_are_zero_when_answers_do_not_match() -> None:
    results = compute_score_batch(
        [
            {
                "idx": 0,
                "response_str": '<answer>["London"]</answer>',
                "ground_truth": {"target": ["Paris"]},
            }
        ]
    )

    assert results[0]["score"] == 0.0
    assert results[0]["answer_correct"] is False
    assert results[0]["any_overlap"] == 0.0
    assert results[0]["hit_at_1"] == 0.0
    assert results[0]["token_f1"] == 0.0
    assert results[0]["entity_f1"] == 0.0


@pytest.mark.unit
def test_parse_predicted_answers_supports_plain_string_payload() -> None:
    assert parse_predicted_answers("<answer>Paris</answer>") == ["Paris"]


@pytest.mark.unit
def test_parse_predicted_answers_ignores_answer_tag_in_graphwalker_continuation() -> None:
    transcript = """<think>query first</think><kg-query>get_relations(\"Paris\")</kg-query>
user
<information>Here are the query results:\nParis</information>

Review the results. If the answer is found, provide it in <answer> tags.
assistant
<think>done</think><answer>[\"Paris\"]</answer>"""

    assert parse_predicted_answers(transcript) == ["Paris"]
