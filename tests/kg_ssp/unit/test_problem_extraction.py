import json
from pathlib import Path

import pytest

from quarl.utils.problem_extraction import (
    ProblemExtractor,
    extract_problems_batch,
    render_structured_react_trajectory,
)
from quarl.utils.llm_as_a_judge import evaluate_kgqa_triple_answer, parse_kgqa_triple_answer


TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "training" / "quarl" / "template"


class _FakeJudge:
    def __init__(self, answer: str):
        self.answer = answer
        self.model = "fake-judge"

    def model_based_answer(self, materials, question) -> str:
        return f"Answer: {self.answer}"

    def model_based_match(self, question, golden_answer, model_answer) -> bool:
        return False

    def model_based_kgqa_triple_answer(self, **kwargs) -> str:
        if isinstance(self.answer, list):
            return f"Answer: {json.dumps(self.answer)}"
        return f'Answer: ["{self.answer}"]'


class _FailingJudge(_FakeJudge):
    def model_based_kgqa_triple_answer(self, **kwargs) -> str:
        raise RuntimeError("synthetic verifier outage")


@pytest.mark.unit
def test_solver_record_loads_and_renders_canonical_jinja_prompts() -> None:
    extractor = ProblemExtractor(answer_pattern="result")
    record = extractor.build_kgqa_solver_record(
        question="Who founded Microsoft?",
        question_entities=[{"name": "Microsoft", "id": "m.04sv4"}],
        answer_targets=["Bill Gates"],
        answer_entities=[{"name": "Bill Gates", "id": "m.01g20"}],
        data_source="kgqa-test",
        dataset_type="webqsp",
        qid="demo",
    )

    expected_system = (TEMPLATE_DIR / "kgqa_ssp_solver_sys.jinja").read_text(encoding="utf-8").strip()
    assert record["prompt"][0] == {"role": "system", "content": expected_system}
    assert record["prompt"][1] == {
        "role": "user",
        "content": "Question: Who founded Microsoft?\nInitial Entities: Microsoft",
    }
    assert "{{" not in record["prompt"][1]["content"]


@pytest.mark.unit
def test_problem_extractor_builds_kgqa_problem_from_result_payload() -> None:
    extractor = ProblemExtractor(answer_pattern="result")
    trajectory = {
        "input": "seed prompt",
        "output": """
        <think>reasoning</think>
        <result>
        {"question": "Which medicine is marketed in this formulation?"}
        </result>
        """,
        "metadata": {
            "trajectory_index": 7,
            "data_source": "kgqa_svts_freebase",
            "ability": "kgqa-self-play",
            "reward_model": {
                "ground_truth": {"target": ["Oxybutynin Oral", "Oxybutynin Chloride Oral"]},
                "style": "rule",
            },
            "extra_info": {
                "qid": 2101535001000,
                "split": "train",
                "dataset_type": "grailqa",
                "question": "placeholder_question_to_be_generated",
                "source_question": "Which drug formulation is associated with this source path?",
                "question_entities": [
                    {"name": "Oxybutynin chloride 5 extended release film coated tablet", "id": "m.0hqs1x_"}
                ],
                "answer_entities": [
                    {"name": "Oxybutynin Oral", "id": "m.0z3xfvs"},
                    {"name": "Oxybutynin Chloride Oral", "id": "m.0z3xm0m"},
                ],
                "path": [
                    "Oxybutynin chloride 5 extended release film coated tablet <--medicine.routed_drug.marketed_formulations-- Ans"
                ],
                "triples": [["Oxybutynin chloride 5 extended release film coated tablet", "medicine.routed_drug.marketed_formulations", "Ans"]],
                "sparql_query": "SELECT ?x WHERE {}",
                "tools_kwargs": {
                    "get_relations": {
                        "create_kwargs": {
                            "question": "Which drug formulation is associated with this source path?"
                        }
                    },
                    "get_triples": {
                        "create_kwargs": {
                            "question": "Which drug formulation is associated with this source path?"
                        }
                    },
                },
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert stats["successful_problems_count"] == 1
    assert len(problems) == 1
    problem = problems[0]
    assert problem["problem_type"] == "kgqa"
    assert problem["trajectory_index"] == 7
    assert problem["reward_model"]["ground_truth"]["target"] == [
        "Oxybutynin Oral",
        "Oxybutynin Chloride Oral",
    ]
    assert problem["extra_info"]["question"] == "Which medicine is marketed in this formulation?"
    assert problem["extra_info"]["source_question"] == (
        "Which drug formulation is associated with this source path?"
    )
    assert problem["extra_info"]["source_gold_answers"] == [
        "Oxybutynin Oral",
        "Oxybutynin Chloride Oral",
    ]
    assert "proposer_answers" not in problem["extra_info"]
    assert sorted(problem["extra_info"]["tools_kwargs"].keys()) == ["get_relations", "get_triples"]
    assert problem["extra_info"]["tools_kwargs"]["get_relations"]["create_kwargs"]["qid"] == 2101535001000
    assert problem["extra_info"]["tools_kwargs"]["get_relations"]["create_kwargs"]["question_entities"] == [
        {"name": "Oxybutynin chloride 5 extended release film coated tablet", "id": "m.0hqs1x_"}
    ]
    assert (
        problem["extra_info"]["tools_kwargs"]["get_relations"]["create_kwargs"]["question"]
        == "Which medicine is marketed in this formulation?"
    )
    assert (
        problem["extra_info"]["tools_kwargs"]["get_triples"]["create_kwargs"]["question"]
        == "Which medicine is marketed in this formulation?"
    )
    assert "<kg-query>" in problem["prompt"][0]["content"]
    assert "Initial Entities:" in problem["prompt"][1]["content"]
    assert "Oxybutynin chloride 5 extended release film coated tablet" in problem["prompt"][1]["content"]


@pytest.mark.unit
def test_problem_extractor_keeps_legacy_question_path() -> None:
    extractor = ProblemExtractor(answer_pattern="question")
    trajectory = {
        "input": "seed prompt",
        "output": """
        <think>reasoning</think>
        <search>first search</search>
        <information>material one</information>
        <search>second search</search>
        <information>material two</information>
        <question>Which city hosted the international cultural event after the committee changed the final venue decision?</question>
        """,
        "metadata": {
            "trajectory_index": 3,
            "data_source": "legacy_search_selfplay",
            "reward_model": {"ground_truth": {"target": ["Paris"]}, "style": "rule"},
            "extra_info": {"split": "train"},
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert stats["successful_problems_count"] == 1
    assert len(problems) == 1
    assert problems[0]["problem_type"] == "search"
    assert problems[0]["extra_info"]["tools_kwargs"]["search"]["create_kwargs"]["question"] == (
        "Which city hosted the international cultural event after the committee changed the final venue decision?"
    )


@pytest.mark.unit
def test_problem_extractor_rejects_invalid_result_payload_without_matching_predicted_answers() -> None:
    extractor = ProblemExtractor(answer_pattern="result")
    trajectory = {
        "input": "seed prompt",
        "output": "<think>reasoning</think><result>{not valid json}</result>",
        "metadata": {
            "trajectory_index": 9,
            "data_source": "kgqa_svts_freebase",
            "ability": "kgqa-self-play",
            "reward_model": {"ground_truth": {"target": ["placeholder"]}, "style": "rule"},
            "extra_info": {
                "qid": 1,
                "split": "train",
                "dataset_type": "grailqa",
                "source_question": "Which formulation markets this medicine?",
                "answer_entities": [{"name": "Answer A", "id": "m.answer"}],
                "question_entities": [{"name": "Entity A", "id": "m.entity"}],
                "path": ["Entity A --demo.rel--> Answer A"],
                "triples": [["Entity A", "demo.rel", "Answer A"]],
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert stats["successful_problems_count"] == 0
    assert len(problems) == 0
    assert stats["format_error_count"] == 1
    assert stats["failure_reasons"] == {"format_rule_failed": 1}


@pytest.mark.unit
def test_problem_extractor_rejects_payload_when_answers_field_is_present() -> None:
    extractor = ProblemExtractor(answer_pattern="result")
    trajectory = {
        "input": "seed prompt",
        "output": """
        <think>reasoning</think>
        <result>
        {"question": "Which formulation markets this medicine?", "answers": ["Wrong Answer"]}
        </result>
        """,
        "metadata": {
            "trajectory_index": 10,
            "data_source": "kgqa_svts_freebase",
            "ability": "kgqa-self-play",
            "reward_model": {"ground_truth": {"target": ["Answer A"]}, "style": "rule"},
            "extra_info": {
                "qid": 1,
                "split": "train",
                "dataset_type": "grailqa",
                "source_question": "Which formulation markets this medicine?",
                "answer_entities": [{"name": "Answer A", "id": "m.answer"}],
                "question_entities": [{"name": "Entity A", "id": "m.entity"}],
                "path": ["Entity A --demo.rel--> Answer A"],
                "triples": [["Entity A", "demo.rel", "Answer A"]],
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert problems == []
    assert stats["successful_problems_count"] == 0
    assert stats["format_error_count"] == 1
    assert stats["failure_reasons"] == {"format_rule_failed": 1}


@pytest.mark.unit
def test_problem_extractor_accepts_gtsqa_question_only_result_using_path_gold() -> None:
    extractor = ProblemExtractor(answer_pattern="result", synthesis_mode="kg_path")
    trajectory = {
        "input": "seed prompt",
        "output": """
        <think>The path already identifies the answer entity.</think>
        <result>{"question": "Which municipality shares a border with Gorduno and Arbedo-Castione?"}</result>
        """,
        "metadata": {
            "trajectory_index": 12,
            "data_source": "kgqa_svts_gtsqa",
            "ability": "kgqa-self-play",
            "reward_model": {"ground_truth": {"target": ["Gnosca"]}, "style": "rule"},
            "extra_info": {
                "qid": 8000000012267,
                "split": "train",
                "dataset_type": "gtsqa",
                "source_question": "placeholder_question_to_be_generated",
                "answer_entities": [{"name": "Gnosca", "id": "Q67754"}],
                "question_entities": [
                    {"name": "Gorduno", "id": "Q67523"},
                    {"name": "Arbedo-Castione", "id": "Q67021"},
                ],
                "path": [
                    "Gorduno <--shares border with-- Ans@",
                    "Arbedo-Castione <--shares border with-- Ans@",
                ],
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert stats["successful_problems_count"] == 1
    assert stats["failure_reasons"] == {}
    assert stats["pass_count"] == 1
    assert stats["trajectory_outcomes"][0]["failure_reason"] is None
    problem = problems[0]
    assert problem["extra_info"]["question"] == (
        "Which municipality shares a border with Gorduno and Arbedo-Castione?"
    )
    assert problem["reward_model"]["ground_truth"]["target"] == ["Gnosca"]
    assert problem["extra_info"]["source_gold_answers"] == ["Gnosca"]
    assert "proposer_answers" not in problem["extra_info"]


@pytest.mark.unit
def test_problem_extractor_rejects_question_that_leaks_path_gold() -> None:
    extractor = ProblemExtractor(answer_pattern="result", synthesis_mode="kg_path")
    trajectory = {
        "input": "seed prompt",
        "output": '<result>{"question": "Is Gnosca the shared neighbor?"}</result>',
        "metadata": {
            "trajectory_index": 13,
            "data_source": "kgqa_svts_gtsqa",
            "ability": "kgqa-self-play",
            "reward_model": {"ground_truth": {"target": ["Gnosca"]}, "style": "rule"},
            "extra_info": {
                "qid": 13,
                "split": "train",
                "dataset_type": "gtsqa",
                "answer_entities": [{"name": "Gnosca", "id": "Q67754"}],
                "question_entities": [{"name": "Gorduno", "id": "Q67523"}],
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert problems == []
    assert stats["valid_questions_count"] == 1
    assert stats["successful_problems_count"] == 0
    assert stats["failure_reasons"] == {"question_leaks_gold_answer": 1}


@pytest.mark.unit
def test_problem_extractor_accepts_kg_explore_rag_with_evidence_judge() -> None:
    extractor = ProblemExtractor(
        answer_pattern="result",
        synthesis_mode="kg_explore_rag",
        use_rag_filter=True,
    )
    extractor.llm_judge = _FakeJudge("Discovered Answer")
    trajectory = {
        "input": "seed prompt",
        "output": """
        <think>I will inspect the initial entity.</think>
        <kg-query>get_relations("Entity A")</kg-query>
        <information>["demo.relation"]</information>
        <think>The relation reveals the answer.</think>
        <kg-query>get_triples("Entity A", ["demo.relation"])</kg-query>
        <information>[{"head": "Entity A", "relation": "demo.relation", "tail": "Discovered Answer"}]</information>
        <result>
        {
          "question": "Which discovered entity is connected to Entity A by the demo relation?",
          "answers": ["Discovered Answer"],
          "question_entities": [{"name": "Entity A", "id": "m.entity"}]
        }
        </result>
        """,
        "metadata": {
            "trajectory_index": 11,
            "data_source": "kgqa_svts_freebase",
            "ability": "kgqa-explore-rag-self-play",
            "reward_model": {"ground_truth": {"target": ["Source Gold"]}, "style": "rule"},
            "extra_info": {
                "qid": 2,
                "split": "train",
                "dataset_type": "grailqa",
                "synthesis_mode": "kg_explore_rag",
                "source_question": "Source question",
                "answer_entities": [{"name": "Source Gold", "id": "m.source"}],
                "question_entities": [{"name": "Entity A", "id": "m.entity"}],
                "path": ["Entity A --source.rel--> Source Gold"],
                "triples": [["Entity A", "source.rel", "Source Gold"]],
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert stats["successful_problems_count"] == 1
    assert len(problems) == 1
    problem = problems[0]
    assert problem["problem_type"] == "kg_explore_rag"
    assert problem["reward_model"]["ground_truth"]["target"] == ["Discovered Answer"]
    assert problem["extra_info"]["proposer_answers"] == ["Discovered Answer"]
    assert problem["extra_info"]["source_gold_answers"] == ["Source Gold"]
    assert problem["extra_info"]["kg_evidence_rag_pass"] is True
    assert problem["extra_info"]["kg_evidence_chars"] > 0
    assert sorted(problem["extra_info"]["tools_kwargs"].keys()) == ["get_relations", "get_triples"]


def _kgqa_trajectory_output(question: str, triples: list[list[str]]) -> str:
    triple_lines = json.dumps(triples)
    return f"""
    <think>reasoning</think>
    <kg-query>get_relations("Entity A")</kg-query>
    <information>["demo.rel"]</information>
    <kg-query>get_triples("Entity A", ["demo.rel"])</kg-query>
    <information>{triple_lines}</information>
    <result>
    {{"question": "{question}"}}
    </result>
    """


@pytest.mark.unit
def test_structured_messages_render_canonical_inline_react_trajectory() -> None:
    messages = {
        "messages": [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "USER"},
            {
                "role": "assistant",
                "content": "<think>inspect</think>",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_triples",
                            "arguments": '{"entity":"Entity A","relations":["demo.rel"]}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": '{"result":"[[\\"Entity A\\", \\"demo.rel\\", \\"Entity B\\"]]"}'},
            {
                "role": "assistant",
                "content": '<think>done</think><result>{"question":"Who is related?"}</result>',
            },
        ]
    }

    rendered = render_structured_react_trajectory(messages)

    assert rendered == (
        '<think>inspect</think><kg-query>get_triples("Entity A", ["demo.rel"])</kg-query>'
        '\n<information>\n[["Entity A", "demo.rel", "Entity B"]]\n</information>\n\n'
        '<think>done</think><result>{"question":"Who is related?"}</result>'
    )
    assert "SYSTEM" not in rendered
    assert "USER" not in rendered


@pytest.mark.unit
def test_kg_path_skips_rag_when_filter_is_disabled() -> None:
    extractor = ProblemExtractor(answer_pattern="result", use_rag_filter=False)
    trajectory = {
        "input": "seed prompt",
        "output": '<result>{"question": "Which municipality shares a border with Gorduno?"}</result>',
        "metadata": {
            "trajectory_index": 40,
            "data_source": "kgqa_svts_gtsqa",
            "reward_model": {"ground_truth": {"target": ["Gnosca"]}, "style": "rule"},
            "extra_info": {
                "qid": 40,
                "dataset_type": "gtsqa",
                "answer_entities": [{"name": "Gnosca", "id": "Q67754"}],
                "question_entities": [{"name": "Gorduno", "id": "Q67523"}],
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert stats["successful_problems_count"] == 1
    assert problems[0]["extra_info"]["rag_diagnostics"]["status"] == "disabled"
    assert problems[0]["extra_info"]["rag_diagnostics"]["accepted"] is True
    assert problems[0]["extra_info"]["rag_hit_at_1"] == 0.0


@pytest.mark.unit
def test_kg_path_rag_filter_rejects_missing_triples() -> None:
    extractor = ProblemExtractor(answer_pattern="result", use_rag_filter=True)
    extractor.llm_judge = _FakeJudge("Gnosca")
    trajectory = {
        "input": "seed prompt",
        "output": '<result>{"question": "Which municipality shares a border with Gorduno?"}</result>',
        "metadata": {
            "trajectory_index": 41,
            "data_source": "kgqa_svts_gtsqa",
            "reward_model": {"ground_truth": {"target": ["Gnosca"]}, "style": "rule"},
            "extra_info": {
                "qid": 41,
                "dataset_type": "gtsqa",
                "answer_entities": [{"name": "Gnosca", "id": "Q67754"}],
                "question_entities": [{"name": "Gorduno", "id": "Q67523"}],
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert problems == []
    assert stats["failure_reasons"] == {"rag_no_triples": 1}


@pytest.mark.unit
def test_kg_path_rag_filter_accepts_hit_at_1() -> None:
    extractor = ProblemExtractor(answer_pattern="result", use_rag_filter=True)
    extractor.llm_judge = _FakeJudge("Oxybutynin Oral")
    trajectory = {
        "input": "seed prompt",
        "output": _kgqa_trajectory_output(
            "Which medicine is marketed in this formulation?",
            [
                [
                    "Oxybutynin chloride 5 extended release film coated tablet",
                    "medicine.routed_drug.marketed_formulations",
                    "Oxybutynin Oral",
                ]
            ],
        ),
        "metadata": {
            "trajectory_index": 42,
            "data_source": "kgqa_svts_freebase",
            "reward_model": {
                "ground_truth": {"target": ["Oxybutynin Oral"]},
                "style": "rule",
            },
            "extra_info": {
                "qid": 42,
                "dataset_type": "grailqa",
                "answer_entities": [{"name": "Oxybutynin Oral", "id": "m.0z3xfvs"}],
                "question_entities": [
                    {"name": "Oxybutynin chloride 5 extended release film coated tablet", "id": "m.0hqs1x_"}
                ],
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert stats["successful_problems_count"] == 1
    assert problems[0]["extra_info"]["rag_hit_at_1"] == 1.0
    assert problems[0]["extra_info"]["rag_diagnostics"]["status"] == "passed"


@pytest.mark.unit
def test_kg_path_rag_filter_rejects_wrong_hit_at_1() -> None:
    extractor = ProblemExtractor(answer_pattern="result", use_rag_filter=True)
    extractor.llm_judge = _FakeJudge("Wrong Answer")
    trajectory = {
        "input": "seed prompt",
        "output": _kgqa_trajectory_output(
            "Which formulation markets this medicine?",
            [["Entity A", "demo.rel", "Wrong Answer"]],
        ),
        "metadata": {
            "trajectory_index": 43,
            "data_source": "kgqa_svts_freebase",
            "reward_model": {"ground_truth": {"target": ["Answer A"]}, "style": "rule"},
            "extra_info": {
                "qid": 43,
                "dataset_type": "grailqa",
                "answer_entities": [{"name": "Answer A", "id": "m.answer"}],
                "question_entities": [{"name": "Entity A", "id": "m.entity"}],
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert problems == []
    assert stats["failure_reasons"] == {"rag_hit_at_1_failed": 1}


@pytest.mark.unit
def test_kg_path_rag_filter_classifies_api_failure_separately() -> None:
    extractor = ProblemExtractor(
        answer_pattern="result", use_rag_filter=True, rag_filter_fail_open=False
    )
    extractor.llm_judge = _FailingJudge("unused")
    trajectory = {
        "input": "seed prompt",
        "output": _kgqa_trajectory_output(
            "Which entity is related?", [["Entity A", "demo.rel", "Answer A"]]
        ),
        "metadata": {
            "trajectory_index": 44,
            "data_source": "kgqa_svts_gtsqa",
            "reward_model": {"ground_truth": {"target": ["Answer A"]}, "style": "rule"},
            "extra_info": {
                "qid": 44,
                "dataset_type": "gtsqa",
                "answer_entities": [{"name": "Answer A", "id": "Q2"}],
                "question_entities": [{"name": "Entity A", "id": "Q1"}],
            },
        },
    }

    problems, stats = extract_problems_batch([trajectory], extractor=extractor, max_workers=1)

    assert problems == []
    assert stats["failure_reasons"] == {"rag_api_failure": 1}
    assert stats["trajectory_outcomes"][0]["rag_diagnostics"]["status"] == "api_error_fail_closed"


@pytest.mark.unit
def test_kg_path_rag_filter_classifies_unparseable_answer_separately() -> None:
    extractor = ProblemExtractor(answer_pattern="result", use_rag_filter=True)
    extractor.llm_judge = _FakeJudge([])

    passed, reason, _, _, _, diagnostics = extractor._validate_rag_hit_at_1(
        question_text="Which entity is related?",
        question_entities=[{"name": "Entity A", "id": "Q1"}],
        source_gold_answers=["Answer A"],
        output_text=_kgqa_trajectory_output(
            "Which entity is related?", [["Entity A", "demo.rel", "Answer A"]]
        ),
        qid=45,
    )

    assert not passed
    assert reason == "rag_parse_failure"
    assert diagnostics["status"] == "empty_or_unparseable_answer"


@pytest.mark.unit
def test_parse_and_evaluate_kgqa_triple_answer() -> None:
    assert parse_kgqa_triple_answer('Answer: ["Entity A", "Entity B"]') == ["Entity A", "Entity B"]
    evaluation = evaluate_kgqa_triple_answer(
        _FakeJudge(["Entity A", "Wrong"]),
        question="Which entity?",
        question_entities="Seed",
        triples_text="- [Seed, demo.rel, Entity A]",
        gold_answers=["Entity A", "Entity B"],
    )
    assert evaluation.predicted_answers == ["Entity A", "Wrong"]
    assert evaluation.hit_at_1 == 1.0
