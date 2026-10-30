import pytest

from quarl.trainer.ppo.ssp_ray_trainer import SSPRayPPOTrainer


@pytest.mark.unit
def test_dummy_problem_for_failed_extraction_uses_kgqa_tools_for_kgqa_metadata() -> None:
    trainer = object.__new__(SSPRayPPOTrainer)

    dummy_problem = trainer._create_dummy_problem_for_failed_extraction(
        trajectory={
            "metadata": {
                "reward_model": {"ground_truth": {"target": ["dummy_answer"], "style": "rule"}},
                "data_source": "kgqa_svts_freebase",
                "ability": "kgqa-self-play",
                "extra_info": {
                    "split": "train",
                    "dataset_type": "grailqa",
                    "qid": 7,
                    "question_entities": [{"name": "Entity A", "id": "m.entity_a"}],
                    "answer_entities": [{"name": "Answer A", "id": "m.answer_a"}],
                    "path": ["Entity A --demo.rel--> Answer A"],
                    "triples": [["Entity A", "demo.rel", "Answer A"]],
                },
            }
        },
        trajectory_index=3,
    )

    assert dummy_problem["problem_type"] == "kgqa"
    assert dummy_problem["extra_info"]["need_tools_kwargs"] is True
    assert sorted(dummy_problem["extra_info"]["tools_kwargs"].keys()) == ["get_relations", "get_triples"]
    assert dummy_problem["extra_info"]["tools_kwargs"]["get_relations"]["create_kwargs"]["dataset_type"] == "grailqa"
    assert dummy_problem["extra_info"]["tools_kwargs"]["get_relations"]["create_kwargs"]["qid"] == 7
    assert dummy_problem["extra_info"]["extraction_failed"] is True


@pytest.mark.unit
def test_template_extraction_metrics_count_valid_problems_by_iso_template() -> None:
    trainer = object.__new__(SSPRayPPOTrainer)

    metrics = trainer._build_template_extraction_metrics(
        [
            {"extra_info": {"template_id": "Iso-0"}},
            {"extra_info": {"template_id": "Iso-0"}},
            {"extra_info": {"template_id": "Iso-3"}},
        ]
    )

    assert metrics["self_play/extraction_by_template/Iso-0/extracted_count"] == 2.0
    assert metrics["self_play/extraction_by_template/Iso-1/extracted_count"] == 0.0
    assert metrics["self_play/extraction_by_template/Iso-2/extracted_count"] == 0.0
    assert metrics["self_play/extraction_by_template/Iso-3/extracted_count"] == 1.0
    assert metrics["self_play/extraction_by_template/Iso-5/extracted_count"] == 0.0


@pytest.mark.unit
def test_template_extraction_metrics_preserve_unknown_template_for_diagnostics() -> None:
    trainer = object.__new__(SSPRayPPOTrainer)

    metrics = trainer._build_template_extraction_metrics(
        [
            {"extra_info": {"template_id": "Iso-5"}},
            {"extra_info": {"template_id": "Iso-unknown"}},
            {"extra_info": {}},
        ]
    )

    assert metrics["self_play/extraction_by_template/Iso-5/extracted_count"] == 1.0
    assert metrics["self_play/extraction_by_template/Iso-unknown/extracted_count"] == 1.0
    assert metrics["self_play/extraction_by_template/unknown/extracted_count"] == 1.0


@pytest.mark.unit
def test_extraction_by_template_summary_counts_totals_and_successes() -> None:
    trainer = object.__new__(SSPRayPPOTrainer)

    summary = trainer._build_extraction_by_template_summary(
        trajectories=[
            {"metadata": {"extra_info": {"template_id": "Iso-0"}}},
            {"metadata": {"extra_info": {"template_id": "Iso-0"}}},
            {"metadata": {"extra_info": {"template_id": "Iso-1"}}},
        ],
        valid_problems=[
            {"extra_info": {"template_id": "Iso-0", "qid": 101}},
            {"extra_info": {"template_id": "Iso-1", "qid": 202}},
        ],
        step=3,
    )

    assert summary["step"] == 3
    assert summary["total_trajectories"] == 3
    assert summary["final_valid_count"] == 2
    assert summary["by_template_id"]["Iso-0"]["total"] == 2
    assert summary["by_template_id"]["Iso-0"]["extracted_success"] == 1
    assert summary["by_template_id"]["Iso-0"]["success_rate"] == 0.5
    assert summary["by_template_id"]["Iso-0"]["qids_success"] == [101]
    assert summary["by_template_id"]["Iso-1"]["total"] == 1
    assert summary["by_template_id"]["Iso-1"]["extracted_success"] == 1


@pytest.mark.unit
def test_success_question_rows_are_compact_and_filter_solver_correctness() -> None:
    trainer = object.__new__(SSPRayPPOTrainer)
    trainer.rollout_n = 2

    proposer_rows = trainer._build_proposer_success_rows(
        valid_problems=[
            {
                "trajectory_index": 1,
                "extracted_question": "Generated question?",
                "extra_info": {
                    "template_id": "Iso-2",
                    "qid": 77,
                    "source_question": "Source question?",
                    "source_gold_answers": ["Gold"],
                    "proposer_answers": ["Gold"],
                    "path": ["A --r--> B"],
                },
            }
        ],
        trajectories=[{"output": "bad"}, {"output": "<result>{}</result>"}],
        step=4,
    )

    assert len(proposer_rows) == 1
    assert proposer_rows[0]["trajectory_index"] == 1
    assert proposer_rows[0]["template_id"] == "Iso-2"
    assert proposer_rows[0]["generated_question"] == "Generated question?"
    assert proposer_rows[0]["path"] == ["A --r--> B"]
    assert proposer_rows[0]["question_entities"] == []
    assert proposer_rows[0]["output"] == "<result>{}</result>"

    solver_rows = trainer._build_solver_success_rows(
        outputs=["wrong", "right"],
        scores=[0.2, 1.0],
        raw_answer_corrects=[False, True],
        reward_eligibilities=[True, True],
        extra_infos=[
            {"template_id": "Iso-0", "qid": 1, "question": "Wrong?", "source_gold_answers": ["A"]},
            {"template_id": "Iso-3", "qid": 2, "question": "Right?", "source_gold_answers": ["B"]},
        ],
        data_sources=["kgqa_svts_gtsqa", "kgqa_svts_gtsqa"],
        step=5,
    )

    assert len(solver_rows) == 1
    assert solver_rows[0]["solver_trajectory_index"] == 1
    assert solver_rows[0]["problem_index"] == 0
    assert solver_rows[0]["template_id"] == "Iso-3"
    assert solver_rows[0]["qid"] == 2
    assert solver_rows[0]["answer_correct"] is True
    assert solver_rows[0]["reward_eligible"] is True
    assert solver_rows[0]["data_source"] == "kgqa_svts_gtsqa"
    assert solver_rows[0]["output"] == "right"


@pytest.mark.unit
def test_failure_and_rag_rows_preserve_gtsqa_metadata_and_stable_status() -> None:
    trainer = object.__new__(SSPRayPPOTrainer)
    trajectories = [
        {
            "input": "seed",
            "output": "<result>{}</result>",
            "metadata": {
                "data_source": "kgqa_svts_gtsqa",
                "reward_model": {"ground_truth": {"target": ["Gold"]}},
                "extra_info": {
                    "qid": 88,
                    "template_id": "Iso-4",
                    "dataset_type": "gtsqa",
                    "question_entities": [{"name": "Seed", "id": "Q1"}],
                    "answer_entities": [{"name": "Gold", "id": "Q2"}],
                    "path": ["Seed --r--> Gold"],
                },
            },
        }
    ]
    outcomes = [
        {
            "trajectory_index": 0,
            "answer_matches": 1,
            "valid_questions": 1,
            "format_error": 0,
            "failure_reason": "rag_api_failure",
            "rag_diagnostics": {
                "status": "api_error_fail_closed",
                "rag_invoked": True,
                "accepted": False,
                "error_type": "TimeoutError",
                "error_message": "timeout",
            },
        }
    ]

    fail_rows = trainer._build_proposer_fail_rows(trajectories, outcomes, step=6)
    rag_rows = trainer._build_rag_filter_rows(trajectories, outcomes, step=6)

    assert fail_rows[0]["failure_reason"] == "rag_api_failure"
    assert fail_rows[0]["dataset_type"] == "gtsqa"
    assert fail_rows[0]["question_entities"] == [{"name": "Seed", "id": "Q1"}]
    assert fail_rows[0]["path"] == ["Seed --r--> Gold"]
    assert rag_rows[0]["status"] == "api_error_fail_closed"
    assert rag_rows[0]["rag_invoked"] is True
    assert rag_rows[0]["error_type"] == "TimeoutError"
    assert rag_rows[0]["gold_answers"] == ["Gold"]
