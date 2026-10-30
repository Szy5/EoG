import pytest

from training.examples.data_preprocess.webqsp_cwq_val.convert import (
    PACKAGE_DIR,
    build_solver_record,
    load_prompt_templates,
)


@pytest.mark.unit
def test_build_solver_record_uses_templates_and_minimal_extra_info() -> None:
    templates = load_prompt_templates(PACKAGE_DIR)
    record = build_solver_record(
        question="Who founded Microsoft?",
        question_entities=[{"name": "Microsoft", "id": "m.04sv4"}],
        answer_targets=["Bill Gates"],
        answer_entities=[{"name": "Bill Gates", "id": "m.01g20"}],
        data_source="kgqa_val_webqsp",
        dataset_type="webqsp",
        qid="WebQTest-demo",
        split="val",
        index=0,
        sparql_query="SELECT ?x WHERE {}",
        prompt_templates=templates,
    )

    assert record["data_source"] == "kgqa_val_webqsp"
    assert record["prompt"][0]["role"] == "system"
    assert record["prompt"][0]["content"] == templates.system
    assert "Question: Who founded Microsoft?" in record["prompt"][1]["content"]
    assert "Initial Entities: Microsoft" in record["prompt"][1]["content"]
    assert record["reward_model"]["ground_truth"]["target"] == ["Bill Gates"]

    extra = record["extra_info"]
    assert "path" not in extra
    assert "triples" not in extra
    assert "source_question" not in extra
    assert extra["dataset_type"] == "webqsp"
    assert extra["need_tools_kwargs"] is True

    create_kwargs = extra["tools_kwargs"]["get_relations"]["create_kwargs"]
    assert "path" not in create_kwargs
    assert "triples" not in create_kwargs
    assert create_kwargs["question_entities"] == [{"name": "Microsoft", "id": "m.04sv4"}]
