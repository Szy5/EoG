import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from training.examples.data_preprocess.convert_svts_to_kg_ssp import convert


SVTS_TASK = {
    "qid": 2101535001000,
    "task_id": "iso-0_00001",
    "template_id": "Iso-0",
    "question_entity": [
        {"name": "Oxybutynin chloride 5 extended release film coated tablet", "id": "m.0hqs1x_"}
    ],
    "answer_entity": [
        {"name": "Oxybutynin Oral", "id": "m.0hqs1y1"},
        {"name": "Oxybutynin Chloride Oral", "id": "m.0hqs1y5"},
    ],
    "path": ["Oxybutynin chloride 5 extended release film coated tablet <--medicine.routed_drug.marketed_formulations-- Ans"],
    "triples": [
        ["Ans", "medicine.routed_drug.marketed_formulations", "Oxybutynin chloride 5 extended release film coated tablet"]
    ],
    "num_edge": 1,
    "domains": "medicine",
    "sparql_query": "SELECT DISTINCT ?x WHERE { ?x ns:medicine.routed_drug.marketed_formulations ns:m.0hqs1x_ . }",
}


@pytest.mark.unit
def test_svts_conversion_preserves_core_contract(tmp_path: Path) -> None:
    input_path = tmp_path / "svts_tasks.jsonl"
    input_path.write_text(
        "".join(json.dumps({**SVTS_TASK, "qid": SVTS_TASK["qid"] + i}, ensure_ascii=False) + "\n" for i in range(3)),
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "train.jsonl"
    output_parquet = tmp_path / "train.parquet"

    total, records = convert(
        input_jsonl=input_path,
        output_jsonl=output_jsonl,
        sys_template_path=Path("training/quarl/template/kgqa_ssp_proposer_sys.jinja"),
        user_template_path=Path("training/quarl/template/kgqa_ssp_proposer_user.jinja"),
        data_source="kgqa_svts_freebase",
        split="train",
        ability="kgqa-self-play",
        question_placeholder="placeholder_question_to_be_generated",
        dataset_type="kgqa",
        max_samples=3,
        collect_records_for_parquet=True,
    )

    assert total == 3
    assert records is not None
    assert len(records) == 3

    first_record = records[0]
    assert first_record["data_source"] == "kgqa_svts_freebase"
    assert first_record["reward_model"]["ground_truth"]["target"] == [
        "Oxybutynin Oral",
        "Oxybutynin Chloride Oral",
    ]
    assert first_record["extra_info"]["qid"] == 2101535001000
    assert first_record["extra_info"]["dataset_type"] == "kgqa"
    assert "source_question" not in first_record["extra_info"]
    assert first_record["extra_info"]["template_id"] == "Iso-0"
    assert first_record["metadata"]["task_id"] == "iso-0_00001"
    assert first_record["extra_info"]["question_entities"] == [
        {"name": "Oxybutynin chloride 5 extended release film coated tablet", "id": "m.0hqs1x_"}
    ]
    assert sorted(first_record["extra_info"]["tools_kwargs"].keys()) == ["get_relations", "get_triples"]
    assert first_record["extra_info"]["tools_kwargs"]["get_relations"]["create_kwargs"]["path"] == [
        "Oxybutynin chloride 5 extended release film coated tablet <--medicine.routed_drug.marketed_formulations-- Ans@"
    ]
    assert first_record["extra_info"]["tools_kwargs"]["get_relations"]["create_kwargs"]["triples"] == [
        ["Ans", "medicine.routed_drug.marketed_formulations", "Oxybutynin chloride 5 extended release film coated tablet"]
    ]
    assert "Path：" in first_record["prompt"][1]["content"]
    assert (
        "Oxybutynin chloride 5 extended release film coated tablet <--medicine.routed_drug.marketed_formulations-- Ans@"
        in first_record["prompt"][1]["content"]
    )
    assert "question entity（`Que@`）：" in first_record["prompt"][1]["content"]
    assert "Oxybutynin chloride 5 extended release film coated tablet" in first_record["prompt"][1]["content"]
    assert "<result>" in first_record["prompt"][0]["content"]

    lines = output_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    reloaded_first = json.loads(lines[0])
    assert reloaded_first["extra_info"]["answer_entities"][0]["name"] == "Oxybutynin Oral"

    from training.examples.data_preprocess.convert_svts_to_kg_ssp import write_parquet

    write_parquet(records, output_parquet)
    parquet_table = pq.read_table(output_parquet)
    assert parquet_table.num_rows == 3


@pytest.mark.unit
def test_conversion_handles_missing_question_entities(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "sample.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "qid": 1,
                "question": "demo",
                "question_entity": [],
                "answer_entity": [{"Answer A": "m.answer"}],
                "path": [],
                "triples": [["Ans", "demo.rel", "42"]],
                "domains": "demo",
                "sparql_query": "SELECT ?x WHERE {}",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "out.jsonl"

    total, records = convert(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        sys_template_path=Path("training/quarl/template/kgqa_ssp_proposer_sys.jinja"),
        user_template_path=Path("training/quarl/template/kgqa_ssp_proposer_user.jinja"),
        data_source="kgqa_svts_freebase",
        split="train",
        ability="kgqa-self-play",
        question_placeholder="placeholder",
        dataset_type="kgqa",
        collect_records_for_parquet=False,
    )

    assert total == 1
    assert records is None
    parsed = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
    assert parsed["extra_info"]["question_entities"] == []
    assert parsed["reward_model"]["ground_truth"]["target"] == ["Answer A"]


@pytest.mark.unit
def test_svts_conversion_supports_kg_explore_rag_contract(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "sample.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "qid": 2,
                "question": "Which organization is connected to Entity A?",
                "question_entity": [{"name": "Entity A", "id": "m.entity"}],
                "answer_entity": [{"Answer A": "m.answer"}],
                "path": ["Entity A --demo.hidden_relation--> Ans"],
                "triples": [["Entity A", "demo.hidden_relation", "Answer A"]],
                "domains": "demo",
                "sparql_query": "SELECT ?x WHERE {}",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "out.jsonl"

    total, records = convert(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        sys_template_path=Path("training/quarl/template/kgqa_explore_rag_proposer_sys.jinja"),
        user_template_path=Path("training/quarl/template/kgqa_explore_rag_proposer_user.jinja"),
        data_source="kgqa_svts_freebase",
        split="train",
        ability="kgqa-explore-rag-self-play",
        question_placeholder="placeholder",
        dataset_type="kgqa",
        synthesis_mode="kg_explore_rag",
        collect_records_for_parquet=True,
    )

    assert total == 1
    assert records is not None
    record = records[0]
    assert record["metadata"]["source_format"] == "svts_kg_explore_rag_jsonl"
    assert record["extra_info"]["synthesis_mode"] == "kg_explore_rag"
    assert sorted(record["extra_info"]["tools_kwargs"].keys()) == ["get_relations", "get_triples"]
    assert "Entity A" in record["prompt"][1]["content"]
    assert "Which organization is connected to Entity A?" not in record["prompt"][1]["content"]
    assert "source_question" not in record["extra_info"]
    assert "demo.hidden_relation" not in record["prompt"][1]["content"]
    assert "Path" not in record["prompt"][1]["content"]


@pytest.mark.unit
def test_seed_json_templates_discard_legacy_prewritten_questions() -> None:
    from training.examples.data_preprocess.convert_svts_to_kg_ssp import render_user_template

    task = {**SVTS_TASK, "question": "PRIVATE_QUESTION", "source_question": "PRIVATE_SOURCE"}
    for template in ("{{ kgqa_sample_json }}", "{{ kgqa_sample_pretty_json }}"):
        rendered = json.loads(render_user_template(template, task))
        assert "question" not in rendered
        assert "source_question" not in rendered
        assert rendered["path"] == task["path"]
    assert task["question"] == "PRIVATE_QUESTION"


@pytest.mark.unit
def test_seed_templates_reject_prewritten_question_placeholder() -> None:
    from training.examples.data_preprocess.convert_svts_to_kg_ssp import render_user_template

    with pytest.raises(ValueError, match="Prewritten question templates"):
        render_user_template("{{ source_question }}", SVTS_TASK)
