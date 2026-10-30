import json

import pytest

from quarl.utils.problem_extraction import ProblemExtractor


@pytest.mark.unit
def test_extract_result_payload_parses_question_and_answers() -> None:
    extractor = ProblemExtractor(answer_pattern="result")
    output_text = """
    <think>hidden reasoning</think>
    <result>
    {"question": "Which medicine is marketed in this formulation?", "answers": ["Oxybutynin Oral", "Oxybutynin Chloride Oral"]}
    </result>
    """

    payload = extractor._extract_result_payload(output_text)

    assert payload == {
        "question": "Which medicine is marketed in this formulation?",
        "answers": ["Oxybutynin Oral", "Oxybutynin Chloride Oral"],
    }


@pytest.mark.unit
def test_extract_result_payload_rejects_invalid_json() -> None:
    extractor = ProblemExtractor(answer_pattern="result")

    assert extractor._extract_result_payload("<result>{not valid json}</result>") is None


@pytest.mark.unit
def test_user_template_contains_path_and_question_entity_placeholders() -> None:
    template_path = "training/quarl/template/kgqa_ssp_proposer_user.jinja"
    with open(template_path, "r", encoding="utf-8") as handle:
        template = handle.read()

    assert "{{ path }}" in template
    assert "{{ question_entity }}" in template
    assert "placeholders, not entity names for tool calls" in template
