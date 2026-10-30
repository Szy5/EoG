import pytest

from quarl.utils.chat_template import SearchChatTemplate
from quarl.tool.query_parser import parse_kg_query_call


def test_parse_kg_query_call_supports_both_functions() -> None:
    assert parse_kg_query_call('get_relations("Barack Obama")') == (
        "get_relations",
        {"entity": "Barack Obama"},
    )
    assert parse_kg_query_call(
        'get_triples("Barack Obama", ["people.person.place_of_birth", "people.person.nationality"])'
    ) == (
        "get_triples",
        {
            "entity": "Barack Obama",
            "relations": ["people.person.place_of_birth", "people.person.nationality"],
        },
    )



@pytest.mark.unit
def test_chat_template_renders_kg_query_tool_calls() -> None:
    rendered = SearchChatTemplate.apply_chat_template(
        [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "<think>reasoning</think>",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_triples",
                            "arguments": {
                                "entity": "Barack Obama",
                                "relations": ["people.person.place_of_birth", "people.person.nationality"],
                            },
                        }
                    }
                ],
            },
        ]
    )

    assert '<kg-query>get_triples("Barack Obama", ["people.person.place_of_birth", "people.person.nationality"])</kg-query>' in rendered


@pytest.mark.unit
def test_chat_template_rollout_helpers_include_kgqa_tags() -> None:
    stop_tokens = SearchChatTemplate.get_generation_stop_tokens()

    assert "</search>" in stop_tokens
    assert "</answer>" in stop_tokens
    assert "</kg-query>" in stop_tokens
    assert "</result>" in stop_tokens


@pytest.mark.unit
def test_chat_template_restores_kgqa_generation_suffixes_without_duplication() -> None:
    assert (
        SearchChatTemplate.restore_stopped_generation_suffix('<think>inspect entity</think><kg-query>get_relations("Barack Obama")')
        == '<think>inspect entity</think><kg-query>get_relations("Barack Obama")</kg-query>'
    )
    assert (
        SearchChatTemplate.restore_stopped_generation_suffix(
            '<think>abstract path</think><result>{"question":"Q","answers":["A"]}'
        )
        == '<think>abstract path</think><result>{"question":"Q","answers":["A"]}</result>'
    )
    assert (
        SearchChatTemplate.restore_stopped_generation_suffix(
            '<think>done</think><result>{"question":"Q","answers":["A"]}</result>'
        )
        == '<think>done</think><result>{"question":"Q","answers":["A"]}</result>'
    )


@pytest.mark.unit
def test_llama3p1_chat_template_uses_native_turn_boundaries() -> None:
    SearchChatTemplate.set_template("llama3p1")
    try:
        rendered = SearchChatTemplate.apply_chat_template(
            [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "USER"},
                {
                    "role": "assistant",
                    "content": "<think>inspect</think>",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_relations",
                                "arguments": {"entity": "Barack Obama"},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": '{"result": "people.person.place_of_birth"}'},
            ],
            add_generation_prompt=True,
        )
        needs_post_tool_assistant = SearchChatTemplate.needs_generation_prompt_after("tool")
    finally:
        SearchChatTemplate.set_template("default")

    assert needs_post_tool_assistant
    assert rendered == (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\nSYSTEM<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nUSER<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        '<think>inspect</think><kg-query>get_relations("Barack Obama")</kg-query><|eot_id|>'
        "<|start_header_id|>ipython<|end_header_id|>\n\n"
        "<information>\npeople.person.place_of_birth\n</information><|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@pytest.mark.unit
def test_llama3p1_react_inline_keeps_one_open_assistant_stream() -> None:
    SearchChatTemplate.set_template("llama3p1_react_inline")
    try:
        rendered = SearchChatTemplate.apply_chat_template(
            [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "USER"},
                {
                    "role": "assistant",
                    "content": "<think>inspect</think>",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_relations",
                                "arguments": {"entity": "Barack Obama"},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": '{"result": "people.person.place_of_birth"}'},
                {
                    "role": "assistant",
                    "content": "<think>expand</think>",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_triples",
                                "arguments": {
                                    "entity": "Barack Obama",
                                    "relations": ["people.person.place_of_birth"],
                                },
                            }
                        }
                    ],
                },
                {"role": "tool", "content": '{"result": "[Barack Obama, place_of_birth, Honolulu]"}'},
                {
                    "role": "assistant",
                    "content": '<think>done</think><result>{"question":"Where was Barack Obama born?"}</result>',
                },
            ],
            add_generation_prompt=True,
        )
        assert SearchChatTemplate.uses_inline_react()
        assert not SearchChatTemplate.uses_separate_tool_turns()
        assert not SearchChatTemplate.needs_generation_prompt_after("tool")
    finally:
        SearchChatTemplate.set_template("default")

    assert rendered == (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\nSYSTEM<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nUSER<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        '<think>inspect</think><kg-query>get_relations("Barack Obama")</kg-query>'
        "\n<information>\npeople.person.place_of_birth\n</information>\n\n"
        '<think>expand</think><kg-query>get_triples("Barack Obama", '
        '["people.person.place_of_birth"])</kg-query>'
        "\n<information>\n[Barack Obama, place_of_birth, Honolulu]\n</information>\n\n"
        '<think>done</think><result>{"question":"Where was Barack Obama born?"}</result><|eot_id|>'
    )
    assert rendered.count("<|start_header_id|>assistant<|end_header_id|>") == 1
    assert "<|start_header_id|>ipython<|end_header_id|>" not in rendered


@pytest.mark.unit
def test_qwen2p5_chat_template_keeps_legacy_continuation_behavior() -> None:
    SearchChatTemplate.set_template("qwen2p5")
    try:
        rendered = SearchChatTemplate.apply_chat_template(
            [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "USER"},
            ],
            add_generation_prompt=True,
        )
        assert not SearchChatTemplate.uses_separate_tool_turns()
        assert not SearchChatTemplate.needs_generation_prompt_after("tool")
        assert SearchChatTemplate.needs_generation_prompt_after("user")
    finally:
        SearchChatTemplate.set_template("default")

    assert rendered == "<|im_start|>system\nSYSTEM<|im_start|>user\nUSER<|im_start|>assistant\n"


@pytest.mark.unit
def test_graphwalker_chat_template_matches_turn_boundaries_and_continuation() -> None:
    SearchChatTemplate.set_template("graphwalker")
    try:
        rendered = SearchChatTemplate.apply_chat_template(
            [
                {"role": "user", "content": "Question"},
                {
                    "role": "assistant",
                    "content": "<think>inspect entity</think>",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_relations",
                                "arguments": {"entity": "Barack Obama"},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": '{"result": "people.person.place_of_birth"}'},
            ],
            add_generation_prompt=True,
        )
    finally:
        SearchChatTemplate.set_template("default")

    assert rendered.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n")
    assert '<|im_start|>assistant\n<think>inspect entity</think><kg-query>get_relations("Barack Obama")</kg-query><|im_end|>' in rendered
    assert "<|im_start|>user\n<information>Here are the query results:\npeople.person.place_of_birth\n</information>" in rendered
    assert "Review the results. If the answer is found" in rendered
    assert rendered.endswith("<|im_end|>\n<|im_start|>assistant\n")
