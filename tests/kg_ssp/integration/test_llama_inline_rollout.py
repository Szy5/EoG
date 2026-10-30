import os

import pytest


LLAMA_TEST_MODEL_PATH = os.getenv("LLAMA_TEST_MODEL_PATH")


@pytest.mark.skipif(not LLAMA_TEST_MODEL_PATH, reason="LLAMA_TEST_MODEL_PATH is not set")
def test_llama_inline_incremental_tokens_and_loss_mask() -> None:
    from transformers import AutoTokenizer
    from verl.tools.schemas import OpenAIFunctionToolCall
    from verl.workers.rollout.schemas import AsyncRolloutRequestStateEnum

    from quarl.utils.chat_template import SearchChatTemplate
    from quarl.utils.patch.chat_template import switch_apply_chat_template
    from quarl.worker.rollout.schema import SSPAsyncRolloutRequest

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_TEST_MODEL_PATH, local_files_only=True)
    SearchChatTemplate.set_template("llama3p1_react_inline")
    tokenizer = switch_apply_chat_template(tokenizer, SearchChatTemplate.apply_chat_template)

    try:
        request = SSPAsyncRolloutRequest(
            request_id="inline-mask-smoke",
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=[
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "USER"},
            ],
            reward_scores={},
            max_prompt_len=4096,
            max_response_len=1024,
            max_model_len=8192,
            use_inference_chat_template=False,
            tokenization_sanity_check_mode="disable",
            processing_class=tokenizer,
        )
        response_start = request.input_ids.shape[-1]

        tool_call = OpenAIFunctionToolCall.model_validate(
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "get_relations",
                    "arguments": {"entity": "Barack Obama"},
                },
            }
        )
        request.add_assistant_message(tokenizer, "<think>inspect</think>", [tool_call])
        assistant_end = request.input_ids.shape[-1]
        assistant_ids = request.input_ids[..., response_start:assistant_end]
        assistant_text = tokenizer.decode(assistant_ids.squeeze(0), skip_special_tokens=False)
        assert request.loss_mask[..., response_start:assistant_end].all()
        assert assistant_text == '<think>inspect</think><kg-query>get_relations("Barack Obama")</kg-query>'

        request.add_tool_response_messages(tokenizer, ['{"result":"people.person.place_of_birth"}'])
        tool_end = request.input_ids.shape[-1]
        tool_ids = request.input_ids[..., assistant_end:tool_end]
        tool_text = tokenizer.decode(tool_ids.squeeze(0), skip_special_tokens=False)
        assert not request.loss_mask[..., assistant_end:tool_end].any()
        assert tool_text == "\n<information>\npeople.person.place_of_birth\n</information>\n\n"
        assert "ipython" not in tool_text
        assert "<|eot_id|>" not in tool_text
        assert "<|start_header_id|>assistant<|end_header_id|>" not in tool_text

        request.get_generation_prompt_ids(tokenizer)
        assert request.input_ids.shape[-1] == tool_end

        terminal = '<think>done</think><result>{"question":"Where was he born?"}</result>'
        request.add_assistant_message(tokenizer, terminal)
        terminal_end = request.input_ids.shape[-1]
        terminal_ids = request.input_ids[..., tool_end:terminal_end]
        terminal_text = tokenizer.decode(terminal_ids.squeeze(0), skip_special_tokens=False)
        assert request.loss_mask[..., tool_end:terminal_end].all()
        assert terminal_text == terminal + "<|eot_id|>"

        full_text = tokenizer.decode(request.input_ids.squeeze(0), skip_special_tokens=False)
        assert full_text.count("<|start_header_id|>assistant<|end_header_id|>") == 1
    finally:
        SearchChatTemplate.set_template("default")
