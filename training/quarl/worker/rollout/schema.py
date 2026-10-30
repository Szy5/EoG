import json
import random
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin
from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.schemas import BASE_CHAT_HISTORY, AsyncRolloutRequest, FinishReasonTypeEnum, Message

from quarl.utils.chat_template import SearchChatTemplate as QSCT
from quarl.utils.data_utils import remove_prefix


class SSPAsyncRolloutRequest(AsyncRolloutRequest):
    ground_truth: Optional[dict] = None
    timing_info: Optional[Dict[str, float]] = None
    last_generation_output: Optional[Dict[str, Any]] = None
    # Populated by finalize(); "length" means generation was cut off by the token
    # budget (max_response_length) rather than terminating on its own. Used
    # downstream to zero out reward/loss for overlong-truncated rollouts.
    finish_reason_type: Optional[str] = None

    def finalize(self, processing_class, reward_scores, finish_reason_type=FinishReasonTypeEnum.STOP) -> None:
        super().finalize(processing_class, reward_scores, finish_reason_type)
        self.finish_reason_type = (
            finish_reason_type.value if isinstance(finish_reason_type, FinishReasonTypeEnum) else finish_reason_type
        )

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        return self.input_ids

    def get_generation_prompt_ids(
        self, processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin]
    ) -> List[int]:
        """
        Get the generation prompt ids for rollout engine.

        Because rollout engine(SGLang) requires the ids to be a list, we need to convert the tensor to a list.
        """

        last_role = self.messages[-1].role
        uses_separate_tool_turns = QSCT.uses_separate_tool_turns()
        continues_existing_assistant_turn = last_role == "assistant" or (
            last_role == "tool" and not uses_separate_tool_turns
        )

        if continues_existing_assistant_turn:
            # 如果是工具调用，不加generation_prefix
            if self.input_ids[..., -self.generation_prompt_ids.shape[-1] :].eq(self.generation_prompt_ids).all():
                self.input_ids = self.input_ids[..., : -self.generation_prompt_ids.shape[-1]]

        generation_prompt_ids = (
            None
            if self.input_ids[..., -self.generation_prompt_ids.shape[-1] :].eq(self.generation_prompt_ids).all()
            else self.generation_prompt_ids
        )
        needs_generation_prompt = QSCT.needs_generation_prompt_after(last_role)
        if generation_prompt_ids is not None and needs_generation_prompt:
            self._update_input_ids(processing_class, generation_prompt_ids, attention_mask=True, loss_mask=False)

        if self.use_inference_chat_template:
            messages = [msg.model_dump() for msg in self.messages]
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
            generation_prompt_ids = self._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=self.multi_modal_data,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )
            return generation_prompt_ids.squeeze(0).tolist()
        else:
            return self.input_ids.squeeze(0).tolist()

    def add_assistant_message(
        self,
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> None:

        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        if self.messages[-1].role in ["tool", "assistant"]:
            content = remove_prefix(content, QSCT.assist_prefix)
            if not QSCT.add_think_after_tool:
                content = remove_prefix(content, QSCT.think_prefix)
            # add tool return to base_messages
            base_messages = [*BASE_CHAT_HISTORY, {"role": "tool", "content": json.dumps({"result": ""})}]
        else:
            base_messages = BASE_CHAT_HISTORY

        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))

        messages = [*base_messages, self.messages[-1]]

        base_prompt = processing_class.apply_chat_template(
            base_messages, tools=tools, add_generation_prompt=True, tokenize=False
        )

        raw_prompt = processing_class.apply_chat_template(
            messages, tools=tools, add_generation_prompt=False, tokenize=False
        )

        content_prompt = raw_prompt[len(base_prompt) :]

        # We don't need to pass multi_modal_data here because we don't have any multi-modal data from Engine
        # Inference, it is pure text.
        # This is an incremental continuation, not a new sequence. Llama
        # tokenizers otherwise prepend <|begin_of_text|> to every assistant
        # segment, while Qwen tokenizers typically hide this bug.
        content_ids = processing_class(
            text=[content_prompt],
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        # self._handle_apply_chat_template(
        #     processing_class, messages, multi_modal_data={}, tools=tools, add_generation_prompt=False, tokenize=True
        # )[..., base_conv_with_gen_prompt_end_pos:]

        self._update_input_ids(processing_class, content_ids, attention_mask=True, loss_mask=True)
