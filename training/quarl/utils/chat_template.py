import json
import os

from pydantic import BaseModel


class SearchChatTemplate:

    _GRAPHWALKER_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    _GRAPHWALKER_CONTINUATION_TEMPLATE = """<information>Here are the query results:
{query_results}
</information>

Review the results. If the answer is found, provide it in <answer> tags. Otherwise, continue reasoning in <think> tags and issue your next <kg-query>.

Reminder: After `get_relations`, you must rank all returned relations and call `get_triples` with the full ranked list. We will execute a query for the top 4.
"""

    TEMPLATES = {
        "default": {
            "system_prefix": "<|im_start|>system\n",
            "assist_prefix": "<|im_start|>assistant\n",
            "user_prefix": "<|im_start|>user\n",
            "tool_prefix": "<|im_start|>function\n",
            "bos": "",
            "eos": "<|im_end|>",
            "think_prefix": "<think>",
            "think_suffix": "</think>",
            "search_prefix": "<search>",
            "search_suffix": "</search>",
            "kg_query_prefix": "<kg-query>",
            "kg_query_suffix": "</kg-query>",
            "answer_prefix": "<answer>",
            "answer_suffix": "</answer>",
            "result_prefix": "<result>",
            "result_suffix": "</result>",
            "search_result_prefix": "\n<information>\n",
            "search_result_suffix": "</information>\n\n",
            "add_think_after_tool": True,  # whether to add think_prefix after tool call
        },
        "qwen2p5": {
            "system_prefix": "<|im_start|>system\n",
            "assist_prefix": "<|im_start|>assistant\n",
            "user_prefix": "<|im_start|>user\n",
            "tool_prefix": "<|im_start|>function\n",
            "bos": "",
            "eos": "<|im_end|>",
            "think_prefix": "<think>",
            "think_suffix": "</think>",
            "search_prefix": "<search>",
            "search_suffix": "</search>",
            "kg_query_prefix": "<kg-query>",
            "kg_query_suffix": "</kg-query>",
            "answer_prefix": "<answer>",
            "answer_suffix": "</answer>",
            "result_prefix": "<result>",
            "result_suffix": "</result>",
            "search_result_prefix": "\n<information>\n",
            "search_result_suffix": "</information>\n\n",
            "add_think_after_tool": True,  
        },
        # Validation-only GraphWalker compatibility mode. The renderer below
        # uses this marker to reproduce GraphWalker's Qwen message boundaries
        # and user continuation after a KG query.
        "graphwalker": {
            "system_prefix": "<|im_start|>system\n",
            "assist_prefix": "<|im_start|>assistant\n",
            "user_prefix": "<|im_start|>user\n",
            "tool_prefix": "<|im_start|>function\n",
            "bos": "",
            "eos": "<|im_end|>",
            "think_prefix": "<think>",
            "think_suffix": "</think>",
            "search_prefix": "<search>",
            "search_suffix": "</search>",
            "kg_query_prefix": "<kg-query>",
            "kg_query_suffix": "</kg-query>",
            "answer_prefix": "<answer>",
            "answer_suffix": "</answer>",
            "result_prefix": "<result>",
            "result_suffix": "</result>",
            "search_result_prefix": "\n<information>\n",
            "search_result_suffix": "</information>\n\n",
            "add_think_after_tool": True,
        },
        "R-Search": {
            "system_prefix": "<|im_start|>system\n",
            "assist_prefix": "<|im_start|>assistant\n",
            "user_prefix": "<|im_start|>user\n",
            "tool_prefix": "<|im_start|>function\n",
            "bos": "",
            "eos": "<|im_end|>",
            "think_prefix": "<think>",
            "think_suffix": "</think>",
            "search_prefix": "<search>",
            "search_suffix": "</search>",
            "kg_query_prefix": "<kg-query>",
            "kg_query_suffix": "</kg-query>",
            "answer_prefix": "<answer>",
            "answer_suffix": "</answer>",
            "result_prefix": "<result>",
            "result_suffix": "</result>",
            "search_result_prefix": "\n<observation>",
            "search_result_suffix": "</observation>\n",
            "add_think_after_tool": True,  
        },
        "llama3p1": {
            "system_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
            "assist_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "user_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
            "tool_prefix": "<|start_header_id|>ipython<|end_header_id|>\n\n",
            "bos": "<|begin_of_text|>",
            "eos": "<|eot_id|>",
            "think_prefix": "<think>",
            "think_suffix": "</think>",
            "search_prefix": "<search>",
            "search_suffix": "</search>",
            "kg_query_prefix": "<kg-query>",
            "kg_query_suffix": "</kg-query>",
            "answer_prefix": "<answer>",
            "answer_suffix": "</answer>",
            "result_prefix": "<result>",
            "result_suffix": "</result>",
            "search_result_prefix": "\n<information>\n",
            "search_result_suffix": "</information>\n\n",
            "add_think_after_tool": True,  
        },
        # Textual ReAct mode for Llama 3.1. Only the initial chat turn uses
        # native Llama headers; tool observations are injected into the same
        # assistant stream and are masked by the rollout request.
        "llama3p1_react_inline": {
            "system_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
            "assist_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "user_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
            "tool_prefix": "",
            "bos": "<|begin_of_text|>",
            "eos": "<|eot_id|>",
            "think_prefix": "<think>",
            "think_suffix": "</think>",
            "search_prefix": "<search>",
            "search_suffix": "</search>",
            "kg_query_prefix": "<kg-query>",
            "kg_query_suffix": "</kg-query>",
            "answer_prefix": "<answer>",
            "answer_suffix": "</answer>",
            "result_prefix": "<result>",
            "result_suffix": "</result>",
            "search_result_prefix": "\n<information>\n",
            "search_result_suffix": "</information>\n\n",
            "add_think_after_tool": True,
        },
    }

    # The current template type is obtained from the environment variable CHAT_TEMPLATE
    _current_template = os.getenv("SEARCH_CHAT_TEMPLATE", "default")

    @classmethod
    def set_template(cls, template_type="default"):
        if template_type in cls.TEMPLATES:
            cls._current_template = template_type
            cls._update_class_attributes()
            print(f"[SearchChatTemplate] Switched to template: {template_type}")
            return True
        else:
            print(
                f"[SearchChatTemplate] Template '{template_type}' not found. Available templates: {list(cls.TEMPLATES.keys())}"
            )
            return False

    @classmethod
    def _update_class_attributes(cls):
        config = cls.TEMPLATES[cls._current_template]
        cls.system_prefix = config["system_prefix"]
        cls.assist_prefix = config["assist_prefix"]
        cls.user_prefix = config["user_prefix"]
        cls.tool_prefix = config["tool_prefix"]
        cls.bos = config["bos"]
        cls.eos = config["eos"]
        cls.think_prefix = config["think_prefix"]
        cls.think_suffix = config["think_suffix"]
        cls.search_prefix = config["search_prefix"]
        cls.search_suffix = config["search_suffix"]
        cls.kg_query_prefix = config["kg_query_prefix"]
        cls.kg_query_suffix = config["kg_query_suffix"]
        cls.answer_prefix = config["answer_prefix"]
        cls.answer_suffix = config["answer_suffix"]
        cls.result_prefix = config["result_prefix"]
        cls.result_suffix = config["result_suffix"]
        cls.search_result_prefix = config["search_result_prefix"]
        cls.search_result_suffix = config["search_result_suffix"]
        cls.add_think_after_tool = config["add_think_after_tool"]

    @classmethod
    def get_available_templates(cls):
        return list(cls.TEMPLATES.keys())

    @classmethod
    def get_current_template(cls):
        return cls._current_template

    @classmethod
    def uses_separate_tool_turns(cls):
        """Whether tool observations must be followed by a fresh assistant header."""
        return cls._current_template == "llama3p1"

    @classmethod
    def uses_inline_react(cls):
        """Whether tool observations continue the current assistant token stream."""
        return cls._current_template == "llama3p1_react_inline"

    @classmethod
    def needs_generation_prompt_after(cls, role):
        """Return whether an incremental rollout should open a new assistant turn."""
        if role == "assistant":
            return False
        if role in {"tool", "function", "ipython"}:
            return cls.uses_separate_tool_turns()
        return True

    @classmethod
    def get_generation_tag_pairs(cls):
        return [
            (cls.search_prefix, cls.search_suffix),
            (cls.answer_prefix, cls.answer_suffix),
            (cls.kg_query_prefix, cls.kg_query_suffix),
            (cls.result_prefix, cls.result_suffix),
        ]

    @classmethod
    def get_generation_stop_tokens(cls):
        return [suffix for _, suffix in cls.get_generation_tag_pairs()]

    @classmethod
    def restore_stopped_generation_suffix(cls, content: str) -> str:
        if not content:
            return content

        unmatched_pairs = []
        for prefix, suffix in cls.get_generation_tag_pairs():
            if content.count(prefix) > content.count(suffix):
                unmatched_pairs.append((prefix, suffix))

        if not unmatched_pairs:
            return content

        stopped_suffix = None
        for i in reversed(range(len(content))):
            for prefix, suffix in unmatched_pairs:
                if prefix in content[i:]:
                    stopped_suffix = suffix
                    break
            if stopped_suffix is not None:
                break

        if stopped_suffix is None:
            return content
        return content + stopped_suffix

    @classmethod
    def _tool_result_text(cls, content):
        try:
            result = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return str(content)
        return str(result.get("result", content)) if isinstance(result, dict) else str(content)

    @classmethod
    def _render_kg_tool_calls(cls, tool_calls) -> str:
        rendered = ""
        for tool_call in tool_calls or []:
            function = tool_call.get("function", tool_call)
            name = function.get("name", "")
            arguments = function.get("arguments", {})
            if name == "get_relations":
                rendered += f'<kg-query>get_relations("{arguments.get("entity", "")}")</kg-query>'
            elif name == "get_triples":
                relations = arguments.get("relations", [])
                serialized = ", ".join(json.dumps(str(item), ensure_ascii=False) for item in relations)
                rendered += f'<kg-query>get_triples("{arguments.get("entity", "")}", [{serialized}])</kg-query>'
        return rendered

    @classmethod
    def _apply_graphwalker_template(cls, messages, add_generation_prompt=False):
        """Render the message structure used by GraphWalker's vLLM runner."""
        rendered_messages = [message.model_dump() if isinstance(message, BaseModel) else message for message in messages]
        prompt = ""
        if not rendered_messages or rendered_messages[0]["role"] != "system":
            prompt += f"<|im_start|>system\n{cls._GRAPHWALKER_DEFAULT_SYSTEM}<|im_end|>\n"

        for message in rendered_messages:
            role = message["role"]
            content = str(message.get("content", ""))
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                if message.get("tool_calls") and "<kg-query>" not in content:
                    content += cls._render_kg_tool_calls(message["tool_calls"])
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "tool":
                query_results = cls._tool_result_text(content)
                continuation = cls._GRAPHWALKER_CONTINUATION_TEMPLATE.format(query_results=query_results)
                prompt += f"<|im_start|>user\n{continuation}<|im_end|>\n"
            else:
                raise ValueError("no such role!!!", message)

        if add_generation_prompt and rendered_messages and rendered_messages[-1]["role"] in {"user", "tool"}:
            prompt += "<|im_start|>assistant\n"
        return prompt

    @classmethod
    def _apply_llama3p1_template(cls, messages, add_generation_prompt=False):
        """Render SSP-KG turns using Llama 3.1's native message boundaries."""
        rendered_messages = [message.model_dump() if isinstance(message, BaseModel) else message for message in messages]
        prompt = cls.bos

        for message in rendered_messages:
            role = message["role"]
            content = str(message.get("content", ""))

            if role == "system":
                prompt += f"{cls.system_prefix}{content}{cls.eos}"
            elif role == "user":
                prompt += f"{cls.user_prefix}{content}{cls.eos}"
            elif role == "assistant":
                if message.get("tool_calls") and "<kg-query>" not in content:
                    content += cls._render_kg_tool_calls(message["tool_calls"])
                prompt += f"{cls.assist_prefix}{content}{cls.eos}"
            elif role in {"function", "tool", "ipython"}:
                query_results = cls._tool_result_text(message.get("content", ""))
                observation = f"<information>\n{query_results}\n</information>"
                prompt += f"{cls.tool_prefix}{observation}{cls.eos}"
            else:
                raise ValueError("no such role!!!", role)

        if add_generation_prompt and rendered_messages and rendered_messages[-1]["role"] in {
            "user",
            "function",
            "tool",
            "ipython",
        }:
            prompt += cls.assist_prefix
        return prompt

    @classmethod
    def _apply_llama3p1_react_inline_template(cls, messages, add_generation_prompt=False):
        """Render native Llama headers with an inline textual ReAct loop."""
        rendered_messages = [message.model_dump() if isinstance(message, BaseModel) else message for message in messages]
        prompt = cls.bos

        for index, message in enumerate(rendered_messages):
            role = message["role"]
            content = str(message.get("content", ""))

            if role == "system":
                prompt += f"{cls.system_prefix}{content}{cls.eos}"
            elif role == "user":
                prompt += f"{cls.user_prefix}{content}{cls.eos}"
            elif role == "assistant":
                if message.get("tool_calls") and "<kg-query>" not in content:
                    content += cls._render_kg_tool_calls(message["tool_calls"])
                previous_role = rendered_messages[index - 1]["role"] if index > 0 else None
                if previous_role not in {"assistant", "function", "tool", "ipython"}:
                    prompt += cls.assist_prefix
                prompt += content
                if not message.get("tool_calls"):
                    prompt += cls.eos
            elif role in {"function", "tool", "ipython"}:
                query_results = cls._tool_result_text(message.get("content", ""))
                prompt += f"{cls.search_result_prefix}{query_results}\n{cls.search_result_suffix}"
            else:
                raise ValueError("no such role!!!", role)

        if add_generation_prompt and rendered_messages and rendered_messages[-1]["role"] == "user":
            prompt += cls.assist_prefix
        return prompt

    @classmethod
    def apply_chat_template(cls, messages, tools=None, add_generation_prompt=False, **kwargs):
        if cls._current_template == "llama3p1_react_inline":
            return cls._apply_llama3p1_react_inline_template(
                messages, add_generation_prompt=add_generation_prompt
            )
        if cls._current_template == "llama3p1":
            return cls._apply_llama3p1_template(messages, add_generation_prompt=add_generation_prompt)
        if cls._current_template == "graphwalker":
            return cls._apply_graphwalker_template(messages, add_generation_prompt=add_generation_prompt)

        prompt = cls.bos
        messages = [message.model_dump() if isinstance(message, BaseModel) else message for message in messages]

        for i, msg in enumerate(messages):

            if msg["role"] == "system":
                prompt += cls.system_prefix + msg["content"]

            elif msg["role"] == "function":
                prompt += cls.tool_prefix + msg["content"]

            elif msg["role"] == "user":
                prompt += cls.user_prefix + msg["content"]

            elif msg["role"] == "assistant":
                if i - 1 >= 0 and not messages[i - 1]["role"] == "user":
                    prompt += msg["content"]
                else:
                    prompt += cls.assist_prefix + msg["content"]

                if isinstance(msg.get("tool_calls", None), list) and len(msg["tool_calls"]) > 0:
                    search_calls = [
                        tool_call
                        for tool_call in msg["tool_calls"]
                        if tool_call["function"].get("name", None) == "search"
                    ]
                    kg_calls = [
                        tool_call
                        for tool_call in msg["tool_calls"]
                        if tool_call["function"].get("name", None) in {"get_relations", "get_triples"}
                    ]
                    for search_call in search_calls:
                        search_query = search_call["function"]["arguments"].get("query_list", "")
                        if isinstance(search_query, list):
                            search_query = "\n".join(search_query)

                        prompt += cls.search_prefix + str(search_query) + cls.search_suffix
                    for kg_call in kg_calls:
                        function_name = kg_call["function"]["name"]
                        arguments = kg_call["function"]["arguments"]
                        if function_name == "get_relations":
                            kg_query = f'get_relations("{arguments.get("entity", "")}")'
                        else:
                            relations = arguments.get("relations", [])
                            serialized_relations = ", ".join(json.dumps(str(item), ensure_ascii=False) for item in relations)
                            kg_query = f'get_triples("{arguments.get("entity", "")}", [{serialized_relations}])'
                        prompt += cls.kg_query_prefix + kg_query + cls.kg_query_suffix

            elif msg["role"] == "tool":
                result = json.loads(msg["content"])
                prompt += cls.search_result_prefix + result.get("result", "") + cls.search_result_suffix

            else:
                raise ValueError("no such role!!!", msg["role"])
        if add_generation_prompt and messages[-1]["role"] == "user":
            prompt = prompt + cls.assist_prefix

        return prompt

SearchChatTemplate._update_class_attributes()


class QwenChatTemplate(SearchChatTemplate):
    system_prefix = "<｜System｜>"
    assist_prefix = "<｜Assistant｜>"
    user_prefix = "<｜User｜>"
    tool_prefix = "<｜Function｜>"
    bos = "<｜begin▁of▁sentence｜>"
    eos = "<｜end▁of▁sentence｜>"
    think_prefix = "<think>"
    think_suffix = "</think>"
    search_prefix = "<search>"
    search_suffix = "</search>"
    kg_query_prefix = "<kg-query>"
    kg_query_suffix = "</kg-query>"
    answer_prefix = "<answer>"
    answer_suffix = "</answer>"
    result_prefix = "<result>"
    result_suffix = "</result>"
    search_result_prefix = "\n<information>\n"
    search_result_suffix = "</information>\n\n"
