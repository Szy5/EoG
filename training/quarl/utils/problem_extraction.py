"""
Problem extraction utilities for self-play training.

This module provides utilities to extract search-based problems from
proposer generation trajectories and format them for the solver phase.
"""

import json
import logging
import os
import random
import re
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, StrictUndefined

from quarl.reward.score.kgqa_answer_em import (
    _collect_normalized_entities,
    compute_entity_f1,
    is_exact_answer_match,
)
from quarl.reward.score.outcome_score import em_check
from quarl.utils.llm_as_a_judge import evaluate_kgqa_triple_answer, get_global_judge

logger = logging.getLogger(__name__)

# Enable debug logging for self-play
SELF_PLAY_DEBUG = os.environ.get("SELF_PLAY_DEBUG", "False").lower() == "true"
if SELF_PLAY_DEBUG:
    logger.setLevel(logging.DEBUG)

_KG_QUERY_INFORMATION_PAIR_PATTERN = re.compile(
    r"<kg-query>\s*(.*?)\s*</kg-query>\s*<information>\s*(.*?)\s*</information>",
    re.DOTALL | re.IGNORECASE,
)

_KGQA_SOLVER_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "template"
_KGQA_SOLVER_SYS_TEMPLATE = _KGQA_SOLVER_TEMPLATE_DIR / "kgqa_ssp_solver_sys.jinja"
_KGQA_SOLVER_USER_TEMPLATE = _KGQA_SOLVER_TEMPLATE_DIR / "kgqa_ssp_solver_user.jinja"
_STRICT_JINJA_ENV = Environment(undefined=StrictUndefined, autoescape=False)


def _as_message_dict(value: Any) -> Dict[str, Any]:
    """Normalize rollout message/tool-call models without coupling to pydantic."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _tool_result_text(content: Any) -> str:
    """Unwrap the JSON envelope used by KG tools while preserving plain text."""
    try:
        payload = json.loads(content) if isinstance(content, str) else content
    except (TypeError, json.JSONDecodeError):
        return str(content)
    if isinstance(payload, dict) and "result" in payload:
        return str(payload["result"])
    return str(content)


def _render_structured_kg_tool_calls(tool_calls: Any) -> str:
    """Render structured KG calls into the textual protocol seen by the model."""
    rendered = []
    for tool_call in tool_calls or []:
        call = _as_message_dict(tool_call)
        function = _as_message_dict(call.get("function", call))
        name = str(function.get("name", ""))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        entity = json.dumps(str(arguments.get("entity", "")), ensure_ascii=False)
        if name == "get_relations":
            rendered.append(f"<kg-query>get_relations({entity})</kg-query>")
        elif name == "get_triples":
            relations = arguments.get("relations", [])
            serialized = ", ".join(json.dumps(str(item), ensure_ascii=False) for item in relations)
            rendered.append(f"<kg-query>get_triples({entity}, [{serialized}])</kg-query>")
    return "".join(rendered)


def render_structured_react_trajectory(messages: Any) -> str:
    """Build extraction text from the rollout's structured message sidecar.

    System/user prompt turns and model-specific role tokens are intentionally
    omitted. Assistant generations and tool observations retain the exact
    textual ReAct ordering expected by problem extraction.
    """
    if isinstance(messages, dict) and "messages" in messages:
        messages = messages["messages"]
    if not isinstance(messages, (list, tuple)):
        return ""

    parts: List[str] = []
    for raw_message in messages:
        message = _as_message_dict(raw_message)
        role = str(message.get("role", ""))
        if role == "assistant":
            content = str(message.get("content", ""))
            if message.get("tool_calls") and "<kg-query>" not in content:
                content += _render_structured_kg_tool_calls(message["tool_calls"])
            parts.append(content)
        elif role in {"tool", "function", "ipython"}:
            result = _tool_result_text(message.get("content", ""))
            parts.append(f"\n<information>\n{result}\n</information>\n\n")
    return "".join(parts)


def _read_required_prompt_template(path: Path) -> str:
    """Read a required training prompt and fail early if packaging omitted it."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required KGQA Solver prompt template not found: {path}") from exc


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class ProblemExtractor:
    """
    Extracts proposer problems from self-play trajectories.

    It supports the legacy search-style formats that emit `<answer>` or
    `<question>`, and the KGQA proposer format that emits `<result>` with a JSON
    payload for the solver phase.
    """

    def __init__(
        self,
        lang="zh",
        use_rag_filter=False,
        use_search_terms_filter=False,
        noisy_rag_materials=0,
        answer_pattern="answer",
        synthesis_mode="kg_path",
        rag_filter_fail_open=False,
        gate_entity_f1_min=0.0,
        gate_reject_question_entity_answers=False,
        gate_reject_question_leaking_gold=False,
    ):
        """Initialize the extractor for legacy search and KGQA proposer outputs."""
        if answer_pattern == "question":
            self.answer_pattern = r"<question>\s*(.*?)\s*</question>"
        elif answer_pattern == "result":
            self.answer_pattern = r"<result>\s*(.*?)\s*</result>"
        else:
            self.answer_pattern = r"<answer>\s*(.*?)\s*</answer>"

        self.material_pattern = r"<(?:information|observation)>\s*(.*?)\s*</(?:information|observation)>"
        self.information_pattern = r"<information>\s*(.*?)\s*</information>"
        self.result_pattern = r"<result>\s*(.*?)\s*</result>"

        self.synthesis_mode = str(synthesis_mode or "kg_path").strip()
        self.rag_filter_fail_open = _to_bool(rag_filter_fail_open)
        self._last_failure_reason: Optional[str] = None

        # Gate applied to synthesized KGQA problems. kg_path no longer checks proposer
        # answers against gold; these knobs only affect leftover answer-list gates
        # such as ``_check_kgqa_gate``. Extra kg_path validation is ``use_rag_filter``.
        self.gate_entity_f1_min = float(gate_entity_f1_min or 0.0)
        self.gate_reject_question_entity_answers = _to_bool(gate_reject_question_entity_answers)
        self.gate_reject_question_leaking_gold = _to_bool(gate_reject_question_leaking_gold)

        # Question verifier: an OpenAI-compatible endpoint, by default a local
        # SGLang server for Qwen2.5-32B-Instruct (scripts/start_verifier_server.sh).
        # Local servers ignore the API key, so a placeholder is accepted.
        judge_base_url = os.getenv("JUDGE_BASE_URL") or "http://127.0.0.1:5000/v1"
        judge_api_key = os.getenv("JUDGE_API_KEY") or "EMPTY"
        judge_model = os.getenv("JUDGE_MODEL")
        if judge_model:
            self.llm_judge = get_global_judge(base_url=judge_base_url, api_key=judge_api_key, model=judge_model)
        else:
            self.llm_judge = None

        self.use_rag_filter = _to_bool(use_rag_filter)

        self.use_search_terms_filter = _to_bool(use_search_terms_filter)

        self.noisy_rag_materials = noisy_rag_materials

        if lang == "R-Search":
            self.solver_system_prompt = """You are a helpful assistant that can solve the given question step by step. For each step, start by explaining your thought process. If additional information is needed, provide a specific query enclosed in <search> and </search>. The system will return the top search results within <observation> and </observation>. You can perform multiple searches as needed. When you know the final answer, use <original_evidence> and </original_evidence> to provide all potentially relevant original information from the observations. Ensure the information is complete and preserves the original wording without modification. If no searches were conducted or observations were made, omit the evidence section. Finally, provide the final answer within <answer> and </answer> tags."""
            self.solver_user_prompt = """{}"""
        else:
            self.solver_system_prompt = """You are a helpful and harmless assistant."""
            self.solver_user_prompt = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> xxx </answer>. Question: {}"""
        # Canonical Solver prompts shared by self-play training and offline KGQA val conversion.
        self.kgqa_solver_system_prompt = _read_required_prompt_template(_KGQA_SOLVER_SYS_TEMPLATE)
        self.kgqa_solver_user_prompt = _read_required_prompt_template(_KGQA_SOLVER_USER_TEMPLATE)

    def _render_kgqa_solver_user_prompt(self, *, question: str, question_entities: Any) -> str:
        return _STRICT_JINJA_ENV.from_string(self.kgqa_solver_user_prompt).render(
            question=question,
            initial_entities=self._format_initial_entities(question_entities),
        )

    @staticmethod
    def _format_initial_entities(question_entities: Any) -> str:
        if not isinstance(question_entities, list) or not question_entities:
            return "(none provided)"

        entity_names = []
        for entity in question_entities:
            if isinstance(entity, dict):
                name = str(entity.get("name", "")).strip()
                if name:
                    entity_names.append(name)
            else:
                normalized = str(entity).strip()
                if normalized:
                    entity_names.append(normalized)
        return ", ".join(entity_names) if entity_names else "(none provided)"

    def build_kgqa_solver_record(
        self,
        *,
        question: str,
        question_entities: List[Dict[str, str]],
        answer_targets: List[str],
        answer_entities: List[Dict[str, str]],
        data_source: str,
        dataset_type: str,
        qid: Any,
        split: str = "val",
        index: int = 0,
        sparql_query: Optional[str] = None,
        path: Optional[List[str]] = None,
        triples: Optional[List[Any]] = None,
        source_question: str = "",
        ability: str = "kgqa-self-play",
    ) -> Dict[str, Any]:
        """Build a solver-phase RL record using the canonical KGQA solver prompts."""
        if not question.strip():
            raise ValueError("question must be non-empty")
        if not answer_targets:
            raise ValueError("answer_targets must be non-empty")

        ground_truth = {"target": list(answer_targets)}

        kgqa_create_kwargs = {
            "data_source": data_source,
            "dataset_type": dataset_type,
            "ground_truth": ground_truth,
            "question": question,
            "qid": qid,
            "question_entities": question_entities,
            "answer_entities": answer_entities,
            "path": path or [],
            "triples": triples or [],
            "sparql_query": sparql_query,
        }
        extra_info = {
            "index": index,
            "qid": qid,
            "split": split,
            "question": question,
            "source_question": source_question or question,
            "source_gold_answers": list(answer_targets),
            "question_entities": question_entities,
            "answer_entities": answer_entities,
            "path": path or [],
            "triples": triples or [],
            "sparql_query": sparql_query,
            "dataset_type": dataset_type,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "get_relations": {"create_kwargs": dict(kgqa_create_kwargs)},
                "get_triples": {"create_kwargs": dict(kgqa_create_kwargs)},
            },
        }

        formatted_prompt = []
        if self.kgqa_solver_system_prompt:
            formatted_prompt.append({"role": "system", "content": self.kgqa_solver_system_prompt})
        formatted_prompt.append(
            {
                "role": "user",
                "content": self._render_kgqa_solver_user_prompt(
                    question=question, question_entities=question_entities
                ),
            }
        )

        return {
            "data_source": data_source,
            "prompt": formatted_prompt,
            "ability": ability,
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "extra_info": extra_info,
            "metadata": {"source_format": f"{dataset_type}_solver_val", "qid": qid},
        }

    def extract_problems_from_trajectory_with_stats(
        self, input_text: str, output_text: str, metadata: Dict[str, Any] = None, batch_materials: List[str] = None
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        从轨迹中抽取生成的问题目前有两种方式：
        1. kg_path：解析 question-only 的 <result>{"question": "..."}</result>，solver gold 沿用 seed path
        2. kg_explore_rag：解析带 answers 的 <result>，再基于证据让外部 LLM 判断问题是否可回答
        
        Extract search-based problems from proposer generation trajectory with detailed statistics.

        Args:
            input_text: The input prompt used for proposer generation
            output_text: The proposer's generated output containing <answer></answer>
            metadata: Additional metadata about the generation
            batch_materials: List of materials from other trajectories in the batch

        Returns:
            Tuple of (extracted problems list, trajectory statistics dict)
        """
        extracted_problems = []
        metadata = metadata or {}

        traj_stats = {
            "answer_matches": 0,
            "format_error": 0,
            "valid_questions": 0,
            "successful_problems": 0,
            "failure_reason": None,  # 添加失败原因
            "rag_diagnostics": None,
        }

        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracting problems from trajectory {metadata.get('trajectory_index', 'unknown')}")

        try:
            if self._should_use_kgqa_result_extraction(output_text, metadata):
                result_match = re.search(self.result_pattern, output_text, re.DOTALL)
                traj_stats["answer_matches"] = int(result_match is not None)
                if result_match is None:
                    traj_stats["format_error"] = 1
                    traj_stats["failure_reason"] = "no_pattern_match"
                    return extracted_problems, traj_stats

                payload = self._parse_result_payload(result_match.group(1))
                if payload is None:
                    traj_stats["format_error"] = 1
                    traj_stats["failure_reason"] = "format_rule_failed"
                    return extracted_problems, traj_stats

                # kg_explore_rag still emits question + answers; kg_path is question-only.
                if self._is_kg_explore_rag_mode(metadata):
                    formatted_problem = self._extract_kg_explore_rag_problem(input_text, output_text, metadata)
                    if formatted_problem:
                        traj_stats["valid_questions"] = 1
                        traj_stats["successful_problems"] = 1
                        extracted_problems.append(formatted_problem)
                    else:
                        traj_stats["format_error"] += 1
                        traj_stats["failure_reason"] = self._last_failure_reason or "invalid_result_payload"
                    return extracted_problems, traj_stats

                question_text, format_reason = self._validate_question_only_payload(payload)
                if format_reason is not None:
                    traj_stats["format_error"] = 1
                    traj_stats["failure_reason"] = format_reason
                    return extracted_problems, traj_stats

                traj_stats["valid_questions"] = 1
                formatted_problem, failure_reason, rag_diagnostics = self._extract_kgqa_problem(
                    output_text, metadata, question_text
                )
                traj_stats["rag_diagnostics"] = rag_diagnostics
                if formatted_problem is not None:
                    traj_stats["successful_problems"] = 1
                    extracted_problems.append(formatted_problem)
                else:
                    traj_stats["failure_reason"] = failure_reason or "problem_construction_failed"
                return extracted_problems, traj_stats

            answer_matches = re.finditer(self.answer_pattern, output_text, re.DOTALL)

            match_list = list(answer_matches)
            traj_stats["answer_matches"] = len(match_list)

            if len(match_list) == 0:
                traj_stats["format_error"] += 1
                traj_stats["failure_reason"] = "no_pattern_match"
                return extracted_problems, traj_stats

            if SELF_PLAY_DEBUG:
                logger.debug(f"Found {len(match_list)} <answer></answer> matches")

            for match_idx, match in enumerate(match_list[:1]):
                question_text = match.group(1).strip()

                question_text = re.sub(r"<\\?/?answer\s*>", "", question_text)
                question_text = re.sub(r"</\s*$", "", question_text)
                question_text = re.sub(r"</answer\s*$", "", question_text)

                # 使用带失败原因的验证方法
                is_valid, failure_reason = self._is_valid_search_question_with_reason(
                    question_text, output_text, metadata, batch_materials
                )
                
                if is_valid:
                    traj_stats["valid_questions"] += 1
                    traj_stats["failure_reason"] = None

                    formatted_problem = self._format_search_problem(question_text, input_text, output_text, metadata)
                    if formatted_problem:
                        extracted_problems.append(formatted_problem)
                        traj_stats["successful_problems"] += 1
                    else:
                        traj_stats["failure_reason"] = "format_problem_failed"
                else:
                    traj_stats["failure_reason"] = failure_reason

        except Exception as e:
            logger.warning(f"Error extracting search problems with stats: {e}")
            traj_stats["failure_reason"] = f"exception: {str(e)}"

        return extracted_problems, traj_stats

    def _extract_information_content(self, output_text: str) -> str:
        """Extract all content from <information>/<observation> tags in the trajectory."""
        information_matches = re.findall(self.material_pattern, output_text, re.DOTALL)
        if information_matches:
            return "\n\n".join(match.strip() for match in information_matches if match.strip())
        return None

    def _is_kg_explore_rag_mode(self, metadata: Optional[Dict[str, Any]]) -> bool:
        """
        判断是否是 kg_explore_rag 模式
        Args:
            metadata: 元数据
        Returns:
            bool: 是否是 kg_explore_rag 模式
        """
        metadata = metadata or {}
        extra_info = metadata.get("extra_info", {})
        mode = self.synthesis_mode
        if isinstance(extra_info, dict) and extra_info.get("synthesis_mode"):
            mode = str(extra_info.get("synthesis_mode", "")).strip()
        return mode == "kg_explore_rag"

    def _extract_search_terms(self, output_text: str) -> List[str]:
        """Extract search terms from <search></search> tags in the output text."""
        search_pattern = r"<search>\s*(.*?)\s*</search>"
        search_matches = re.findall(search_pattern, output_text, re.DOTALL)

        search_terms = []
        for match in search_matches:
            try:
                terms = json.loads(match)
                if isinstance(terms, list):
                    search_terms.extend([term.strip() for term in terms if term.strip()])
                else:
                    search_terms.append(match.strip())
            except (json.JSONDecodeError, ValueError):
                search_terms.append(match.strip())

        return search_terms

    def _validate_with_external_llm(
        self, question_text: str, materials: str, ground_truth: Any, assigned_noisy_materials: List[str] = None
    ) -> bool:
        """Use external LLM to validate if the question can be answered correctly with the materials."""
        if not materials or not ground_truth:
            if SELF_PLAY_DEBUG:
                logger.debug(f"Skipping external validation - no judge or no materials/ground_truth")
            return False

        try:
            final_materials = materials

            if self.noisy_rag_materials > 0 and assigned_noisy_materials:
                all_materials = [materials] + assigned_noisy_materials
                random.shuffle(all_materials)
                final_materials = "\n\n".join(all_materials)

                if SELF_PLAY_DEBUG:
                    logger.debug(
                        f"Added {len(assigned_noisy_materials)} pre-assigned noisy RAG materials to validation, material: {final_materials[:100]}..."
                    )

            llm_answer = self.llm_judge.model_based_answer(final_materials, question_text)

            if not llm_answer:
                if SELF_PLAY_DEBUG:
                    logger.debug(f"External LLM returned empty answer for question: {question_text[:100]}...")
                return False

            answer_match = re.search(r"Answer[:：]?\s*(.*)", llm_answer, re.IGNORECASE)
            if answer_match:
                llm_answer = answer_match.group(1).strip()

            is_consistent = (
                em_check(
                    prediction=llm_answer,
                    golden_answers=ground_truth,
                )
                == 1
            )

            if not is_consistent:
                is_consistent = self.llm_judge.model_based_match(
                    question=question_text, golden_answer=ground_truth, model_answer=llm_answer
                )

            if SELF_PLAY_DEBUG:
                logger.debug(f"External validation result for question '{question_text[:50]}...': {is_consistent}")
                logger.debug(f"  Ground truth: {ground_truth}")
                logger.debug(f"  LLM answer: {llm_answer[:100]}...")
                logger.debug(f"LLM Judge is consistent: {is_consistent}")

            return is_consistent

        except Exception as e:
            logger.warning(f"External LLM validation failed: {e}")
            return True

    def _is_valid_search_question_with_reason(
        self,
        question_text: str,
        output_text: str = "",
        metadata: Dict = None,
        assigned_noisy_materials: List[str] = None,
    ) -> tuple:
        """
        Check if the extracted question is a valid search-based question.
        Returns: (is_valid: bool, failure_reason: str or None)
        """
        # Step 1: 检查问题长度和单词数
        if not question_text:
            return False, "no_question_text"
        if len(question_text.strip()) < 10:
            return False, "question_too_short"
        if len(question_text.strip()) > 1000:
            return False, "question_too_long"
        if len(question_text.split(" ")) < 10:
            return False, "word_count_less_than_10"

        # Step 2: 处理 ground_truth
        reward_model = metadata.get("reward_model", {})
        ground_truth = None
        ground_truth_str = None
        if isinstance(reward_model, dict):
            ground_truth_info = reward_model.get("ground_truth")
            if isinstance(ground_truth_info, dict):
                ground_truth = ground_truth_info.get("target")
            elif ground_truth_info is not None:
                ground_truth = ground_truth_info

        if ground_truth is not None:
            if isinstance(ground_truth, list):
                ground_truth_str = " ".join(str(g) for g in ground_truth)
            else:
                ground_truth_str = str(ground_truth)
            
            # Step 3: 检查 ground_truth 是否在问题中
            if ground_truth_str.lower() in question_text.lower():
                return False, "ground_truth_in_question"

        # Step 4: 检查 materials
        materials = self._extract_information_content(output_text)
        if not materials:
            return False, "no_materials"

        # Step 5: 检查搜索次数
        search_terms = self._extract_search_terms(output_text)
        if len(search_terms) <= 1:
            return False, "search_count_le_1"

        # Step 6: 检查重复搜索词
        if len(search_terms) != len(set(search_terms)):
            return False, "duplicate_search_terms"

        # Step 7: 检查 ground_truth 是否在搜索词中
        if self.use_search_terms_filter and ground_truth is not None:
            for search_term in search_terms:
                if ground_truth_str.lower() in search_term.lower():
                    return False, "ground_truth_in_search_term"

        # Step 8: RAG Filter 验证
        if self.llm_judge and self.use_rag_filter:
            rag_result = self._validate_with_external_llm(question_text, materials, ground_truth_str, assigned_noisy_materials)
            if not rag_result:
                return False, "rag_filter_failed"

        return True, None

    def _is_valid_search_question(
        self,
        question_text: str,
        output_text: str = "",
        metadata: Dict = None,
        assigned_noisy_materials: List[str] = None,
    ) -> bool:
        """Check if the extracted question is a valid search-based question."""
        is_valid, reason = self._is_valid_search_question_with_reason(
            question_text, output_text, metadata, assigned_noisy_materials
        )
        if not is_valid and SELF_PLAY_DEBUG:
            logger.debug(f"[Question Validation] Failed: {reason}, question: {question_text[:100] if question_text else 'None'}...")
        return is_valid

    def _should_use_kgqa_result_extraction(self, output_text: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """
        把 Proposer 的输出结构按照格式进行解析

        """
        if "<result>" in output_text and "</result>" in output_text:
            return True

        metadata = metadata or {}
        extra_info = metadata.get("extra_info", {})
        dataset_type = ""
        if isinstance(extra_info, dict):
            dataset_type = str(extra_info.get("dataset_type", "")).strip().lower()
        return dataset_type in {
            "grailqa",
            "webqsp",
            "graphqa",
            "kgqa",
            "gtsqa",
            "cwq",
            "car_kgqa",
        }

    @staticmethod
    def _parse_result_payload(payload_text: str) -> Optional[Dict[str, Any]]:
        """Parse the content of an already matched ``<result>`` tag."""
        payload_str = str(payload_text or "").strip()
        if not payload_str:
            return None

        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None
        return payload

    def _extract_result_payload(self, output_text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse a complete ``<result>`` payload."""
        match = re.search(self.result_pattern, output_text, re.DOTALL)
        if match is None:
            return None
        return self._parse_result_payload(match.group(1))

    @staticmethod
    def _validate_question_only_payload(payload: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
        """Validate the kg_path Proposer ``<result>`` JSON contract: ``{"question": "..."}`` only."""
        if payload is None:
            return None, "format_rule_failed"
        if set(payload.keys()) != {"question"}:
            return None, "format_rule_failed"
        raw_question = payload.get("question")
        if not isinstance(raw_question, str) or not raw_question.strip():
            return None, "format_rule_failed"
        return raw_question.strip(), None

    @staticmethod
    def question_leaks_gold_answer(question_text: str, source_gold_answers: List[str]) -> bool:
        """Return True when any gold answer appears in the question text."""
        if not question_text:
            return False
        lowered_question = question_text.lower()
        for gold in source_gold_answers:
            gold_text = str(gold).strip()
            if gold_text and gold_text.lower() in lowered_question:
                return True
        return False

    @staticmethod
    def _parse_information_triples(info_text: str) -> List[Dict[str, str]]:
        info_text = str(info_text or "").strip()
        if not info_text:
            return []

        triples: List[Dict[str, str]] = []
        try:
            decoded = json.loads(info_text)
        except json.JSONDecodeError:
            decoded = None

        if isinstance(decoded, list):
            for item in decoded:
                if isinstance(item, dict):
                    subject = item.get("head") or item.get("subject") or item.get("entity")
                    relation = item.get("relation")
                    obj = item.get("tail") or item.get("object")
                    if subject and relation and obj:
                        triples.append(
                            {
                                "subject": str(subject).strip(),
                                "relation": str(relation).strip(),
                                "object": str(obj).strip(),
                            }
                        )
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    triples.append(
                        {
                            "subject": str(item[0]).strip(),
                            "relation": str(item[1]).strip(),
                            "object": str(item[2]).strip(),
                        }
                    )
            if triples:
                return triples

        for line in info_text.splitlines():
            line = line.strip()
            if not (line.startswith("[") and line.endswith("]")):
                continue
            inner = line[1:-1]
            parts = [part.strip() for part in inner.split(",")]
            if len(parts) < 3:
                continue
            if len(parts) == 3:
                subject, relation, obj = parts
            else:
                subject, obj = parts[0], parts[-1]
                relation = ",".join(parts[1:-1]).strip()
            triples.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                }
            )
        return triples

    @classmethod
    def _extract_exploration_triples_from_trajectory(cls, output_text: str) -> List[Dict[str, str]]:
        """Collect triples returned by real get_triples tool calls in the trajectory."""
        triples: List[Dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for query, info in _KG_QUERY_INFORMATION_PAIR_PATTERN.findall(output_text or ""):
            if "get_triples" not in query:
                continue
            for triple in cls._parse_information_triples(info):
                key = (triple["subject"], triple["relation"], triple["object"])
                if key in seen:
                    continue
                seen.add(key)
                triples.append(triple)
        return triples

    @staticmethod
    def _format_exploration_triples_for_rag(triples: List[Dict[str, str]]) -> str:
        if not triples:
            return "(no triples available)"
        return "\n".join(
            f"- [{triple['subject']}, {triple['relation']}, {triple['object']}]" for triple in triples
        )

    def _validate_rag_hit_at_1(
        self,
        *,
        question_text: str,
        question_entities: List[Dict[str, str]],
        source_gold_answers: List[str],
        output_text: str,
        qid: Any = None,
    ) -> tuple[bool, Optional[str], List[str], float, str, Dict[str, Any]]:
        """Optionally run the triple-answer verifier and compare with seed gold via Hit@1.

        When ``use_rag_filter`` is off, skip the LLM call and accept after the
        question-only / leak / path-gold checks. Main's copy of this method
        returned ``rag_hit_at_1_failed`` when the switch was False; that would
        zero out GTSQA extraction.
        """
        diagnostics: Dict[str, Any] = {
            "status": "pending",
            "rag_invoked": False,
            "accepted": False,
            "fail_open": self.rag_filter_fail_open,
            "model": str(
                getattr(self.llm_judge, "model", "")
                or os.getenv("JUDGE_MODEL")
                or ""
            ),
            "question": question_text,
            "qid": qid,
            "question_entities": deepcopy(question_entities),
            "gold_answers": list(source_gold_answers),
            "triple_count": 0,
            "triples": [],
            "raw_answer": "",
            "predicted_answers": [],
            "hit_at_1": 0.0,
            "duration_seconds": 0.0,
            "error_type": None,
            "error_message": None,
        }
        if not self.use_rag_filter:
            diagnostics["status"] = "disabled"
            diagnostics["accepted"] = True
            return True, None, [], 0.0, "", diagnostics
        if self.llm_judge is None:
            diagnostics["status"] = (
                "judge_unavailable_fail_open" if self.rag_filter_fail_open else "judge_unavailable"
            )
            diagnostics["accepted"] = self.rag_filter_fail_open
            if self.rag_filter_fail_open:
                return True, None, [], 0.0, "", diagnostics
            return False, "rag_judge_unavailable", [], 0.0, "", diagnostics

        exploration_triples = self._extract_exploration_triples_from_trajectory(output_text)
        diagnostics["triple_count"] = len(exploration_triples)
        diagnostics["triples"] = deepcopy(exploration_triples)
        if not exploration_triples:
            diagnostics["status"] = "no_triples"
            return False, "rag_no_triples", [], 0.0, "", diagnostics

        triples_text = self._format_exploration_triples_for_rag(exploration_triples)
        initial_entities = self._format_initial_entities(question_entities)
        diagnostics["rag_invoked"] = True
        started_at = time.perf_counter()
        try:
            evaluation = evaluate_kgqa_triple_answer(
                self.llm_judge,
                question=question_text,
                question_entities=initial_entities,
                triples_text=triples_text,
                gold_answers=source_gold_answers,
            )
        except Exception as exc:
            diagnostics["duration_seconds"] = time.perf_counter() - started_at
            diagnostics["status"] = (
                "api_error_fail_open" if self.rag_filter_fail_open else "api_error_fail_closed"
            )
            diagnostics["accepted"] = self.rag_filter_fail_open
            diagnostics["error_type"] = type(exc).__name__
            diagnostics["error_message"] = str(exc)
            logger.warning(
                "KGQA RAG verifier call failed: qid=%s model=%s question=%r error_type=%s error=%s",
                qid,
                diagnostics["model"],
                question_text[:200],
                diagnostics["error_type"],
                exc,
                exc_info=SELF_PLAY_DEBUG,
            )
            if self.rag_filter_fail_open:
                return True, None, [], 0.0, "", diagnostics
            return False, "rag_api_failure", [], 0.0, "", diagnostics

        diagnostics["duration_seconds"] = time.perf_counter() - started_at
        llm_answer = str(evaluation.raw_answer or "")
        predicted_answers = list(evaluation.predicted_answers)
        hit_at_1 = float(evaluation.hit_at_1)
        diagnostics.update(
            {
                "raw_answer": llm_answer,
                "predicted_answers": predicted_answers,
                "hit_at_1": hit_at_1,
            }
        )
        if not predicted_answers:
            diagnostics["status"] = "empty_or_unparseable_answer"
            return False, "rag_parse_failure", [], 0.0, llm_answer, diagnostics

        if hit_at_1 < 1.0:
            diagnostics["status"] = "hit_at_1_failed"
            return (
                False,
                "rag_hit_at_1_failed",
                predicted_answers,
                hit_at_1,
                llm_answer,
                diagnostics,
            )
        diagnostics["status"] = "passed"
        diagnostics["accepted"] = True
        return True, None, predicted_answers, hit_at_1, llm_answer, diagnostics

    def _apply_kgqa_verifier_gates(
        self,
        *,
        question_text: str,
        source_gold_answers: List[str],
        question_entities: List[Dict[str, str]],
        output_text: str,
        qid: Any = None,
    ) -> tuple[bool, Optional[str], List[str], float, str, Dict[str, Any]]:
        """Apply leak-gold then optional RAG Hit@1 when ``PROPOSER_RAG_FILTER`` is on."""
        if not source_gold_answers:
            return False, "no_source_gold_answers", [], 0.0, "", {
                "status": "skipped_no_source_gold_answers",
                "rag_invoked": False,
                "accepted": False,
                "question": question_text,
                "gold_answers": [],
                "raw_answer": "",
            }
        if self.question_leaks_gold_answer(question_text, source_gold_answers):
            return False, "question_leaks_gold_answer", [], 0.0, "", {
                "status": "skipped_question_leaks_gold",
                "rag_invoked": False,
                "accepted": False,
                "question": question_text,
                "gold_answers": list(source_gold_answers),
                "raw_answer": "",
            }
        return self._validate_rag_hit_at_1(
            question_text=question_text,
            question_entities=question_entities,
            source_gold_answers=source_gold_answers,
            output_text=output_text,
            qid=qid,
        )

    @staticmethod
    def _normalize_answer_targets(raw_answers: Any) -> List[str]:
        """
        标准化答案列表：
        1. 如果答案是字符串，则转换为列表 (不区分大小写)
        2. 如果答案是列表，则转换为列表 (不区分大小写)
        3. 如果答案是其他类型，则返回空列表
        """
        if isinstance(raw_answers, str):
            raw_text = raw_answers.strip()
            try:
                decoded = json.loads(raw_text)
            except json.JSONDecodeError:
                decoded = raw_text
            if isinstance(decoded, list):
                answers = decoded
            else:
                answers = [decoded]
        elif isinstance(raw_answers, list):
            answers = raw_answers
        else:
            return []

        normalized_answers = []
        for answer in answers:
            normalized_answer = str(answer).strip()
            if normalized_answer:
                normalized_answers.append(normalized_answer)
        return normalized_answers

    @staticmethod
    def _normalize_entity_records(raw_entities: Any) -> List[Dict[str, str]]:
        if not isinstance(raw_entities, list):
            return []

        normalized: List[Dict[str, str]] = []
        for entity in raw_entities:
            if isinstance(entity, dict):
                if "name" in entity:
                    name = str(entity.get("name", "")).strip()
                    entity_id = str(entity.get("id", "")).strip()
                    if name:
                        normalized.append({"name": name, "id": entity_id})
                else:
                    for name, entity_id in entity.items():
                        normalized_name = str(name).strip()
                        if normalized_name:
                            normalized.append({"name": normalized_name, "id": str(entity_id).strip()})
            else:
                name = str(entity).strip()
                if name:
                    normalized.append({"name": name, "id": ""})
        return normalized

    @staticmethod
    def _answer_entities_from_targets(answer_targets: List[str]) -> List[Dict[str, str]]:
        return [{"name": str(answer).strip(), "id": ""} for answer in answer_targets if str(answer).strip()]

    @staticmethod
    def _strip_answer_prefix(answer_text: str) -> str:
        answer_text = str(answer_text or "").strip()
        answer_match = re.search(r"Answer[:：]?\s*(.*)", answer_text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        return answer_text

    def _validate_kg_evidence_with_external_llm(
        self,
        *,
        question_text: str,
        question_entities: List[Dict[str, str]],
        evidence: str,
        proposer_answers: List[str],
    ) -> tuple[bool, Optional[str], str]:
        """
        基于证据让外部LLM来判断问题是否可回答
        Args:
            question_text: 问题文本
            question_entities: 问题实体
            evidence: 证据
            proposer_answers: Proposer 的答案
        Returns:
            tuple[bool, Optional[str], str]: 是否通过，失败原因，LLM 答案
        """
        if not self.use_rag_filter:
            return True, None, ""

        if not evidence:
            return False, "no_kg_evidence", ""
        if not proposer_answers:
            return False, "no_proposer_answers", ""
        if self.llm_judge is None:
            if self.rag_filter_fail_open:
                return True, "kg_evidence_rag_judge_unavailable_fail_open", ""
            return False, "kg_evidence_rag_judge_unavailable", ""

        initial_entities = self._format_initial_entities(question_entities)
        judge_question = (
            "Answer the KGQA question using only the provided KG evidence. "
            "If the evidence is insufficient, say 'Answer: unknown'. "
            "Return the final answer after 'Answer:'.\n"
            f"Question: {question_text}\n"
            f"Initial Entities: {initial_entities}"
        )

        try:
            llm_answer = self.llm_judge.model_based_answer(evidence, judge_question)
            normalized_llm_answer = self._strip_answer_prefix(llm_answer)
            if not normalized_llm_answer:
                return False, "kg_evidence_rag_empty_answer", ""

            parsed_answers = self._normalize_answer_targets(normalized_llm_answer)
            is_consistent = is_exact_answer_match(parsed_answers, proposer_answers)
            if not is_consistent:
                is_consistent = (
                    em_check(prediction=normalized_llm_answer, golden_answers=proposer_answers) == 1
                )
            if not is_consistent and hasattr(self.llm_judge, "model_based_match"):
                is_consistent = self.llm_judge.model_based_match(
                    question=question_text,
                    golden_answer=proposer_answers,
                    model_answer=normalized_llm_answer,
                )
            if not is_consistent:
                return False, "kg_evidence_rag_filter_failed", normalized_llm_answer
            return True, None, normalized_llm_answer
        except Exception as exc:
            logger.warning("KG evidence RAG validation failed: %s", exc)
            if self.rag_filter_fail_open:
                return True, "kg_evidence_rag_exception_fail_open", ""
            return False, "kg_evidence_rag_filter_exception", ""

    def _extract_source_gold_answers(
        self, reward_model: Dict[str, Any], formatted_extra_info: Dict[str, Any]
    ) -> List[str]:
        answer_entity_targets = self._normalize_answer_targets(
            [entity.get("name", "") for entity in formatted_extra_info.get("answer_entities", []) if isinstance(entity, dict)]
        )
        if answer_entity_targets:
            return answer_entity_targets

        gold_answers: Any = []
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict):
            gold_answers = ground_truth.get("target", [])
        elif ground_truth is not None:
            gold_answers = ground_truth

        return self._normalize_answer_targets(gold_answers)

    def _check_kgqa_gate(
        self,
        question_text: str,
        proposer_answers: List[str],
        source_gold_answers: List[str],
        extra_info: Dict[str, Any],
    ) -> Optional[str]:
        """Decide whether a synthesized KGQA problem may enter the solver batch.

        Returns a failure reason string, or None when the problem passes.

        The synthesized problem's gold answers are copied verbatim from the seed KG path, so a
        proposer that merely mentions one of them somewhere in its own answer list produces a
        mislabelled problem. That was invisible while the solver was scored by "at least one
        overlapping entity"; it is not once the solver is scored by entity F1.
        """
        if not proposer_answers:
            return "no_proposer_answers"

        if self.gate_reject_question_entity_answers:
            question_entity_names = [
                entity.get("name")
                for entity in (extra_info.get("question_entities") or [])
                if isinstance(entity, dict) and entity.get("name")
            ]
            if _collect_normalized_entities(question_entity_names) & _collect_normalized_entities(proposer_answers):
                # A question entity is given to the solver in the prompt, so it can never be the
                # answer. Pure error signal, no false positives.
                return "answer_contains_question_entity"

        if self.gate_reject_question_leaking_gold and self.question_leaks_gold_answer(
            question_text, source_gold_answers
        ):
            return "question_leaks_gold_answer"

        if self.gate_entity_f1_min > 0.0:
            if compute_entity_f1(proposer_answers, source_gold_answers) < self.gate_entity_f1_min:
                return "answer_entity_f1_below_threshold"
            return None

        if not is_exact_answer_match(proposer_answers, source_gold_answers):
            return "answer_gold_mismatch"
        return None

    def _extract_kgqa_problem(
        self,
        output_text: str,
        metadata: Optional[Dict[str, Any]],
        question_text: str,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        """Build a kg_path solver sample from a question-only Proposer result.

        The synthesized problem inherits gold answers from the seed KG path. The
        Proposer is not asked to emit ``answers``; missing that field is not a
        gate failure.
        """
        metadata = metadata or {}
        self._last_failure_reason = None
        source_extra_info = deepcopy(metadata.get("extra_info", {}))
        formatted_extra_info = source_extra_info if isinstance(source_extra_info, dict) else {}
        reward_model = deepcopy(metadata.get("reward_model", {"ground_truth": {"style": "rule"}}))
        source_gold_answers = self._extract_source_gold_answers(reward_model, formatted_extra_info)
        question_entities = self._normalize_entity_records(formatted_extra_info.get("question_entities", []))

        if not question_text:
            self._last_failure_reason = "no_question_text"
            return None, self._last_failure_reason, {"status": "skipped_no_question_text", "rag_invoked": False}
        if not source_gold_answers:
            self._last_failure_reason = "no_source_gold_answers"
            return None, self._last_failure_reason, {"status": "skipped_no_source_gold_answers", "rag_invoked": False}

        passed, failure_reason, rag_predicted_answers, rag_hit_at_1, rag_raw_answer, rag_diagnostics = (
            self._apply_kgqa_verifier_gates(
                question_text=question_text,
                source_gold_answers=source_gold_answers,
                question_entities=question_entities,
                output_text=output_text,
                qid=formatted_extra_info.get("qid"),
            )
        )
        if not passed:
            self._last_failure_reason = failure_reason or "problem_construction_failed"
            if SELF_PLAY_DEBUG:
                logger.debug(
                    "Rejecting KGQA proposer result (%s). trajectory_index=%s gold_answers=%s",
                    failure_reason,
                    metadata.get("trajectory_index", -1),
                    source_gold_answers,
                )
            return None, self._last_failure_reason, deepcopy(rag_diagnostics)

        ground_truth = reward_model.get("ground_truth")
        if not isinstance(ground_truth, dict):
            ground_truth = {"target": source_gold_answers}
            reward_model["ground_truth"] = ground_truth
        else:
            ground_truth["target"] = source_gold_answers

        data_source = metadata.get("data_source", "self_generated")
        kgqa_create_kwargs = {
            "data_source": data_source,
            "dataset_type": formatted_extra_info.get("dataset_type", "kgqa"),
            "qid": formatted_extra_info.get("qid"),
            "question": question_text,
            "ground_truth": {"target": source_gold_answers},
            "question_entities": question_entities,
            "answer_entities": formatted_extra_info.get("answer_entities", []),
            "path": formatted_extra_info.get("path", []),
            "triples": formatted_extra_info.get("triples", []),
            "sparql_query": formatted_extra_info.get("sparql_query"),
        }

        formatted_extra_info.update(
            {
                "question": question_text,
                "source_question": formatted_extra_info.get("source_question", ""),
                "source_gold_answers": source_gold_answers,
                "rag_predicted_answers": rag_predicted_answers,
                "rag_hit_at_1": rag_hit_at_1,
                "rag_verifier_answer": rag_raw_answer,
                "rag_diagnostics": deepcopy(rag_diagnostics),
                "need_tools_kwargs": True,
                "split": formatted_extra_info.get("split", "train"),
                "tools_kwargs": {
                    "get_relations": {"create_kwargs": dict(kgqa_create_kwargs)},
                    "get_triples": {"create_kwargs": dict(kgqa_create_kwargs)},
                },
            }
        )

        formatted_prompt = []
        if self.kgqa_solver_system_prompt:
            formatted_prompt.append({"role": "system", "content": self.kgqa_solver_system_prompt})
        formatted_prompt.append(
            {
                "role": "user",
                "content": self._render_kgqa_solver_user_prompt(
                    question=question_text, question_entities=question_entities
                ),
            }
        )

        problem = {
            "data_source": data_source,
            "prompt": formatted_prompt,
            "ability": metadata.get("ability", "kgqa-self-play"),
            "reward_model": reward_model,
            "extra_info": formatted_extra_info,
            "metadata": None,
            "extracted_question": question_text,
            "formatted_prompt": formatted_prompt,
            "problem_type": "kgqa",
            "trajectory_index": metadata.get("trajectory_index", -1),
        }
        return problem, None, deepcopy(rag_diagnostics)

    def _extract_kg_explore_rag_problem(
        self, input_text: str, output_text: str, metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract a KG free-exploration proposer result validated by KG evidence RAG.

        """
        self._last_failure_reason = None
        metadata = metadata or {}
        payload = self._extract_result_payload(output_text)
        if payload is None:
            self._last_failure_reason = "invalid_result_payload"
            return None

        question_text = str(payload.get("question", "")).strip()
        proposer_answers = self._normalize_answer_targets(payload.get("answers"))
        if not question_text:
            self._last_failure_reason = "no_question_text"
            return None
        if not proposer_answers:
            self._last_failure_reason = "no_proposer_answers"
            return None

        source_extra_info = deepcopy(metadata.get("extra_info", {}))
        formatted_extra_info = source_extra_info if isinstance(source_extra_info, dict) else {}
        payload_question_entities = self._normalize_entity_records(payload.get("question_entities"))
        question_entities = payload_question_entities or formatted_extra_info.get("question_entities", [])
        question_entities = self._normalize_entity_records(question_entities)
        answer_entities = self._normalize_entity_records(payload.get("answer_entities"))
        if not answer_entities:
            answer_entities = self._answer_entities_from_targets(proposer_answers)
        # Step 1： 提取 探索过程中的三元组 证据
        evidence = self._extract_information_content(output_text) or ""
        # Step 2：基于证据让外部LLM来判断问题是否可回答
        rag_pass, failure_reason, llm_answer = self._validate_kg_evidence_with_external_llm(
            question_text=question_text,
            question_entities=question_entities,
            evidence=evidence,
            proposer_answers=proposer_answers,
        )
        if not rag_pass:
            self._last_failure_reason = failure_reason or "kg_evidence_rag_filter_failed"
            return None

        reward_model = deepcopy(metadata.get("reward_model", {"ground_truth": {"style": "rule"}}))
        ground_truth = reward_model.get("ground_truth")
        if not isinstance(ground_truth, dict):
            ground_truth = {}
            reward_model["ground_truth"] = ground_truth
        ground_truth["target"] = proposer_answers
        reward_model["style"] = reward_model.get("style", "rule")

        data_source = metadata.get("data_source", "self_generated")
        kgqa_create_kwargs = {
            "data_source": data_source,
            "dataset_type": formatted_extra_info.get("dataset_type", "kgqa"),
            "qid": formatted_extra_info.get("qid"),
            "question": question_text,
            "ground_truth": {"target": proposer_answers},
            "question_entities": question_entities,
            "answer_entities": answer_entities,
            "path": formatted_extra_info.get("path", []),
            "triples": formatted_extra_info.get("triples", []),
            "sparql_query": formatted_extra_info.get("sparql_query"),
        }

        formatted_extra_info.update(
            {
                "question": question_text,
                "source_question": formatted_extra_info.get("source_question", ""),
                "source_gold_answers": self._extract_source_gold_answers(
                    metadata.get("reward_model", {}), formatted_extra_info
                ),
                "proposer_answers": proposer_answers,
                "question_entities": question_entities,
                "answer_entities": answer_entities,
                "synthesis_mode": "kg_explore_rag",
                "kg_evidence_rag_pass": True,
                "kg_evidence_rag_answer": llm_answer,
                "kg_evidence_chars": len(evidence),
                "need_tools_kwargs": True,
                "split": formatted_extra_info.get("split", "train"),
                "tools_kwargs": {
                    "get_relations": {"create_kwargs": dict(kgqa_create_kwargs)},
                    "get_triples": {"create_kwargs": dict(kgqa_create_kwargs)},
                },
            }
        )

        formatted_prompt = []
        if self.kgqa_solver_system_prompt:
            formatted_prompt.append({"role": "system", "content": self.kgqa_solver_system_prompt})
        formatted_prompt.append(
            {
                "role": "user",
                "content": self._render_kgqa_solver_user_prompt(
                    question=question_text, question_entities=question_entities
                ),
            }
        )

        return {
            "data_source": data_source,
            "prompt": formatted_prompt,
            "ability": metadata.get("ability", "kgqa-explore-rag-self-play"),
            "reward_model": reward_model,
            "extra_info": formatted_extra_info,
            "metadata": None,
            "extracted_question": question_text,
            "formatted_prompt": formatted_prompt,
            "problem_type": "kg_explore_rag",
            "trajectory_index": metadata.get("trajectory_index", -1),
        }

    def _format_search_problem(
        self, question_text: str, input_text: str, output_text: str, metadata: Dict
    ) -> Optional[Dict[str, Any]]:
        """Format an extracted search question into solver phase format."""
        try:
            if SELF_PLAY_DEBUG:
                logger.debug(f"Formatting search problem: {question_text[:100]}...")
                logger.debug(f"Metadata received: {metadata}")

            formatted_prompt = []
            if self.solver_system_prompt:
                formatted_prompt.append({"role": "system", "content": self.solver_system_prompt})

            formatted_prompt.append(
                {
                    "role": "user",
                    "content": self.solver_user_prompt.format(question_text),
                }
            )

            reward_model = metadata.get("reward_model", {"ground_truth": {"style": "rule"}})

            data_source = metadata.get("data_source", "self_generated")
            source_extra_info = deepcopy(metadata.get("extra_info", {}))
            formatted_extra_info = source_extra_info if isinstance(source_extra_info, dict) else {}
            formatted_extra_info.update(
                {
                    "question": question_text,
                    "need_tools_kwargs": True,
                    "split": formatted_extra_info.get("split", "train"),
                    "tools_kwargs": {
                        "search": {
                            "create_kwargs": {
                                "data_source": "self_generated",
                                "question": question_text,
                                "ground_truth": reward_model.get("ground_truth"),
                            }
                        }
                    },
                }
            )

            if SELF_PLAY_DEBUG:
                logger.debug(f"Using reward_model: {reward_model}")
                logger.debug(f"Using data_source: {data_source}")

            formatted_problem = {
                "data_source": data_source,
                "prompt": formatted_prompt,
                "ability": "fact-reasoning",
                "reward_model": reward_model,
                "extra_info": formatted_extra_info,
                "metadata": None,
                "extracted_question": question_text,
                "formatted_prompt": formatted_prompt,
                "problem_type": "search",
                "trajectory_index": metadata.get("trajectory_index", -1),
            }

            if SELF_PLAY_DEBUG:
                logger.debug(f"Successfully formatted problem: {formatted_problem['extracted_question']}")

            return formatted_problem

        except Exception as e:
            logger.warning(f"Error formatting search problem: {e}")
            if SELF_PLAY_DEBUG:
                logger.debug(f"Exception details: {e}", exc_info=True)
                logger.debug(f"Failed on question: {question_text}")
                logger.debug(f"Failed with metadata: {metadata}")
            return None


def _process_single_trajectory(
    trajectory_data: tuple[int, Dict[str, Any], ProblemExtractor, List[str]],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process a single trajectory for problem extraction.

    Args:
        trajectory_data: Tuple of (trajectory_index, trajectory_dict, extractor, assigned_noisy_materials)

    Returns:
        Tuple of (extracted problems list, trajectory statistics dict)
    """
    i, trajectory, extractor, assigned_noisy_materials = trajectory_data
    problems = []

    traj_stats = {
        "answer_matches": 0,
        "valid_questions": 0,
        "successful_problems": 0,
        "format_error": 0,
    }

    try:
        input_text = trajectory.get("input", "")
        output_text = trajectory.get("output", "")
        metadata = trajectory.get("metadata", {}).copy()

        if SELF_PLAY_DEBUG:
            logger.debug(f"Processing trajectory {i}")
        #-----------------------------------------------------------------------------------------
        #  Proposer： 从轨迹中进行问题抽取
        #-----------------------------------------------------------------------------------------
        problems, traj_stats = extractor.extract_problems_from_trajectory_with_stats(
            input_text, output_text, metadata, assigned_noisy_materials
        )

        if SELF_PLAY_DEBUG:
            logger.debug(f"Trajectory {i} yielded {len(problems)} problems")

    except Exception as e:
        logger.warning(f"Error processing trajectory {i}: {e}")
        if SELF_PLAY_DEBUG:
            logger.debug(f"Exception details for trajectory {i}: {e}", exc_info=True)

    return problems, traj_stats


def extract_problems_batch(
    trajectories: List[Dict[str, Any]], extractor: ProblemExtractor = None, max_workers: Optional[int] = None
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract search-based problems from a batch of proposer generation trajectories using multi-threading.

    Args:
        trajectories: List of trajectory dictionaries with 'input', 'output', and optional 'metadata'
        extractor: ProblemExtractor instance (creates default if None)
        max_workers: Maximum number of threads to use (defaults to min(32, number of trajectories))

    Returns:
        Tuple of (extracted problems list, extraction statistics dict)
    """
    if extractor is None:
        extractor = ProblemExtractor()

    if not trajectories:
        return [], {
            "trajectories_count": 0,
            "answer_matches_count": 0,
            "valid_questions_count": 0,
            "successful_problems_count": 0,
            "format_error_count": 0,
            "failure_reasons": {},
            "pass_count": 0,
            "trajectory_outcomes": [],
        }

    if max_workers is None:
        max_workers = min(32, len(trajectories))

    all_problems = []

    stats = {
        "trajectories_count": len(trajectories),
        "answer_matches_count": 0,
        "valid_questions_count": 0,
        "successful_problems_count": 0,
        "format_error_count": 0,
        "failure_reasons": {},  # 详细失败原因统计
        "pass_count": 0,
        "trajectory_outcomes": [None] * len(trajectories),
    }

    if SELF_PLAY_DEBUG:
        logger.debug(f"Starting batch extraction from {len(trajectories)} trajectories using {max_workers} threads")

    trajectory_data = []
    if extractor.noisy_rag_materials > 0:
        all_materials = []
        for trajectory in trajectories:
            output_text = trajectory.get("output", "")
            materials = extractor._extract_information_content(output_text)
            if materials:
                all_materials.append(materials)

        if SELF_PLAY_DEBUG:
            logger.debug(
                f"Extracted {len(all_materials)} materials for noisy RAG from {len(trajectories)} trajectories"
            )

        for i, trajectory in enumerate(trajectories):
            current_materials = extractor._extract_information_content(trajectory.get("output", ""))

            available_materials = [mat for mat in all_materials if mat != current_materials]
            assigned_noisy_materials = []

            if available_materials and current_materials:
                num_to_select = min(extractor.noisy_rag_materials, len(available_materials))
                if num_to_select > 0:
                    assigned_noisy_materials = random.sample(available_materials, num_to_select)

                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: assigned {num_to_select} noisy materials")

            trajectory_data.append((i, trajectory, extractor, assigned_noisy_materials))
    else:
        trajectory_data = [(i, trajectory, extractor, []) for i, trajectory in enumerate(trajectories)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(_process_single_trajectory, data): data[0] for data in trajectory_data}

        for future in as_completed(future_to_index):
            trajectory_index = future_to_index[future]
            try:
                problems, traj_stats = future.result()

                all_problems.extend(problems)

                if traj_stats["answer_matches"] > 0:
                    stats["answer_matches_count"] += 1
                stats["valid_questions_count"] += traj_stats["valid_questions"]
                stats["successful_problems_count"] += traj_stats["successful_problems"]
                stats["format_error_count"] += traj_stats["format_error"]
                
                # 收集失败原因统计
                failure_reason = traj_stats.get("failure_reason")
                if failure_reason:
                    if failure_reason not in stats["failure_reasons"]:
                        stats["failure_reasons"][failure_reason] = 0
                    stats["failure_reasons"][failure_reason] += 1
                elif traj_stats["successful_problems"] > 0:
                    stats["pass_count"] += 1
                else:
                    failure_reason = "unknown_extraction_failure"
                    stats["failure_reasons"].setdefault(failure_reason, 0)
                    stats["failure_reasons"][failure_reason] += 1

                stats["trajectory_outcomes"][trajectory_index] = {
                    "trajectory_index": trajectory_index,
                    "answer_matches": int(traj_stats.get("answer_matches", 0)),
                    "valid_questions": int(traj_stats.get("valid_questions", 0)),
                    "successful_problems": int(traj_stats.get("successful_problems", 0)),
                    "format_error": int(traj_stats.get("format_error", 0)),
                    "failure_reason": failure_reason,
                    "rag_diagnostics": deepcopy(traj_stats.get("rag_diagnostics")),
                }

            except Exception as e:
                logger.warning(f"Thread processing trajectory {trajectory_index} failed: {e}")
                if SELF_PLAY_DEBUG:
                    logger.debug(f"Thread exception details for trajectory {trajectory_index}: {e}", exc_info=True)
                # 记录异常
                if "thread_exception" not in stats["failure_reasons"]:
                    stats["failure_reasons"]["thread_exception"] = 0
                stats["failure_reasons"]["thread_exception"] += 1
                stats["trajectory_outcomes"][trajectory_index] = {
                    "trajectory_index": trajectory_index,
                    "answer_matches": 0,
                    "valid_questions": 0,
                    "successful_problems": 0,
                    "format_error": 0,
                    "failure_reason": "thread_exception",
                    "rag_diagnostics": {
                        "status": "thread_exception",
                        "rag_invoked": False,
                        "accepted": False,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                }

    if SELF_PLAY_DEBUG:
        logger.debug(f"Batch extraction complete: {len(all_problems)} total problems extracted")
        logger.debug(f"Extraction statistics: {stats}")

    classified_count = stats["pass_count"] + sum(stats["failure_reasons"].values())
    if classified_count != stats["trajectories_count"]:
        raise RuntimeError(
            "Extraction outcomes do not cover the batch exactly: "
            f"pass={stats['pass_count']} failures={sum(stats['failure_reasons'].values())} "
            f"trajectories={stats['trajectories_count']}"
        )
    if stats["pass_count"] != stats["successful_problems_count"]:
        raise RuntimeError(
            "Extraction pass count differs from successful problem count: "
            f"pass={stats['pass_count']} successful={stats['successful_problems_count']}"
        )

    # 打印失败原因摘要
    logger.info(f"Extraction failure reasons: {stats['failure_reasons']}")
    
    logger.info(
        f"Extracted {len(all_problems)} search problems from {len(trajectories)} trajectories using {max_workers} threads"
    )
    return all_problems, stats
