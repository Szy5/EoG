import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List

from quarl.reward.score.outcome_score import em_check, extract_math_solution, extract_solution
from quarl.utils.llm_as_a_judge import get_global_judge

# Broken tool-tag patterns that indicate the model has entered a degenerate
# tool loop. The model generates malformed tags such as </search| or
# </kg-query| instead of the expected closing tag, causing the tool parser to
# miss a complete call, so the environment cannot respond and the model may
# retry until the token limit is exhausted.
#
# Note: </think| is NOT included here because some model checkpoints consistently
# emit </think| as their normal think-closing token; flagging it would incorrectly
# penalise healthy outputs. Only broken tool-call tags cause actual tool failure.
_BROKEN_TOOL_TAG_RE = re.compile(
    r"</(?:search|kg-query)[^>a-zA-Z]"
    r"|<(?:search|kg-query)[^>/a-zA-Z\s]",
    re.IGNORECASE,
)

_MIN_BROKEN_TAG_COUNT = 5  # require several occurrences to avoid false positives


def has_broken_format(response_str: str) -> bool:
    """Return True if the response contains repeated broken tool-call tags that
    signal a degenerate tool-loop collapse (reward-hacking via malformed tags)."""
    if not response_str:
        return False
    matches = _BROKEN_TOOL_TAG_RE.findall(response_str)
    return len(matches) >= _MIN_BROKEN_TAG_COUNT


def has_format_error(response_str: str) -> bool:
    """Return True if the response has structural format errors indicating
    the model abandoned the expected agent output structure.

    Checks beyond has_broken_format / answer-is-None:
      - Empty or whitespace-only response
      - No <think> tag at all (complete format collapse)
    """
    if not response_str or not response_str.strip():
        return True
    if not re.search(r"<think>", response_str, re.IGNORECASE):
        return True
    return False


def is_overlength_rollout(
    response_str: str,
    valid_response_length: int = 0,
    max_search_calls: int = 0,
    max_response_tokens: int = 0,
) -> bool:
    """Return True if the rollout exceeds token or tool-call limits.

    Following E-GRPO (Zhao et al., 2025), overlength rollouts are assigned
    reward=0.  They typically indicate unproductive reasoning loops where
    the model searches in circles without converging on an answer.

    Args:
        max_search_calls: Upper bound on <search> invocations.  0 = disabled.
        max_response_tokens: Upper bound on response token count.  0 = disabled.
    """
    if max_search_calls > 0:
        search_count = len(re.findall(r"<search>", response_str, re.IGNORECASE))
        if search_count > max_search_calls:
            return True
    if max_response_tokens > 0 and valid_response_length > 0:
        if int(valid_response_length) >= max_response_tokens:
            return True
    return False


MATH_DATA_SOURCES = {
    "Algebra",
    "Intermediate Algebra",
    "Prealgebra",
    "Number Theory",
    "Geometry",
    "Precalculus",
    "Counting & Probability",
}


def extract_think_text(solution_str: str) -> str:
    think_matches = re.findall(r"<think>(.*?)</think>", solution_str or "", re.DOTALL | re.IGNORECASE)
    return "\n".join(match.strip() for match in think_matches if match.strip())


def extract_answer_for_data_source(data_source: str, solution_str: str):
    if data_source in MATH_DATA_SOURCES:
        return extract_math_solution(solution_str=solution_str)
    return extract_solution(solution_str=solution_str)


def is_answer_correct(data_source: str, answer: str, ground_truth: Dict[str, Any], prompt_str: str) -> bool:
    if em_check(answer, ground_truth["target"]):
        return True

    api_key = os.getenv("JUDGE_API_KEY", "dummy_api_key")
    judge = get_global_judge(
        base_url=os.getenv("JUDGE_BASE_URL"),
        api_key=api_key,
        model=os.getenv("JUDGE_MODEL"),
    )

    question = prompt_str
    if "Question: " in prompt_str:
        question = prompt_str.split("Question: ", 1)[1]
    elif "<｜User｜>" in prompt_str:
        question = prompt_str.replace("<｜User｜>", "")

    return judge.model_based_match(question=question, golden_answer=ground_truth["target"], model_answer=answer)


def compute_hit_nodes(think_text: str, precursor_nodes: List[str], case_sensitive: bool = True) -> List[str]:
    if not think_text:
        return []

    matched_nodes = []
    seen = set()
    haystack = think_text if case_sensitive else think_text.lower()

    for node in precursor_nodes:
        if not isinstance(node, str):
            continue
        normalized_node = node.strip()
        if not normalized_node or normalized_node in seen:
            continue

        needle = normalized_node if case_sensitive else normalized_node.lower()
        if needle in haystack:
            matched_nodes.append(normalized_node)
            seen.add(normalized_node)

    return matched_nodes


def compute_think_length_penalty(
    think_length: int,
    group_think_lengths: List[int],
    beta: float,
) -> tuple:
    """Compute the group-normalized think length penalty factor.

    Uses min-max normalization within the group (same uid) so that the
    shortest rollout gets penalty=1.0 and the longest gets penalty=exp(-beta).
    This is model-agnostic: no absolute thresholds needed.

    Args:
        think_length: Character count of <think> text for this rollout.
        group_think_lengths: Character counts of <think> text for all
            rollouts sharing the same uid.
        beta: Exponential decay rate.  0 = disabled (penalty always 1.0).

    Returns:
        (penalty_factor, normalized_length)
    """
    if beta <= 0.0 or not group_think_lengths:
        return 1.0, 0.0

    L_min = min(group_think_lengths)
    L_max = max(group_think_lengths)

    if L_max == L_min:
        return 1.0, 0.0

    normalized_length = (think_length - L_min) / (L_max - L_min)
    penalty_factor = math.exp(-beta * normalized_length)
    return penalty_factor, normalized_length


def compute_score_batch(
    batch_data,
    alpha: float = 0.5,
    think_only: bool = True,
    case_sensitive: bool = True,
    max_search_calls: int = 0,
    max_response_tokens: int = 0,
    node_filter_mode: str = "none",
    think_length_penalty_beta: float = 0.0,
):
    sample_infos = []
    uid2raw_scores = defaultdict(list)
    uid2think_lengths = defaultdict(list)

    for data in batch_data:
        extra_info = data.get("extra_info", {}) or {}
        effective_node_filter_mode = extra_info.get("process_reward_node_filter_mode", node_filter_mode) or "none"
        raw_precursor_nodes = extra_info.get("raw_precursor_nodes", []) or []
        named_entity_precursor_nodes = extra_info.get("named_entity_precursor_nodes", []) or []
        filtered_out_precursor_nodes = extra_info.get("filtered_out_precursor_nodes", []) or []

        if effective_node_filter_mode == "named_entity_only" and named_entity_precursor_nodes:
            precursor_nodes = named_entity_precursor_nodes
        else:
            precursor_nodes = extra_info.get("precursor_nodes", []) or []

        precursor_nodes = [node.strip() for node in precursor_nodes if isinstance(node, str) and node.strip()]
        raw_precursor_nodes = [node.strip() for node in raw_precursor_nodes if isinstance(node, str) and node.strip()]
        named_entity_precursor_nodes = [
            node.strip() for node in named_entity_precursor_nodes if isinstance(node, str) and node.strip()
        ]
        filtered_out_precursor_nodes = [
            node.strip() for node in filtered_out_precursor_nodes if isinstance(node, str) and node.strip()
        ]

        response_str = data["response_str"]
        think_text = extract_think_text(response_str) if think_only else response_str
        hit_nodes = compute_hit_nodes(think_text, precursor_nodes, case_sensitive=case_sensitive)
        raw_match_rate = len(hit_nodes) / len(precursor_nodes) if precursor_nodes else 0.0

        # v4p1: collect think length for group normalization
        think_length = len(think_text)

        group_uid = data.get("uid")
        if group_uid is None:
            group_uid = extra_info.get("index", data["idx"])
        group_uid = str(group_uid)
        uid2raw_scores[group_uid].append(raw_match_rate)
        uid2think_lengths[group_uid].append(think_length)

        answer = extract_answer_for_data_source(data.get("data_source", ""), response_str)
        format_error = (
            answer is None
            or has_broken_format(response_str)
            or has_format_error(response_str)
        )
        overlength = is_overlength_rollout(
            response_str,
            valid_response_length=int(data.get("valid_response_length", 0)),
            max_search_calls=max_search_calls,
            max_response_tokens=max_response_tokens,
        )
        invalid_output = format_error or overlength
        extraction_failed = bool(extra_info.get("extraction_failed", False))
        answer_correct = False
        if not invalid_output and not extraction_failed:
            answer_correct = is_answer_correct(
                data_source=data.get("data_source", ""),
                answer=answer,
                ground_truth=data["ground_truth"],
                prompt_str=data.get("prompt_str", ""),
            )

        sample_infos.append(
            {
                "idx": data["idx"],
                "uid": group_uid,
                "raw_match_rate": raw_match_rate,
                "hit_nodes": hit_nodes,
                "precursor_nodes": precursor_nodes,
                "precursor_node_count": len(precursor_nodes),
                "raw_precursor_nodes": raw_precursor_nodes,
                "raw_precursor_node_count": len(raw_precursor_nodes),
                "named_entity_precursor_nodes": named_entity_precursor_nodes,
                "filtered_out_precursor_nodes": filtered_out_precursor_nodes,
                "node_filter_mode": effective_node_filter_mode,
                "answer_correct": answer_correct,
                "invalid_output": invalid_output,
                "format_error": format_error,
                "overlength": overlength,
                "extraction_failed": extraction_failed,
                "think_length": think_length,
            }
        )

    results = []
    for info in sample_infos:
        group_max = max(uid2raw_scores[info["uid"]]) if uid2raw_scores[info["uid"]] else 0.0
        normalized_match_rate = info["raw_match_rate"] / group_max if group_max > 0 else 0.0

        if info["invalid_output"] or info["extraction_failed"]:
            base_score = 0.0
        elif info["answer_correct"]:
            base_score = 1.0
        else:
            base_score = alpha * normalized_match_rate

        # v4p1: apply think length penalty via group-normalized exponential decay
        think_length_penalty_factor, think_length_normalized = compute_think_length_penalty(
            think_length=info["think_length"],
            group_think_lengths=uid2think_lengths[info["uid"]],
            beta=think_length_penalty_beta,
        )
        final_score = base_score * think_length_penalty_factor

        results.append(
            {
                "idx": info["idx"],
                "score": float(final_score),
                "raw_match_rate": float(info["raw_match_rate"]),
                "normalized_match_rate": float(normalized_match_rate),
                "group_max_match_rate": float(group_max),
                "precursor_node_count": info["precursor_node_count"],
                "raw_precursor_node_count": info["raw_precursor_node_count"],
                "hit_node_count": len(info["hit_nodes"]),
                "hit_nodes": info["hit_nodes"],
                "precursor_nodes": info["precursor_nodes"],
                "raw_precursor_nodes": info["raw_precursor_nodes"],
                "named_entity_precursor_nodes": info["named_entity_precursor_nodes"],
                "filtered_out_precursor_nodes": info["filtered_out_precursor_nodes"],
                "node_filter_mode": info["node_filter_mode"],
                "answer_correct": info["answer_correct"],
                "invalid_output": info["invalid_output"],
                "format_error": info["format_error"],
                "overlength": info["overlength"],
                "extraction_failed": info["extraction_failed"],
                "group_uid": info["uid"],
                "think_length": info["think_length"],
                "think_length_normalized": float(think_length_normalized),
                "think_length_penalty_factor": float(think_length_penalty_factor),
            }
        )

    return results
