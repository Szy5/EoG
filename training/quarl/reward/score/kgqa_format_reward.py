"""KGQA 多轮交互的格式奖励（format reward）。

本模块校验 Solver 解码后的完整 response 是否符合 SSP-KG 交互协议，并给出
[0, max_score] 的稠密格式分。与 ``kgqa_answer_em`` 的结果奖励（EM）互补：
EM 只看最终答案是否正确，格式奖励鼓励模型按规范组织每一轮输出。

一轮合法交互的典型结构::

    <think>...</think><kg-query>...</kg-query>
    <information>...</information>   # 环境反馈，不计入模型轮次
    <think>...</think><answer>...</answer>

评分要点：
- 按 ``<information>`` 块切分模型轮次，环境块本身不参与格式校验。
- 每个模型轮次必须是「thinking + kg-query」或「thinking + answer」之一，
  且各标签恰好出现一次，前后不能有多余文本或其它标签。
- 若某轮后面紧跟 ``<information>``，该轮必须是合法的 kg-query 轮。
- 默认仅对 ``split=train`` 样本把 measured score 计入训练 reward；
  val/test 仍记录 ``measured_format_score`` 等指标，但 ``score`` 为 0。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

# 环境返回的信息块；用于切分模型轮次，块内内容不参与格式判定。
INFO_BLOCK_RE = re.compile(r"<information>.*?</information>", re.DOTALL | re.IGNORECASE)

ADJACENT_THINK_PATTERNS = (
    re.compile(r"</think>\s*<think>", re.IGNORECASE),
    re.compile(r"<think>\s*<think>", re.IGNORECASE),
    re.compile(r"</think>\s*</think>", re.IGNORECASE),
)

PROTOCOL_TAG_PAIRS = {
    "think": ("<think>", "</think>"),
    "kg_query": ("<kg-query>", "</kg-query>"),
    "answer": ("<answer>", "</answer>"),
    "information": ("<information>", "</information>"),
}

# 合法 kg-query 轮：整轮仅含一对 thinking + 一对 kg-query，无其它前后缀。
KG_QUERY_TURN_RE = re.compile(
    r"^<think>.*?</think>\s*<kg-query>.*?</kg-query>$",
    re.DOTALL | re.IGNORECASE,
)
# 合法 answer 轮：整轮仅含一对 thinking + 一对 answer，无其它前后缀。
ANSWER_TURN_RE = re.compile(
    r"^<think>.*?</think>\s*<answer>.*?</answer>$",
    re.DOTALL | re.IGNORECASE,
)


def _count_tag(text: str, tag: str) -> int:
    """统计开/闭标签在文本中出现次数（大小写不敏感）。"""
    return len(re.findall(re.escape(tag), text or "", flags=re.IGNORECASE))


def _split_model_turns_with_boundaries(response_str: str) -> Tuple[List[str], List[bool]]:
    """按 ``<information>`` 切分模型轮次，并记录每轮后是否紧跟环境块。

    Returns:
        turns: 非空的模型输出片段列表。
        followed_by_information: 与 ``turns`` 等长；``True`` 表示该轮后紧跟
            ``<information>``，此时该轮必须是合法 kg-query 轮。
    """
    turns = []
    followed_by_information = []
    start = 0
    for match in INFO_BLOCK_RE.finditer(response_str or ""):
        turns.append((response_str or "")[start:match.start()].strip())
        followed_by_information.append(True)
        start = match.end()
    turns.append((response_str or "")[start:].strip())
    followed_by_information.append(False)

    non_empty_turns = []
    non_empty_followed_by_information = []
    for turn, has_next_information in zip(turns, followed_by_information):
        if turn:
            non_empty_turns.append(turn)
            non_empty_followed_by_information.append(has_next_information)
    return non_empty_turns, non_empty_followed_by_information


def split_model_turns(response_str: str) -> List[str]:
    """将解码后的 KGQA response 按 information 块切分为模型轮次。"""
    turns, _ = _split_model_turns_with_boundaries(response_str)
    return turns


def classify_turn_action(turn_text: str) -> str:
    """粗分类单轮动作类型，用于选择对应的格式校验规则。

    Returns:
        ``"answer"`` | ``"kg-query"`` | ``"search"`` | ``"error"``
        （``search`` 为旧协议残留，一律视为非法。）
    """
    if re.search(r"<answer>.*?</answer>", turn_text or "", flags=re.DOTALL | re.IGNORECASE):
        return "answer"
    if re.search(r"<kg-query>.*?</kg-query>", turn_text or "", flags=re.DOTALL | re.IGNORECASE):
        return "kg-query"
    if re.search(r"<search>.*?</search>", turn_text or "", flags=re.DOTALL | re.IGNORECASE):
        return "search"
    return "error"


def _has_information_tag(turn_text: str) -> bool:
    """模型轮次内不得伪造 ``<information>``（该标签仅由环境注入）。"""
    return bool(re.search(r"</?information>", turn_text or "", flags=re.IGNORECASE))


def is_valid_kg_query_turn(turn_text: str) -> bool:
    """判定单轮是否为合法 kg-query 轮。

    除正则整体匹配外，还要求 think / kg-query 各恰好一对，
    且不能混入 answer 标签。
    """
    turn = (turn_text or "").strip()
    if _has_information_tag(turn):
        return False
    if not KG_QUERY_TURN_RE.fullmatch(turn):
        return False
    return (
        _count_tag(turn, "<think>") == 1
        and _count_tag(turn, "</think>") == 1
        and _count_tag(turn, "<kg-query>") == 1
        and _count_tag(turn, "</kg-query>") == 1
        and _count_tag(turn, "<answer>") == 0
        and _count_tag(turn, "</answer>") == 0
    )

def is_valid_answer_turn(turn_text: str) -> bool:
    """判定单轮是否为合法 answer 轮（规则同 kg-query 轮，标签换成 answer）。"""
    turn = (turn_text or "").strip()
    if _has_information_tag(turn):
        return False
    if not ANSWER_TURN_RE.fullmatch(turn):
        return False
    return (
        _count_tag(turn, "<think>") == 1
        and _count_tag(turn, "</think>") == 1
        and _count_tag(turn, "<answer>") == 1
        and _count_tag(turn, "</answer>") == 1
        and _count_tag(turn, "<kg-query>") == 0
        and _count_tag(turn, "</kg-query>") == 0
    )


def _validate_single_turn(turn_text: str) -> Dict[str, Any]:
    """校验单轮格式，返回动作类型与是否通过。"""
    action = classify_turn_action(turn_text)
    if action == "kg-query":
        valid = is_valid_kg_query_turn(turn_text)
    elif action == "answer":
        valid = is_valid_answer_turn(turn_text)
    else:
        valid = False
    return {"action": action, "format_valid": valid}


def validate_model_turns(turns: Sequence[str]) -> Dict[str, Any]:
    """汇总多轮格式校验结果。

    Returns:
        checked_turn_count: 参与校验的轮次数。
        valid_turn_count: 通过单轮格式规则的轮次数。
        turn_format_valid_rate: 合法轮次占比（尚未考虑 information 序列约束）。
        turn_results: 每轮的 ``action`` / ``format_valid`` 明细。
    """
    turn_results = [_validate_single_turn(turn) for turn in turns]
    checked_turn_count = len(turn_results)
    valid_turn_count = sum(1 for result in turn_results if result["format_valid"])
    turn_format_valid_rate = valid_turn_count / checked_turn_count if checked_turn_count else 0.0
    return {
        "checked_turn_count": checked_turn_count,
        "valid_turn_count": valid_turn_count,
        "turn_format_valid_rate": turn_format_valid_rate,
        "turn_results": turn_results,
    }


def _tag_matches(text: str, tag: str) -> List[re.Match]:
    """Return case-insensitive literal-tag matches in source order."""
    return list(re.finditer(re.escape(tag), text or "", flags=re.IGNORECASE))


def _information_sequence_valid(response_str: str) -> bool:
    """Require every balanced information block to follow one valid query turn."""
    response = response_str or ""
    information_open = _tag_matches(response, "<information>")
    information_close = _tag_matches(response, "</information>")
    information_blocks = list(INFO_BLOCK_RE.finditer(response))
    if not len(information_open) == len(information_close) == len(information_blocks):
        return False

    previous_end = 0
    for block in information_blocks:
        preceding_turn = response[previous_end:block.start()].strip()
        if not preceding_turn or not is_valid_kg_query_turn(preceding_turn):
            return False
        previous_end = block.end()
    return True


def analyze_protocol_critical_errors(response_str: str) -> Dict[str, Any]:
    """Detect structural protocol failures that must gate the full reward."""
    response = response_str or ""
    turns, _ = _split_model_turns_with_boundaries(response)
    reasons = set()

    adjacent_think = any(
        pattern.search(turn)
        for turn in turns
        for pattern in ADJACENT_THINK_PATTERNS
    )
    if adjacent_think:
        reasons.add("adjacent_think_tags")

    duplicate_tags = False
    unclosed_tags = False
    invalid_tag_order = False
    mixed_query_and_answer = False
    for turn in turns:
        matches = {
            name: (_tag_matches(turn, opening), _tag_matches(turn, closing))
            for name, (opening, closing) in PROTOCOL_TAG_PAIRS.items()
            if name != "information"
        }
        if any(len(opening) > 1 or len(closing) > 1 for opening, closing in matches.values()):
            duplicate_tags = True
        if any(len(opening) != len(closing) for opening, closing in matches.values()):
            unclosed_tags = True

        has_query = bool(matches["kg_query"][0] or matches["kg_query"][1])
        has_answer = bool(matches["answer"][0] or matches["answer"][1])
        if has_query and has_answer:
            mixed_query_and_answer = True

        for opening, closing in matches.values():
            if len(opening) == 1 and len(closing) == 1 and opening[0].start() > closing[0].start():
                invalid_tag_order = True

        think_open, think_close = matches["think"]
        for action_name in ("kg_query", "answer"):
            action_open, action_close = matches[action_name]
            if all(len(items) == 1 for items in (think_open, think_close, action_open, action_close)):
                if not (
                    think_open[0].start()
                    < think_close[0].start()
                    < action_open[0].start()
                    < action_close[0].start()
                ):
                    invalid_tag_order = True

    if duplicate_tags:
        reasons.add("duplicate_protocol_tags")
    if unclosed_tags:
        reasons.add("unclosed_protocol_tags")
    if invalid_tag_order:
        reasons.add("invalid_tag_order")
    if mixed_query_and_answer:
        reasons.add("query_and_answer_same_turn")

    information_open = _tag_matches(response, "<information>")
    information_close = _tag_matches(response, "</information>")
    unbalanced_information_tags = len(information_open) != len(information_close)
    if unbalanced_information_tags:
        reasons.add("unbalanced_information_tags")

    information_sequence_valid = _information_sequence_valid(response)
    if not information_sequence_valid:
        reasons.add("invalid_information_sequence")

    return {
        "critical_error": bool(reasons),
        "critical_error_reasons": sorted(reasons),
        "adjacent_think_tags": adjacent_think,
        "duplicate_protocol_tags": duplicate_tags,
        "unclosed_protocol_tags": unclosed_tags,
        "invalid_tag_order": invalid_tag_order,
        "query_and_answer_same_turn": mixed_query_and_answer,
        "unbalanced_information_tags": unbalanced_information_tags,
        "information_sequence_valid": information_sequence_valid,
    }


def _split_names(extra_info: Dict[str, Any]) -> List[str]:
    """从 extra_info 提取可能的数据划分字段（小写）。"""
    candidates = [
        extra_info.get("split"),
        extra_info.get("data_split"),
        extra_info.get("mode"),
    ]
    return [str(candidate).lower() for candidate in candidates if candidate is not None]


def _format_reward_applies(extra_info: Dict[str, Any], apply_to_splits: Sequence[str]) -> bool:
    """判断当前样本是否应把格式分计入训练 reward。

    无 split 信息时默认视为 train；val/test 仅记录指标，不叠加到 ``score``。
    """
    normalized_allowed = tuple(str(split).lower() for split in apply_to_splits)
    split_names = _split_names(extra_info)
    if not split_names:
        return "train" in normalized_allowed
    return any(any(allowed in split_name for allowed in normalized_allowed) for split_name in split_names)


def compute_format_score(
    response_str: str,
    extra_info: Dict[str, Any] | None = None,
    *,
    max_score: float = 0.2,
    apply_to_splits: Sequence[str] = ("train",),
) -> Dict[str, Any]:
    """计算单条样本的格式奖励与诊断字段。

    最终 ``score = max_score * valid_turn_rate``（在 information 序列合法且``format_reward_applied`` 为真时）；
    否则 ``score`` 为 0，但``measured_format_score`` 仍反映真实格式质量，便于 val 监控。

    Args:
        response_str: 解码后的完整模型输出。
        extra_info: 样本元信息，用于判断是否应用训练奖励。
        max_score: 全部轮次格式均合法时的满分。
        apply_to_splits: 允许叠加训练 reward 的数据划分白名单。
    """
    extra_info = extra_info or {}
    turns, _ = _split_model_turns_with_boundaries(response_str)
    validation = validate_model_turns(turns)
    information_sequence_valid = _information_sequence_valid(response_str)

    checked_turn_count = validation["checked_turn_count"]
    # information 序列非法时，有效轮次清零（例如 answer 后又出现 information）。
    # 合法轮次数量
    valid_turn_count = validation["valid_turn_count"] if information_sequence_valid else 0
    # 合法轮次百分比
    turn_format_valid_rate = valid_turn_count / checked_turn_count if checked_turn_count else 0.0
    # 格式得分 = 满分 * 合法轮次百分比
    measured_score = float(max_score) * turn_format_valid_rate
    # 格式奖励应用
    format_reward_applied = _format_reward_applies(extra_info, apply_to_splits)
    # 得分
    score = measured_score if format_reward_applied else 0.0

    return {
        "score": score,
        "format_score": score,
        "measured_format_score": measured_score,
        "checked_turn_count": checked_turn_count,
        "valid_turn_count": valid_turn_count,
        "turn_format_valid_rate": turn_format_valid_rate,
        "format_reward_applied": format_reward_applied,
        "information_sequence_valid": information_sequence_valid,
    }


def compute_score_batch(
    batch_data,
    max_score: float = 0.2,
    apply_to_splits: Sequence[str] = ("train",),
):
    """
    SSPRewardManager 批量入口：对每条轨迹计算格式奖励。

    ``kwargs`` 中的 ``max_score`` / ``apply_to_splits`` 由 Hydra 配置注入。
    返回字段中的 ``score`` 会经 manager 累加到 token 级 reward。
    """
    results = []
    for batch_idx, data in enumerate(batch_data):
        score_info = compute_format_score(
            response_str=data["response_str"],
            extra_info=data.get("extra_info", {}),
            max_score=max_score,
            apply_to_splits=apply_to_splits,
        )
        results.append(
            {
                "idx": batch_data[batch_idx]["idx"],
                "score": float(score_info["score"]),
                "format_score": float(score_info["format_score"]),
                "measured_format_score": float(score_info["measured_format_score"]),
                "checked_turn_count": int(score_info["checked_turn_count"]),
                "valid_turn_count": int(score_info["valid_turn_count"]),
                "turn_format_valid_rate": float(score_info["turn_format_valid_rate"]),
                "format_reward_applied": bool(score_info["format_reward_applied"]),
                "information_sequence_valid": bool(score_info["information_sequence_valid"]),
            }
        )
    return results
