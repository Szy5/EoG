import json
import re
import string
from typing import Any, List


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punc = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punc)
    return " ".join(without_articles.split())


def extract_answer_payload(solution_str: str) -> str | None:
    """Return the payload of the final answer tag in a rollout transcript.

    Multi-turn rollouts are persisted as one decoded transcript.  A user-side
    continuation may therefore literally say ``provide it in <answer> tags``
    before the model emits its actual ``<answer>...</answer>``.  Pairing tags
    with a non-greedy regex makes that instructional opening tag consume the
    final closing tag.  Select the final opening tag instead: it is the model's
    answer tag in a well-formed final response, and preserves the previous
    behaviour for a normal single-turn answer.
    """
    text = solution_str or ""
    opening_matches = list(re.finditer(r"<answer>", text, re.IGNORECASE))
    if not opening_matches:
        return None
    opening = opening_matches[-1]
    closing = re.search(r"</answer>", text[opening.end() :], re.IGNORECASE)
    if closing is None:
        return None
    return text[opening.end() : opening.end() + closing.start()].strip()


def parse_predicted_answers(solution_str: str) -> List[str]:
    payload = extract_answer_payload(solution_str)
    if payload is None or not payload:
        return []

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        decoded = payload

    if isinstance(decoded, str):
        answers = [decoded]
    elif isinstance(decoded, list):
        answers = decoded
    else:
        return []

    normalized = []
    for answer in answers:
        candidate = str(answer).strip()
        if candidate:
            normalized.append(candidate)
    return normalized


def _normalize_answer_list(answers: Any) -> List[str]:
    if isinstance(answers, str):
        raw_answers = [answers]
    else:
        raw_answers = list(answers or [])

    normalized = []
    for answer in raw_answers:
        candidate = normalize_answer(str(answer).strip())
        if candidate:
            normalized.append(candidate)
    return normalized


def _collect_normalized_entities(answers: Any) -> set[str]:
    return set(_normalize_answer_list(answers))


def _collect_answer_tokens(answers: Any) -> set[str]:
    tokens: set[str] = set()
    for normalized_answer in _normalize_answer_list(answers):
        tokens.update(normalized_answer.split())
    return tokens


def _compute_f1(predicted_count: int, target_count: int, common_count: int) -> float:
    if predicted_count == 0 or target_count == 0 or common_count == 0:
        return 0.0

    precision = common_count / predicted_count
    recall = common_count / target_count
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_hit_at_1(predicted_answers: List[str], target_answers: Any) -> float:
    """
    计算指标：HIT@1。
    
    将预测答案与标准答案都归一化为实体集合；只要存在至少一个完全相同的实体
    （归一化后字符串相等），即视为命中并返回 1.0。
    """
    normalized_predicted = _collect_normalized_entities(predicted_answers)
    normalized_target = _collect_normalized_entities(target_answers)
    if not normalized_predicted or not normalized_target:
        return 0.0
    return 1.0 if normalized_predicted & normalized_target else 0.0


def compute_token_f1(predicted_answers: List[str], target_answers: Any) -> float:
    """
    计算指标：Token F1。

    将多个答案中的 token 合并为集合，再计算集合级 precision、recall 和 F1。
    """
    predicted_tokens = _collect_answer_tokens(predicted_answers)
    target_tokens = _collect_answer_tokens(target_answers)
    common_count = len(predicted_tokens & target_tokens)
    return _compute_f1(len(predicted_tokens), len(target_tokens), common_count)


def compute_entity_f1(predicted_answers: List[str], target_answers: Any) -> float:
    """
    计算指标：Entity F1。
    将每个答案视为一个完整实体，先做归一化和去重，再基于实体集合计算
    precision、recall 和 F1。
    """
    normalized_predicted = _collect_normalized_entities(predicted_answers)
    normalized_target = _collect_normalized_entities(target_answers)
    common_count = len(normalized_predicted & normalized_target)
    return _compute_f1(len(normalized_predicted), len(normalized_target), common_count)


def is_exact_answer_match(predicted_answers: List[str], target_answers: Any) -> bool:
    """
    计算指标：Related EM
    预测与标准答案归一化为实体集合后，只要存在至少一个相同实体即视为命中。
    """
    normalized_predicted = _collect_normalized_entities(predicted_answers)
    normalized_target = _collect_normalized_entities(target_answers)
    if not normalized_predicted or not normalized_target:
        return False
    return bool(normalized_predicted & normalized_target)


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    """
    计算指标汇总
    """
    predicted_answers = parse_predicted_answers(solution_str or "")
    target_answers = (ground_truth or {}).get("target", [])
    answer_correct = is_exact_answer_match(predicted_answers, target_answers)
    entity_f1 = compute_entity_f1(predicted_answers, target_answers)
    return {
        "score": entity_f1,
        "answer_correct": answer_correct,
        # Formerly "em". Renamed because it is neither EM nor Hits@1 — it only asks whether the
        # prediction and the gold set intersect. Kept for comparison against historical runs.
        "any_overlap": 1.0 if answer_correct else 0.0,
        "hit_at_1": compute_hit_at_1(predicted_answers, target_answers),
        "token_f1": compute_token_f1(predicted_answers, target_answers),
        "entity_f1": entity_f1,
        "predicted_answers": predicted_answers,
    }


def compute_score_batch(batch_data, score_coef: float = 1.0):
    results = []
    for batch_idx, data in enumerate(batch_data):
        score_info = compute_score(
            data_source=data.get("data_source", ""),
            solution_str=data["response_str"],
            ground_truth=data["ground_truth"],
            extra_info=data.get("extra_info", {}),
            sandbox_fusion_url=data.get("sandbox_fusion_url", None),
            concurrent_semaphore=data.get("concurrent_semaphore", None),
        )
        results.append(
            {
                "score": float(score_info["score"]) * score_coef,
                "idx": batch_data[batch_idx]["idx"],
                "answer_correct": score_info["answer_correct"],
                "any_overlap": score_info["any_overlap"],
                "hit_at_1": score_info["hit_at_1"],
                "token_f1": score_info["token_f1"],
                "entity_f1": score_info["entity_f1"],
                "predicted_answers": score_info["predicted_answers"],
            }
        )
    return results
