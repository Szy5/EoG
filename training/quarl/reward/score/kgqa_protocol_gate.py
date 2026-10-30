"""Binary reward gate for critical KGQA interaction-protocol failures."""

from typing import Any, Dict, List

from .kgqa_format_reward import analyze_protocol_critical_errors


def compute_score_batch(batch_data, enabled: bool = True) -> List[Dict[str, Any]]:
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}

    results = []
    for data in batch_data:
        diagnostics = analyze_protocol_critical_errors(data["response_str"])
        critical_error = bool(diagnostics["critical_error"]) if enabled else False
        results.append(
            {
                "idx": data["idx"],
                "score": 0.0 if critical_error else 1.0,
                "enabled": bool(enabled),
                "critical_error": critical_error,
                "critical_error_reasons": diagnostics["critical_error_reasons"] if enabled else [],
                "adjacent_think_tags": bool(diagnostics["adjacent_think_tags"]),
                "duplicate_protocol_tags": bool(diagnostics["duplicate_protocol_tags"]),
                "unclosed_protocol_tags": bool(diagnostics["unclosed_protocol_tags"]),
                "invalid_tag_order": bool(diagnostics["invalid_tag_order"]),
                "query_and_answer_same_turn": bool(diagnostics["query_and_answer_same_turn"]),
                "unbalanced_information_tags": bool(diagnostics["unbalanced_information_tags"]),
                "information_sequence_valid": bool(diagnostics["information_sequence_valid"]),
            }
        )
    return results
