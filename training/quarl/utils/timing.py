"""Timing aggregation helpers for self-play training."""

from numbers import Real
from typing import Any, MutableMapping


def accumulate_rollout_timing(
    timing_raw: MutableMapping[str, float],
    role: str,
    rollout_timing: MutableMapping[str, Any] | None,
) -> None:
    """Merge one rollout call's timings without losing role or retry information.

    Worker timings such as ``generate_sequences`` and ``reshard`` use generic
    names. Prefixing them keeps proposer and solver measurements separate, and
    adding rather than updating preserves every sequential dynamic-sampling
    attempt within the current training step.
    """
    if role not in {"proposer", "solver"}:
        raise ValueError(f"Unsupported rollout timing role: {role}")

    rollout_timing = rollout_timing or {}
    for name, value in rollout_timing.items():
        if isinstance(value, Real):
            key = f"{role}_{name}"
            timing_raw[key] = timing_raw.get(key, 0.0) + float(value)

    duration = rollout_timing.get("generate_sequences", 0.0)
    if not isinstance(duration, Real):
        duration = 0.0
    duration = float(duration)

    total_key = f"{role}_rollout_total"
    attempts_key = f"{role}_rollout_attempts"
    mean_key = f"{role}_rollout_mean"
    max_key = f"{role}_rollout_max"

    timing_raw[total_key] = timing_raw.get(total_key, 0.0) + duration
    timing_raw[attempts_key] = timing_raw.get(attempts_key, 0.0) + 1.0
    timing_raw[max_key] = max(timing_raw.get(max_key, 0.0), duration)
    timing_raw[mean_key] = timing_raw[total_key] / timing_raw[attempts_key]
