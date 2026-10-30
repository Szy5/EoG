import ast
from pathlib import Path

import pytest

from quarl.utils.timing import accumulate_rollout_timing


@pytest.mark.unit
def test_rollout_timing_accumulates_dynamic_sampling_attempts_by_role() -> None:
    timing_raw = {}

    accumulate_rollout_timing(
        timing_raw,
        "proposer",
        {"generate_sequences": 10.0, "reshard": 1.5},
    )
    accumulate_rollout_timing(
        timing_raw,
        "proposer",
        {"generate_sequences": 14.0, "reshard": 2.5},
    )
    accumulate_rollout_timing(
        timing_raw,
        "solver",
        {"generate_sequences": 20.0, "reshard": 3.0},
    )

    assert timing_raw["proposer_generate_sequences"] == pytest.approx(24.0)
    assert timing_raw["proposer_reshard"] == pytest.approx(4.0)
    assert timing_raw["proposer_rollout_total"] == pytest.approx(24.0)
    assert timing_raw["proposer_rollout_attempts"] == pytest.approx(2.0)
    assert timing_raw["proposer_rollout_mean"] == pytest.approx(12.0)
    assert timing_raw["proposer_rollout_max"] == pytest.approx(14.0)

    assert timing_raw["solver_generate_sequences"] == pytest.approx(20.0)
    assert timing_raw["solver_reshard"] == pytest.approx(3.0)
    assert timing_raw["solver_rollout_total"] == pytest.approx(20.0)
    assert timing_raw["solver_rollout_attempts"] == pytest.approx(1.0)
    assert timing_raw["solver_rollout_mean"] == pytest.approx(20.0)
    assert timing_raw["solver_rollout_max"] == pytest.approx(20.0)


@pytest.mark.unit
def test_rollout_timing_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="Unsupported rollout timing role"):
        accumulate_rollout_timing({}, "critic", {"generate_sequences": 1.0})


@pytest.mark.unit
def test_self_play_fit_records_total_timers_after_their_full_scopes() -> None:
    trainer_path = Path("training/quarl/trainer/ppo/ssp_ray_trainer.py")
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    trainer_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SSPRayPPOTrainer")
    fit = next(node for node in trainer_class.body if isinstance(node, ast.FunctionDef) and node.name == "fit")

    assignments = {
        target.slice.value: node.lineno
        for node in ast.walk(fit)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == "timing_raw"
        and isinstance(target.slice, ast.Constant)
        and target.slice.value in {"step_total", "model_updates"}
    }
    calls = {
        node.func.attr: node.lineno
        for node in ast.walk(fit)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert assignments["model_updates"] > calls["_update_on_trajectories"]
    assert assignments["step_total"] > calls["_validate"]
    assert assignments["step_total"] > calls["_save_checkpoint"]
