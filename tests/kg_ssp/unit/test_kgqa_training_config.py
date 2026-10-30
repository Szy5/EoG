from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_gtsqa_v9_config_selects_embedded_backend_without_freebase() -> None:
    env = os.environ.copy()
    env.update(
        {
            "KGQA_CONFIG_ONLY": "1",
            "KGQA_EXPERIMENT_CONFIG": str(REPO_ROOT / "scripts/experiments/gtsqa_v9/.env.example"),
            "REPO_ROOT": str(REPO_ROOT),
        }
    )
    result = subprocess.run(
        ["bash", "scripts/platforms/local.sh"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert values["kgqa_backend"] == "gtsqa"
    assert values["freebase_lifecycle_enabled"] == "False"
    assert values["train_parquet"].endswith("/data/train/train_gtsqa_v9.parquet")
    assert values["val_parquet"].endswith("/data/val/gtsqa/gtsqa_test.parquet")
    assert values["generation_data_path"].endswith("/data/train/train_gtsqa_v9.jsonl")
    assert values["kgqa_profile_path"].endswith("/configs/kg/profiles/gtsqa_training.yaml")
    assert values["proposer_rag_filter"] == "True"
    assert values["proposer_rag_fail_open"] == "False"
    assert values["rag_max_workers"] == "300"
    assert values["kgqa_protocol_gate_enable"] == "True"


def test_gtsqa_training_profile_is_embedded_and_has_no_freebase_whitelist() -> None:
    profile = (REPO_ROOT / "configs/kg/profiles/gtsqa_training.yaml").read_text()

    assert "adapter: gtsqa_array" in profile
    assert "kind: embedded" in profile
    assert "max_relations: 50" in profile
    assert "freebase" not in profile.casefold()
    assert "whitelist" not in profile.casefold()


def test_car_v1_config_selects_embedded_backend_without_freebase() -> None:
    env = os.environ.copy()
    env.update(
        {
            "KGQA_CONFIG_ONLY": "1",
            "KGQA_EXPERIMENT_CONFIG": str(REPO_ROOT / "scripts/experiments/car_v1/.env.example"),
            "REPO_ROOT": str(REPO_ROOT),
        }
    )
    result = subprocess.run(
        ["bash", "scripts/platforms/local.sh"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert values["kgqa_backend"] == "car"
    assert values["freebase_lifecycle_enabled"] == "False"
    assert values["train_parquet"].endswith("/data/train/car_train.parquet")
    assert values["val_parquet"].endswith("/data/val/val_carbench.parquet")
    assert values["generation_data_path"].endswith("/data/train/car_train.jsonl")
    assert values["kgqa_profile_path"].endswith("/configs/kg/profiles/car_kg_training.yaml")
    assert values["car_kg_index_path"].endswith("/data/database/kg-platform-car-kg-index")
    assert values["proposer_rag_filter"] == "True"
    assert values["proposer_rag_fail_open"] == "False"
    assert values["rag_max_workers"] == "300"


def test_car_training_profile_is_embedded_global_array() -> None:
    profile = (REPO_ROOT / "configs/kg/profiles/car_kg_training.yaml").read_text()

    assert "adapter: global_array" in profile
    assert "kind: embedded" in profile
    assert "CAR_KG_INDEX_PATH" in profile
    assert "max_relations: 50" in profile
    assert "freebase_direct_sparql" not in profile
    assert "freebase_sparql" not in profile
    assert "whitelist" not in profile.casefold()


def test_local_runtime_runs_without_cluster_tools_and_preserves_training_failure(tmp_path: Path) -> None:
    """Exercise the full launcher with a fake Python runtime, including pipefail."""
    tools_dir = tmp_path / "bin"
    tools_dir.mkdir()
    calls = tmp_path / "python_calls.txt"
    fake_python = tools_dir / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$LAUNCH_CHECK_CALLS"\n'
        'if [[ "$*" == *"quarl.main_rl"* ]]; then\n'
        '  [[ -z "${RAY_ADDRESS:-}" ]] || exit 99\n'
        '  exit "${LAUNCH_CHECK_TRAIN_EXIT:-0}"\n'
        'fi\n'
    )
    fake_python.chmod(0o755)
    (tools_dir / "python3").symlink_to(fake_python)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text("{}")
    data_path = tmp_path / "data.placeholder"
    data_path.touch()
    env = {
        **os.environ,
        "PATH": str(tools_dir) + os.pathsep + os.environ["PATH"],
        "REPO_ROOT": str(REPO_ROOT),
        "PROJECT_ROOT": str(tmp_path),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "RAY_NODE_TMP_BASE": str(tmp_path / "ray"),
        "ENV_FILE": str(tmp_path / "no.env"),
        "KGQA_EXPERIMENT_CONFIG": str(REPO_ROOT / "scripts/experiments/baseline/.env.example"),
        "KGQA_CONFIG_ONLY": "0",
        "KGQA_PREFLIGHT_ONLY": "0",
        "KGQA_BACKEND": "gtsqa",
        "KGQA_PROFILE_PATH": str(REPO_ROOT / "configs/kg/profiles/gtsqa_training.yaml"),
        "GTSQA_INDEX_PATH": str(index_dir),
        "MODEL_PATH": str(model_dir),
        "TRAIN_PARQUET": str(data_path),
        "VAL_PARQUET": str(data_path),
        "GENERATION_DATA_PATH": str(data_path),
        "PROPOSER_RAG_FILTER": "False",
        "NUM_GPUS_PER_NODE": "1",
        "ROLLOUT_TP": "1",
        "RUN_ID": "launcher_check",
        "LAUNCH_CHECK_CALLS": str(calls),
        "RAY_ADDRESS": "stale-cluster:6379",
    }
    result = subprocess.run(
        ["bash", "scripts/platforms/local.sh"], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = calls.read_text()
    assert "kg_platform.cli.kgctl materialize" in commands
    assert "kg_platform.cli.kgctl prepare" in commands
    assert "quarl.main_rl" in commands
    assert "trainer.nnodes=1" in commands
    assert "self_play.proposer.enable=True" in commands
    assert "self_play.solver.enable=True" in commands
    assert (tmp_path / "output/launcher_check/run_summary.md").is_file()
    assert not (tmp_path / "output/launcher_check/diagnostics/env.txt").exists()

    env.update({"RUN_ID": "failed_check", "LAUNCH_CHECK_TRAIN_EXIT": "23"})
    result = subprocess.run(
        ["bash", "scripts/platforms/local.sh"], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    assert not (tmp_path / "output/failed_check/run_summary.md").exists()
