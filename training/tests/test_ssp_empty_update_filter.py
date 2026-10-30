from types import SimpleNamespace

import numpy as np
import torch

from verl import DataProto

from quarl.trainer.ppo.ssp_ray_trainer import SSPRayPPOTrainer


def _make_trainer(world_size: int = 1) -> SSPRayPPOTrainer:
    trainer = SSPRayPPOTrainer.__new__(SSPRayPPOTrainer)
    trainer.rollout_n = 5
    trainer.global_steps = 20
    trainer.actor_rollout_wg = SimpleNamespace(world_size=world_size)
    return trainer


def _make_batch(data_sources: list[str], uids: list[str], nonempty: list[bool]) -> DataProto:
    response_mask = torch.zeros((len(data_sources), 3), dtype=torch.long)
    for index, has_tokens in enumerate(nonempty):
        if has_tokens:
            response_mask[index, :2] = 1
    return DataProto.from_dict(
        tensors={"response_mask": response_mask},
        non_tensors={
            "data_source": np.asarray(data_sources, dtype=object),
            "uid": np.asarray(uids, dtype=object),
        },
    )


def test_prepare_solver_update_batch_removes_dummy_groups_after_reward_alignment():
    batch = _make_batch(
        data_sources=["dummy"] * 5 + ["kgqa_selfplay_v2"] * 5,
        uids=["dummy-group"] * 5 + ["real-group"] * 5,
        nonempty=[False] * 5 + [True] * 5,
    )

    filtered, metrics = _make_trainer()._prepare_solver_update_batch(batch)

    assert filtered is not None
    assert len(filtered) == 5
    assert set(filtered.non_tensor_batch["uid"].tolist()) == {"real-group"}
    assert metrics["dummy_update_sequence_count"] == 5
    assert metrics["empty_update_sequence_count"] == 5
    assert metrics["retained_update_sequence_count"] == 5


def test_prepare_solver_update_batch_skips_all_dummy_batch():
    batch = _make_batch(
        data_sources=["dummy"] * 5,
        uids=["dummy-group"] * 5,
        nonempty=[False] * 5,
    )

    filtered, metrics = _make_trainer()._prepare_solver_update_batch(batch)

    assert filtered is None
    assert metrics["actor/update_skipped_empty_batch"] == 1
    assert metrics["nonempty_update_sequence_count"] == 0


def test_prepare_solver_update_batch_drops_whole_group_for_one_unexpected_empty_row():
    batch = _make_batch(
        data_sources=["kgqa_selfplay_v2"] * 10,
        uids=["bad-group"] * 5 + ["good-group"] * 5,
        nonempty=[False] + [True] * 9,
    )

    filtered, metrics = _make_trainer()._prepare_solver_update_batch(batch)

    assert filtered is not None
    assert set(filtered.non_tensor_batch["uid"].tolist()) == {"good-group"}
    assert metrics["unexpected_empty_update_sequence_count"] == 1
    assert metrics["retained_update_sequence_count"] == 5


def test_prepare_solver_update_batch_drops_minimum_complete_groups_for_world_size():
    data_sources = ["dummy"] * (167 * 5) + ["kgqa_selfplay_v2"] * (89 * 5)
    uids = [uid for i in range(167) for uid in [f"dummy-{i}"] * 5] + [
        uid for i in range(89) for uid in [f"real-{i}"] * 5
    ]
    batch = _make_batch(
        data_sources=data_sources,
        uids=uids,
        nonempty=[False] * (167 * 5) + [True] * (89 * 5),
    )

    filtered, metrics = _make_trainer(world_size=8)._prepare_solver_update_batch(batch)

    assert filtered is not None
    assert len(filtered) == 440
    assert len(filtered) % 8 == 0
    assert len(set(filtered.non_tensor_batch["uid"].tolist())) == 88
    assert metrics["dropped_group_count"] == 1
    assert metrics["dropped_sequence_count"] == 5
