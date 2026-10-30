"""Advantage-baseline behaviour for the proposer under ``proposer.n == 1``.

Locks in the two properties the 20260728 rework depends on:

1. ``grpo`` with one rollout per prompt leaves the advantage equal to the raw, all-positive
   proposer reward — every group has size 1, so verl skips centering entirely.
2. ``reinforce_plus_plus_baseline`` centers it, and grouping the baseline by ``template_id``
   additionally removes the per-topology offset that a mixed-topology batch introduces.
"""

import numpy as np
import pytest
import torch

from verl.trainer.ppo import core_algos

TEMPLATES = ["Iso-0", "Iso-1", "Iso-2", "Iso-3", "Iso-5"]
# Per-topology mean proposer reward, measured on run_20260722_055535_job306859 at step 200.
TEMPLATE_REWARD_MEAN = {"Iso-0": 0.48, "Iso-1": 0.52, "Iso-2": 0.45, "Iso-3": 0.63, "Iso-5": 0.56}
# The only three values gaussian_difficulty can take with solver rollout.n == 5.
REWARD_VALUES = np.array([0.044, 0.667, 0.995])

BATCH_SIZE = 256
RESPONSE_LEN = 40


def _mixed_topology_batch(seed: int = 0):
    rng = np.random.default_rng(seed)
    templates = np.array([TEMPLATES[i % len(TEMPLATES)] for i in range(BATCH_SIZE)], dtype=object)

    rewards = []
    for template in templates:
        mean = TEMPLATE_REWARD_MEAN[template]
        weights = np.array([max(0.05, 0.9 - mean), 0.35, max(0.05, mean - 0.25)])
        rewards.append(rng.choice(REWARD_VALUES, p=weights / weights.sum()))

    response_mask = torch.ones(BATCH_SIZE, RESPONSE_LEN)
    token_level_rewards = torch.zeros(BATCH_SIZE, RESPONSE_LEN)
    token_level_rewards[:, -1] = torch.tensor(np.array(rewards), dtype=torch.float32)

    # proposer.n == 1 means one uid per seed KG path, i.e. every group holds a single rollout.
    uid = np.array([f"path-{i}" for i in range(BATCH_SIZE)], dtype=object)
    return token_level_rewards, response_mask, uid, templates


def _sequence_advantages(advantages, response_mask):
    return (advantages * response_mask).sum(-1) / response_mask.sum(-1)


def _topology_spread(seq_adv, templates):
    means = [float(seq_adv[torch.tensor(templates == t)].mean()) for t in TEMPLATES]
    return max(means) - min(means)


@pytest.mark.unit
def test_grpo_with_single_rollout_groups_leaves_advantage_uncentered() -> None:
    token_level_rewards, response_mask, uid, _ = _mixed_topology_batch()

    advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards, response_mask, index=uid, norm_adv_by_std_in_grpo=True
    )
    seq_adv = _sequence_advantages(advantages, response_mask)

    # Not a single negative advantage: the policy is pushed to raise the likelihood of every
    # rollout it just produced, in proportion to the raw reward.
    assert float((seq_adv < 0).float().mean()) == 0.0
    assert float(seq_adv.mean()) > 0.4


@pytest.mark.unit
def test_reinforce_plus_plus_baseline_centers_the_advantage() -> None:
    token_level_rewards, response_mask, uid, _ = _mixed_topology_batch()

    advantages, _ = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
        token_level_rewards, response_mask, index=uid
    )
    seq_adv = _sequence_advantages(advantages, response_mask)

    assert float(seq_adv.mean()) == pytest.approx(0.0, abs=1e-5)
    assert 0.3 < float((seq_adv < 0).float().mean()) < 0.7
    assert float(seq_adv.std()) == pytest.approx(1.0, abs=0.05)


@pytest.mark.unit
def test_template_id_baseline_removes_the_per_topology_offset() -> None:
    token_level_rewards, response_mask, uid, templates = _mixed_topology_batch()

    by_path, _ = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
        token_level_rewards.clone(), response_mask, index=uid
    )
    by_template, _ = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
        token_level_rewards.clone(), response_mask, index=templates
    )

    path_spread = _topology_spread(_sequence_advantages(by_path, response_mask), templates)
    template_spread = _topology_spread(_sequence_advantages(by_template, response_mask), templates)

    # A batch-mean baseline hands every topology a constant offset; grouping by template cancels it.
    # The advantage std is ~1 here, so path_spread reads directly in sigma — the run measured
    # 0.33-0.69 sigma on real trajectories.
    assert path_spread > 0.25
    assert template_spread == pytest.approx(0.0, abs=1e-5)
