"""Overlong rollouts must remain trainable failures in their original GRPO groups.

Production forces a length-truncated rollout's reward to zero but keeps its normal response mask.
The original group can therefore assign it negative advantage when sibling rollouts perform better.
No UID rewrite or no-contrast mask is applied.
"""

import numpy as np
import pytest
import torch

from verl.trainer.ppo import core_algos

GROUP_COUNT = 64
ROLLOUT_N = 5
RESPONSE_LEN = 16
NORMAL_REWARD = 0.6


def _batch(overlong_per_group):
    """Build GRPO groups in which truncated rollouts have reward zero and a normal mask."""
    rewards, uid, overlong = [], [], []
    rng = np.random.default_rng(0)
    for group in range(GROUP_COUNT):
        n_cut = overlong_per_group[group]
        cut = set(rng.choice(ROLLOUT_N, size=n_cut, replace=False).tolist()) if n_cut else set()
        for rollout in range(ROLLOUT_N):
            is_cut = rollout in cut
            overlong.append(is_cut)
            uid.append(f"group-{group}")
            rewards.append(0.0 if is_cut else NORMAL_REWARD + 0.1 * (rollout % 3))

    batch_size = len(rewards)
    response_mask = torch.ones(batch_size, RESPONSE_LEN)
    token_level_rewards = torch.zeros(batch_size, RESPONSE_LEN)
    token_level_rewards[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    return token_level_rewards, response_mask, np.array(uid, dtype=object), np.array(overlong, dtype=bool)


def _sequence_advantages(advantages, response_mask):
    return (advantages * response_mask).sum(-1) / response_mask.sum(-1)


@pytest.mark.unit
def test_overlong_rollouts_receive_negative_advantage_in_original_group() -> None:
    token_level_rewards, response_mask, uid, overlong = _batch([1] * GROUP_COUNT)

    advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards,
        response_mask,
        index=uid,
        norm_adv_by_std_in_grpo=True,
    )
    seq_adv = _sequence_advantages(advantages, response_mask)

    assert torch.all(seq_adv[torch.tensor(overlong)] < 0)
    assert torch.all(response_mask[torch.tensor(overlong)].sum(-1) == RESPONSE_LEN)
    assert float(seq_adv.mean()) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
def test_one_good_rollout_and_four_overlong_rollouts_keep_group_contrast() -> None:
    per_group = [4] + [1] * (GROUP_COUNT - 1)
    token_level_rewards, response_mask, uid, overlong = _batch(per_group)

    advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards,
        response_mask,
        index=uid,
        norm_adv_by_std_in_grpo=True,
    )
    seq_adv = _sequence_advantages(advantages, response_mask)
    first_group_overlong = torch.tensor(overlong[:ROLLOUT_N])

    assert int((~first_group_overlong).sum()) == 1
    assert torch.all(seq_adv[:ROLLOUT_N][first_group_overlong] < 0)
    assert torch.all(seq_adv[:ROLLOUT_N][~first_group_overlong] > 0)
    assert torch.all(response_mask[:ROLLOUT_N].sum(-1) == RESPONSE_LEN)


@pytest.mark.unit
def test_all_overlong_group_is_finite_zero_advantage_not_an_empty_mask() -> None:
    per_group = [ROLLOUT_N] + [1] * (GROUP_COUNT - 1)
    token_level_rewards, response_mask, uid, _ = _batch(per_group)

    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards,
        response_mask,
        index=uid,
        norm_adv_by_std_in_grpo=True,
    )
    seq_adv = _sequence_advantages(advantages, response_mask)

    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()
    assert torch.all(seq_adv[:ROLLOUT_N] == 0)
    assert int(response_mask[:ROLLOUT_N].sum()) == ROLLOUT_N * RESPONSE_LEN


@pytest.mark.unit
def test_all_overlong_batch_keeps_valid_tokens() -> None:
    token_level_rewards, response_mask, uid, _ = _batch([ROLLOUT_N] * GROUP_COUNT)

    advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards,
        response_mask,
        index=uid,
        norm_adv_by_std_in_grpo=True,
    )

    assert torch.isfinite(advantages).all()
    assert torch.all(advantages == 0)
    assert int(response_mask.sum()) == GROUP_COUNT * ROLLOUT_N * RESPONSE_LEN
