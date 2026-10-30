import pytest
import torch

from quarl.utils.proposer_reward import compute_proposer_reward_values


@pytest.mark.unit
def test_proposer_reward_default_bridge_is_one_minus_accuracy() -> None:
    success_rates = torch.tensor([0.0, 0.25, 1.0])

    rewards = compute_proposer_reward_values(success_rates, reward_type="1-acc")

    assert torch.allclose(rewards, torch.tensor([1.0, 0.75, 0.0]))


@pytest.mark.unit
def test_proposer_reward_intermediate_difficulty_peaks_in_middle() -> None:
    success_rates = torch.tensor([0.0, 0.5, 1.0])

    rewards = compute_proposer_reward_values(success_rates, reward_type="intermediate_difficulty")

    assert rewards[0].item() == 0.0
    assert rewards[1].item() > 0.0
    assert rewards[2].item() == 0.0


@pytest.mark.unit
def test_proposer_reward_gaussian_difficulty_prefers_boundary() -> None:
    success_rates = torch.tensor([0.0, 0.5, 1.0])

    rewards = compute_proposer_reward_values(
        success_rates,
        reward_type="gaussian_difficulty",
        proposer_tau=0.01,
    )

    assert rewards[1].item() == pytest.approx(1.0)
    assert rewards[0].item() < rewards[1].item()
    assert rewards[2].item() < rewards[1].item()
    assert rewards[0].item() == pytest.approx(rewards[2].item())


@pytest.mark.unit
def test_proposer_reward_gaussian_tau_is_configurable() -> None:
    success_rates = torch.tensor([0.0])

    sharp_reward = compute_proposer_reward_values(
        success_rates,
        reward_type="gaussian_difficulty",
        proposer_tau=0.005,
    )
    smooth_reward = compute_proposer_reward_values(
        success_rates,
        reward_type="gaussian_difficulty",
        proposer_tau=0.05,
    )

    assert smooth_reward.item() > sharp_reward.item()


@pytest.mark.unit
def test_proposer_reward_triangular_difficulty_peaks_at_half() -> None:
    success_rates = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    rewards = compute_proposer_reward_values(success_rates, reward_type="triangular_difficulty")

    assert torch.allclose(rewards, torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0]))


@pytest.mark.unit
def test_gaussian_score_variance_matches_gaussian_difficulty_on_binary_scores() -> None:
    """The continuous variant is a strict generalisation: feed it binary scores and it agrees.

    gaussian_difficulty is SPICE's Challenger reward (arXiv:2510.24684), whose Var is taken over
    binary correctness indicators — and the biased sample variance of a binary indicator is
    exactly p(1-p).
    """
    solver_scores = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    success_rates = solver_scores.mean(dim=-1)

    binary = compute_proposer_reward_values(success_rates, reward_type="gaussian_difficulty")
    continuous = compute_proposer_reward_values(
        success_rates, reward_type="gaussian_score_variance", solver_scores=solver_scores
    )

    assert torch.allclose(binary, continuous, atol=1e-6)


@pytest.mark.unit
def test_gaussian_score_variance_sees_signal_that_binarisation_destroys() -> None:
    """Four half-right answers plus one fully-right answer is not a trivial problem.

    Every rollout overlaps the gold set, so the binary success rate is 1.0 and the binary reward
    writes the problem off as too easy — even though one rollout recovered the full answer set.
    """
    solver_scores = torch.tensor([[0.5, 0.5, 0.5, 0.5, 1.0]])
    success_rates = torch.tensor([1.0])  # every rollout overlapped the gold set

    binary = compute_proposer_reward_values(success_rates, reward_type="gaussian_difficulty")
    continuous = compute_proposer_reward_values(
        success_rates, reward_type="gaussian_score_variance", solver_scores=solver_scores
    )

    assert binary[0].item() == pytest.approx(0.044, abs=1e-3)  # the floor: "too easy"
    assert continuous[0].item() == pytest.approx(0.110, abs=1e-3)  # within-group variance is 0.04
    assert continuous[0].item() > 2 * binary[0].item()


@pytest.mark.unit
def test_gaussian_score_variance_is_monotone_in_the_within_group_variance() -> None:
    solver_scores = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5, 0.5],  # var 0.0
            [0.5, 0.5, 0.5, 0.5, 0.55],  # var ~0.0004
            [0.4, 0.45, 0.5, 0.55, 0.6],  # var 0.005
            [0.0, 0.0, 0.0, 1.0, 1.0],  # var 0.24
        ]
    )
    rewards = compute_proposer_reward_values(
        solver_scores.mean(dim=-1), reward_type="gaussian_score_variance", solver_scores=solver_scores
    )

    assert rewards[0] < rewards[1] < rewards[2] < rewards[3]


@pytest.mark.unit
def test_gaussian_score_variance_requires_the_score_matrix() -> None:
    with pytest.raises(ValueError, match="gaussian_score_variance"):
        compute_proposer_reward_values(torch.tensor([0.5]), reward_type="gaussian_score_variance")
