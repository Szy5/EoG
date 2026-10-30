from typing import Optional

import torch


def compute_proposer_reward_values(
    success_rates: torch.Tensor,
    reward_type: str = "1-acc",
    proposer_left: float = 0.0,
    proposer_right: float = 1.0,
    proposer_tau: float = 0.01,
    solver_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Score each generated problem from how the solver fared on it.

    ``success_rates`` is ``[B]`` — the fraction of solver rollouts judged correct.
    ``solver_scores`` is ``[B, n]`` — the raw per-rollout solver scores, required by the
    reward types that need a continuous signal rather than a binary success rate.
    """
    if reward_type == "intermediate_difficulty":
        """
        中间难度奖励函数
        """
        proposer_reward_values = torch.zeros_like(success_rates)
        intermediate_mask = (success_rates > 0) & (success_rates < 1)
        proposer_reward_values[intermediate_mask] = (
            4.0
            * (proposer_left + success_rates[intermediate_mask])
            * (proposer_right - success_rates[intermediate_mask])
        )
        return proposer_reward_values

    if reward_type == "format_only":
        return torch.ones_like(success_rates)

    if reward_type == "gaussian_difficulty":
        """
        高斯难度奖励函数（SPICE 的 Challenger reward，https://arxiv.org/abs/2510.24684）

            r = exp( -(Var - 0.25)^2 / (2*tau) )

        其中 Var 是同一道题 n 次 solver rollout 的【二值】正确性的方差。二值指示变量
        的有偏样本方差恰好等于 p(1-p)，所以下面直接用 success_rates 算。

        注意分辨率：Var 的可取值个数由 solver rollout 数 n 决定 ——
        n=5 只有 3 个值 [0, 0.16, 0.24]（p=0.4 与 p=0.6 完全不可分），
        n=8 有 5 个值，n=16 有 9 个。SPICE 用的是 n=8。
        """
        tau = float(proposer_tau)
        if tau <= 0.0:
            raise ValueError("proposer_tau must be positive for gaussian_difficulty reward")
        sigma2 = success_rates * (1.0 - success_rates)
        return torch.exp(-((sigma2 - 0.25) ** 2) / (2.0 * tau))

    if reward_type == "gaussian_score_variance":
        """
        gaussian_difficulty 在集合值答案下的推广。

        形状与上面完全相同，只把「二值正确性的方差」换成「连续 entity F1 的方差」。
        动机是二值化本身丢信息：KGQA 的答案是集合，solver 五次分别拿到
        [0.5, 0.5, 0.5, 0.5, 1.0] 这种情况，二值化后 p=1（每次都至少蒙对一个实体），
        高斯奖励判定「太简单」，但其中显然有真实的学习信号。

        两者对题目的排序在 Var∈[0, 0.25] 上一致（该区间内二者都严格单调递增），
        差别只在打分疏密，所以这是最小偏离。保留 gaussian_difficulty 便于做消融。
        """
        tau = float(proposer_tau)
        if tau <= 0.0:
            raise ValueError("proposer_tau must be positive for gaussian_score_variance reward")
        if solver_scores is None:
            raise ValueError("gaussian_score_variance reward requires solver_scores of shape [B, n]")
        if solver_scores.dim() != 2:
            raise ValueError(f"solver_scores must be 2-D [B, n], got shape {tuple(solver_scores.shape)}")
        sigma2 = solver_scores.var(dim=-1, unbiased=False)
        return torch.exp(-((sigma2 - 0.25) ** 2) / (2.0 * tau))

    if reward_type == "triangular_difficulty":
        """
        三角形中间难度奖励函数：p=0.5 时最高，p=0 或 p=1 时为 0。
        """
        rewards = 1.0 - 2.0 * torch.abs(success_rates - 0.5)
        return torch.clamp(rewards, min=0.0, max=1.0)

    if reward_type == "1-acc":
        """
        奖励函数为1-success_rate
        难度越高的问题，奖励高
        """
        return 1.0 - success_rates

    raise ValueError(f"Invalid reward type: {reward_type}")
