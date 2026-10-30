from types import SimpleNamespace

import numpy as np
import pytest
import torch

import quarl.trainer.ppo.ssp_ray_trainer as trainer_module
from quarl.trainer.ppo.ssp_ray_trainer import SSPRayPPOTrainer


class _ConfigNode(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class _FakeBatch:
    def __init__(self, batch, *, non_tensor_batch=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = {}

    def ensure_unlocked(self):
        return self

    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update(other.non_tensor_batch)
        return self


class _FakeActorWorkerGroup:
    def __init__(self):
        self.compute_log_prob_calls = 0
        self.update_actor_calls = 0
        self.updated_batch = None

    def compute_log_prob(self, gen_batch):
        self.compute_log_prob_calls += 1
        response_mask = gen_batch.batch["response_mask"]
        return _FakeBatch(
            {
                "old_log_probs": torch.full_like(response_mask, -0.5, dtype=torch.float32),
                "entropys": torch.full_like(response_mask, 0.25, dtype=torch.float32),
            }
        )

    def update_actor(self, gen_batch):
        self.update_actor_calls += 1
        self.updated_batch = gen_batch
        return SimpleNamespace(meta_info={"metrics": {}})


@pytest.mark.unit
def test_capture_role_log_prob_is_idempotent_and_keeps_entropy() -> None:
    trainer = object.__new__(SSPRayPPOTrainer)
    trainer.actor_rollout_wg = _FakeActorWorkerGroup()
    batch = _FakeBatch({"response_mask": torch.ones((1, 2), dtype=torch.long)})

    captured = trainer._capture_role_log_prob(batch, "solver", {})
    captured_again = trainer._capture_role_log_prob(captured, "solver", {})

    assert captured_again is captured
    assert trainer.actor_rollout_wg.compute_log_prob_calls == 1
    assert torch.equal(captured.batch["old_log_probs"], torch.full((1, 2), -0.5))
    assert torch.equal(captured.batch["entropys"], torch.full((1, 2), 0.25))


@pytest.mark.unit
def test_role_update_reuses_precaptured_old_log_probs(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = object.__new__(SSPRayPPOTrainer)
    trainer.global_steps = 3
    trainer.rollout_n = 1
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.actor_rollout_wg = _FakeActorWorkerGroup()
    trainer.config = _ConfigNode(
        trainer=_ConfigNode(balance_batch=False, critic_warmup=0),
        actor_rollout_ref=_ConfigNode(
            actor=_ConfigNode(loss_agg_mode="token-mean"),
            rollout=_ConfigNode(multi_turn=_ConfigNode(enable=True)),
        ),
        algorithm=_ConfigNode(
            use_kl_in_reward=False,
            norm_adv_by_std_in_grpo=True,
            adv_estimator="grpo",
            gamma=1.0,
            lam=1.0,
        ),
    )
    trainer.sp_config = _ConfigNode(proposer=_ConfigNode(n=1, adv_estimator="grpo"))
    trainer._swap_proposer_baseline_key = lambda *args, **kwargs: None
    trainer._compute_overlong_metrics = lambda *args, **kwargs: {}
    trainer._compute_advantage_metrics = lambda *args, **kwargs: {}
    monkeypatch.setattr(trainer_module, "compute_advantage", lambda batch, **kwargs: batch)

    old_log_probs = torch.tensor([[-0.7, -0.8]], dtype=torch.float32)
    batch = _FakeBatch(
        {
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "response_mask": torch.ones((1, 2), dtype=torch.long),
            "token_level_scores": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "old_log_probs": old_log_probs.clone(),
            "entropys": torch.tensor([[0.2, 0.4]], dtype=torch.float32),
        },
        non_tensor_batch={"uid": np.array(["solver-problem-0"], dtype=object)},
    )

    metrics = trainer._update_on_trajectories(batch, "solver", {})

    assert trainer.actor_rollout_wg.compute_log_prob_calls == 0
    assert trainer.actor_rollout_wg.update_actor_calls == 1
    assert torch.equal(trainer.actor_rollout_wg.updated_batch.batch["old_log_probs"], old_log_probs)
    assert "entropys" not in trainer.actor_rollout_wg.updated_batch.batch
    assert metrics["solver/actor/entropy"] == pytest.approx(0.3)


@pytest.mark.unit
def test_protocol_or_effective_reward_rejection_masks_solver_signal() -> None:
    trainer = object.__new__(SSPRayPPOTrainer)
    rewards = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)
    extra_infos = {
        "kgqa_protocol_gate_score": [1.0, 0.0, 1.0],
        "kgqa_answer_em_entity_f1": [1.0, 1.0, 1.0],
    }

    eligibility = trainer._extract_solver_reward_eligibility_tensor(rewards, extra_infos)
    proposer_scores = trainer._extract_solver_score_tensor(rewards, extra_infos)

    assert eligibility.tolist() == [True, False, False]
    assert proposer_scores.tolist() == [1.0, 0.0, 0.0]


@pytest.mark.unit
def test_solver_outcome_rows_classify_rejected_correct_as_failure() -> None:
    trainer = object.__new__(SSPRayPPOTrainer)
    trainer.rollout_n = 1
    extra_infos = [
        {
            "qid": "accepted",
            "dataset_type": "gtsqa",
            "kgqa_backend": "gtsqa",
            "question_entities": [{"id": "Q1", "name": "Seed"}],
            "answer_entities": [{"id": "Q2", "name": "Gold"}],
            "path": ["Seed --r--> Gold"],
        },
        {"qid": "rejected", "dataset_type": "gtsqa", "kgqa_backend": "gtsqa"},
    ]

    success_rows = trainer._build_solver_success_rows(
        outputs=["ok", "bad protocol"],
        scores=[1.0, 0.0],
        raw_answer_corrects=[True, True],
        reward_eligibilities=[True, False],
        extra_infos=extra_infos,
        data_sources=["kgqa_svts_gtsqa", "kgqa_svts_gtsqa"],
        step=4,
    )
    fail_rows = trainer._build_solver_fail_rows(
        outputs=["ok", "bad protocol"],
        scores=[1.0, 0.0],
        raw_answer_corrects=[True, True],
        reward_eligibilities=[True, False],
        extra_infos=extra_infos,
        data_sources=["kgqa_svts_gtsqa", "kgqa_svts_gtsqa"],
        step=4,
    )

    assert [row["qid"] for row in success_rows] == ["accepted"]
    assert [row["qid"] for row in fail_rows] == ["rejected"]
    assert fail_rows[0]["raw_answer_correct"] is True
    assert fail_rows[0]["reward_eligible"] is False
    assert fail_rows[0]["rejected_correct"] is True
    assert fail_rows[0]["failure_reason"] == "reward_rejected_correct"
