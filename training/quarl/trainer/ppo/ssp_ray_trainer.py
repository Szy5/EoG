"""
Self-Play Ray PPO Trainer for QUARL.

This trainer implements the self-play algorithm where a model acts as both
problem generator and problem solver in alternating phases within each global step.

Self-play flow:
1. Problem Generation Phase: Model generates problems from seeded data
2. Problem Extraction: Extract questions from generation trajectories
3. Problem Solving Phase: Model solves the extracted problems
4. Dual Training: Update model twice with rewards from both phases
"""

import json
import logging
import math
import os
import time
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tracking import Tracking
from verl.workers.rollout.schemas import FinishReasonTypeEnum

from quarl.utils.process_reward import enrich_extra_info_with_process_reward
from quarl.utils.proposer_reward import compute_proposer_reward_values
from quarl.utils.ssp_data_manager import SelfPlayPhase, SSPDataManager
from quarl.utils.problem_extraction import (
    ProblemExtractor,
    extract_problems_batch,
    render_structured_react_trajectory,
)
from quarl.utils.timing import accumulate_rollout_timing

logger = logging.getLogger(__name__)

# Enable debug logging for self-play
SELF_PLAY_DEBUG = os.environ.get("SELF_PLAY_DEBUG", "False").lower() == "true"
if SELF_PLAY_DEBUG:
    logger.setLevel(logging.DEBUG)


class SSPRayPPOTrainer(RayPPOTrainer):
    """
    Self-Play Ray PPO Trainer that implements the dual-phase training process.

    This trainer orchestrates the self-play loop:
    - Phase 1: 问题生成和抽取
    - Phase 2: 问题解答
    - 双阶段模型更新，分别使用生成和解答的奖励
    """

    # Preset buckets for the per-template stats table, matching the Iso-N
    # template ids carried by SVTS tasks in ``extra_info.template_id``.
    # Unknown ids still work: the counters below use setdefault, so this only
    # fixes the reported column set.
    # Iso-7 and Iso-10 were retired into Iso-16 and Iso-4 respectively.
    KGQA_TEMPLATE_IDS = (
        "Iso-0", "Iso-1", "Iso-2", "Iso-3", "Iso-4", "Iso-5", "Iso-6", "Iso-8",
        "Iso-9", "Iso-11", "Iso-12", "Iso-13", "Iso-14", "Iso-15", "Iso-16",
        "Iso-17", "Iso-18", "Iso-19",
    )

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
    ):
        self.sp_config = config.self_play

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn,
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            device_name,
        )

        self.current_phase = SelfPlayPhase.PROBLEM_GENERATION

        self.sp_data_manager = SSPDataManager.remote()
        # 问题抽取器，负责从生成轨迹中抽取问题
        self.problem_extractor = ProblemExtractor(
            lang=self.config.self_play.lang,
            use_rag_filter=self.sp_config.use_rag_filter,
            use_search_terms_filter=self.sp_config.use_search_terms_filter,
            noisy_rag_materials=self.sp_config.get("noisy_RAG_materials", 0),
            answer_pattern=self.sp_config.get("answer_pattern", "answer"),
            synthesis_mode=self.sp_config.get("synthesis_mode", "kg_path"),
            rag_filter_fail_open=self.sp_config.get("rag_filter_fail_open", False),
            gate_entity_f1_min=self.sp_config.get("extraction_gate", {}).get("entity_f1_min", 0.0),
            gate_reject_question_entity_answers=self.sp_config.get("extraction_gate", {}).get(
                "reject_question_entity_answers", False
            ),
            gate_reject_question_leaking_gold=self.sp_config.get("extraction_gate", {}).get(
                "reject_question_leaking_gold", False
            ),
        )
        self.process_reward_node_filter_mode = self._get_process_reward_node_filter_mode()

        self._load_problem_generation_data()

    def _get_process_reward_node_filter_mode(self) -> str:
        node_filter_mode = OmegaConf.select(
            self.config,
            "custom_reward_functions.process_score.kwargs.node_filter_mode",
            default="none",
        )
        if not isinstance(node_filter_mode, str) or not node_filter_mode.strip():
            return "none"
        return node_filter_mode.strip().lower()

    def _validate_config(self):
        val = self.sp_config.get("validate_config", True)
        if isinstance(val, str):
            val = val.lower() in ("1", "true", "yes", "y")
        if val:
            super()._validate_config()
        else:
            print("Skipping config validation as per self_play.validate_config=False")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, extra_fields=None):
        filename, n = self._write_generation_jsonl(
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
            extra_fields=extra_fields,
            mode="w",
            fsync_file=False,
        )
        print(f"Dumped {n} generations to {filename}")

    @staticmethod
    def _json_dump_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return str(obj)

    def _generation_dump_filename(self, dump_path):
        os.makedirs(dump_path, exist_ok=True)
        return os.path.join(dump_path, f"{self.global_steps}.jsonl")

    @staticmethod
    def _as_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    def _read_generation_jsonl(self, filename):
        if not os.path.exists(filename):
            return []

        entries = []
        with open(filename, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"WARNING: Ignoring invalid JSONL line {line_no} in {filename}: {exc}")
                    break
        return entries

    def _rewrite_generation_jsonl(self, filename, entries):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False, default=self._json_dump_default) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _append_loaded_validation_entries(
        self,
        entries,
        sample_inputs,
        sample_outputs,
        sample_scores,
        reward_extra_infos_dict,
        data_source_lst,
        reward_model_lst,
    ):
        if not entries:
            return

        loaded_scores = []
        loaded_data_sources = []
        loaded_reward_models = []

        for entry in entries:
            sample_inputs.append(entry.get("input", ""))
            sample_outputs.append(entry.get("output", ""))
            score = entry.get("score", entry.get("reward", 0.0))
            loaded_scores.append(score)
            sample_scores.append(score)
            loaded_data_sources.append(entry.get("data_source", "unknown"))
            loaded_reward_models.append(entry.get("reward_model", "unknown"))

            for key, value in entry.items():
                if key in {"input", "output", "score", "step", "data_source", "reward_model"}:
                    continue
                reward_extra_infos_dict[key].append(value)

            if "reward" not in entry:
                reward_extra_infos_dict["reward"].append(score)

        data_source_lst.append(np.array(loaded_data_sources, dtype=object))
        reward_model_lst.append(np.array(loaded_reward_models, dtype=object))

    def _write_generation_jsonl(
        self,
        inputs,
        outputs,
        scores,
        reward_extra_infos_dict,
        dump_path,
        extra_fields=None,
        mode="a",
        fsync_file=False,
    ):
        filename = self._generation_dump_filename(dump_path)
        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in (reward_extra_infos_dict or {}).items():
            if len(v) == n:
                base_data[k] = v

        if extra_fields:
            for k, v in extra_fields.items():
                if len(v) == n:
                    base_data[k] = v

        with open(filename, mode, encoding="utf-8") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=self._json_dump_default) + "\n")
            f.flush()
            if fsync_file:
                os.fsync(f.fileno())

        return filename, n

    def _validate(self):
        data_source_lst = []
        reward_model_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        streaming_val_filename = None
        validation_resume = self._as_bool(OmegaConf.select(self.config, "trainer.validation_resume", default=False))
        resume_entries = []
        resume_entry_cursor = 0
        resume_rows_remaining = 0

        if val_data_dir:
            streaming_val_filename = self._generation_dump_filename(val_data_dir)
            if validation_resume:
                resume_entries = self._read_generation_jsonl(streaming_val_filename)
                resume_rows_remaining = len(resume_entries)
                print(
                    f"Streaming validation generations to {streaming_val_filename} "
                    f"(resume enabled, found {resume_rows_remaining} existing rows)"
                )
            else:
                with open(streaming_val_filename, "w", encoding="utf-8"):
                    pass
                print(f"Streaming validation generations to {streaming_val_filename}")

        for test_data in tqdm(self.val_dataloader, desc="Validation Progress"):
            test_batch = DataProto.from_single_dict(test_data)

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
            current_batch_rows = len(input_ids)

            if resume_rows_remaining >= current_batch_rows:
                loaded_entries = resume_entries[resume_entry_cursor : resume_entry_cursor + current_batch_rows]
                self._append_loaded_validation_entries(
                    entries=loaded_entries,
                    sample_inputs=sample_inputs,
                    sample_outputs=sample_outputs,
                    sample_scores=sample_scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    data_source_lst=data_source_lst,
                    reward_model_lst=reward_model_lst,
                )
                resume_entry_cursor += current_batch_rows
                resume_rows_remaining -= current_batch_rows
                print(
                    "Skipping completed validation batch from resume file "
                    f"(rows={current_batch_rows}, consumed={resume_entry_cursor})"
                )
                continue

            if resume_rows_remaining > 0:
                kept_entries = resume_entries[:resume_entry_cursor]
                if streaming_val_filename:
                    self._rewrite_generation_jsonl(streaming_val_filename, kept_entries)
                print(
                    "WARNING: Existing validation file ends inside a batch; "
                    f"kept {len(kept_entries)} complete rows and will rerun the partial batch."
                )
                resume_entries = kept_entries
                resume_rows_remaining = 0

            if input_ids.dtype != torch.long:
                print(f"WARNING: Converting input_ids from {input_ids.dtype} to torch.long")
                input_ids = input_ids.long()

            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            val_batch_index = len(data_source_lst) + 1
            val_batch_size = len(test_gen_batch.batch)
            val_padded_batch_size = len(test_gen_batch_padded.batch)
            val_generation_start_time = time.time()
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            val_generation_elapsed = time.time() - val_generation_start_time
            print(
                "VALIDATION_BATCH_TIMING "
                f"batch={val_batch_index} "
                f"samples={val_batch_size} "
                f"padded_samples={val_padded_batch_size} "
                f"seconds={val_generation_elapsed:.3f}"
            )

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            output_ids = test_output_gen_batch.batch["responses"]

            print(f"DEBUG: Validation batch #{len(data_source_lst) + 1}")
            print(f"DEBUG: output_ids type: {type(output_ids)}")
            print(f"DEBUG: output_ids dtype: {output_ids.dtype}")
            print(f"DEBUG: output_ids shape: {output_ids.shape}")
            print(f"DEBUG: output_ids device: {output_ids.device}")
            print(f"DEBUG: test_gen_batch meta_info: {test_gen_batch.meta_info}")

            if output_ids.dtype != torch.long:
                print(f"ERROR: Found float32 output_ids!")
                print(f"DEBUG: output_ids min/max: {output_ids.min().item()} / {output_ids.max().item()}")
                print(f"DEBUG: Contains non-integer values: {not torch.all(output_ids == output_ids.long())}")
                print(f"DEBUG: First few values: {output_ids[0, :100]}")

                is_integer_valued = torch.all(output_ids == output_ids.round())
                print(f"DEBUG: All values are integer-valued: {is_integer_valued}")

                print(f"WARNING: Converting output_ids from {output_ids.dtype} to torch.long")
                if is_integer_valued:
                    output_ids = output_ids.long()
                else:
                    print("ERROR: Cannot safely convert non-integer floats to long!")
                    output_ids = output_ids.round().long()

            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # Ensure batch is unlocked before modification
            test_batch.ensure_unlocked()
            for key in test_batch.batch.keys():
                if key not in ["old_log_probs", "ref_log_prob"]:
                    test_batch.batch[key] = test_batch.batch[key].long()

            test_batch.meta_info["validate"] = True
            # 使用val_reward_fn计算奖励
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            batch_reward_extra_infos = {"reward": scores}
            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    batch_reward_extra_infos[key] = lst
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            batch_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            batch_reward_models = test_batch.non_tensor_batch.get("reward_model", ["unknown"] * reward_tensor.shape[0])
            data_source_lst.append(batch_data_sources)
            reward_model_lst.append(batch_reward_models)

            if val_data_dir:
                extra_fields = {
                    "data_source": batch_data_sources,
                    "reward_model": batch_reward_models,
                }
                filename, n_written = self._write_generation_jsonl(
                    inputs=input_texts,
                    outputs=output_texts,
                    scores=scores,
                    reward_extra_infos_dict=batch_reward_extra_infos,
                    dump_path=val_data_dir,
                    extra_fields=extra_fields,
                    mode="a",
                    fsync_file=True,
                )
                print(f"Appended {n_written} validation generations to {filename}")

        if validation_resume and resume_rows_remaining > 0 and streaming_val_filename:
            kept_entries = resume_entries[:resume_entry_cursor]
            self._rewrite_generation_jsonl(streaming_val_filename, kept_entries)
            print(
                "WARNING: Existing validation file has extra rows beyond the current dataloader; "
                f"kept {len(kept_entries)} rows and dropped {resume_rows_remaining} extra rows."
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        if streaming_val_filename:
            print(f"Dumped {len(sample_inputs)} streamed validation generations to {streaming_val_filename}")

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        avg_core_metrics = defaultdict(list)
        avg_core_metric_var = None
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            aggregate_var = (
                "reward"
                if "reward" in var2metric2val
                else ("acc" if "acc" in var2metric2val else None)
            )
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    if (
                        aggregate_var is not None
                        and var_name == aggregate_var
                        and metric_name.startswith("mean")
                        and (f"@{n_max}" in metric_name)
                    ):
                        if aggregate_var in {"reward", "acc"}:
                            avg_core_metrics[metric_name].append(metric_val)
                            avg_core_metric_var = aggregate_var
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        for metric_name, metric_vals in avg_core_metrics.items():
            if metric_vals:
                metric_var = avg_core_metric_var or "reward"
                metric_dict[f"val-core/all_datasets/{metric_var}/{metric_name}"] = float(np.mean(metric_vals))

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _load_problem_generation_data(self):
        try:
            generation_data_path = self.sp_config.get("generation_data_path")
            if generation_data_path and os.path.exists(generation_data_path):
                with open(generation_data_path, "r") as f:
                    if generation_data_path.endswith(".jsonl"):
                        generation_data = [json.loads(line) for line in f]
                    else:
                        generation_data = json.load(f)

                ray.get(self.sp_data_manager.set_problem_generation_data.remote(generation_data))
                logger.info(f"Loaded {len(generation_data)} problem generation samples")
            else:
                logger.warning("No generation data path specified or file not found")
                dummy_data = self._create_dummy_generation_data()
                ray.get(self.sp_data_manager.set_problem_generation_data.remote(dummy_data))

        except Exception as e:
            logger.error(f"Error loading problem generation data: {e}")
            raise

    def _create_dummy_generation_data(self) -> List[Dict[str, Any]]:
        dummy_data = [
            {
                "prompt": [{"role": "user", "content": "What are the main causes of the Great Depression, and how did government policies influence its course?"}],
                "data_source": "dummy_generation",
                "reward_model": {"style": "rule"},
                "extra_info": {"type": "qa_search"},
            },
            {
                "prompt": [{"role": "user", "content": "How does CRISPR gene editing work, and what are its potential applications in medicine?"}],
                "data_source": "dummy_generation",
                "reward_model": {"style": "rule"},
                "extra_info": {"type": "qa_search"},
            },
        ]
        return dummy_data

    def fit(self):
        from verl.trainer.ppo.metric_utils import (
            compute_throughout_metrics,
            compute_timing_metrics,
        )

        from quarl.utils.patch.metric_patch import quarl_compute_data_metrics

        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name + "_selfplay",
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                tracking_logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Self-Play Training")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        reward_dynamic_sampling_enabled = self.sp_config.get("reward_dynamic_sampling", {}).get("enable", False)

        if reward_dynamic_sampling_enabled:
            accumulated_proposer_batch = None
            accumulated_solver_batch = None
            last_step_proposer_batch = None
            last_step_solver_batch = None
            num_prompt_in_batch = 0
            num_gen_batches = 0
            target_prompt_batch_size = self.config.data.train_batch_size
            max_num_gen_batches = self.sp_config.get("reward_dynamic_sampling", {}).get("max_num_gen_batches", 20)

        for epoch in range(self.config.trainer.total_epochs):
            dataloader_iter = iter(self.train_dataloader)
            print("Epoch:", epoch)
            while True:
                try:
                    timing_raw = {}
                    all_metrics = {}
                    step_total_start = time.perf_counter()

                    is_last_step = self.global_steps >= self.total_training_steps

                    step_metrics = {}

                    with marked_timer("proposer_generation", timing_raw):
                        proposer_gen_batch, solver_batch, sampling_metrics = (
                            # 对Proposer 进行Rollout轨迹生成并进行问题抽取
                            self._generate_proposer_with_dynamic_sampling(dataloader_iter, timing_raw)
                        )

                        if proposer_gen_batch is None and solver_batch is None:
                            print(
                                "No valid problems extracted after dynamic sampling, dataloader exhausted, breaking to next epoch"
                            )
                            break

                        step_metrics["self_play/proposer_samples"] = len(proposer_gen_batch.batch)
                        step_metrics["self_play/extracted_problems"] = len(solver_batch.batch)
                        step_metrics.update(sampling_metrics)
                        step_metrics.update(solver_batch.meta_info.get("metrics", {}))
                    mini_epoch = 0
                    while mini_epoch < self.sp_config.mini_epochs:

                        mini_epoch += 1

                        if mini_epoch == 1:
                            proposer_gen_batch_copy = deepcopy(proposer_gen_batch)
                            solver_batch_copy = deepcopy(solver_batch)
                        else:
                            proposer_gen_batch = deepcopy(proposer_gen_batch_copy)
                            solver_batch = deepcopy(solver_batch_copy)

                        with marked_timer("solver_generation", timing_raw):
                            # 对Solver 进行Rollout轨迹生成
                            solver_gen_batch = self._generate_solver_trajectories(solver_batch, timing_raw)
                            step_metrics["self_play/solver_samples"] = len(solver_gen_batch.batch)

                        with marked_timer("reward_calculation", timing_raw):
                            # 计算SSP的奖励
                            proposer_rewards, solver_rewards, solver_extra_infos = self._calculate_self_play_rewards(
                                proposer_gen_batch, solver_gen_batch, timing_raw
                            )
                            step_metrics.update(getattr(self, "current_solver_process_reward_metrics", {}))
                            step_metrics.update(getattr(self, "current_solver_reward_extra_metrics", {}))
                            step_metrics.update(getattr(self, "current_proposer_reward_metrics", {}))

                        # Ensure batch is unlocked before modification (may be locked after Ray transfer)
                        proposer_gen_batch.ensure_unlocked()
                        solver_gen_batch.ensure_unlocked()
                        proposer_gen_batch.batch["token_level_scores"] = proposer_rewards
                        solver_gen_batch.batch["token_level_scores"] = solver_rewards
                        self._attach_reward_extra_infos(solver_gen_batch, solver_extra_infos)
                        self._dump_solver_success_questions(
                            solver_gen_batch=solver_gen_batch,
                            solver_rewards=solver_rewards,
                            solver_extra_infos=solver_extra_infos,
                            step=self.global_steps,
                        )
                        # 可选- 动态采样:过滤零方差的Reward的组
                        if reward_dynamic_sampling_enabled:
                            filtered_proposer_batch, filtered_solver_batch = self._apply_reward_dynamic_sampling(
                                proposer_gen_batch, solver_gen_batch, timing_raw
                            )

                            unique_prompts = set(filtered_proposer_batch.non_tensor_batch["uid"])
                            num_prompt_in_batch += len(unique_prompts)
                            num_gen_batches += 1

                            if last_step_proposer_batch is not None and accumulated_proposer_batch is None:
                                saved_unique_prompts = set(last_step_proposer_batch.non_tensor_batch["uid"])
                                num_prompt_in_batch += len(saved_unique_prompts)
                                print(f"Added {len(saved_unique_prompts)} prompts from saved data")

                            if accumulated_proposer_batch is None:
                                if last_step_proposer_batch is not None:
                                    print(
                                        f"Using {len(last_step_proposer_batch.batch)} saved proposer trajectories from last step"
                                    )
                                    accumulated_proposer_batch = DataProto.concat(
                                        [last_step_proposer_batch, filtered_proposer_batch]
                                    )
                                else:
                                    accumulated_proposer_batch = filtered_proposer_batch

                                if last_step_solver_batch is not None:
                                    print(
                                        f"Using {len(last_step_solver_batch.batch)} saved solver trajectories from last step"
                                    )
                                    accumulated_solver_batch = DataProto.concat(
                                        [last_step_solver_batch, filtered_solver_batch]
                                    )
                                else:
                                    accumulated_solver_batch = filtered_solver_batch
                            else:
                                accumulated_proposer_batch = DataProto.concat(
                                    [accumulated_proposer_batch, filtered_proposer_batch]
                                )
                                accumulated_solver_batch = DataProto.concat(
                                    [accumulated_solver_batch, filtered_solver_batch]
                                )

                            print(
                                f"Reward filtering: {len(unique_prompts)} prompts kept, total: {num_prompt_in_batch}/{target_prompt_batch_size}"
                            )

                            # Check if we have enough prompts
                            if num_prompt_in_batch < target_prompt_batch_size:
                                print(f"Need more prompts: {num_prompt_in_batch} < {target_prompt_batch_size}")
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"Batch {num_gen_batches}/{max_num_gen_batches}. Continue generating...")
                                    progress_bar.update(1)
                                    self.global_steps += 1
                                    continue
                                else:
                                    print(
                                        f"Max batches reached ({num_gen_batches}/{max_num_gen_batches}). Using accumulated data."
                                    )

                            # Use accumulated batches for training
                            proposer_gen_batch = accumulated_proposer_batch
                            solver_gen_batch = accumulated_solver_batch

                            # Truncate to target size if we have more than needed
                            proposer_n = self.sp_config.get("proposer", {}).get(
                                "n", self.config.actor_rollout_ref.rollout.n
                            )
                            target_traj_size = target_prompt_batch_size * proposer_n

                            if len(proposer_gen_batch.batch) > target_traj_size:
                                print(
                                    f"Truncating proposer batch from {len(proposer_gen_batch.batch)} to {target_traj_size}"
                                )
                                last_step_proposer_batch = proposer_gen_batch[target_traj_size:]
                                proposer_gen_batch = proposer_gen_batch[:target_traj_size]
                                print(
                                    f"Saved {len(last_step_proposer_batch.batch)} proposer trajectories for next round"
                                )
                            else:
                                last_step_proposer_batch = None

                            solver_n = self.config.actor_rollout_ref.rollout.n
                            target_solver_size = target_prompt_batch_size * proposer_n * solver_n

                            if len(solver_gen_batch.batch) > target_solver_size:
                                print(
                                    f"Truncating solver batch from {len(solver_gen_batch.batch)} to {target_solver_size}"
                                )
                                last_step_solver_batch = solver_gen_batch[target_solver_size:]
                                solver_gen_batch = solver_gen_batch[:target_solver_size]
                                print(f"Saved {len(last_step_solver_batch.batch)} solver trajectories for next round")
                            else:
                                last_step_solver_batch = None

                            print(
                                f"Final batch sizes: proposer={len(proposer_gen_batch.batch)}, solver={len(solver_gen_batch.batch)}"
                            )

                            step_metrics["self_play/reward_dynamic_sampling/batches_used"] = num_gen_batches
                            step_metrics["self_play/reward_dynamic_sampling/final_prompt_count"] = num_prompt_in_batch
                            step_metrics["self_play/reward_dynamic_sampling/target_prompt_count"] = (
                                target_prompt_batch_size
                            )

                            step_metrics["self_play/reward_dynamic_sampling/final_proposer_trajectories"] = len(
                                proposer_gen_batch.batch
                            )
                            step_metrics["self_play/reward_dynamic_sampling/final_solver_trajectories"] = len(
                                solver_gen_batch.batch
                            )

                            if last_step_proposer_batch is not None:
                                step_metrics["self_play/reward_dynamic_sampling/saved_proposer_trajectories"] = len(
                                    last_step_proposer_batch.batch
                                )
                            if last_step_solver_batch is not None:
                                step_metrics["self_play/reward_dynamic_sampling/saved_solver_trajectories"] = len(
                                    last_step_solver_batch.batch
                                )

                        if self.sp_config.save_freq > 0 and (
                            self.global_steps % self.sp_config.save_freq == 0
                            or self.sp_config.reward_dynamic_sampling.enable
                        ):
                            try:
                                inputs = self.tokenizer.batch_decode(
                                    proposer_gen_batch.batch["prompts"], skip_special_tokens=True
                                )
                                outputs = self.tokenizer.batch_decode(
                                    proposer_gen_batch.batch["responses"], skip_special_tokens=True
                                )

                                reward_extra_infos_dict = {}
                                if (
                                    hasattr(proposer_gen_batch, "non_tensor_batch")
                                    and proposer_gen_batch.non_tensor_batch
                                ):
                                    for key in ["data_source", "reward_model", "extra_info", "uid", "messages"]:
                                        if key in proposer_gen_batch.non_tensor_batch:
                                            reward_extra_infos_dict[key] = proposer_gen_batch.non_tensor_batch[
                                                key
                                            ].tolist()

                                self._dump_trajectories(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=proposer_gen_batch.batch["token_level_scores"].sum(-1).cpu().tolist(),
                                    role="proposer",
                                    step=self.global_steps,
                                    reward_extra_infos_dict=(
                                        reward_extra_infos_dict if reward_extra_infos_dict else None
                                    ),
                                )
                            except Exception as e:
                                logger.warning(f"Failed to dump proposer trajectories: {e}")

                            try:
                                inputs = self.tokenizer.batch_decode(
                                    solver_gen_batch.batch["prompts"], skip_special_tokens=True
                                )
                                outputs = self.tokenizer.batch_decode(
                                    solver_gen_batch.batch["responses"], skip_special_tokens=True
                                )

                                reward_extra_infos_dict = {}
                                if hasattr(solver_gen_batch, "non_tensor_batch") and solver_gen_batch.non_tensor_batch:
                                    for key in solver_gen_batch.non_tensor_batch.keys():
                                        if key in solver_gen_batch.non_tensor_batch:
                                            value = solver_gen_batch.non_tensor_batch[key]
                                            try:
                                                field_len = len(value)
                                            except TypeError:
                                                continue
                                            if field_len == len(solver_gen_batch.batch):
                                                reward_extra_infos_dict[key] = value.tolist()

                                self._dump_trajectories(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=solver_gen_batch.batch["token_level_scores"].sum(-1).cpu().tolist(),
                                    role="solver",
                                    step=self.global_steps,
                                    reward_extra_infos_dict=(
                                        reward_extra_infos_dict if reward_extra_infos_dict else None
                                    ),
                                )
                            except Exception as e:
                                logger.warning(f"Failed to dump proposer trajectories: {e}")

                        # Both roles were sampled from the same behavior policy. Freeze their
                        # old log-probabilities before either update mutates the shared actor.
                        solver_update_batch = solver_gen_batch
                        if (
                            self.sp_config.solver.enable
                            and solver_update_batch is not None
                            and (
                                self.global_steps > self.sp_config.proposer.warm_up_steps
                                or not self.sp_config.proposer.enable
                            )
                        ):
                            # Reward calculation and artifact dumps need the full dense
                            # [problem, rollout] grid. Remove dummy padding only at the policy
                            # boundary, and only as complete GRPO groups.
                            solver_update_batch, update_filter_metrics = self._prepare_solver_update_batch(
                                solver_update_batch
                            )
                            step_metrics.update(
                                {f"solver/{key}": value for key, value in update_filter_metrics.items()}
                            )

                        if self.sp_config.proposer.enable and proposer_gen_batch is not None:
                            proposer_gen_batch = self._capture_role_log_prob(
                                proposer_gen_batch, "proposer", timing_raw
                            )

                        if (
                            self.sp_config.solver.enable
                            and solver_update_batch is not None
                            and (
                                self.global_steps > self.sp_config.proposer.warm_up_steps
                                or not self.sp_config.proposer.enable
                            )
                        ):
                            solver_update_batch = self._capture_role_log_prob(
                                solver_update_batch, "solver", timing_raw
                            )

                        model_updates_start = time.perf_counter()
                        representative_batch = (
                            solver_update_batch
                            if solver_update_batch is not None
                            else proposer_gen_batch
                            if proposer_gen_batch is not None
                            else solver_gen_batch
                        )

                        combine_updates = getattr(self.sp_config, "combine_update", False)
                        # 分别对 Proposer 轨迹和 Solver 轨迹做 PPO/GRPO 更新
                        if combine_updates:
                            batches_to_combine = []
                            if self.sp_config.proposer.enable and proposer_gen_batch is not None:
                                batches_to_combine.append(proposer_gen_batch)
                            if (
                                self.sp_config.solver.enable
                                and solver_update_batch is not None
                                and (
                                    self.global_steps > self.sp_config.proposer.warm_up_steps
                                    or not self.sp_config.proposer.enable
                                )
                            ):
                                batches_to_combine.append(solver_update_batch)

                            if len(batches_to_combine) > 1:
                                for batch in batches_to_combine:
                                    for key in [
                                        "extracted_question",
                                        "formatted_prompt",
                                        "problem_type",
                                        "trajectory_index",
                                        "success_rate",
                                        "generation_step",
                                    ]:
                                        if key in batch.non_tensor_batch:
                                            del batch.non_tensor_batch[key]

                                combined_batch = DataProto.concat(batches_to_combine)

                                combined_metrics = self._update_on_trajectories(combined_batch, "combined", timing_raw)
                                step_metrics.update({f"combined/{k}": v for k, v in combined_metrics.items()})
                                representative_batch = combined_batch
                            else:
                                if self.sp_config.proposer.enable and proposer_gen_batch is not None:
                                    proposer_metrics = self._update_on_trajectories(
                                        proposer_gen_batch, "proposer", timing_raw
                                    )
                                    step_metrics.update({f"proposer/{k}": v for k, v in proposer_metrics.items()})
                                    representative_batch = proposer_gen_batch

                                if (
                                    self.sp_config.solver.enable
                                    and solver_update_batch is not None
                                    and (
                                        self.global_steps > self.sp_config.proposer.warm_up_steps
                                        or not self.sp_config.proposer.enable
                                    )
                                ):
                                    solver_metrics = self._update_on_trajectories(
                                        solver_update_batch, "solver", timing_raw
                                    )
                                    step_metrics.update({f"solver/{k}": v for k, v in solver_metrics.items()})
                                    representative_batch = solver_update_batch

                        else:
                            if self.sp_config.proposer.enable:
                                proposer_metrics = self._update_on_trajectories(
                                    proposer_gen_batch, "proposer", timing_raw
                                )
                                step_metrics.update({f"proposer/{k}": v for k, v in proposer_metrics.items()})
                                representative_batch = proposer_gen_batch

                            if self.sp_config.solver.enable and solver_update_batch is not None and (
                                self.global_steps > self.sp_config.proposer.warm_up_steps
                                or not self.sp_config.proposer.enable
                            ):
                                solver_metrics = self._update_on_trajectories(
                                    solver_update_batch, "solver", timing_raw
                                )
                                step_metrics.update({f"solver/{k}": v for k, v in solver_metrics.items()})
                                representative_batch = solver_update_batch

                        timing_raw["model_updates"] = timing_raw.get("model_updates", 0.0) + (
                            time.perf_counter() - model_updates_start
                        )

                    print(f"Representative batch size: {len(representative_batch.batch)}")
                    all_metrics.update(step_metrics)

                    ray.get(self.sp_data_manager.update_global_step.remote(self.global_steps))

                    all_metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )

                    sp_stats = ray.get(self.sp_data_manager.get_statistics.remote())
                    for key, value in sp_stats.items():
                        if isinstance(value, (int, float)):
                            all_metrics[f"self_play/stats/{key}"] = value

                    if hasattr(solver_gen_batch, "meta_info") and solver_gen_batch.meta_info:
                        solver_metrics = solver_gen_batch.meta_info.get("metrics", {})
                        for key, value in solver_metrics.items():
                            if key.startswith("dynamic_sampling/") and isinstance(value, (int, float, bool)):
                                all_metrics[f"self_play/{key}"] = value
                            if key.startswith("reward_dynamic_sampling/") and isinstance(value, (int, float, bool)):
                                all_metrics[f"self_play/{key}"] = value

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                            or self.sp_config.reward_dynamic_sampling.enable
                        )
                    ):
                        with marked_timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        all_metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                        or self.sp_config.reward_dynamic_sampling.enable
                    ):

                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                    timing_raw["step_total"] = time.perf_counter() - step_total_start
                    # Keep the historical name for existing dashboards while making
                    # its value represent the complete step rather than proposer-only time.
                    timing_raw["self_play_step"] = timing_raw["step_total"]

                    steps_duration = timing_raw["step_total"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    print(f"[sp_ray_trainer] compute data metrics")
                    all_metrics.update(
                        quarl_compute_data_metrics(
                            tokenizer=self.tokenizer, batch=representative_batch, use_critic=self.use_critic
                        )
                    )

                    # Attempt counts are dimensionless, so keep them out of the
                    # timing_s/* namespace produced by compute_timing_metrics.
                    for role in ("proposer", "solver"):
                        attempts_key = f"{role}_rollout_attempts"
                        if attempts_key in timing_raw:
                            all_metrics[f"self_play/{attempts_key}"] = int(timing_raw.pop(attempts_key))

                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    adapted_timing_raw = timing_raw.copy()

                    if "step_total" in adapted_timing_raw and "step" not in adapted_timing_raw:
                        adapted_timing_raw["step"] = adapted_timing_raw["step_total"]

                    timer_mapping = {
                        "solver_generation": "gen",
                        "solver_ref": "ref",
                        "solver_values": "values",
                        "solver_adv": "adv",
                        "solver_update_critic": "update_critic",
                        "solver_update_actor": "update_actor",
                    }

                    for our_name, expected_name in timer_mapping.items():
                        if our_name in adapted_timing_raw and expected_name not in adapted_timing_raw:
                            adapted_timing_raw[expected_name] = adapted_timing_raw[our_name]

                    all_metrics.update(
                        compute_timing_metrics(batch=representative_batch, timing_raw=adapted_timing_raw)
                    )
                    all_metrics.update(
                        compute_throughout_metrics(
                            batch=representative_batch, timing_raw=adapted_timing_raw, n_gpus=n_gpus
                        )
                    )

                    # Log metrics
                    tracking_logger.log(data=all_metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if self.sp_config.extraction_failure.strategy == "reuse":
                        if self.global_steps % self.sp_config.extraction_failure.pool_clear_interval == 0:

                            ray.get(
                                self.sp_data_manager.clear_solving_pool.remote(
                                    keep_ratio=self.sp_config.extraction_failure.keep_ratio
                                )
                            )

                            print(f"Cleared solving pool")

                            has_existing = ray.get(self.sp_data_manager.has_existing_problems.remote())
                            print(f"Has existing problems: {has_existing}")

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

                    if reward_dynamic_sampling_enabled:
                        accumulated_proposer_batch = None
                        accumulated_solver_batch = None
                        num_prompt_in_batch = 0
                        num_gen_batches = 0

                except StopIteration:
                    break

    def _generate_proposer_with_dynamic_sampling(
        self, dataloader_iter, timing_raw: Dict
    ) -> Tuple[DataProto, Optional[DataProto], Dict[str, Any]]:
        dynamic_sampling_config = self.sp_config.get("dynamic_sampling", {})
        enable_dynamic_sampling = dynamic_sampling_config.get("enable", False)

        try:
            batch_dict = next(dataloader_iter)
        except StopIteration:
            return None, None, {}

        batch = DataProto.from_single_dict(batch_dict)

        if not enable_dynamic_sampling:
            # 1. Proposer Rollout 轨迹生成
            proposer_gen_batch = self._generate_proposer_trajectories(batch, timing_raw)
            # 2. 问题抽取
            solver_batch = self._extract_and_assemble_solver_data(proposer_gen_batch, timing_raw)
            return proposer_gen_batch, solver_batch, {}

        max_retry_attempts = dynamic_sampling_config.get("max_retry_attempts", 3)
        min_valid_ratio = dynamic_sampling_config.get("min_valid_ratio", 1.0)
        target_batch_size = len(batch.batch)

        print(
            f"=== Dynamic Sampling Enabled: max_batches={max_retry_attempts + 1}, min_valid_ratio={min_valid_ratio}, target_size={target_batch_size} ==="
        )

        all_valid_problems = []
        all_trajectories = []
        all_proposer_batches = []
        sampling_metrics = {
            "dynamic_sampling/batches_used": 0,
            "dynamic_sampling/total_trajectories": 0,
            "dynamic_sampling/total_valid_problems": 0,
            "dynamic_sampling/final_valid_count": 0,
        }
        
        # Accumulate extraction stats across all dynamic sampling batches
        accumulated_extraction_stats = {
            "trajectories_count": 0,
            "answer_matches_count": 0,
            "valid_questions_count": 0,
            "successful_problems_count": 0,
            "format_error_count": 0,
            "pass_count": 0,
            "failure_reasons": defaultdict(int),
            "trajectory_outcomes": [],
        }

        current_batch = batch
        batch_count = 0

        while batch_count <= max_retry_attempts:
            batch_count += 1
            sampling_metrics["dynamic_sampling/batches_used"] = batch_count

            print(f"Dynamic sampling batch {batch_count}/{max_retry_attempts + 1}")

            proposer_gen_batch = self._generate_proposer_trajectories(current_batch, timing_raw)

            trajectories = self._extract_trajectories_from_batch(proposer_gen_batch)
            trajectory_offset = len(all_trajectories)
            all_trajectories.extend(trajectories)
            # Don't record stats yet - we'll accumulate across all batches
            extracted_problems, batch_extraction_stats = self._extract_and_process_problems(trajectories, record_stats=False)
            
            # Accumulate extraction stats
            accumulated_extraction_stats["trajectories_count"] += batch_extraction_stats["trajectories_count"]
            accumulated_extraction_stats["answer_matches_count"] += batch_extraction_stats["answer_matches_count"]
            accumulated_extraction_stats["valid_questions_count"] += batch_extraction_stats["valid_questions_count"]
            accumulated_extraction_stats["successful_problems_count"] += batch_extraction_stats["successful_problems_count"]
            accumulated_extraction_stats["format_error_count"] += batch_extraction_stats["format_error_count"]
            accumulated_extraction_stats["pass_count"] += batch_extraction_stats["pass_count"]
            for reason, count in batch_extraction_stats.get("failure_reasons", {}).items():
                accumulated_extraction_stats["failure_reasons"][reason] += count
            for outcome in batch_extraction_stats.get("trajectory_outcomes", []):
                global_outcome = dict(outcome)
                global_outcome["trajectory_index"] = trajectory_offset + int(
                    outcome.get("trajectory_index", -1)
                )
                accumulated_extraction_stats["trajectory_outcomes"].append(global_outcome)

            valid_problems_this_batch = len(extracted_problems)
            total_trajectories_this_batch = len(trajectories)

            sampling_metrics["dynamic_sampling/total_trajectories"] += total_trajectories_this_batch
            sampling_metrics["dynamic_sampling/total_valid_problems"] += valid_problems_this_batch

            print(f"Batch {batch_count}: {valid_problems_this_batch}/{total_trajectories_this_batch} valid problems")

            if extracted_problems:
                valid_trajectory_indices = [problem["trajectory_index"] for problem in extracted_problems]
                valid_proposer_batch = self._filter_proposer_batch_by_indices(
                    proposer_gen_batch, valid_trajectory_indices
                )
                all_proposer_batches.append(valid_proposer_batch)

            for problem in extracted_problems:
                global_problem = deepcopy(problem)
                global_problem["trajectory_index"] = trajectory_offset + int(problem.get("trajectory_index", -1))
                all_valid_problems.append(global_problem)

            current_valid_count = len(all_valid_problems)
            required_valid_count = int(target_batch_size * min_valid_ratio)

            if current_valid_count >= required_valid_count:
                print(
                    f"✅ Dynamic sampling succeeded! Collected {current_valid_count} valid problems (required: {required_valid_count})"
                )
                break

            if batch_count <= max_retry_attempts:
                try:
                    batch_dict = next(dataloader_iter)
                    current_batch = DataProto.from_single_dict(batch_dict)
                    print(
                        f"❌ Need more valid problems: {current_valid_count}/{required_valid_count}, using next batch..."
                    )
                except StopIteration:
                    print(f"❌ No more batches available, collected {current_valid_count} valid problems")
                    break
            else:
                print(f"❌ Max batches reached, collected {current_valid_count} valid problems")

        if len(all_valid_problems) == 0 or len(all_proposer_batches) == 0:
            print("❌ No valid problems found from any batch")
            # Still record accumulated stats even if no valid problems
            ray.get(self.sp_data_manager.reset_current_step_stats.remote())
            ray.get(
                self.sp_data_manager.record_extraction_stats.remote(
                    trajectories_count=accumulated_extraction_stats["trajectories_count"],
                    answer_matches_count=accumulated_extraction_stats["answer_matches_count"],
                    valid_questions_count=accumulated_extraction_stats["valid_questions_count"],
                    format_error_count=accumulated_extraction_stats["format_error_count"],
                    successful_problems_count=accumulated_extraction_stats["successful_problems_count"],
                )
            )
            self._save_extraction_stats(
                accumulated_extraction_stats,
                accumulated_extraction_stats["trajectories_count"],
                0,
            )
            self._save_extraction_by_template(all_trajectories, [], self.global_steps)
            self._dump_proposer_success_questions([], all_trajectories, self.global_steps)
            self._dump_proposer_fail_questions(
                all_trajectories, accumulated_extraction_stats["trajectory_outcomes"], self.global_steps
            )
            self._dump_rag_filter_artifacts(
                all_trajectories, accumulated_extraction_stats["trajectory_outcomes"], self.global_steps
            )
            return None, None, sampling_metrics

        # Record accumulated extraction stats for all dynamic sampling batches
        ray.get(self.sp_data_manager.reset_current_step_stats.remote())
        ray.get(
            self.sp_data_manager.record_extraction_stats.remote(
                trajectories_count=accumulated_extraction_stats["trajectories_count"],
                answer_matches_count=accumulated_extraction_stats["answer_matches_count"],
                valid_questions_count=accumulated_extraction_stats["valid_questions_count"],
                format_error_count=accumulated_extraction_stats["format_error_count"],
                successful_problems_count=accumulated_extraction_stats["successful_problems_count"],
            )
        )
        print(f"📊 Accumulated extraction stats: {accumulated_extraction_stats}")

        self._save_extraction_stats(
            accumulated_extraction_stats,
            accumulated_extraction_stats["trajectories_count"],
            len(all_valid_problems),
        )
        sampling_metrics.update(self._build_template_extraction_metrics(all_valid_problems))
        self._save_extraction_by_template(all_trajectories, all_valid_problems, self.global_steps)
        self._dump_proposer_success_questions(all_valid_problems, all_trajectories, self.global_steps)
        self._dump_proposer_fail_questions(
            all_trajectories, accumulated_extraction_stats["trajectory_outcomes"], self.global_steps
        )
        self._dump_rag_filter_artifacts(
            all_trajectories, accumulated_extraction_stats["trajectory_outcomes"], self.global_steps
        )

        final_proposer_batch = self._combine_proposer_batches(all_proposer_batches)

        final_batch_size = len(all_valid_problems)
        aligned_problems = self._align_problems_with_proposer_batch(all_valid_problems, final_batch_size)

        replicated_count = 0
        if final_batch_size < target_batch_size:
            print(f"🔄 Replicating {final_batch_size} valid problems to reach target size {target_batch_size}")

            needed_count = target_batch_size - final_batch_size
            replicated_count = needed_count

            import random

            replicated_problems = []
            replicated_proposer_trajectories = []

            for i in range(needed_count):
                source_idx = random.randint(0, final_batch_size - 1)

                replicated_problem = deepcopy(aligned_problems[source_idx])
                replicated_problem["trajectory_index"] = final_batch_size + i
                replicated_problems.append(replicated_problem)

                source_proposer_data = final_proposer_batch[source_idx : source_idx + 1]
                replicated_proposer_trajectories.append(source_proposer_data)

            aligned_problems.extend(replicated_problems)

            for replicated_traj in replicated_proposer_trajectories:
                final_proposer_batch = DataProto.concat([final_proposer_batch, replicated_traj])

            print(f"🔄 Replicated {replicated_count} problems, now have {len(aligned_problems)} total")

        else:
            if final_batch_size > target_batch_size:
                print(f"✂️ Truncating {final_batch_size} problems to target size {target_batch_size}")
                aligned_problems = aligned_problems[:target_batch_size]
                final_proposer_batch = final_proposer_batch[:target_batch_size]
                print(f"✂️ Truncated to {len(aligned_problems)} problems")
        self.generate_problem = aligned_problems

        solver_batch = self._prepare_solving_batch_from_data(aligned_problems)

        sampling_metrics["dynamic_sampling/original_valid_count"] = final_batch_size
        sampling_metrics["dynamic_sampling/final_valid_count"] = len(aligned_problems)
        sampling_metrics["dynamic_sampling/replicated_count"] = replicated_count

        if solver_batch is not None:
            solver_batch.meta_info.setdefault("metrics", {}).update(sampling_metrics)

        return final_proposer_batch, solver_batch, sampling_metrics

    def _apply_reward_dynamic_sampling(
        self, proposer_gen_batch: DataProto, solver_gen_batch: DataProto, timing_raw: Dict
    ) -> Tuple[DataProto, DataProto]:
        print("=== Applying Reward-based Dynamic Sampling ===")

        reward_config = self.sp_config.get("reward_dynamic_sampling", {})
        metric_name = reward_config.get("metric", "seq_final_reward")

        if metric_name == "seq_final_reward":
            metric_values = proposer_gen_batch.batch["token_level_scores"].sum(dim=-1).numpy()
        elif metric_name == "seq_reward":
            metric_values = proposer_gen_batch.batch["token_level_scores"].sum(dim=-1).numpy()
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        proposer_gen_batch.non_tensor_batch[metric_name] = metric_values

        uid2metric_vals = defaultdict(list)
        uid2traj_indices = defaultdict(list)

        for idx, (uid, metric_val) in enumerate(
            zip(proposer_gen_batch.non_tensor_batch["uid"], proposer_gen_batch.non_tensor_batch[metric_name])
        ):
            uid2metric_vals[uid].append(metric_val)
            uid2traj_indices[uid].append(idx)

        uid2variance = {}
        for uid, metric_vals in uid2metric_vals.items():
            uid2variance[uid] = np.std(metric_vals)

        kept_uids = [uid for uid, variance in uid2variance.items() if variance > 0 or len(uid2metric_vals[uid]) == 1]

        kept_traj_indices = []
        for uid in kept_uids:
            kept_traj_indices.extend(uid2traj_indices[uid])

        kept_traj_indices.sort()

        print(
            f"Reward filtering: {len(uid2metric_vals)} groups, {len(kept_uids)} kept, {len(kept_traj_indices)}/{len(proposer_gen_batch.batch)} trajectories"
        )

        filtered_proposer_batch = proposer_gen_batch[kept_traj_indices]

        solver_n = self.config.actor_rollout_ref.rollout.n
        kept_solver_indices = []

        for proposer_idx in kept_traj_indices:
            solver_start_idx = proposer_idx * solver_n
            solver_end_idx = solver_start_idx + solver_n
            kept_solver_indices.extend(range(solver_start_idx, solver_end_idx))

        filtered_solver_batch = solver_gen_batch[kept_solver_indices]

        print(f"Solver filtering: {len(kept_solver_indices)}/{len(solver_gen_batch.batch)} trajectories kept")

        filtering_metrics = {
            "reward_dynamic_sampling/original_proposer_groups": len(uid2metric_vals),
            "reward_dynamic_sampling/kept_proposer_groups": len(kept_uids),
            "reward_dynamic_sampling/original_proposer_trajectories": len(proposer_gen_batch.batch),
            "reward_dynamic_sampling/kept_proposer_trajectories": len(filtered_proposer_batch.batch),
            "reward_dynamic_sampling/original_solver_trajectories": len(solver_gen_batch.batch),
            "reward_dynamic_sampling/kept_solver_trajectories": len(filtered_solver_batch.batch),
            "reward_dynamic_sampling/filtering_ratio": (
                len(kept_traj_indices) / len(proposer_gen_batch.batch) if len(proposer_gen_batch.batch) > 0 else 0.0
            ),
        }

        pprint(filtering_metrics)

        return filtered_proposer_batch, filtered_solver_batch

    def _combine_proposer_batches(self, proposer_batches: List[DataProto]) -> DataProto:
        if len(proposer_batches) == 1:
            return proposer_batches[0]

        combined_batch = proposer_batches[0]
        for batch in proposer_batches[1:]:
            combined_batch = DataProto.concat([combined_batch, batch])

        return combined_batch

    def _filter_proposer_batch_by_indices(self, proposer_batch: DataProto, valid_indices: List[int]) -> DataProto:
        if not valid_indices:
            return None

        filtered_batch = proposer_batch[valid_indices]

        return filtered_batch

    def _align_problems_with_proposer_batch(
        self, all_valid_problems: List[Dict[str, Any]], target_size: int
    ) -> List[Dict[str, Any]]:
        aligned_problems = []

        for i, problem in enumerate(all_valid_problems[:target_size]):
            aligned_problem = problem.copy()
            aligned_problem["trajectory_index"] = i
            aligned_problems.append(aligned_problem)

        return aligned_problems
    
    def _generate_proposer_trajectories(self, batch: DataProto, timing_raw: Dict) -> DataProto:
        """
        Proposer 轨迹生成
        """
        print(f"=== Step 1: Proposer Generation at step {self.global_steps} ===")

        ray.get(self.sp_data_manager.switch_to_phase.remote(SelfPlayPhase.PROBLEM_GENERATION))

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "index" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("index")
        if "agent_name" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch.meta_info["phase"] = "proposer"
        gen_batch.meta_info["do_sample"] = self.sp_config.get("proposer", {}).get("do_sample", True)
        gen_batch.meta_info["temperature"] = self.sp_config.get("proposer", {}).get("temperature", 0.8)

        self.rollout_n = self.config.actor_rollout_ref.rollout.n
        proposer_n = self.sp_config.get("proposer", {}).get("n", self.rollout_n)

        if SELF_PLAY_DEBUG:
            logger.debug(f"Proposer gen_batch size before repeat: {len(gen_batch)}, proposer_n: {proposer_n}")

        gen_batch = gen_batch.repeat(repeat_times=proposer_n, interleave=True)

        if SELF_PLAY_DEBUG:
            logger.debug(f"Proposer gen_batch size after repeat: {len(gen_batch)}")

        if not self.async_rollout_mode:
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

        rollout_timing = gen_batch_output.meta_info.pop("timing", {})
        accumulate_rollout_timing(timing_raw, "proposer", rollout_timing)

        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        batch = batch.repeat(repeat_times=proposer_n, interleave=True)

        batch = batch.union(gen_batch_output)

        # Ensure batch is unlocked before modification
        batch.ensure_unlocked()
        for key in batch.batch.keys():
            if key not in ["old_log_probs", "ref_log_prob"]:
                batch.batch[key] = batch.batch[key].long()

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        print(f"Generated {len(batch.batch)} proposer trajectories")

        return batch

    def _extract_and_assemble_solver_data(self, proposer_gen_batch: DataProto, timing_raw: Dict) -> Optional[DataProto]:
        """
        问题抽取和组装
        """
        print(f"=== Step 2: Problem Extraction and Assembly at step {self.global_steps} ===")
        
        trajectories = self._extract_trajectories_from_batch(proposer_gen_batch)
        batch_size = len(trajectories)

        extracted_problems = self._extract_and_process_problems(trajectories)

        extraction_success_mask = [False] * batch_size

        problem_to_trajectory_map = {}
        for problem in extracted_problems:
            traj_idx = problem.get("trajectory_index", -1)
            if 0 <= traj_idx < batch_size:
                extraction_success_mask[traj_idx] = True
                problem_to_trajectory_map[traj_idx] = problem

        solving_data = []
        failed_extraction_count = 0
        reused_problem_count = 0
        metrics = {}

        for i in range(batch_size):
            if i in problem_to_trajectory_map:
                solving_data.append(problem_to_trajectory_map[i])
            else:
                failed_extraction_count += 1
                fallback_problem, reused = self._create_fallback_problem_for_failed_extraction(trajectories[i], i)
                solving_data.append(fallback_problem)
                if reused:
                    reused_problem_count += 1

        print(
            f"Failed extractions: {failed_extraction_count} (reused {reused_problem_count} existing problems, created {failed_extraction_count - reused_problem_count} dummy problems)"
        )
        metrics["self_play/reused_problem_count"] = reused_problem_count
        metrics["self_play/dummy_problem_count"] = failed_extraction_count - reused_problem_count

        print(f"Extracted {len(extracted_problems)} valid problems, padded to {len(solving_data)} total")
        print(
            f"Extraction success rate: {sum(extraction_success_mask)}/{batch_size} = {sum(extraction_success_mask) / batch_size:.2%}"
        )
        metrics["self_play/extraction_success_rate"] = sum(extraction_success_mask) / batch_size
        metrics.update(self._build_template_extraction_metrics(extracted_problems))
        self._save_extraction_by_template(trajectories, extracted_problems, self.global_steps)
        self._dump_proposer_success_questions(extracted_problems, trajectories, self.global_steps)
        self._dump_proposer_fail_questions(
            trajectories, getattr(self, "current_extraction_outcomes", []), self.global_steps
        )
        self._dump_rag_filter_artifacts(
            trajectories, getattr(self, "current_extraction_outcomes", []), self.global_steps
        )

        self.current_extraction_success_mask = extraction_success_mask

        solver_batch = self._prepare_solving_batch_from_data(solving_data)

        self.generate_problem = solving_data

        solver_batch.meta_info.setdefault("metrics", {}).update(metrics)

        return solver_batch

    def _generate_solver_trajectories(self, solver_batch: DataProto, timing_raw: Dict) -> DataProto:
        print(f"=== Step 3: Solver Generation at step {self.global_steps} ===")

        if SELF_PLAY_DEBUG:
            logger.debug(f"Input solver_batch size: {len(solver_batch)}")

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "index" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("index")
        if "agent_name" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")

        gen_batch = solver_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch.meta_info["phase"] = "solver"
        gen_batch.meta_info["do_sample"] = self.sp_config.get("solver", {}).get("do_sample", True)
        gen_batch.meta_info["temperature"] = self.sp_config.get("solver", {}).get("temperature", 0.7)

        self.rollout_n = self.config.actor_rollout_ref.rollout.n

        if SELF_PLAY_DEBUG:
            logger.debug(f"gen_batch size before repeat: {len(gen_batch)}, rollout_n: {self.rollout_n}")

        gen_batch = gen_batch.repeat(repeat_times=self.rollout_n, interleave=True)

        if SELF_PLAY_DEBUG:
            logger.debug(f"gen_batch size after repeat: {len(gen_batch)}")

        if not self.async_rollout_mode:
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

        rollout_timing = gen_batch_output.meta_info.pop("timing", {})
        accumulate_rollout_timing(timing_raw, "solver", rollout_timing)

        solver_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(solver_batch.batch))], dtype=object
        )
        solver_batch = solver_batch.repeat(repeat_times=self.rollout_n, interleave=True)
        solver_batch = solver_batch.union(gen_batch_output)

        # Ensure batch is unlocked before modification
        solver_batch.ensure_unlocked()
        for key in solver_batch.batch.keys():
            if key not in ["old_log_probs", "ref_log_prob"]:
                solver_batch.batch[key] = solver_batch.batch[key].long()

        if "response_mask" not in solver_batch.batch.keys():
            solver_batch.batch["response_mask"] = compute_response_mask(solver_batch)

        print(f"Generated {len(solver_batch.batch)} solver trajectories")

        return solver_batch

    def _calculate_self_play_rewards(
        self, proposer_gen_batch: DataProto, solver_gen_batch: DataProto, timing_raw: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[Any]]]:
        print(f"=== Step 4: Reward Calculation at step {self.global_steps} ===")

        with marked_timer("solver_reward_computation", timing_raw, color="yellow"):
            if self.config.reward_model.launch_reward_fn_async:
                future_solver_reward = compute_reward_async.remote(data=solver_gen_batch, reward_fn=self.reward_fn)
                solver_rewards, solver_extra_infos = ray.get(future_solver_reward)
            else:
                solver_rewards, solver_extra_infos = compute_reward(solver_gen_batch, self.reward_fn)
            self.current_solver_process_reward_metrics = self._extract_solver_process_reward_metrics(
                solver_rewards, solver_extra_infos
            )
            self.current_solver_reward_extra_metrics = self._summarize_solver_reward_extra_infos(solver_extra_infos)

        with marked_timer("proposer_reward_computation", timing_raw, color="orange"):
            proposer_rewards = self._compute_proposer_rewards(proposer_gen_batch, solver_rewards, solver_extra_infos)

        print(f"Computed rewards: proposer_mean={proposer_rewards.mean():.3f}, solver_mean={solver_rewards.mean():.3f}")

        return proposer_rewards, solver_rewards, solver_extra_infos

    def _attach_reward_extra_infos(self, batch: DataProto, reward_extra_infos: Dict[str, List[Any]]) -> None:
        if not reward_extra_infos:
            return

        batch_size = len(batch.batch)
        for key, values in reward_extra_infos.items():
            if values is None or len(values) != batch_size:
                continue
            batch.non_tensor_batch[key] = np.array(values, dtype=object)

    def _build_template_extraction_metrics(self, valid_problems: List[Dict[str, Any]]) -> Dict[str, float]:
        counts = {template_id: 0 for template_id in self.KGQA_TEMPLATE_IDS}

        for problem in valid_problems or []:
            extra_info = problem.get("extra_info") or {}
            template_id = str(extra_info.get("template_id") or "unknown")
            counts.setdefault(template_id, 0)
            counts[template_id] += 1

        return {
            f"self_play/extraction_by_template/{template_id}/extracted_count": float(count)
            for template_id, count in counts.items()
        }

    def _get_run_artifact_dir(self) -> str:
        trajectory_dir = os.environ.get("TRAJECTORY_DIR")
        if trajectory_dir:
            return os.path.dirname(os.path.abspath(trajectory_dir))

        stats_dir = os.environ.get("EXTRACTION_STATS_DIR")
        if stats_dir:
            return os.path.dirname(os.path.abspath(stats_dir))

        return os.path.abspath("./output")

    def _get_template_id_from_problem(self, problem: Dict[str, Any]) -> str:
        extra_info = problem.get("extra_info") or {}
        return str(extra_info.get("template_id") or "unknown")

    def _get_template_id_from_trajectory(self, trajectory: Dict[str, Any]) -> str:
        metadata = trajectory.get("metadata") or {}
        extra_info = metadata.get("extra_info") or {}
        return str(extra_info.get("template_id") or "unknown")

    def _build_extraction_by_template_summary(
        self, trajectories: List[Dict[str, Any]], valid_problems: List[Dict[str, Any]], step: int
    ) -> Dict[str, Any]:
        by_template = {
            template_id: {"total": 0, "extracted_success": 0, "success_rate": 0.0, "qids_success": []}
            for template_id in self.KGQA_TEMPLATE_IDS
        }

        for trajectory in trajectories or []:
            template_id = self._get_template_id_from_trajectory(trajectory)
            by_template.setdefault(
                template_id,
                {"total": 0, "extracted_success": 0, "success_rate": 0.0, "qids_success": []},
            )
            by_template[template_id]["total"] += 1

        for problem in valid_problems or []:
            template_id = self._get_template_id_from_problem(problem)
            by_template.setdefault(
                template_id,
                {"total": 0, "extracted_success": 0, "success_rate": 0.0, "qids_success": []},
            )
            by_template[template_id]["extracted_success"] += 1

            extra_info = problem.get("extra_info") or {}
            qid = extra_info.get("qid")
            if qid is not None:
                by_template[template_id]["qids_success"].append(qid)

        for bucket in by_template.values():
            total = bucket["total"]
            bucket["success_rate"] = bucket["extracted_success"] / total if total else 0.0

        return {
            "step": step,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_trajectories": len(trajectories or []),
            "final_valid_count": len(valid_problems or []),
            "by_template_id": by_template,
        }

    def _build_proposer_success_rows(
        self, valid_problems: List[Dict[str, Any]], trajectories: List[Dict[str, Any]], step: int
    ) -> List[Dict[str, Any]]:
        trajectory_by_index = {
            index: trajectory
            for index, trajectory in enumerate(trajectories or [])
        }

        rows = []
        for problem in valid_problems or []:
            extra_info = problem.get("extra_info") or {}
            trajectory_index = int(problem.get("trajectory_index", -1))
            trajectory = trajectory_by_index.get(trajectory_index, {})

            rows.append(
                {
                    "step": step,
                    "trajectory_index": trajectory_index,
                    "template_id": self._get_template_id_from_problem(problem),
                    "qid": extra_info.get("qid"),
                    "data_source": problem.get("data_source")
                    or (trajectory.get("metadata") or {}).get("data_source"),
                    "dataset_type": extra_info.get("dataset_type"),
                    "kgqa_backend": extra_info.get("kgqa_backend"),
                    "source_question": extra_info.get("source_question", ""),
                    "generated_question": problem.get("extracted_question") or extra_info.get("question", ""),
                    "gold_answers": extra_info.get("source_gold_answers", []),
                    "proposer_answers": extra_info.get("proposer_answers", []),
                    "question_entities": extra_info.get("question_entities", []),
                    "answer_entities": extra_info.get("answer_entities", []),
                    "path": extra_info.get("path", []),
                    "output": trajectory.get("output", ""),
                }
            )

        return rows

    def _save_extraction_by_template(
        self, trajectories: List[Dict[str, Any]], valid_problems: List[Dict[str, Any]], step: int
    ) -> None:
        try:
            stats_dir = os.environ.get("EXTRACTION_STATS_DIR", "./output/extraction_stats")
            os.makedirs(stats_dir, exist_ok=True)
            filepath = os.path.join(stats_dir, f"extraction_by_template_step_{step}.json")

            summary = self._build_extraction_by_template_summary(trajectories, valid_problems, step)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

            print(f"Saved extraction by template summary to: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save extraction by template summary for step {step}: {e}")

    def _dump_jsonl_rows(self, rows: List[Dict[str, Any]], filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _build_rag_filter_rows(
        self,
        trajectories: List[Dict[str, Any]],
        trajectory_outcomes: List[Dict[str, Any]],
        step: int,
    ) -> List[Dict[str, Any]]:
        """Build one auditable RAG-verifier row per proposer trajectory."""
        outcome_by_index = {
            int(outcome.get("trajectory_index", -1)): outcome
            for outcome in trajectory_outcomes or []
            if isinstance(outcome, dict)
        }
        skipped_statuses = {
            "no_pattern_match": "skipped_no_pattern_match",
            "format_rule_failed": "skipped_format_failure",
            "question_leaks_gold_answer": "skipped_question_leaks_gold",
            "no_source_gold_answers": "skipped_no_source_gold_answers",
        }

        rows: List[Dict[str, Any]] = []
        for trajectory_index, trajectory in enumerate(trajectories or []):
            metadata = trajectory.get("metadata") or {}
            extra_info = metadata.get("extra_info") or {}
            reward_model = metadata.get("reward_model") or {}
            ground_truth = reward_model.get("ground_truth") or {}
            outcome = outcome_by_index.get(trajectory_index, {})
            diagnostics = outcome.get("rag_diagnostics")
            diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
            failure_reason = outcome.get("failure_reason")
            if not diagnostics:
                diagnostics = {
                    "status": skipped_statuses.get(str(failure_reason), "skipped_before_rag"),
                    "rag_invoked": False,
                    "accepted": False,
                    "raw_answer": "",
                    "predicted_answers": [],
                    "hit_at_1": 0.0,
                    "triple_count": 0,
                    "duration_seconds": 0.0,
                    "error_type": None,
                    "error_message": None,
                }

            rows.append(
                {
                    "step": step,
                    "trajectory_index": trajectory_index,
                    "template_id": self._get_template_id_from_trajectory(trajectory),
                    "qid": extra_info.get("qid"),
                    "data_source": metadata.get("data_source"),
                    "dataset_type": extra_info.get("dataset_type"),
                    "kgqa_backend": extra_info.get("kgqa_backend"),
                    "synthesis_mode": extra_info.get("synthesis_mode"),
                    "status": diagnostics.get("status", "unknown"),
                    "rag_invoked": bool(diagnostics.get("rag_invoked", False)),
                    "accepted": bool(diagnostics.get("accepted", False)),
                    "fail_open": bool(diagnostics.get("fail_open", False)),
                    "failure_reason": failure_reason,
                    "model": diagnostics.get("model", ""),
                    "question": diagnostics.get("question", ""),
                    "question_entities": diagnostics.get(
                        "question_entities", extra_info.get("question_entities", [])
                    ),
                    "answer_entities": extra_info.get("answer_entities", []),
                    "path": extra_info.get("path", []),
                    "gold_answers": diagnostics.get("gold_answers")
                    or extra_info.get("source_gold_answers")
                    or ground_truth.get("target")
                    or [],
                    "triple_count": int(diagnostics.get("triple_count", 0) or 0),
                    "triples": diagnostics.get("triples", []),
                    "raw_answer": diagnostics.get("raw_answer", ""),
                    "predicted_answers": diagnostics.get("predicted_answers", []),
                    "hit_at_1": float(diagnostics.get("hit_at_1", 0.0) or 0.0),
                    "duration_seconds": float(diagnostics.get("duration_seconds", 0.0) or 0.0),
                    "error_type": diagnostics.get("error_type"),
                    "error_message": diagnostics.get("error_message"),
                }
            )
        return rows

    def _dump_rag_filter_artifacts(
        self,
        trajectories: List[Dict[str, Any]],
        trajectory_outcomes: List[Dict[str, Any]],
        step: int,
    ) -> None:
        """Write per-trajectory RAG details and a compact step summary."""
        try:
            artifact_dir = os.path.join(self._get_run_artifact_dir(), "rag_filter")
            rows = self._build_rag_filter_rows(trajectories, trajectory_outcomes, step)
            detail_path = os.path.join(artifact_dir, f"rag_filter_step_{step}.jsonl")
            self._dump_jsonl_rows(rows, detail_path)

            status_counts: Dict[str, int] = {}
            invoked_rows = []
            for row in rows:
                status = str(row.get("status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                if row.get("rag_invoked"):
                    invoked_rows.append(row)

            summary = {
                "step": step,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_trajectories": len(rows),
                "rag_invoked_count": len(invoked_rows),
                "accepted_count": sum(bool(row.get("accepted")) for row in rows),
                "api_error_count": sum(
                    count for status, count in status_counts.items() if status.startswith("api_error")
                ),
                "status_counts": dict(sorted(status_counts.items())),
                "mean_duration_seconds": (
                    sum(float(row.get("duration_seconds", 0.0)) for row in invoked_rows)
                    / len(invoked_rows)
                    if invoked_rows
                    else 0.0
                ),
            }
            os.makedirs(artifact_dir, exist_ok=True)
            summary_path = os.path.join(artifact_dir, f"rag_filter_summary_step_{step}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            print(
                f"Dumped {len(rows)} RAG filter records to {detail_path}; summary to {summary_path}"
            )
        except Exception as exc:
            logger.warning(
                "Failed to dump RAG filter artifacts for step %s: %s",
                step,
                exc,
                exc_info=SELF_PLAY_DEBUG,
            )

    def _dump_proposer_success_questions(
        self, valid_problems: List[Dict[str, Any]], trajectories: List[Dict[str, Any]], step: int
    ) -> None:
        try:
            success_dir = os.path.join(self._get_run_artifact_dir(), "successful_questions")
            filepath = os.path.join(success_dir, f"proposer_success_step_{step}.jsonl")
            rows = self._build_proposer_success_rows(valid_problems, trajectories, step)
            self._dump_jsonl_rows(rows, filepath)
            print(f"Dumped {len(rows)} proposer success questions to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to dump proposer success questions for step {step}: {e}")

    def _build_proposer_fail_rows(
        self,
        trajectories: List[Dict[str, Any]],
        trajectory_outcomes: List[Dict[str, Any]],
        step: int,
    ) -> List[Dict[str, Any]]:
        trajectory_by_index = {index: trajectory for index, trajectory in enumerate(trajectories or [])}
        rows = []
        for outcome in trajectory_outcomes or []:
            if not isinstance(outcome, dict) or not outcome.get("failure_reason"):
                continue
            trajectory_index = int(outcome.get("trajectory_index", -1))
            trajectory = trajectory_by_index.get(trajectory_index, {})
            metadata = trajectory.get("metadata") or {}
            extra_info = metadata.get("extra_info") or {}
            reward_model = metadata.get("reward_model") or {}
            ground_truth = reward_model.get("ground_truth") or {}
            rows.append(
                {
                    "step": step,
                    "trajectory_index": trajectory_index,
                    "template_id": self._get_template_id_from_trajectory(trajectory),
                    "qid": extra_info.get("qid"),
                    "data_source": metadata.get("data_source"),
                    "dataset_type": extra_info.get("dataset_type"),
                    "kgqa_backend": extra_info.get("kgqa_backend"),
                    "source_question": extra_info.get("source_question", ""),
                    "question_entities": extra_info.get("question_entities", []),
                    "answer_entities": extra_info.get("answer_entities", []),
                    "path": extra_info.get("path", []),
                    "gold_answers": extra_info.get("source_gold_answers")
                    or ground_truth.get("target")
                    or [],
                    "failure_reason": str(outcome["failure_reason"]),
                    "answer_matches": int(outcome.get("answer_matches", 0)),
                    "valid_questions": int(outcome.get("valid_questions", 0)),
                    "format_error": int(outcome.get("format_error", 0)),
                    "input": trajectory.get("input", ""),
                    "output": trajectory.get("output", ""),
                }
            )
        return rows

    def _dump_proposer_fail_questions(
        self,
        trajectories: List[Dict[str, Any]],
        trajectory_outcomes: List[Dict[str, Any]],
        step: int,
    ) -> None:
        try:
            fail_dir = os.path.join(self._get_run_artifact_dir(), "fail_questions")
            filepath = os.path.join(fail_dir, f"proposer_fail_step_{step}.jsonl")
            rows = self._build_proposer_fail_rows(trajectories, trajectory_outcomes, step)
            self._dump_jsonl_rows(rows, filepath)
            print(f"Dumped {len(rows)} proposer failed questions to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to dump proposer failed questions for step {step}: {e}")

    def _build_solver_success_rows(
        self,
        outputs: List[str],
        scores: List[float],
        raw_answer_corrects: List[bool],
        reward_eligibilities: List[bool],
        extra_infos: List[Any],
        data_sources: List[Any],
        step: int,
    ) -> List[Dict[str, Any]]:
        rows = []
        rollout_n = int(getattr(self, "rollout_n", 1) or 1)

        for index, raw_correct in enumerate(raw_answer_corrects):
            reward_eligible = bool(
                reward_eligibilities[index] if index < len(reward_eligibilities) else False
            )
            if not (raw_correct and reward_eligible):
                continue

            extra_info = extra_infos[index] if index < len(extra_infos) else {}
            if not isinstance(extra_info, dict):
                extra_info = {}
            data_source = data_sources[index] if index < len(data_sources) else None

            rows.append(
                {
                    "step": step,
                    "solver_trajectory_index": index,
                    "problem_index": index // rollout_n,
                    "template_id": str(extra_info.get("template_id") or "unknown"),
                    "qid": extra_info.get("qid"),
                    "data_source": data_source,
                    "dataset_type": extra_info.get("dataset_type"),
                    "kgqa_backend": extra_info.get("kgqa_backend"),
                    "question": extra_info.get("question", ""),
                    "gold_answers": extra_info.get("source_gold_answers") or [],
                    "question_entities": extra_info.get("question_entities", []),
                    "answer_entities": extra_info.get("answer_entities", []),
                    "path": extra_info.get("path", []),
                    "solver_score": scores[index] if index < len(scores) else None,
                    "answer_correct": True,
                    "raw_answer_correct": True,
                    "reward_eligible": True,
                    "rejected_correct": False,
                    "output": outputs[index] if index < len(outputs) else "",
                }
            )

        return rows

    def _build_solver_fail_rows(
        self,
        outputs: List[str],
        scores: List[float],
        raw_answer_corrects: List[bool],
        reward_eligibilities: List[bool],
        extra_infos: List[Any],
        data_sources: List[Any],
        step: int,
    ) -> List[Dict[str, Any]]:
        rows = []
        rollout_n = int(getattr(self, "rollout_n", 1) or 1)
        for index, raw_correct in enumerate(raw_answer_corrects):
            reward_eligible = bool(
                reward_eligibilities[index] if index < len(reward_eligibilities) else False
            )
            if raw_correct and reward_eligible:
                continue
            extra_info = extra_infos[index] if index < len(extra_infos) else {}
            if not isinstance(extra_info, dict):
                extra_info = {}
            data_source = data_sources[index] if index < len(data_sources) else None
            rejected_correct = bool(raw_correct and not reward_eligible)
            rows.append(
                {
                    "step": step,
                    "solver_trajectory_index": index,
                    "problem_index": index // rollout_n,
                    "template_id": str(extra_info.get("template_id") or "unknown"),
                    "qid": extra_info.get("qid"),
                    "data_source": data_source,
                    "dataset_type": extra_info.get("dataset_type"),
                    "kgqa_backend": extra_info.get("kgqa_backend"),
                    "is_dummy": str(data_source).strip().lower() == "dummy",
                    "question": extra_info.get("question", ""),
                    "gold_answers": extra_info.get("source_gold_answers") or [],
                    "question_entities": extra_info.get("question_entities", []),
                    "answer_entities": extra_info.get("answer_entities", []),
                    "path": extra_info.get("path", []),
                    "solver_score": scores[index] if index < len(scores) else None,
                    "answer_correct": False,
                    "raw_answer_correct": bool(raw_correct),
                    "reward_eligible": reward_eligible,
                    "rejected_correct": rejected_correct,
                    "failure_reason": "reward_rejected_correct" if rejected_correct else "answer_incorrect",
                    "output": outputs[index] if index < len(outputs) else "",
                }
            )
        return rows

    def _dump_solver_success_questions(
        self, solver_gen_batch: DataProto, solver_rewards: torch.Tensor, solver_extra_infos: Dict, step: int
    ) -> None:
        try:
            raw_answer_correct_tensor = self._extract_solver_answer_correct_tensor(
                solver_rewards, solver_extra_infos
            )
            reward_eligibility_tensor = self._extract_solver_reward_eligibility_tensor(
                solver_rewards, solver_extra_infos
            )
            raw_answer_corrects = raw_answer_correct_tensor.detach().cpu().tolist()
            reward_eligibilities = reward_eligibility_tensor.detach().cpu().tolist()
            scores = solver_rewards.sum(-1).detach().cpu().tolist()
            outputs = self.tokenizer.batch_decode(
                solver_gen_batch.batch["responses"], skip_special_tokens=True
            )

            extra_infos = []
            data_sources = []
            if hasattr(solver_gen_batch, "non_tensor_batch") and solver_gen_batch.non_tensor_batch is not None:
                raw_extra_infos = solver_gen_batch.non_tensor_batch.get("extra_info", [])
                extra_infos = list(raw_extra_infos)
                data_sources = list(solver_gen_batch.non_tensor_batch.get("data_source", []))

            rows = self._build_solver_success_rows(
                outputs=outputs,
                scores=scores,
                raw_answer_corrects=[bool(value) for value in raw_answer_corrects],
                reward_eligibilities=[bool(value) for value in reward_eligibilities],
                extra_infos=extra_infos,
                data_sources=data_sources,
                step=step,
            )

            success_dir = os.path.join(self._get_run_artifact_dir(), "successful_questions")
            filepath = os.path.join(success_dir, f"solver_success_step_{step}.jsonl")
            self._dump_jsonl_rows(rows, filepath)
            print(f"Dumped {len(rows)} solver success questions to {filepath}")

            fail_rows = self._build_solver_fail_rows(
                outputs=outputs,
                scores=scores,
                raw_answer_corrects=[bool(value) for value in raw_answer_corrects],
                reward_eligibilities=[bool(value) for value in reward_eligibilities],
                extra_infos=extra_infos,
                data_sources=data_sources,
                step=step,
            )
            fail_dir = os.path.join(self._get_run_artifact_dir(), "fail_questions")
            fail_filepath = os.path.join(fail_dir, f"solver_fail_step_{step}.jsonl")
            self._dump_jsonl_rows(fail_rows, fail_filepath)
            print(f"Dumped {len(fail_rows)} solver failed questions to {fail_filepath}")
        except Exception as e:
            logger.warning(f"Failed to dump solver question outcomes for step {step}: {e}")

    def _summarize_solver_reward_extra_infos(self, solver_extra_infos: Dict[str, List[Any]]) -> Dict[str, float]:
        metrics = {}
        for key, values in (solver_extra_infos or {}).items():
            numeric_values = []
            for value in values:
                if value is None:
                    continue
                if isinstance(value, (bool, np.bool_)):
                    numeric_values.append(float(value))
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_values.append(float(value))

            if numeric_values:
                metrics[f"solver/reward_extra/{key}_mean"] = float(np.mean(numeric_values))

        return metrics

    def _extract_solver_answer_correct_tensor(
        self, solver_rewards: torch.Tensor, solver_extra_infos: Dict[str, List[Any]]
    ) -> torch.Tensor:
        answer_correct_key = None
        if solver_extra_infos:
            for key in solver_extra_infos.keys():
                if key.endswith("_answer_correct"):
                    answer_correct_key = key
                    break

        if answer_correct_key is not None:
            answer_corrects = solver_extra_infos[answer_correct_key]
            assert len(answer_corrects) == solver_rewards.size(
                0
            ), f"Expected {solver_rewards.size(0)} answer_correct values, got {len(answer_corrects)}"
            return torch.tensor(
                [bool(val) if val is not None else False for val in answer_corrects],
                dtype=torch.bool,
                device=solver_rewards.device,
            )

        # Fallback for reward functions that only expose final binary reward.
        seq_rewards = solver_rewards.sum(-1)
        return seq_rewards >= (1.0 - 1e-6)

    def _extract_solver_reward_eligibility_tensor(
        self, solver_rewards: torch.Tensor, solver_extra_infos: Dict[str, List[Any]]
    ) -> torch.Tensor:
        """Return which Solver rollouts may influence Proposer success signals."""
        protocol_gate_key = None
        if solver_extra_infos:
            for key in solver_extra_infos.keys():
                if key.endswith("kgqa_protocol_gate_score"):
                    protocol_gate_key = key
                    break

        if protocol_gate_key is None:
            protocol_eligible = torch.ones(
                solver_rewards.size(0), dtype=torch.bool, device=solver_rewards.device
            )
        else:
            values = solver_extra_infos[protocol_gate_key]
            assert len(values) == solver_rewards.size(
                0
            ), f"Expected {solver_rewards.size(0)} {protocol_gate_key} values, got {len(values)}"
            protocol_eligible = torch.tensor(
                [float(value) > 0.0 if value is not None else False for value in values],
                dtype=torch.bool,
                device=solver_rewards.device,
            )

        # This also rejects response-overlong trajectories, whose effective reward
        # is zeroed by SSPRewardManager independently of the text protocol scorer.
        effective_reward_positive = solver_rewards.sum(-1) > 1e-8
        return protocol_eligible & effective_reward_positive

    def _extract_solver_score_tensor(
        self, solver_rewards: torch.Tensor, solver_extra_infos: Dict[str, List[Any]]
    ) -> torch.Tensor:
        """Continuous per-rollout solver score, used for the proposer's difficulty signal.

        Deliberately kept separate from ``_extract_solver_answer_correct_tensor``: that one must
        stay a bool because the process-reward diagnostics negate it with ``~``.
        """
        score_key = None
        if solver_extra_infos:
            for key in solver_extra_infos.keys():
                if key.endswith("_entity_f1"):
                    score_key = key
                    break

        eligibility = self._extract_solver_reward_eligibility_tensor(
            solver_rewards, solver_extra_infos
        ).float()

        if score_key is None:
            # No continuous score exposed; fall back to the binary signal.
            score_tensor = self._extract_solver_answer_correct_tensor(
                solver_rewards, solver_extra_infos
            ).float()
            return score_tensor * eligibility

        values = solver_extra_infos[score_key]
        assert len(values) == solver_rewards.size(
            0
        ), f"Expected {solver_rewards.size(0)} {score_key} values, got {len(values)}"
        score_tensor = torch.tensor(
            [float(val) if val is not None else 0.0 for val in values],
            dtype=torch.float32,
            device=solver_rewards.device,
        )
        return score_tensor * eligibility

    def _extract_solver_process_reward_metrics(
        self, solver_rewards: torch.Tensor, solver_extra_infos: Dict[str, List[Any]]
    ) -> Dict[str, float]:
        metrics = {}

        raw_match_rate_key = None
        if solver_extra_infos:
            for key in solver_extra_infos.keys():
                if key.endswith("_raw_match_rate"):
                    raw_match_rate_key = key

        if raw_match_rate_key is not None:
            raw_match_rates = [val for val in solver_extra_infos[raw_match_rate_key] if val is not None]
            if raw_match_rates:
                metrics["solver/process_reward/raw_match_rate_mean"] = float(np.mean(raw_match_rates))

        seq_rewards = solver_rewards.sum(-1)
        answer_correct_tensor = self._extract_solver_answer_correct_tensor(solver_rewards, solver_extra_infos)
        reward_eligible_tensor = self._extract_solver_reward_eligibility_tensor(
            solver_rewards, solver_extra_infos
        )
        eligible_correct_tensor = answer_correct_tensor & reward_eligible_tensor
        wrong_mask = ~answer_correct_tensor
        wrong_sample_rewards = seq_rewards[wrong_mask]
        positive_wrong_mask = wrong_mask & (seq_rewards > 0)

        metrics["solver/final_correct_rate"] = float(answer_correct_tensor.float().mean().item())
        metrics["solver/protocol_eligible_final_correct_rate"] = float(
            eligible_correct_tensor.float().mean().item()
        )
        metrics["solver/rejected_correct_rate"] = float(
            (answer_correct_tensor & ~reward_eligible_tensor).float().mean().item()
        )
        metrics["solver/process_reward/positive_reward_wrong_rate"] = float(positive_wrong_mask.float().mean().item())
        metrics["solver/process_reward/positive_reward_wrong_count"] = float(positive_wrong_mask.sum().item())

        if wrong_sample_rewards.numel() > 0:
            metrics["solver/process_reward/wrong_sample_reward_mean"] = float(wrong_sample_rewards.mean().item())

        return metrics

    def _compute_proposer_rewards(
        self, proposer_gen_batch: DataProto, solver_rewards: torch.Tensor, solver_extra_infos: Dict
    ) -> torch.Tensor:
        proposer_batch_size = len(proposer_gen_batch.batch)
        response_length = proposer_gen_batch.batch["responses"].size(1)

        solver_n_attempts = self.rollout_n
        solver_batch_size = solver_rewards.size(0)
        assert (
            solver_batch_size == proposer_batch_size * solver_n_attempts
        ), f"Expected {proposer_batch_size * solver_n_attempts} solver samples, got {solver_batch_size}"

        solver_rewards_per_attempt = solver_rewards.sum(-1)

        solver_rewards_grouped = solver_rewards_per_attempt.view(proposer_batch_size, solver_n_attempts)
        raw_answer_correct_tensor = self._extract_solver_answer_correct_tensor(
            solver_rewards, solver_extra_infos
        )
        reward_eligible_tensor = self._extract_solver_reward_eligibility_tensor(
            solver_rewards, solver_extra_infos
        )
        answer_correct_tensor = raw_answer_correct_tensor & reward_eligible_tensor
        success_rates = answer_correct_tensor.float().view(proposer_batch_size, solver_n_attempts).mean(dim=1)
        raw_success_rates = raw_answer_correct_tensor.float().view(
            proposer_batch_size, solver_n_attempts
        ).mean(dim=1)
        positive_reward_rates = (solver_rewards_grouped > 0).float().mean(dim=1)
        # [B, n] continuous solver scores — the only signal that survives set-valued answers.
        solver_scores = self._extract_solver_score_tensor(solver_rewards, solver_extra_infos).view(
            proposer_batch_size, solver_n_attempts
        )

        reward_type = self.sp_config.get("proposer", {}).get("reward_type", "1-acc")

        proposer_reward_values = compute_proposer_reward_values(
            success_rates=success_rates,
            reward_type=reward_type,
            proposer_left=self.sp_config.proposer.get("left", 0.0),
            proposer_right=self.sp_config.proposer.get("right", 1.0),
            proposer_tau=self.sp_config.proposer.get("tau", 0.01),
            solver_scores=solver_scores,
        )

        extraction_failure_penalty = self.sp_config.proposer.format_penalty

        if hasattr(self, "current_extraction_success_mask"):
            for i, extraction_success in enumerate(self.current_extraction_success_mask):
                if not extraction_success:
                    proposer_reward_values[i] = extraction_failure_penalty
                    if SELF_PLAY_DEBUG:
                        logger.debug(
                            f"Applying extraction failure penalty {extraction_failure_penalty} to proposer generation {i}"
                        )
        elif hasattr(self, "generate_problem") and self.generate_problem:
            assert len(self.generate_problem) == len(
                proposer_reward_values
            ), "generate_problem and proposer_reward_values must have the same length"
            for i, problem in enumerate(self.generate_problem):
                if i < len(proposer_reward_values):
                    data_source = problem.get("data_source", "")
                    if data_source == "dummy":
                        proposer_reward_values[i] = extraction_failure_penalty
                        if SELF_PLAY_DEBUG:
                            logger.debug(
                                f"Applying extraction failure penalty {extraction_failure_penalty} to proposer generation {i} (dummy data_source)"
                            )

        proposer_rewards = torch.zeros(
            (proposer_batch_size, response_length),
            dtype=torch.float32,
            device=solver_rewards.device,
        )

        if "response_mask" in proposer_gen_batch.batch:
            response_mask = proposer_gen_batch.batch["response_mask"]
            last_token_positions = response_mask.sum(-1) - 1
            for i in range(proposer_batch_size):
                last_pos = last_token_positions[i].item()
                if 0 <= last_pos < response_length:
                    proposer_rewards[i, last_pos] = proposer_reward_values[i].item()
        else:
            proposer_rewards[:, -1] = proposer_reward_values

        extraction_failures = 0
        if hasattr(self, "current_extraction_success_mask"):
            extraction_failures = sum(1 for x in self.current_extraction_success_mask if not x)

        self.current_proposer_reward_metrics = {
            "proposer/success_rate_final_correct_mean": float(success_rates.mean().item()),
            "proposer/success_rate_raw_final_correct_mean": float(raw_success_rates.mean().item()),
            "proposer/success_rate_rejected_correct_mean": float(
                (raw_success_rates - success_rates).mean().item()
            ),
            "proposer/success_rate_positive_reward_mean": float(positive_reward_rates.mean().item()),
            "proposer/success_rate_gap_mean": float((positive_reward_rates - success_rates).mean().item()),
            "proposer/all_zero_solver_reward_group_frac": float(
                (solver_rewards_grouped.abs().amax(dim=1) <= 1e-8).float().mean().item()
            ),
        }

        print(f"Proposer reward computation (using formula: {reward_type}):")
        print(
            f"  - Eligible final-correct success rates: mean={success_rates.mean():.3f}, std={success_rates.std():.3f}"
        )
        print(
            f"  - Raw final-correct success rates: mean={raw_success_rates.mean():.3f}, std={raw_success_rates.std():.3f}"
        )
        print(
            f"  - Positive-reward rates (diagnostic only): mean={positive_reward_rates.mean():.3f}, std={positive_reward_rates.std():.3f}"
        )
        print(
            f"  - Proposer rewards: mean={proposer_reward_values.mean():.3f}, std={proposer_reward_values.std():.3f}, max={proposer_reward_values.max():.3f}"
        )
        print(f"  - Total problems: {proposer_batch_size}")
        print(f"  - Extraction failures: {extraction_failures}/{proposer_batch_size}")

        print(f"  - Reward range: [{proposer_reward_values.min():.3f}, {proposer_reward_values.max():.3f}]")

        if reward_type == "intermediate_difficulty":
            zero_success = (success_rates == 0).sum().item()
            full_success = (success_rates == 1).sum().item()
            intermediate_count = ((success_rates > 0) & (success_rates < 1)).sum().item()
            print(f"  - Problems with success_rate=0 (zero reward): {zero_success}")
            print(f"  - Problems with success_rate=1 (zero reward): {full_success}")
            print(f"  - Problems with intermediate difficulty (non-zero reward): {intermediate_count}")
            if intermediate_count > 0:
                intermediate_rates = success_rates[(success_rates > 0) & (success_rates < 1)]
                intermediate_rewards = proposer_reward_values[(success_rates > 0) & (success_rates < 1)]
                print(
                    f"  - Intermediate problems success_rate range: [{intermediate_rates.min():.3f}, {intermediate_rates.max():.3f}]"
                )
                print(
                    f"  - Intermediate problems reward range: [{intermediate_rewards.min():.3f}, {intermediate_rewards.max():.3f}]"
                )
        elif reward_type == "gaussian_difficulty":
            tau = self.sp_config.proposer.get("tau", 0.01)
            boundary_count = ((success_rates > 0) & (success_rates < 1)).sum().item()
            print(f"  - Gaussian difficulty tau: {tau}")
            print(f"  - Problems near boundary difficulty (0<success_rate<1): {boundary_count}")
            print(f"  - Problems with success_rate=0 (low reward): {(success_rates == 0).sum().item()}")
            print(f"  - Problems with success_rate=1 (low reward): {(success_rates == 1).sum().item()}")
        else:
            print(f"  - Problems with success_rate=0 (max reward): {(success_rates == 0).sum().item()}")
            print(f"  - Problems with success_rate=1 (min reward): {(success_rates == 1).sum().item()}")
            print(f"  - Problems with 0<success_rate<1: {((success_rates > 0) & (success_rates < 1)).sum().item()}")

        if hasattr(self, "generate_problem") and self.generate_problem:
            valid_problems_with_success_rate = []
            assert len(self.generate_problem) == len(
                success_rates
            ), f"generate_problem length {len(self.generate_problem)} != success_rates length {len(success_rates)}"

            for i, problem in enumerate(self.generate_problem):
                if i < len(success_rates):
                    data_source = problem.get("data_source", "")
                    if data_source != "dummy":
                        problem_with_rate = problem.copy()
                        problem_with_rate["success_rate"] = success_rates[i].item()
                        problem_with_rate["generation_step"] = self.global_steps
                        valid_problems_with_success_rate.append(problem_with_rate)

            if valid_problems_with_success_rate:
                ray.get(
                    self.sp_data_manager.add_generated_problems_with_success_rate.remote(
                        valid_problems_with_success_rate, generation_step=self.global_steps
                    )
                )
                print(f"Added {len(valid_problems_with_success_rate)} valid problems with success_rate to data manager")

        return proposer_rewards

    def _swap_proposer_baseline_key(self, gen_batch: DataProto, role: str, num_repeat: int) -> Optional[np.ndarray]:
        """Temporarily repoint the advantage grouping key for the proposer.

        The advantage estimators read their grouping key from ``non_tensor_batch["uid"]``, which is
        allocated one-per-seed-path. With ``proposer.n == 1`` every group then holds a single rollout
        and the baseline degenerates to the batch mean — and because the batch mixes iso-templates
        whose expected reward differs, each template picks up a constant advantage offset.

        Grouping by ``template_id`` instead removes that offset exactly. Returns the original uid
        array, or None if no swap was performed; the caller restores uid unconditionally after
        advantage computation.
        """
        if role != "proposer":
            return None

        baseline_key = self.sp_config.get("proposer", {}).get("baseline_key", "uid")
        if not baseline_key or baseline_key == "uid":
            return None

        if "extra_info" not in gen_batch.non_tensor_batch:
            print(f"[proposer_adv] WARNING: baseline_key={baseline_key} requested but extra_info is absent; keeping uid")
            return None

        if num_repeat > 1:
            print(
                f"[proposer_adv] WARNING: baseline_key={baseline_key} together with proposer.n={num_repeat} "
                "overrides the within-path contrast; keep baseline_key=uid when n > 1"
            )

        uid_backup = gen_batch.non_tensor_batch["uid"]
        gen_batch.non_tensor_batch["uid"] = np.array(
            [str((info or {}).get(baseline_key, "unknown")) for info in gen_batch.non_tensor_batch["extra_info"]],
            dtype=object,
        )

        group_sizes = Counter(gen_batch.non_tensor_batch["uid"])
        print(
            f"[proposer_adv] advantage baseline grouped by {baseline_key}: "
            f"{len(group_sizes)} groups, sizes={dict(sorted(group_sizes.items()))}"
        )
        return uid_backup

    def _compute_overlong_metrics(self, gen_batch: DataProto, role: str) -> Dict[str, float]:
        """Report length-truncated rollouts without changing masks or advantage groups."""
        if "finish_reason" not in gen_batch.non_tensor_batch:
            return {}

        finish_reasons = gen_batch.non_tensor_batch["finish_reason"]
        overlong = np.array([fr == FinishReasonTypeEnum.LENGTH.value for fr in finish_reasons], dtype=bool)
        return {f"{role}/overlong/frac": float(overlong.mean()) if len(overlong) else 0.0}

    def _compute_advantage_metrics(self, gen_batch: DataProto, role: str) -> Dict[str, float]:
        """Sequence-level advantage diagnostics.

        ``neg_frac`` is the load-bearing one: an uncentered baseline (every group of size 1 under
        GRPO) leaves the advantage equal to the raw, all-positive reward, so it reads 0. A working
        baseline puts it near the fraction of samples below the batch mean.

        ``seq_mean`` is *not* a sufficient check on its own: REINFORCE++ whitens over tokens, which
        forces the token-weighted mean to exactly 0. This metric averages over sequences instead, so
        what is left is the length/reward correlation rather than a pass/fail signal.

        All four are computed over rollouts carrying at least one valid response token. The current
        overlong policy retains the normal response mask, but filtering from other code paths can
        still produce an all-zero row; excluding such rows prevents a hard 0 from silently
        deflating ``neg_frac``.
        """
        if "advantages" not in gen_batch.batch or "response_mask" not in gen_batch.batch:
            return {}

        advantages = gen_batch.batch["advantages"]
        response_mask = gen_batch.batch["response_mask"].to(advantages.dtype)
        token_counts = response_mask.sum(-1)
        seq_adv = (advantages * response_mask).sum(-1) / token_counts.clamp(min=1.0)

        seq_adv = seq_adv[token_counts > 0]
        if seq_adv.numel() == 0:
            return {}

        return {
            f"{role}/adv/neg_frac": float((seq_adv < 0).float().mean().item()),
            f"{role}/adv/seq_mean": float(seq_adv.mean().item()),
            f"{role}/adv/seq_std": float(seq_adv.std().item()),
            f"{role}/adv/seq_abs_mean": float(seq_adv.abs().mean().item()),
        }

    def _prepare_solver_update_batch(
        self, gen_batch: Optional[DataProto]
    ) -> Tuple[Optional[DataProto], Dict[str, float]]:
        """Remove dummy rows and retain only complete, actor-aligned GRPO groups."""
        if gen_batch is None:
            return None, {}
        if "response_mask" not in gen_batch.batch:
            raise KeyError("Solver update batch is missing response_mask")

        response_mask = gen_batch.batch["response_mask"].bool()
        token_counts = response_mask.sum(dim=-1)
        empty_mask = token_counts == 0
        total_count = int(token_counts.numel())

        data_sources = gen_batch.non_tensor_batch.get("data_source")
        if data_sources is None:
            dummy_mask = torch.zeros_like(empty_mask)
        else:
            dummy_values = np.asarray(data_sources, dtype=object)
            if len(dummy_values) != total_count:
                raise ValueError(
                    f"data_source length {len(dummy_values)} does not match Solver batch size {total_count}"
                )
            dummy_mask = torch.as_tensor(
                [str(value).strip().lower() == "dummy" for value in dummy_values],
                dtype=torch.bool,
                device=empty_mask.device,
            )

        invalid_sequence_mask = dummy_mask | empty_mask
        uid_values = gen_batch.non_tensor_batch.get("uid")
        if uid_values is not None:
            if len(uid_values) != total_count:
                raise ValueError(
                    f"uid length {len(uid_values)} does not match Solver batch size {total_count}"
                )
            invalid_uids = {
                str(uid_values[index])
                for index in torch.nonzero(invalid_sequence_mask, as_tuple=False).squeeze(-1).tolist()
            }
            # PPO/GRPO group normalization must never see a partial uid group.
            retained_mask = torch.as_tensor(
                [str(uid) not in invalid_uids for uid in uid_values],
                dtype=torch.bool,
                device=empty_mask.device,
            )
        else:
            retained_mask = ~invalid_sequence_mask
        retained_nonempty_mask = retained_mask & ~empty_mask
        dummy_count = int(dummy_mask.sum().item())
        empty_count = int(empty_mask.sum().item())
        unexpected_empty_count = int((empty_mask & ~dummy_mask).sum().item())
        retained_count = int(retained_mask.sum().item())
        retained_nonempty_count = int(retained_nonempty_mask.sum().item())

        metrics = {
            "empty_update_sequence_count": float(empty_count),
            "dummy_update_sequence_count": float(dummy_count),
            "unexpected_empty_update_sequence_count": float(unexpected_empty_count),
            "retained_update_sequence_count": float(retained_count),
            "nonempty_update_sequence_count": float(retained_nonempty_count),
            "empty_update_sequence_rate": float(empty_count / total_count) if total_count else 0.0,
            "incomplete_update_group_count": 0.0,
            "dropped_group_count": 0.0,
            "dropped_sequence_count": 0.0,
            "actor/update_skipped_empty_batch": float(retained_nonempty_count == 0),
            "actor/update_skipped_alignment_batch": 0.0,
        }

        if retained_nonempty_count == 0:
            logger.warning(
                "Solver update skipped at step %s: %s/%s rows are dummy and no retained row "
                "contains a trainable response token",
                self.global_steps,
                dummy_count,
                total_count,
            )
            return None, metrics

        if bool(retained_mask.all().item()):
            filtered_batch = gen_batch
        else:
            keep_indices = torch.nonzero(retained_mask, as_tuple=False).squeeze(-1)
            filtered_batch = gen_batch[keep_indices]
        metrics["retained_update_sequence_count"] = float(len(filtered_batch))
        filtered_token_counts = filtered_batch.batch["response_mask"].bool().sum(dim=-1)
        metrics["nonempty_update_sequence_count"] = float((filtered_token_counts > 0).sum().item())

        uid_values = filtered_batch.non_tensor_batch.get("uid")
        incomplete_group_count = 0
        if uid_values is not None:
            group_sizes = Counter(str(uid) for uid in uid_values)
            incomplete_group_count = sum(size != self.rollout_n for size in group_sizes.values())
        metrics["incomplete_update_group_count"] = float(incomplete_group_count)
        if incomplete_group_count:
            raise ValueError(
                f"Filtering dummy Solver rows left {incomplete_group_count} incomplete GRPO groups; "
                f"expected every retained uid to have rollout_n={self.rollout_n}"
            )

        world_size = int(self.actor_rollout_wg.world_size)
        if world_size <= 0:
            raise ValueError(f"Actor world size must be positive, got {world_size}")

        if len(filtered_batch) % world_size != 0:
            if uid_values is None:
                raise ValueError(
                    f"Cannot align Solver update batch of size {len(filtered_batch)} to actor world size "
                    f"{world_size} without uid metadata"
                )

            group_order = list(dict.fromkeys(str(uid) for uid in uid_values))
            group_multiple = world_size // math.gcd(world_size, self.rollout_n)
            dropped_group_count = len(group_order) % group_multiple
            dropped_uids = set(group_order[-dropped_group_count:])
            alignment_keep_mask = torch.as_tensor(
                [str(uid) not in dropped_uids for uid in uid_values],
                dtype=torch.bool,
                device=filtered_batch.batch["response_mask"].device,
            )
            filtered_batch = filtered_batch[alignment_keep_mask]

            dropped_sequence_count = dropped_group_count * self.rollout_n
            metrics["dropped_group_count"] = float(dropped_group_count)
            metrics["dropped_sequence_count"] = float(dropped_sequence_count)
            metrics["retained_update_sequence_count"] = float(len(filtered_batch))
            filtered_token_counts = filtered_batch.batch["response_mask"].bool().sum(dim=-1)
            metrics["nonempty_update_sequence_count"] = float((filtered_token_counts > 0).sum().item())
            logger.warning(
                "Solver update alignment at step %s: dropped %s complete GRPO groups (%s rows) "
                "so %s rows divide evenly across %s actor workers",
                self.global_steps,
                dropped_group_count,
                dropped_sequence_count,
                len(filtered_batch),
                world_size,
            )

        if len(filtered_batch) == 0:
            metrics["actor/update_skipped_alignment_batch"] = 1.0
            logger.warning(
                "Solver update skipped at step %s: fewer than %s aligned GRPO groups remain "
                "for %s actor workers",
                self.global_steps,
                world_size // math.gcd(world_size, self.rollout_n),
                world_size,
            )
            return None, metrics

        if len(filtered_batch) % world_size != 0:
            raise AssertionError(
                f"Aligned Solver update batch size {len(filtered_batch)} is not divisible by actor world size "
                f"{world_size}"
            )

        logger.info(
            "Solver update filtering at step %s: observed %s dummy and %s empty rows; "
            "retained %s rows (%s non-empty) after whole-group filtering",
            self.global_steps,
            dummy_count,
            empty_count,
            len(filtered_batch),
            int(metrics["nonempty_update_sequence_count"]),
        )
        return filtered_batch, metrics

    def _update_on_trajectories(self, gen_batch: DataProto, role: str, timing_raw: Dict) -> Dict[str, Any]:
        print(f"=== Step 5: {role.title()} Update at step {self.global_steps} ===")
        update_total_start = time.perf_counter()

        metrics = {}

        if self.config.trainer.balance_batch:
            self._balance_batch(gen_batch, metrics=metrics)

        gen_batch.meta_info["global_token_num"] = torch.sum(gen_batch.batch["attention_mask"], dim=-1).tolist()

        # Ensure batch is unlocked before modification
        gen_batch.ensure_unlocked()
        for key in gen_batch.batch.keys():
            if key not in [
                "old_log_probs",
                "ref_log_prob",
                "entropys",
                "token_level_scores",
                "token_level_rewards",
                "advantages",
                "returns",
                "values",
            ]:
                gen_batch.batch[key] = gen_batch.batch[key].long()

        with marked_timer(f"{role}_old_log_prob", timing_raw, color="blue"):
            if "old_log_probs" in gen_batch.batch:
                entropys = gen_batch.batch["entropys"]
            else:
                old_log_prob = self.actor_rollout_wg.compute_log_prob(gen_batch)
                entropys = old_log_prob.batch["entropys"]
                old_log_prob.ensure_unlocked()
                old_log_prob.batch.pop("entropys")
                gen_batch = gen_batch.union(old_log_prob)
            response_masks = gen_batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics[f"{role}/actor/entropy"] = entropy_agg.detach().item()
            gen_batch.batch.pop("entropys", None)

        if self.use_reference_policy:
            with marked_timer(f"{role}_ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(gen_batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(gen_batch)
                gen_batch = gen_batch.union(ref_log_prob)

        if self.use_critic:
            with marked_timer(f"{role}_values", timing_raw, color="cyan"):
                values = self.critic_wg.compute_values(gen_batch)
                gen_batch = gen_batch.union(values)

        with marked_timer(f"{role}_adv", timing_raw, color="brown"):
            if self.config.algorithm.use_kl_in_reward:
                gen_batch, kl_metrics = apply_kl_penalty(
                    gen_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update({f"{role}_{k}": v for k, v in kl_metrics.items()})
            else:
                gen_batch.batch["token_level_rewards"] = gen_batch.batch["token_level_scores"]

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            if role == "solver":
                num_repeat = self.rollout_n
                adv_estimator = self.config.algorithm.adv_estimator
            else:
                num_repeat = self.sp_config.proposer.n
                if num_repeat == 1:
                    adv_estimator = self.sp_config.proposer.adv_estimator
                else:
                    adv_estimator = self.config.algorithm.adv_estimator

            uid_original = gen_batch.non_tensor_batch["uid"]
            self._swap_proposer_baseline_key(gen_batch, role, num_repeat)
            overlong_metrics = self._compute_overlong_metrics(gen_batch, role)
            try:
                gen_batch = compute_advantage(
                    gen_batch,
                    adv_estimator=adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=num_repeat,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )
            finally:
                # The proposer baseline helper may repoint uid, which is still consumed
                # downstream (e.g. _apply_reward_dynamic_sampling), so restore it.
                gen_batch.non_tensor_batch["uid"] = uid_original

            metrics.update(overlong_metrics)
            metrics.update(self._compute_advantage_metrics(gen_batch, role))

        if self.use_critic:
            with marked_timer(f"{role}_update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(gen_batch)
            critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update({f"{role}_critic_{k}": v for k, v in critic_metrics.items()})

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer(f"{role}_update_actor", timing_raw, color="red"):
                gen_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(gen_batch)
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update({f"{role}_actor_{k}": v for k, v in actor_metrics.items()})

        scores = gen_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
        metrics.update(
            {
                f"{role}_reward_mean": np.mean(scores),
                f"{role}_reward_std": np.std(scores),
                f"{role}_reward_max": np.max(scores),
                f"{role}_reward_min": np.min(scores),
            }
        )

        update_total_key = f"{role}_update_total"
        timing_raw[update_total_key] = timing_raw.get(update_total_key, 0.0) + (
            time.perf_counter() - update_total_start
        )
        return metrics

    def _capture_role_log_prob(self, gen_batch: DataProto, role: str, timing_raw: Dict) -> DataProto:
        """Attach behavior-policy log-probabilities before the shared actor is updated."""
        if gen_batch is None or "old_log_probs" in gen_batch.batch:
            return gen_batch

        with marked_timer(f"{role}_old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(gen_batch)
            return gen_batch.union(old_log_prob)

    def _dump_trajectories(self, inputs, outputs, scores, role: str, step: int, reward_extra_infos_dict=None):
        try:
            import os

            base_dir = os.environ.get("TRAJECTORY_DIR", "./output/trajectories")
            role_dir = os.path.join(base_dir, role)
            os.makedirs(role_dir, exist_ok=True)

            filename = f"{role}_step_{step}.jsonl"
            filepath = os.path.join(role_dir, filename)

            n = len(inputs)
            base_data = {
                "input": inputs,
                "output": outputs,
                "score": scores,
                "step": [step] * n,
                "role": [role] * n,
                "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * n,
            }

            if reward_extra_infos_dict:
                for k, v in reward_extra_infos_dict.items():
                    if len(v) == n:
                        base_data[k] = v

            lines = []
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                lines.append(json.dumps(entry, ensure_ascii=False, default=str))

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print(f"Dumped {len(inputs)} {role} trajectories to {filepath}")

        except Exception as e:
            logger.warning(f"Failed to dump {role} trajectories for step {step}: {e}")

    def _prepare_solving_batch_from_data(self, solving_data: List[Dict]) -> DataProto:
        assert solving_data is not None, "No solving data available"

        import tempfile

        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_file = f.name
            table = pa.Table.from_pylist(solving_data)
            pq.write_table(table, temp_file)

        try:
            from verl.utils.dataset.rl_dataset import RLHFDataset

            if SELF_PLAY_DEBUG:
                logger.debug(f"Input solving_data length: {len(solving_data)}")

            temp_config = deepcopy(self.config.data)
            temp_config.filter_overlong_prompts = False

            temp_dataset = RLHFDataset(
                data_files=[temp_file], tokenizer=self.tokenizer, config=temp_config, processor=self.processor
            )

            if SELF_PLAY_DEBUG:
                logger.debug(f"RLHFDataset length after creation: {len(temp_dataset)}")

            from torchdata.stateful_dataloader import StatefulDataLoader
            from verl.utils.dataset.rl_dataset import collate_fn

            temp_dataloader = StatefulDataLoader(
                dataset=temp_dataset,
                batch_size=len(solving_data),
                num_workers=0,
                drop_last=False,
                collate_fn=collate_fn,
                shuffle=False,
            )

            batch_dict = next(iter(temp_dataloader))
            batch = DataProto.from_single_dict(batch_dict)

            if SELF_PLAY_DEBUG:
                logger.debug(f"Final batch size after processing: {len(batch)}")

            # Ensure batch is unlocked before modification
            batch.ensure_unlocked()
            for key in batch.batch.keys():
                if key not in [
                    "old_log_probs",
                    "ref_log_prob",
                    "token_level_scores",
                    "token_level_rewards",
                    "advantages",
                    "returns",
                    "values",
                ]:
                    batch.batch[key] = batch.batch[key].long()

            return batch

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _extract_trajectories_from_batch(self, batch: DataProto) -> List[Dict[str, Any]]:
        trajectories = []

        assert "prompts" in batch.batch and "responses" in batch.batch, "prompts and responses must be in batch"

        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracting trajectories from batch with {len(batch.batch['prompts'])} samples")
            logger.debug(f"Batch keys: {list(batch.batch.keys())}")
            if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch:
                logger.debug(f"Non-tensor batch keys: {list(batch.non_tensor_batch.keys())}")

        input_ids = batch.batch["prompts"]
        output_ids = batch.batch["responses"]

        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for i, (input_text, output_text) in enumerate(zip(input_texts, output_texts)):
            metadata = {"trajectory_index": i, "global_step": self.global_steps, "phase": self.current_phase}
            structured_messages = None

            if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch is not None:
                if "messages" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["messages"]):
                    structured_messages = batch.non_tensor_batch["messages"][i]

                if "reward_model" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["reward_model"]):
                    metadata["reward_model"] = batch.non_tensor_batch["reward_model"][i]
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: reward_model = {metadata['reward_model']}")

                if "data_source" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["data_source"]):
                    metadata["data_source"] = batch.non_tensor_batch["data_source"][i]
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: data_source = {metadata['data_source']}")

                if "extra_info" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["extra_info"]):
                    metadata["extra_info"] = enrich_extra_info_with_process_reward(
                        batch.non_tensor_batch["extra_info"][i],
                        input_text,
                        node_filter_mode=self.process_reward_node_filter_mode,
                    )
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: extra_info = {metadata['extra_info']}")
                else:
                    metadata["extra_info"] = enrich_extra_info_with_process_reward(
                        None,
                        input_text,
                        node_filter_mode=self.process_reward_node_filter_mode,
                    )

            structured_output = render_structured_react_trajectory(structured_messages)
            extraction_output = structured_output or output_text
            trajectory = {
                "input": input_text,
                "output": extraction_output,
                "raw_output": output_text,
                "messages": structured_messages,
                "metadata": metadata,
            }
            trajectories.append(trajectory)

            if SELF_PLAY_DEBUG:
                logger.debug(f"Trajectory {i}: input[:100] = {input_text[:100]}...")
                logger.debug(f"Trajectory {i}: output[:200] = {extraction_output[:200]}...")
                logger.debug(f"Trajectory {i}: metadata = {metadata}")

        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracted {len(trajectories)} trajectories total")

        return trajectories

    def _extract_and_process_problems(self, trajectories: List[Dict[str, Any]], record_stats: bool = True) -> tuple:
        """
        Extract and process problems from trajectories.
        
        Args:
            trajectories: List of trajectory dictionaries
            record_stats: Whether to record stats to sp_data_manager (default True)
                         Set to False when accumulating stats across multiple batches in dynamic sampling
        
        Returns:
            If record_stats is True: List of valid problems
            If record_stats is False: Tuple of (valid_problems, extraction_stats_dict)
        """
        if SELF_PLAY_DEBUG:
            logger.debug(f"Starting problem extraction from {len(trajectories)} trajectories")
            for i, traj in enumerate(trajectories[:3]):
                logger.debug(f"Trajectory {i} output for extraction: {traj['output'][:500]}...")

        if record_stats:
            ray.get(self.sp_data_manager.reset_current_step_stats.remote())

        start_time = time.time()
        #-----------------------------------------------------------------------------------------
        #  Proposer： 问题抽取
        #-----------------------------------------------------------------------------------------
        # 降低并发数以避免 API 429 限流错误
        rag_max_workers = self.sp_config.get("rag_max_workers", 4)  # 默认 4 个线程
        extracted_problems, extraction_stats = extract_problems_batch(trajectories, self.problem_extractor, max_workers=rag_max_workers)
        end_time = time.time()
        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracted {len(extracted_problems)} problems in {end_time - start_time} seconds")
            logger.debug(f"Extraction statistics: {extraction_stats}")
            for i, problem in enumerate(extracted_problems[:3]):
                logger.debug(f"Raw problem {i}: {problem}")

        valid_problems = []
        trajectory_outcomes = [
            dict(outcome) if isinstance(outcome, dict) else {}
            for outcome in extraction_stats.get("trajectory_outcomes", [])
        ]
        second_stage_failure_count = 0
        for i, problem in enumerate(extracted_problems):
            if self._validate_extracted_problem(problem):
                valid_problems.append(problem)
                if SELF_PLAY_DEBUG:
                    logger.debug(f"Problem {i} is VALID: {problem.get('extracted_question', '')[:100]}...")
            else:
                if SELF_PLAY_DEBUG:
                    logger.debug(f"Problem {i} is INVALID: {problem}")
                second_stage_failure_count += 1
                trajectory_index = int(problem.get("trajectory_index", -1))
                if 0 <= trajectory_index < len(trajectory_outcomes):
                    trajectory_outcomes[trajectory_index].update(
                        {
                            "trajectory_index": trajectory_index,
                            "successful_problems": 0,
                            "failure_reason": "second_stage_validation_failed",
                        }
                    )

        print(f"Extracted {len(extracted_problems)} problems, {len(valid_problems)} valid")

        failure_reasons = dict(extraction_stats.get("failure_reasons", {}))
        # A trajectory that passed the first stage but failed validation changes
        # classification from pass to this stable second-stage failure bucket.
        if second_stage_failure_count:
            failure_reasons["second_stage_validation_failed"] = (
                failure_reasons.get("second_stage_validation_failed", 0)
                + second_stage_failure_count
            )

        # Build exactly reconcilable stats for this batch. A trajectory is either
        # a final pass or has one first-failure reason, never both.
        batch_stats = {
            "trajectories_count": extraction_stats["trajectories_count"],
            "answer_matches_count": extraction_stats["answer_matches_count"],
            "valid_questions_count": extraction_stats["valid_questions_count"],
            "format_error_count": extraction_stats["format_error_count"],
            "successful_problems_count": len(valid_problems),
            "pass_count": len(valid_problems),
            # Which gate rule rejected the trajectory. Needed to tune the gate threshold —
            # without it a drop in pass rate is indistinguishable from a parse failure.
            "failure_reasons": failure_reasons,
            "trajectory_outcomes": trajectory_outcomes,
        }

        self.current_extraction_outcomes = trajectory_outcomes

        classified_count = batch_stats["pass_count"] + sum(failure_reasons.values())
        if classified_count != batch_stats["trajectories_count"]:
            raise RuntimeError(
                "Extraction outcome counts do not cover the batch exactly: "
                f"pass={batch_stats['pass_count']} failures={sum(failure_reasons.values())} "
                f"trajectories={batch_stats['trajectories_count']}"
            )

        if record_stats:
            # Dynamic sampling writes its accumulated version outside the retry loop.
            self._save_extraction_stats(batch_stats, len(trajectories), len(valid_problems))

        if record_stats:
            ray.get(
                self.sp_data_manager.record_extraction_stats.remote(
                    trajectories_count=batch_stats["trajectories_count"],
                    answer_matches_count=batch_stats["answer_matches_count"],
                    valid_questions_count=batch_stats["valid_questions_count"],
                    format_error_count=batch_stats["format_error_count"],
                    successful_problems_count=batch_stats["successful_problems_count"],
                )
            )

        if SELF_PLAY_DEBUG:
            logger.debug(f"Final valid problems count: {len(valid_problems)}")
            for i, problem in enumerate(valid_problems[:2]):  # Debug first 2 valid problems
                logger.debug(f"Valid problem {i}: {problem}")

        if record_stats:
            return valid_problems
        else:
            return valid_problems, batch_stats

    def _save_extraction_stats(self, extraction_stats: Dict, total_trajectories: int, valid_count: int):
        """保存详细的提取统计信息到文件"""
        import json
        from datetime import datetime
        
        stats_dir = os.environ.get("EXTRACTION_STATS_DIR", "./output/extraction_stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        failure_reasons = extraction_stats.get("failure_reasons", {}).copy()
        pass_count = int(extraction_stats.get("pass_count", valid_count))
        classified_count = pass_count + sum(failure_reasons.values())
        if classified_count != total_trajectories:
            raise RuntimeError(
                "Extraction artifact would contain inconsistent outcome counts: "
                f"pass={pass_count} failures={sum(failure_reasons.values())} "
                f"trajectories={total_trajectories}"
            )
        
        # 计算各阶段统计
        stats_summary = {
            "step": self.global_steps,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_trajectories": total_trajectories,
            "final_valid_count": valid_count,
            "pass_count": pass_count,
            "success_rate": f"{valid_count / total_trajectories * 100:.2f}%" if total_trajectories > 0 else "0%",
            "failure_breakdown": {},
            "raw_stats": {
                "trajectories_count": extraction_stats.get("trajectories_count", 0),
                "answer_matches_count": extraction_stats.get("answer_matches_count", 0),
                "valid_questions_count": extraction_stats.get("valid_questions_count", 0),
                "successful_problems_count": valid_count,
                "format_error_count": extraction_stats.get("format_error_count", 0),
            }
        }
        
        # 按失败数量排序
        sorted_failures = sorted(failure_reasons.items(), key=lambda x: -x[1])
        for reason, count in sorted_failures:
            percentage = count / total_trajectories * 100 if total_trajectories > 0 else 0
            stats_summary["failure_breakdown"][reason] = {
                "count": count,
                "percentage": f"{percentage:.1f}%"
            }
        
        # 保存到文件
        stats_file = os.path.join(stats_dir, f"extraction_stats_step_{self.global_steps}.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_summary, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print(f"\n{'='*60}")
        print(f"📊 Extraction Stats at Step {self.global_steps}")
        print(f"{'='*60}")
        print(f"Total: {total_trajectories} | Valid: {valid_count} | Rate: {stats_summary['success_rate']}")
        print(f"\nFailure Breakdown (by first failure reason):")
        for reason, info in list(stats_summary["failure_breakdown"].items())[:10]:
            print(f"  - {reason}: {info['count']} ({info['percentage']})")
        print(f"\nStats saved to: {stats_file}")
        print(f"{'='*60}\n")

    def _create_fallback_problem_for_failed_extraction(
        self, trajectory: Dict[str, Any], trajectory_index: int
    ) -> Tuple[Dict[str, Any], bool]:
        """
        
        """
        extraction_strategy = self.sp_config.get("extraction_failure", {}).get("strategy", "dummy")
        fallback_to_dummy = self.sp_config.get("extraction_failure", {}).get("fallback_to_dummy", True)

        if SELF_PLAY_DEBUG:
            logger.debug(
                f"Handling extraction failure for trajectory {trajectory_index} with strategy: {extraction_strategy}"
                )

        if extraction_strategy == "reuse":
            has_existing = ray.get(self.sp_data_manager.has_existing_problems.remote())

            if has_existing:
                reuse_success_rate_threshold = self.sp_config.extraction_failure.reuse_success_rate_threshold
                existing_problems = ray.get(
                    self.sp_data_manager.get_problems_by_success_rate.remote(1, reuse_success_rate_threshold)
                )

                if existing_problems:
                    reused_problem = self._adapt_existing_problem_for_trajectory(
                        existing_problems[0], trajectory, trajectory_index
                    )
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Reused existing problem for trajectory {trajectory_index}")
                    return reused_problem, True

            if fallback_to_dummy:
                if SELF_PLAY_DEBUG:
                    logger.debug(
                        f"No existing problems available, falling back to dummy for trajectory {trajectory_index}"
                    )
                return self._create_dummy_problem_for_failed_extraction(trajectory, trajectory_index), False
            else:
                raise RuntimeError(f"Cannot create fallback problem: no existing problems and dummy fallback disabled")

        return self._create_dummy_problem_for_failed_extraction(trajectory, trajectory_index), False

    def _adapt_existing_problem_for_trajectory(
        self, existing_problem: Dict[str, Any], trajectory: Dict[str, Any], trajectory_index: int
    ) -> Dict[str, Any]:
        adapted_problem = deepcopy(existing_problem)

        adapted_problem["trajectory_index"] = trajectory_index
        adapted_problem["data_source"] = f"{adapted_problem.get('data_source', 'unknown')}"

        return adapted_problem

    def _create_dummy_problem_for_failed_extraction(
        self, trajectory: Dict[str, Any], trajectory_index: int
    ) -> Dict[str, Any]:
        metadata = trajectory.get("metadata", {})
        source_extra_info = deepcopy(metadata.get("extra_info", {}))
        dataset_type = str(source_extra_info.get("dataset_type", "")).strip().lower()
        data_source = str(metadata.get("data_source", "")).strip().lower()
        ability = str(metadata.get("ability", "")).strip().lower()
        is_kgqa_like = dataset_type in {"grailqa", "webqsp", "graphqa", "kgqa", "cwq", "simpleqa"} or (
            "kgqa" in data_source or "kgqa" in ability
        )
        dummy_question = "This is a dummy problem. Please directly output </answer>"

        dummy_prompt = [
            {"role": "system", "content": "You are a helpful and harmless assistant."},
            {
                "role": "user",
                "content": dummy_question,
            },
        ]

        dummy_reward_model = metadata.get("reward_model", {"ground_truth": {"target": "dummy_answer", "style": "rule"}})
        dummy_extra_info = source_extra_info if isinstance(source_extra_info, dict) else {}
        if is_kgqa_like:
            kgqa_create_kwargs = {
                "data_source": "dummy",
                "dataset_type": source_extra_info.get("dataset_type", "kgqa"),
                "qid": source_extra_info.get("qid"),
                "question": dummy_question,
                "ground_truth": dummy_reward_model.get("ground_truth"),
                "question_entities": source_extra_info.get("question_entities", []),
                "answer_entities": source_extra_info.get("answer_entities", []),
                "path": source_extra_info.get("path", []),
                "triples": source_extra_info.get("triples", []),
                "sparql_query": source_extra_info.get("sparql_query"),
            }
            tools_kwargs = {
                "get_relations": {"create_kwargs": dict(kgqa_create_kwargs)},
                "get_triples": {"create_kwargs": dict(kgqa_create_kwargs)},
            }
            problem_type = "kgqa"
            problem_ability = metadata.get("ability", "kgqa-self-play")
        else:
            tools_kwargs = {
                "search": {
                    "create_kwargs": {
                        "data_source": "dummy",
                        "question": dummy_question,
                        "ground_truth": dummy_reward_model.get("ground_truth"),
                    }
                }
            }
            problem_type = "search"
            problem_ability = "fact-reasoning"

        dummy_extra_info.update(
            {
                "question": dummy_question,
                "need_tools_kwargs": True,
                "split": dummy_extra_info.get("split", "train"),
                "tools_kwargs": tools_kwargs,
                "extraction_failed": True,
            }
        )

        dummy_problem = {
            "data_source": "dummy",
            "prompt": dummy_prompt,
            "ability": problem_ability,
            "reward_model": dummy_reward_model,
            "extra_info": dummy_extra_info,
            "metadata": None,
            "extracted_question": dummy_question,
            "formatted_prompt": dummy_prompt,
            "problem_type": problem_type,
            "trajectory_index": trajectory_index,
        }

        if SELF_PLAY_DEBUG:
            logger.debug(f"Created dummy problem for failed extraction at trajectory {trajectory_index}")

        return dummy_problem

    def _validate_extracted_problem(self, problem: Dict[str, Any]) -> bool:
        """
        问题验证：
        1. 问题长度大于10：问题无效
        2. 问题答案出现在问题中：问题无效
        """
        question = problem["extracted_question"]
        if len(question.strip()) < 10:
            return False

        answer = problem.get("reward_model", {}).get("ground_truth", {}).get("target", None)
        if answer is not None:
            # Handle both string and list answers
            if isinstance(answer, list):
                # Check if any answer in the list is in the question
                for ans in answer:
                    if isinstance(ans, str) and ans in question:
                        return False
            elif isinstance(answer, str) and answer in question:
                return False

        return True

    def _balance_batch(self, batch: DataProto, metrics):
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size

        from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix="global_seqlen"
        )
        metrics.update(global_balance_stats)
