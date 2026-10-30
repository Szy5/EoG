# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os

import numpy as np
import torch
import torch.distributed as dist
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tensordict import TensorDict

try:
    from sglang.srt.utils import TorchPatchMultiprocessingSerializer as MultiprocessingSerializer
except ImportError:
    from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.utils.torch_functional import check_device_is_available
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

from .base import BaseShardingManager

# from vllm.distributed import parallel_state as sglang_ps
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ROLLOUT_DEBUG_SYNC = os.getenv("VERL_ROLLOUT_DEBUG_SYNC", "0") == "1"


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    if tensor.device.type == "cpu":
        return tensor.to(get_device_id(), non_blocking=True)
    return tensor


class FSDPSGLangShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        rollout_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        multi_stage_wake_up: bool = False,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.multi_stage_wake_up = multi_stage_wake_up

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            )
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()
        self.cpu_data_gather = bool(self.rollout_config.get("cpu_data_gather", False))
        self.cpu_infer_tp_group = None
        if self.tp_size > 1 and self.cpu_data_gather:
            if not dist.is_gloo_available():
                logger.warning("rollout.cpu_data_gather=True but Gloo is unavailable; falling back to NCCL gather")
            else:
                tp_ranks = [int(rank) for rank in self.device_mesh["infer_tp"].mesh.reshape(-1).tolist()]
                self.cpu_infer_tp_group = dist.new_group(ranks=tp_ranks, backend="gloo")
                logger.warning("Using Gloo CPU gather for SGLang TP input data: ranks=%s", tp_ranks)

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None
    import asyncio

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            try:
                # Py3.7+：如果当前线程已有 running loop，会直接拿到
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # 没有 running loop：尝试拿当前 thread 的默认 loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # Py3.12/3.13 常见：当前线程根本没有默认 loop，需要显式创建
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            loop.run_until_complete(self.wake_up())
    # @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    # def __enter__(self):
    #     self.timing = {}
    #     with simple_timer("reshard", self.timing):
    #         loop = asyncio.get_event_loop()
    #         loop.run_until_complete(self.wake_up())

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.sleep())

    async def update_weights(self, params):
        # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
        named_tensors = [(k, v) for k, v in params.items()]
        load_format = None
        # convert megabytes to bytes
        update_weights_bucket_bytes = int(self.rollout_config.update_weights_bucket_megabytes) << 20
        for batch in get_named_tensor_buckets(named_tensors, update_weights_bucket_bytes):
            # On each rank, serialize a batch of (name, tensor) tuples.
            # named_tensors_batch will be a list like:
            # [(name0, serialized_tensor0_tp0), (name1, serialized_tensor1_tp0), ...]
            named_tensors_batch = [
                (name, MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor)))
                for name, tensor in batch
            ]

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                # On rank 0, prepare a list to hold the gathered batches from all ranks.
                gathered_serialized_batches = [None for _ in range(self.device_mesh["infer_tp"].mesh.size()[0])]
            else:
                gathered_serialized_batches = None

            # Gather the named_tensors_batch from all ranks to rank 0.
            # After this, on rank 0, gathered_serialized_batches will be a list of lists:
            # [ [ (name0, s_t0_tp0), (name1, s_t1_tp0), ... ],  # batch from TP rank 0
            #   [ (name0, s_t0_tp1), (name1, s_t1_tp1), ... ],  # batch from TP rank 1
            #   ... ]
            # On other ranks, gathered_serialized_batches will be None.
            dist.gather_object(
                obj=named_tensors_batch,
                object_gather_list=gathered_serialized_batches,
                dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
                group=self.device_mesh["infer_tp"].get_group(),
            )

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                # Use zip(*) to "transpose" the data structure.
                # This groups the serialized parts for each individual tensor across all TP ranks.
                # Example: from [[(n0, t0_tp0), (n1, t1_tp0)], [(n0, t0_tp1), (n1, t1_tp1)]]
                # to [ ( (n0, t0_tp0), (n0, t0_tp1) ), ( (n1, t1_tp0), (n1, t1_tp1) ) ]
                logical_tensors = zip(*gathered_serialized_batches, strict=True)

                await self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        # 'tensor_group' represents a single logical tensor's data from all ranks.
                        (
                            tensor_group[0][0],  # Get the name from the first rank's data.
                            LocalSerializedTensor(
                                # 'rank_part' is the (name, serialized_tensor) tuple from one specific rank.
                                values=[rank_part[1] for rank_part in tensor_group]
                            ),
                        )
                        for tensor_group in logical_tensors
                        # each tensor_group is like ( (n0, t0_tp0), (n0, t0_tp1) )
                    ],
                    load_format=load_format,
                    flush_cache=False,
                )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self.inference_engine.flush_cache()

        # Fence: ensure all TP ranks have finished weight update before any
        # rank proceeds to generation.  Without this barrier the engine on
        # tp-rank 0 could start scheduling prefills while non-zero ranks are
        # still inside gather_object, leading to stale weight reads.
        get_torch_device().synchronize()
        dist.barrier(group=self.device_mesh["infer_tp"].get_group())

    async def release_memory(self):
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            if self.multi_stage_wake_up:
                await self.inference_engine.release_memory_occupation(tags=["kv_cache", "weights"])
            else:
                await self.inference_engine.release_memory_occupation()
            log_gpu_memory_usage("After release memory occupation in sharding manager", logger=logger)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        get_torch_device().empty_cache()

        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            if self.multi_stage_wake_up:
                await self.inference_engine.resume_memory_occupation(tags=["weights"])
                log_gpu_memory_usage("Before resume SGLang weights in sharding manager", logger=logger)
            else:
                await self.inference_engine.resume_memory_occupation()
                log_gpu_memory_usage("Before resume SGLang weights + kv_cache in sharding manager", logger=logger)

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.rollout_config.get("skip_weight_sync", False):
            logger.warning("Skipping FSDP -> SGLang weight sync because rollout.skip_weight_sync=True")
            dist.barrier(group=self.device_mesh["infer_tp"].get_group())
            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)
            return

        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        params = {
            k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()
        }

        # Fence: non_blocking=True copies above may not have completed on the
        # current CUDA stream. Synchronize before serializing tensors for the
        # SGLang weight update so that gather/serialize reads fully-written data.
        get_torch_device().synchronize()
        if _ROLLOUT_DEBUG_SYNC:
            logger.warning("[debug-sync] weight .to(device) fence passed – rank %s", dist.get_rank())

        # convert weight keys to match the model config
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        # Copy, not share memory
        await self.update_weights(params)
        if _ROLLOUT_DEBUG_SYNC:
            logger.warning("[debug-sync] update_weights complete – rank %s", dist.get_rank())
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

        del params
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        if (
            self.multi_stage_wake_up
            and self.rollout_config.free_cache_engine
            and self.device_mesh["infer_tp"].get_local_rank() == 0
        ):
            await self.inference_engine.resume_memory_occupation(tags=["kv_cache"])
            log_gpu_memory_usage("After resume SGLang kv_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        if self.rollout_config.free_cache_engine:
            log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
            await self.release_memory()
            log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()

        if self.cpu_infer_tp_group is not None:
            self._all_gather_data_proto_via_cpu(data=data)
        else:
            all_gather_data_proto(data=data, process_group=group)
        return data

    def _all_gather_data_proto_via_cpu(self, data: DataProto) -> None:
        """Gather small rollout input batches through Gloo to avoid NCCL host-pinned allocations."""
        group_size = dist.get_world_size(group=self.cpu_infer_tp_group)
        prev_device = data.batch.device
        batch_cpu = {key: value.detach().cpu() for key, value in data.batch.contiguous().to_dict().items()}

        gathered_batches = [None for _ in range(group_size)]
        dist.all_gather_object(gathered_batches, batch_cpu, group=self.cpu_infer_tp_group)

        gathered_tensors = {
            key: torch.cat([batch[key] for batch in gathered_batches], dim=0).to(prev_device)
            for key in batch_cpu
        }
        batch_size = next(iter(gathered_tensors.values())).shape[0] if gathered_tensors else 0
        data.batch = TensorDict(source=gathered_tensors, batch_size=[batch_size])

        gathered_non_tensors = [None for _ in range(group_size)]
        dist.all_gather_object(gathered_non_tensors, data.non_tensor_batch, group=self.cpu_infer_tp_group)
        data.non_tensor_batch = {
            key: np.concatenate([batch[key] for batch in gathered_non_tensors])
            for key in data.non_tensor_batch
        }

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
