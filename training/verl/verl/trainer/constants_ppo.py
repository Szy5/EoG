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

import os

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    },
}

PPO_RAY_PASSTHROUGH_ENV_KEYS = (
    "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "TRITON_NUM_COMPILER_WORKERS",
)


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    Prefer explicit passthrough of the current process environment so Ray workers
    inherit launcher-side thread caps instead of falling back to broad defaults.
    """
    runtime_env = {"env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy()}
    for key in PPO_RAY_PASSTHROUGH_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            runtime_env["env_vars"][key] = value
    return runtime_env
