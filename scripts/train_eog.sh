#!/usr/bin/env bash

# Local EoG training runtime. Entry point: scripts/platforms/local.sh.

set -euo pipefail

KGQA_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

best_effort_ulimit() {
  local limit_name="$1"
  local target_value="$2"
  if ! ulimit "${limit_name}" "${target_value}" 2>/dev/null; then
    echo "[WARN] Unable to set ulimit ${limit_name}=${target_value}; continuing with system default." >&2
  fi
}

best_effort_ulimit -n 65535
best_effort_ulimit -u 1048576


unset CC CXX

export TRITON_NUM_COMPILER_WORKERS="${TRITON_NUM_COMPILER_WORKERS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# 设置为False
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"


export ENABLE_LLM_RELATION_FILTER="${ENABLE_LLM_RELATION_FILTER:-TRUE}"

REPO_ROOT="${REPO_ROOT:-$(cd -- "${KGQA_SCRIPT_DIR}/.." && pwd)}"
# Root for local model files and caches.
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"


ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
if [[ ! -f "${ENV_FILE}" && -f "${REPO_ROOT}/training/.env" ]]; then
  ENV_FILE="${REPO_ROOT}/training/.env"
fi
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${KGQA_SCRIPT_DIR}/experiments/baseline/.env.example}"
# Resolve README-era profile names to the published example templates.
# Limit fallback to known configs in this repository; never replace a missing
# user-supplied private config with a different experiment silently.
if [[ ! -f "${EXPERIMENT_CONFIG}" && "${EXPERIMENT_CONFIG}" == "${KGQA_SCRIPT_DIR}/experiments/"*.env ]]; then
  CONFIG_NAME="$(basename "${EXPERIMENT_CONFIG}" .env)"
  case "${CONFIG_NAME}" in
    baseline|baseline_model_qwen25_7b|baseline_model_llama31_8b_instruct|gtsqa_v9|gtsqa_v9_qwen25_3b|gtsqa_v9_llama31_8b|car_v1)
      EXPERIMENT_CONFIG="${KGQA_SCRIPT_DIR}/experiments/${CONFIG_NAME}/.env.example"
      ;;
  esac
fi
if [[ ! -f "${EXPERIMENT_CONFIG}" ]]; then
  echo "[ERROR] Missing experiment config: ${EXPERIMENT_CONFIG}" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "${EXPERIMENT_CONFIG}"
set +a
export EXPERIMENT_CONFIG

if [[ -z "${ENABLE_LLM_RELATION_FILTER:-}" ]]; then
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    ENABLE_LLM_RELATION_FILTER=True
  else
    ENABLE_LLM_RELATION_FILTER=False
  fi
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_MODEL="${OPENAI_MODEL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export ENABLE_LLM_RELATION_FILTER

# Question verifier (PROPOSER_RAG_FILTER): an OpenAI-compatible endpoint that
# answers each Proposer question from the Proposer's retrieved triples only.
# The reference setup serves Qwen2.5-32B-Instruct locally with SGLang; see
# scripts/start_verifier_server.sh. Local servers ignore the API key.
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:5000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen2.5-32B-Instruct}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"

# =========================== Knowledge graph / model / data ===========================
# 基座模型路径
if [[ "${REQUIRE_EXPLICIT_MODEL_PATH:-False}" == "True" && -z "${MODEL_PATH:-}" ]]; then
  echo "[ERROR] MODEL_PATH must be filled in ${EXPERIMENT_CONFIG}" >&2
  exit 1
fi
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/Qwen2.5-7B-Instruct}"
# Multi-turn prompt boundary mode. Keep Qwen behavior explicit and use the
# Llama-native inline ReAct renderer for Llama 3.1 experiments.
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-default}"
case "${CHAT_TEMPLATE_MODE}" in
  default | qwen2p5 | graphwalker | R-Search | llama3p1 | llama3p1_react_inline) ;;
  *)
    echo "[ERROR] Unsupported CHAT_TEMPLATE_MODE: ${CHAT_TEMPLATE_MODE}" >&2
    echo "[ERROR] Expected one of: default, qwen2p5, graphwalker, R-Search, llama3p1, llama3p1_react_inline" >&2
    exit 1
    ;;
esac
# 训练集
TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/data/train/svts_freebase.parquet}"
# 验证集
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/data/val/val_webqsp.parquet}"
# self-play proposer 出题用的原始语料（从中采样 KG 路径来生成新问题）
if [[ "${REQUIRE_EXPLICIT_GENERATION_DATA_PATH:-False}" == "True" && -z "${GENERATION_DATA_PATH:-}" ]]; then
  echo "[ERROR] GENERATION_DATA_PATH must be filled in ${EXPERIMENT_CONFIG}" >&2
  exit 1
fi
GENERATION_DATA_PATH="${GENERATION_DATA_PATH:-${REPO_ROOT}/data/train/svts_freebase.jsonl}"
# KG profile and stable two-operation tool schema.
KGQA_BACKEND="${KGQA_BACKEND:-freebase}"
# Freebase requires an externally managed SPARQL endpoint.
FREEBASE_SPARQL_ENDPOINT="${FREEBASE_SPARQL_ENDPOINT:-}"
KGQA_PREFLIGHT_ONLY="${KGQA_PREFLIGHT_ONLY:-0}"
KGQA_PROFILE_PATH="${KGQA_PROFILE_PATH:-${REPO_ROOT}/configs/kg/profiles/freebase.yaml}"
BASE_TOOL_CONFIG="${BASE_TOOL_CONFIG:-${REPO_ROOT}/training/verl/examples/sglang_multiturn/config/tool_config/kg_explorer_tool_config.yaml}"
GTSQA_INDEX_PATH="${GTSQA_INDEX_PATH:-${REPO_ROOT}/data/database/kg-platform-gtsqa-index}"
CAR_KG_INDEX_PATH="${CAR_KG_INDEX_PATH:-${REPO_ROOT}/data/database/kg-platform-car-kg-index}"
case "${KGQA_BACKEND}" in
  freebase | gtsqa | car) ;;
  *)
    echo "[ERROR] KGQA_BACKEND must be freebase, gtsqa, or car, got: ${KGQA_BACKEND}" >&2
    exit 1
    ;;
esac
export KGQA_BACKEND KGQA_PREFLIGHT_ONLY KGQA_PROFILE_PATH GTSQA_INDEX_PATH CAR_KG_INDEX_PATH
FREEBASE_LIFECYCLE_ENABLED=False
if [[ "${KGQA_BACKEND}" == "freebase" && -z "${FREEBASE_SPARQL_ENDPOINT}" ]]; then
  echo "[ERROR] Set FREEBASE_SPARQL_ENDPOINT to a running Freebase endpoint." >&2
  exit 1
fi
KGQA_LAUNCHER=local
KGQA_CONTAINER=none
WORKSPACE_ROOT="${REPO_ROOT}"
# 输出根目录
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/kgqa_qwen25_7b_8gpu}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-kgqa_qwen25_7b_8gpu}"
PROJECT_NAME="${PROJECT_NAME:-kgqa_selfplay_qwen25_7b}"

# =========================== Self-Play 消融开关 ===========================
# These flags control role updates. Both rollouts still run because proposer
# reward is computed from solver performance and solver data comes from the
# proposer-generated curriculum.
PROPOSER_ENABLE="${PROPOSER_ENABLE:-True}"
SOLVER_ENABLE="${SOLVER_ENABLE:-True}"
PROPOSER_SYNTHESIS_MODE="${PROPOSER_SYNTHESIS_MODE:-kg_path}"
PROPOSER_RAG_FILTER="${PROPOSER_RAG_FILTER:-True}"
# False: a question the verifier cannot check is rejected, never accepted.
PROPOSER_RAG_FAIL_OPEN="${PROPOSER_RAG_FAIL_OPEN:-False}"
RAG_MAX_WORKERS="${RAG_MAX_WORKERS:-8}"
NOISY_RAG_MATERIALS="${NOISY_RAG_MATERIALS:-0}"

case "${PROPOSER_SYNTHESIS_MODE}" in
  kg_path | kg_explore_rag) ;;
  *)
    echo "[ERROR] Unsupported PROPOSER_SYNTHESIS_MODE: ${PROPOSER_SYNTHESIS_MODE}" >&2
    echo "[ERROR] Expected one of: kg_path, kg_explore_rag" >&2
    exit 1
    ;;
esac

if [[ "${KGQA_CONFIG_ONLY:-0}" != "1" && "${PROPOSER_RAG_FILTER}" == "True" && -z "${JUDGE_MODEL}" ]]; then
  echo "[ERROR] PROPOSER_RAG_FILTER=True requires JUDGE_MODEL (and JUDGE_BASE_URL) of the verifier service." >&2
  exit 1
fi

# =========================== 训练规模 ===========================
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-62}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
CRITIC_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
# 提示词最大长度
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1792}"
# 轨迹最大长度
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
# 最多检索的轮次
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-8}"
ROLLOUT_N="${ROLLOUT_N:-5}"
KGQA_PROTOCOL_GATE_ENABLE="${KGQA_PROTOCOL_GATE_ENABLE:-True}"
# 超长样本过滤(参考DAPO)
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-True}"
FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-1}"

# Single-machine GPU and CPU resources.
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$(nproc)}"
if [[ "${NNODES}" != "1" ]]; then
  echo "[ERROR] Local training supports NNODES=1 only, got: ${NNODES}" >&2
  exit 1
fi

ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-12288}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"

# ===== FSDP offload   =====
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
ACTOR_OFFLOAD_POLICY="${ACTOR_OFFLOAD_POLICY:-True}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
CRITIC_PARAM_OFFLOAD="${CRITIC_PARAM_OFFLOAD:-True}"
CRITIC_OPTIMIZER_OFFLOAD="${CRITIC_OPTIMIZER_OFFLOAD:-True}"
# Compilation is opt-in for portability across GPU environments.
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-False}"
USE_LEGACY_WORKER_IMPL="${USE_LEGACY_WORKER_IMPL:-auto}"
ROLLOUT_MULTI_STAGE_WAKE_UP="${ROLLOUT_MULTI_STAGE_WAKE_UP:-False}"

# ===== 验证与保存 =====
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-FALSE}"
# 只做验证不训练（用于单独评测某个 checkpoint）
VAL_ONLY="${VAL_ONLY:-False}"
SAVE_FREQ="${SAVE_FREQ:-20}"
# Save a merged Hugging Face model for validation/inference. This deliberately
# omits FSDP model shards, optimizer state, scheduler state, and RNG state.
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-['hf_model']}"
# SSP 问题池恢复基线不使用显式熵奖励；仅保留 entropy 指标用于监控
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"


# 断点续训开关：resume_path = 从 RESUME_FROM_PATH 指定的 checkpoint 继续；
# resume_path  从零开始训练请改成 disable（auto 则是自动找 default_local_dir 下最新的 checkpoint） 
RESUME_MODE="${RESUME_MODE:-disable}"

# 续训起点（RESUME_MODE=resume_path 时生效）：指向某个 global_step_* checkpoint 目录
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"


MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-1}"

# 设置每次任务的RUN_ID 和相应Artifact 的路径
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
DIAG_DIR="${RUN_DIR}/diagnostics"
CKPT_DIR="${RUN_DIR}/ckpt"
ROLLOUT_DIR="${RUN_DIR}/rollout"
VAL_DIR="${RUN_DIR}/val"
TRAJECTORY_DIR="${RUN_DIR}/trajectories"
EXTRACTION_STATS_DIR="${RUN_DIR}/extraction_stats"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
# Export artifact paths to the trainer and its local Ray workers.
export TRAJECTORY_DIR EXTRACTION_STATS_DIR TENSORBOARD_DIR
TRAIN_LOG="${RUN_DIR}/train.log"
SUMMARY_MD="${RUN_DIR}/run_summary.md"
TOOL_CONFIG_LOCAL="${RUN_DIR}/kgqa_tool_config_remote.yaml"

RAY_NODE_TMP_BASE="${RAY_NODE_TMP_BASE:-/tmp/${USER}/kgqa_7b_$$}"
export TMPDIR="${RAY_NODE_TMP_BASE}/batch/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
# Keep generated caches in the configured project directory.
export FLASHINFER_WORKSPACE_BASE="${XDG_CACHE_HOME}"
export FLASHINFER_WORKSPACE_DIR="${XDG_CACHE_HOME}/flashinfer_workspace"
export WANDB_DIR="${XDG_CACHE_HOME}/wandb"
export WANDB_CACHE_DIR="${WANDB_DIR}"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export RAY_TMPDIR="/tmp/d$$"
export RAY_CHDIR_TO_SESSION_DIR=1
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export VLLM_USAGE_SOURCE=do-not-track
export VLLM_NO_USAGE_STATS=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VERL_VLLM_DISABLE_SLEEP_MODE="${VERL_VLLM_DISABLE_SLEEP_MODE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-72000000}"
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-72000}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ "${KGQA_CONFIG_ONLY:-0}" == "1" ]]; then
  printf 'kgqa_launcher=%s\n' "${KGQA_LAUNCHER}"
  printf 'kgqa_container=%s\n' "${KGQA_CONTAINER}"
  printf 'nnodes=%s\n' "${NNODES}"
  printf 'num_gpus_per_node=%s\n' "${NUM_GPUS_PER_NODE}"
  printf 'freebase_sparql_endpoint=%s\n' "${FREEBASE_SPARQL_ENDPOINT}"
  printf 'experiment_config=%s\n' "${EXPERIMENT_CONFIG}"
  printf 'experiment_name=%s\n' "${EXPERIMENT_NAME}"
  printf 'val_before_train=%s\n' "${VAL_BEFORE_TRAIN}"
  printf 'save_freq=%s\n' "${SAVE_FREQ}"
  printf 'checkpoint_save_contents=%s\n' "${CHECKPOINT_SAVE_CONTENTS}"
  printf 'entropy_coeff=%s\n' "${ENTROPY_COEFF}"
  printf 'proposer_adv_estimator=%s\n' "${PROPOSER_ADV_ESTIMATOR}"
  printf 'proposer_baseline_key=%s\n' "${PROPOSER_BASELINE_KEY}"
  printf 'proposer_enable=%s\n' "${PROPOSER_ENABLE}"
  printf 'solver_enable=%s\n' "${SOLVER_ENABLE}"
  printf 'proposer_synthesis_mode=%s\n' "${PROPOSER_SYNTHESIS_MODE}"
  printf 'proposer_rag_filter=%s\n' "${PROPOSER_RAG_FILTER}"
  printf 'proposer_rag_fail_open=%s\n' "${PROPOSER_RAG_FAIL_OPEN}"
  printf 'judge_base_url=%s\n' "${JUDGE_BASE_URL}"
  printf 'judge_model=%s\n' "${JUDGE_MODEL}"
  printf 'rag_max_workers=%s\n' "${RAG_MAX_WORKERS}"
  printf 'kgqa_protocol_gate_enable=%s\n' "${KGQA_PROTOCOL_GATE_ENABLE}"
  printf 'model_path=%s\n' "${MODEL_PATH}"
  printf 'chat_template_mode=%s\n' "${CHAT_TEMPLATE_MODE}"
  printf 'train_parquet=%s\n' "${TRAIN_PARQUET}"
  printf 'val_parquet=%s\n' "${VAL_PARQUET}"
  printf 'generation_data_path=%s\n' "${GENERATION_DATA_PATH}"
  printf 'kgqa_backend=%s\n' "${KGQA_BACKEND}"
  printf 'kgqa_preflight_only=%s\n' "${KGQA_PREFLIGHT_ONLY}"
  printf 'kgqa_profile_path=%s\n' "${KGQA_PROFILE_PATH}"
  printf 'gtsqa_index_path=%s\n' "${GTSQA_INDEX_PATH}"
  printf 'car_kg_index_path=%s\n' "${CAR_KG_INDEX_PATH}"
  printf 'freebase_lifecycle_enabled=%s\n' "${FREEBASE_LIFECYCLE_ENABLED}"
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'output_root=%s\n' "${OUTPUT_ROOT}"
  exit 0
fi

export BASH_BIN="${BASH_BIN:-/usr/bin/bash}"
export HOSTNAME_BIN="${HOSTNAME_BIN:-/usr/bin/hostname}"
HEAD_NODE="$("${HOSTNAME_BIN}" -s)"
WORKER_NODES=()
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
SPARQL_ENDPOINT="${FREEBASE_SPARQL_ENDPOINT}"
export KGQA_SPARQL_ENDPOINT="${SPARQL_ENDPOINT}"
if [[ "${KGQA_BACKEND}" == "freebase" ]]; then
  KG_ENDPOINT_DESCRIPTION="${SPARQL_ENDPOINT}"
elif [[ "${KGQA_BACKEND}" == "gtsqa" ]]; then
  KG_ENDPOINT_DESCRIPTION="embedded:${GTSQA_INDEX_PATH}"
elif [[ "${KGQA_BACKEND}" == "car" ]]; then
  KG_ENDPOINT_DESCRIPTION="embedded:${CAR_KG_INDEX_PATH}"
else
  echo "[ERROR] Unhandled KGQA_BACKEND for endpoint description: ${KGQA_BACKEND}" >&2
  exit 1
fi
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},$("${HOSTNAME_BIN}" -s),$("${HOSTNAME_BIN}" -f),$("${HOSTNAME_BIN}")"
export NO_PROXY="${no_proxy}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
export PROJECT_ROOT REPO_ROOT RAY_NUM_CPUS

mkdir -p \
  "${REPO_ROOT}/output" \
  "${OUTPUT_ROOT}" \
  "${RUN_DIR}" \
  "${LOG_DIR}" \
  "${DIAG_DIR}" \
  "${CKPT_DIR}" \
  "${ROLLOUT_DIR}" \
  "${VAL_DIR}" \
  "${TRAJECTORY_DIR}" \
  "${EXTRACTION_STATS_DIR}" \
  "${TENSORBOARD_DIR}" \
  "${TMPDIR}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}/hub" \
  "${HF_HOME}/datasets" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${FLASHINFER_WORKSPACE_DIR}" \
  "${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer" \
  "${WANDB_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${MPLCONFIGDIR}" \
  "${RAY_TMPDIR}"

log() {
  echo "[$(date +'%F %T')] $*"
}

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 1
  fi
}

render_tool_config() {
  # Materialize a runtime profile without changing the source configuration.
  require_file "${BASE_TOOL_CONFIG}"
  require_file "${KGQA_PROFILE_PATH}"
  export KGQA_SPARQL_ENDPOINT="${SPARQL_ENDPOINT}"
  build_runtime_cmd "
      export PYTHONPATH='${WORKSPACE_ROOT}':'${WORKSPACE_ROOT}/training':'${WORKSPACE_ROOT}/training/verl':\${PYTHONPATH:-}
      export KGQA_SPARQL_ENDPOINT='${SPARQL_ENDPOINT}'
      export GTSQA_INDEX_PATH='${GTSQA_INDEX_PATH}'
      export CAR_KG_INDEX_PATH='${CAR_KG_INDEX_PATH}'
      python -m kg_platform.cli.kgctl materialize --profile '${KGQA_PROFILE_PATH}' --run-dir '${RUN_DIR}'
    "
  "${RUNTIME_CMD[@]}"
  export KGQA_RESOLVED_CONFIG="${RUN_DIR}/kg_runtime.resolved.yaml"
  cp "${BASE_TOOL_CONFIG}" "${TOOL_CONFIG_LOCAL}"
  log "[KG] profile=${KGQA_PROFILE_PATH} resolved=${KGQA_RESOLVED_CONFIG}"
}

capture_diagnostics() {
  {
    echo "timestamp=$(date +'%F %T')"
    echo "host=$("${HOSTNAME_BIN}" -f)"
    echo "master_addr=${MASTER_ADDR}"
    echo "head_node=${HEAD_NODE}"
    echo "worker_nodes=${WORKER_NODES[*]}"
    echo "kgqa_backend=${KGQA_BACKEND}"
    echo "kg_endpoint=${KG_ENDPOINT_DESCRIPTION}"
    echo "kgqa_profile_path=${KGQA_PROFILE_PATH}"
    echo "gtsqa_index_path=${GTSQA_INDEX_PATH}"
    echo "car_kg_index_path=${CAR_KG_INDEX_PATH}"
    echo "model_path=${MODEL_PATH}"
    echo "nnodes=${NNODES}"
    echo "num_gpus_per_node=${NUM_GPUS_PER_NODE}"
    echo "rollout_tp=${ROLLOUT_TP}"
  } > "${DIAG_DIR}/job_context.txt" 2>/dev/null || true


  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L > "${DIAG_DIR}/nvidia_smi_L.txt" 2>&1 || true
    nvidia-smi > "${DIAG_DIR}/nvidia_smi.txt" 2>&1 || true
  fi

}

# Run commands in the current Python environment.
build_runtime_cmd() {
  RUNTIME_CMD=("${BASH_BIN}" -c "$1")
}

cleanup() {
  local exit_code=$?
  capture_diagnostics
  exit "${exit_code}"
}
trap cleanup EXIT

probe_freebase_remote() {
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_freebase_endpoint.py" \
    --endpoint "${SPARQL_ENDPOINT}" \
    --timeout 10 \
    --query 'SELECT ?predicate ?object WHERE { <http://rdf.freebase.com/ns/m.0hqs1x_> ?predicate ?object . } LIMIT 1' \
    --require-bindings
}

log "============================================"
log "Job ID        : local"
log "Head node     : ${HEAD_NODE}"
log "Worker nodes  : ${WORKER_NODES[*]}"
log "Master addr   : ${MASTER_ADDR}"
log "KG backend    : ${KGQA_BACKEND}"
log "KG endpoint   : ${KG_ENDPOINT_DESCRIPTION}"
log "KG profile    : ${KGQA_PROFILE_PATH}"
log "Model path    : ${MODEL_PATH}"
log "Chat template : ${CHAT_TEMPLATE_MODE}"
log "NNODES        : ${NNODES}"
log "GPUs / node   : ${NUM_GPUS_PER_NODE}"
log "Rollout TP    : ${ROLLOUT_TP}"
log "Run dir       : ${RUN_DIR}"
log "Experiment cfg: ${EXPERIMENT_CONFIG}"
log "============================================"

require_file "${BASH_BIN}"
require_file "${HOSTNAME_BIN}"
require_file "${MODEL_PATH}/config.json"
require_file "${TRAIN_PARQUET}"
require_file "${VAL_PARQUET}"
require_file "${GENERATION_DATA_PATH}"
if [[ "${KGQA_BACKEND}" == "gtsqa" ]]; then
  require_file "${GTSQA_INDEX_PATH}/manifest.json"
elif [[ "${KGQA_BACKEND}" == "car" ]]; then
  require_file "${CAR_KG_INDEX_PATH}/manifest.json"
fi
render_tool_config

if [[ "${KGQA_BACKEND}" == "freebase" ]]; then
  if ! probe_freebase_remote > "${LOG_DIR}/freebase_probe.log" 2>&1; then
    log "[FREEBASE] external endpoint ${SPARQL_ENDPOINT} did not answer the probe query."
    cat "${LOG_DIR}/freebase_probe.log" || true
    exit 1
  fi
  log "[FREEBASE] external endpoint ready: ${SPARQL_ENDPOINT}"
else
  # gtsqa / car: embedded mmap backends — prepare validates the index, no Virtuoso.
  _EMBEDDED_PREPARE_LOG="${LOG_DIR}/${KGQA_BACKEND}_prepare.log"
  log "[${KGQA_BACKEND^^}] validating embedded backend on head node ${HEAD_NODE}"
  build_runtime_cmd "
      export PYTHONPATH='${WORKSPACE_ROOT}':'${WORKSPACE_ROOT}/training':'${WORKSPACE_ROOT}/training/verl':\${PYTHONPATH:-}
      export GTSQA_INDEX_PATH='${GTSQA_INDEX_PATH}'
      export CAR_KG_INDEX_PATH='${CAR_KG_INDEX_PATH}'
      python -m kg_platform.cli.kgctl prepare --config '${KGQA_RESOLVED_CONFIG}'
    "
  "${RUNTIME_CMD[@]}" > "${_EMBEDDED_PREPARE_LOG}" 2>&1
  cat "${_EMBEDDED_PREPARE_LOG}"
  log "[${KGQA_BACKEND^^}] embedded backend ready; no Freebase process was touched"
  unset _EMBEDDED_PREPARE_LOG
fi

if [[ "${PROPOSER_RAG_FILTER}" == "True" ]]; then
  if ! curl -sf --max-time 10 -H "Authorization: Bearer ${JUDGE_API_KEY}" "${JUDGE_BASE_URL%/}/models" > "${LOG_DIR}/verifier_probe.log" 2>&1; then
    log "[VERIFIER] ${JUDGE_BASE_URL} is not reachable; start it with scripts/start_verifier_server.sh or set JUDGE_BASE_URL."
    cat "${LOG_DIR}/verifier_probe.log" || true
    if [[ "${PROPOSER_RAG_FAIL_OPEN}" != "True" ]]; then
      exit 1
    fi
    log "[VERIFIER] PROPOSER_RAG_FAIL_OPEN=True: continuing; unverified questions will be accepted."
  else
    log "[VERIFIER] ${JUDGE_MODEL} ready at ${JUDGE_BASE_URL}"
  fi
fi

if [[ "${KGQA_PREFLIGHT_ONLY}" == "1" ]]; then
  log "[PREFLIGHT] ${KGQA_BACKEND} training inputs, profile materialization, and backend validation passed"
  exit 0
fi

# quarl.main_rl starts and manages its local Ray instance.
log "[TRAIN] launching local KGQA training"
TRAINER_RAY_ENV="unset RAY_ADDRESS"
TRAINER_SCRIPT="
    set -euo pipefail
    export PYTHONPATH='${WORKSPACE_ROOT}':'${WORKSPACE_ROOT}/training':'${WORKSPACE_ROOT}/training/verl':\${PYTHONPATH:-}
    export WORKSPACE='${WORKSPACE_ROOT}/training'
    ${TRAINER_RAY_ENV}
    export RAY_TMPDIR='${RAY_TMPDIR}'
    export RAY_CHDIR_TO_SESSION_DIR=1
    export KGQA_SPARQL_ENDPOINT='${SPARQL_ENDPOINT}'
    export KGQA_RESOLVED_CONFIG='${KGQA_RESOLVED_CONFIG}'
    export GTSQA_INDEX_PATH='${GTSQA_INDEX_PATH}'
    export CAR_KG_INDEX_PATH='${CAR_KG_INDEX_PATH}'
    export OPENAI_BASE_URL='${OPENAI_BASE_URL:-}'
    export OPENAI_MODEL='${OPENAI_MODEL:-}'
    export OPENAI_API_KEY='${OPENAI_API_KEY:-}'
    export JUDGE_BASE_URL='${JUDGE_BASE_URL:-}'
    export JUDGE_MODEL='${JUDGE_MODEL:-}'
    export JUDGE_API_KEY='${JUDGE_API_KEY:-}'
    export no_proxy='${no_proxy}'
    export NO_PROXY='${NO_PROXY}'
    export TMPDIR='${TMPDIR}'
    export TEMP='${TMPDIR}'
    export TMP='${TMPDIR}'
    export XDG_CACHE_HOME='${XDG_CACHE_HOME}'
    export HF_HOME='${HF_HOME}'
    export HF_HUB_CACHE='${HF_HUB_CACHE}'
    export HUGGINGFACE_HUB_CACHE='${HUGGINGFACE_HUB_CACHE}'
    export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
    export HF_DATASETS_CACHE='${HF_DATASETS_CACHE}'
    export TORCH_EXTENSIONS_DIR='${TORCH_EXTENSIONS_DIR}'
    export FLASHINFER_WORKSPACE_DIR='${FLASHINFER_WORKSPACE_DIR}'
    export FLASHINFER_WORKSPACE_BASE='${FLASHINFER_WORKSPACE_BASE}'
    export TENSORBOARD_DIR='${TENSORBOARD_DIR}'
    export EXTRACTION_STATS_DIR='${EXTRACTION_STATS_DIR}'
    export WANDB_DIR='${WANDB_DIR}'
    export WANDB_CACHE_DIR='${WANDB_CACHE_DIR}'
    export TRITON_CACHE_DIR='${TRITON_CACHE_DIR}'
    export MPLCONFIGDIR='${MPLCONFIGDIR}'
    export VLLM_USAGE_SOURCE=do-not-track
    export VLLM_NO_USAGE_STATS=1
    export VLLM_ATTENTION_BACKEND='${VLLM_ATTENTION_BACKEND}'
    export HYDRA_FULL_ERROR=1
    export TOKENIZERS_PARALLELISM='${TOKENIZERS_PARALLELISM}'
    export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'
    export LOG_LEVEL='${LOG_LEVEL:-WARNING}'
    cd '${WORKSPACE_ROOT}/training'
    python3 -c 'from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained(\"${MODEL_PATH}\", local_files_only=True); print(\"[MODEL PREFLIGHT] tokenizer=\" + type(t).__name__)'
    # ==================================================================
    # 关键超参说明：hydra 参数是用反斜杠续行的，行间不能插注释（会截断命令），
    # 所以集中写在这里。下面提到的都是写死的值，要改就直接改对应那一行。
    #
    # -- 优化器 / PPO --
    #   actor.optim.lr=1e-6           actor 学习率，7B RL 的常用量级，调大容易崩
    #   optim.lr_warmup_steps=5       前 5 步线性 warmup
    #   critic.optim.lr=1e-5          critic 学习率；GRPO 下 critic 不参与优势计算，仅占位
    #   use_dynamic_bsz=True          按 token 数动态打包 micro batch，配合 PPO_MAX_TOKEN_LEN_PER_GPU
    #   clip_ratio_high=0.285         clip-higher（DAPO）：放宽重要性比率上界，鼓励低概率 token 的探索
    #   use_kl_loss=True              把 KL 正则加进 loss
    #   kl_loss_coef=0.01             KL 强度，越大越贴近 ref 模型、越保守
    #   kl_loss_type=low_var_kl       低方差 KL 估计量
    #   ulysses_sequence_parallel_size=1  不开序列并行
    #
    # -- 算法 --
    #   algorithm.adv_estimator=grpo      组内归一化优势（组大小 = ROLLOUT_N），不需要 value model
    #                                     norm_adv_by_std_in_grpo 保持默认 True。曾试过按 Dr. GRPO
    #                                     关掉，但实测组内 score 标准差只有 0.17~0.24，关掉相当于
    #                                     把 solver 有效步长砍到 1/4.5，且无文献支持在 self-play
    #                                     里这么配 —— 要做的话应单独做消融
    #   algorithm.use_kl_in_reward=False  KL 只进 loss 不进 reward，避免双重惩罚
    #   algorithm.gamma=1.0               不折扣，整条轨迹同权
    #
    # -- 奖励（总分 = 答案 0.95 + 格式 0.05，纯规则无奖励模型）--
    #   reward_model.enable=False                 答案对错由规则函数判定
    #   kgqa_answer_em.kwargs.score_coef=0.95     答案得分权重。score 已从「至少一个实体重合」
    #                                             换成 entity F1：前者对预测集合单调非减，多列
    #                                             答案永远不掉分，是结构性可钻的奖励
    #   kgqa_format_reward.kwargs.max_score=0.05  格式分。已饱和成常数偏置（满分占比 97%），
    #                                             几乎不产生梯度，把量程让给 F1
    #
    # -- 验证与采样温度 --
    #   trainer.test_freq=20          每 20 步跑一次验证集
    #   val_kwargs.do_sample=False    验证用贪心解码（temperature=0），保证可复现
    #   self_play.solver.temperature=0.6    解题侧采样温度
    #   self_play.proposer.temperature=0.7  出题侧采样温度
    #
    # -- self-play（本项目核心逻辑）--
    #   self_play.proposer.n=1                    每条语料生成 1 个候选问题
    #   proposer.adv_estimator=grpo               本版本用 singleton GRPO 恢复 SSP 的 raw
    #                                             REINFORCE 语义：每个 uid 只有一条轨迹时，
    #                                             verl 令 mean=0/std=1，因此 advantage=原始奖励
    #   proposer.baseline_key=uid                 不再按 template_id 跨题中心化；抽取失败的
    #                                             reward=0 因而保持 advantage=0，不奖也不罚
    #   proposer.reward_type=gaussian_score_variance
    #                                             exp(-(Var-0.25)^2/(2*tau))，即 SPICE 的 Challenger
    #                                             reward（arXiv:2510.24684，其消融里四种 proposer
    #                                             奖励中最好）。唯一改动是 Var 取【连续 entity F1】
    #                                             的组内方差而非【二值】正确性的方差 —— KGQA 答案是
    #                                             集合，二值化会把 [.5,.5,.5,.5,1] 判成「太简单」。
    #                                             设成 gaussian_difficulty 即退回原版做消融
    #   proposer.tau=0.01                         高斯宽度，SPICE 用的就是 0.01
    #   proposer.left=0.0 / right=1.0             仅 intermediate_difficulty 用得到
    #   ⚠️ rollout.n=5 会把【二值】方差压成 3 个取值；连续方差没这个问题，但 n=8（SPICE 的配置）
    #      仍会让方差估计更稳。要不要加取决于 solver rollout 的预算
    #   extraction_gate.* 已弃用：不再核对 proposer answers vs gold。
    #      合成题 gold 沿用 seed path；题干泄漏 gold 由抽取器拒绝。
    #      额外可答性检查与 main 相同：PROPOSER_RAG_FILTER 对探索三元组做 Hit@1。
    #   proposer.warm_up_steps=-1                 关闭 warmup，proposer 从第 0 步就一起训
    #   dynamic_sampling.min_valid_ratio=0.8      一个 batch 有效样本不足 80% 就重新采样
    #   dynamic_sampling.max_retry_attempts=5     最多重采 5 次。n=1 后单次采样产出减半，
    #                                             原来的 3 次在后期只够勉强凑够，会频繁触发复制补齐
    #   extraction_failure.strategy=reuse         KG 路径抽取失败时复用历史成功题目兜底
    #   self_play.save_freq=20                    每 20 步落一次 self-play 生成的数据
    # ==================================================================
    python3 -m quarl.main_rl \
      ssp_kg.task_type=ssp_kg_deep_search \
      ray_init.num_cpus=${RAY_NUM_CPUS} \
      data.train_files=\"['${TRAIN_PARQUET}']\" \
      data.val_files=\"['${VAL_PARQUET}']\" \
      data.train_batch_size=${TRAIN_BATCH_SIZE} \
      data.val_batch_size=${VAL_BATCH_SIZE} \
      data.max_prompt_length=${MAX_PROMPT_LENGTH} \
      data.max_response_length=${MAX_RESPONSE_LENGTH} \
      data.return_raw_chat=True \
      data.dataloader_num_workers=0 \
      data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS} \
      data.filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS} \
      data.truncation=left \
      data.shuffle=False \
      data.reward_fn_key=data_source \
      actor_rollout_ref.model.path='${MODEL_PATH}' \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
      actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
      actor_rollout_ref.actor.strategy=fsdp \
      actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
      actor_rollout_ref.actor.use_dynamic_bsz=True \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.use_torch_compile=${ACTOR_USE_TORCH_COMPILE} \
      actor_rollout_ref.actor.kl_loss_coef=0.01 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
      actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
      actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_OFFLOAD_POLICY} \
      actor_rollout_ref.actor.clip_ratio_high=0.285 \
      actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_SAVE_CONTENTS}" \
      actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
      actor_rollout_ref.rollout.name=sglang_async \
      actor_rollout_ref.rollout.n=${ROLLOUT_N} \
      actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
      actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS} \
      actor_rollout_ref.rollout.multi_stage_wake_up=${ROLLOUT_MULTI_STAGE_WAKE_UP} \
      actor_rollout_ref.rollout.multi_turn.enable=True \
      actor_rollout_ref.rollout.multi_turn.format=ssp_kg \
      actor_rollout_ref.rollout.multi_turn.chat_template_mode=${CHAT_TEMPLATE_MODE} \
      actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS} \
      actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
      actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
      actor_rollout_ref.rollout.multi_turn.tool_config_path='${TOOL_CONFIG_LOCAL}' \
      actor_rollout_ref.rollout.val_kwargs.n=1 \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
      actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
      actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=False \
      +actor_rollout_ref.rollout.val_kwargs.frequency_penalty=0 \
      +actor_rollout_ref.rollout.val_kwargs.repetition_penalty=1 \
      +actor_rollout_ref.rollout.repetition_penalty=1 \
      actor_rollout_ref.hybrid_engine=True \
      actor_rollout_ref.ref.strategy=fsdp \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
      actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD} \
      critic.optim.lr=1e-5 \
      critic.strategy=fsdp \
      critic.ppo_micro_batch_size_per_gpu=${CRITIC_MICRO_BATCH_SIZE_PER_GPU} \
      critic.model.path='${MODEL_PATH}' \
      critic.model.enable_gradient_checkpointing=True \
      critic.model.use_remove_padding=True \
      critic.model.fsdp_config.param_offload=${CRITIC_PARAM_OFFLOAD} \
      critic.model.fsdp_config.optimizer_offload=${CRITIC_OPTIMIZER_OFFLOAD} \
      critic.ulysses_sequence_parallel_size=1 \
      critic.checkpoint.save_contents="${CHECKPOINT_SAVE_CONTENTS}" \
      reward_model.enable=False \
      reward_model.reward_manager=ssp_kg \
      +custom_reward_functions.kgqa_answer_em.labels=\"['unknown']\" \
      +custom_reward_functions.kgqa_answer_em.integration=sum \
      +custom_reward_functions.kgqa_answer_em.kwargs.score_coef=0.95 \
      +custom_reward_functions.kgqa_format_reward.labels=\"['unknown']\" \
      +custom_reward_functions.kgqa_format_reward.integration=sum \
      +custom_reward_functions.kgqa_format_reward.kwargs.max_score=0.05 \
      +custom_reward_functions.kgqa_protocol_gate.labels="['unknown']" \
      +custom_reward_functions.kgqa_protocol_gate.integration=multiply \
      +custom_reward_functions.kgqa_protocol_gate.kwargs.enabled=${KGQA_PROTOCOL_GATE_ENABLE} \
      trainer.logger=\"['console','tensorboard']\" \
      trainer.project_name='${PROJECT_NAME}' \
      trainer.experiment_name='${EXPERIMENT_NAME}_${RUN_ID}' \
      trainer.val_before_train=${VAL_BEFORE_TRAIN} \
      trainer.val_only=${VAL_ONLY} \
      trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
      trainer.nnodes=${NNODES} \
      trainer.use_legacy_worker_impl=${USE_LEGACY_WORKER_IMPL} \
      trainer.save_freq=${SAVE_FREQ} \
      trainer.test_freq=20 \
      trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
      trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP} \
      trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
      trainer.total_epochs=${TOTAL_EPOCHS} \
      trainer.default_local_dir='${CKPT_DIR}' \
      trainer.resume_mode=${RESUME_MODE} \
      trainer.resume_from_path='${RESUME_FROM_PATH}' \
      trainer.rollout_data_dir='${ROLLOUT_DIR}' \
      trainer.validation_data_dir='${VAL_DIR}' \
      algorithm.adv_estimator=grpo \
      algorithm.use_kl_in_reward=False \
      algorithm.kl_ctrl.kl_coef=0.001 \
      algorithm.kl_penalty=low_var_kl \
      algorithm.gamma=1.0 \
      self_play.enable=True \
      self_play.generation_data_path='${GENERATION_DATA_PATH}' \
      self_play.lang=en \
      self_play.save_freq=20 \
      self_play.synthesis_mode=${PROPOSER_SYNTHESIS_MODE} \
      self_play.use_rag_filter=${PROPOSER_RAG_FILTER} \
      self_play.rag_filter_fail_open=${PROPOSER_RAG_FAIL_OPEN} \
      self_play.rag_max_workers=${RAG_MAX_WORKERS} \
      self_play.noisy_RAG_materials=${NOISY_RAG_MATERIALS} \
      self_play.use_search_terms_filter=False \
      self_play.proposer.enable=${PROPOSER_ENABLE} \
      self_play.proposer.n=1 \
      self_play.proposer.do_sample=True \
      self_play.proposer.temperature=0.7 \
      self_play.proposer.warm_up_steps=-1 \
      self_play.proposer.format_penalty=0 \
      self_play.proposer.reward_type=gaussian_score_variance \
      self_play.proposer.tau=0.01 \
      self_play.proposer.adv_estimator=${PROPOSER_ADV_ESTIMATOR:-grpo} \
      self_play.proposer.baseline_key=${PROPOSER_BASELINE_KEY:-uid} \
      self_play.proposer.left=0.0 \
      self_play.proposer.right=1.0 \
      self_play.solver.enable=${SOLVER_ENABLE} \
      self_play.solver.do_sample=True \
      self_play.solver.temperature=0.6 \
      self_play.answer_pattern=result \
      self_play.extraction_gate.entity_f1_min=0.0 \
      self_play.extraction_gate.reject_question_entity_answers=False \
      self_play.extraction_gate.reject_question_leaking_gold=False \
      self_play.dynamic_sampling.enable=False \
      self_play.dynamic_sampling.min_valid_ratio=0.8 \
      self_play.dynamic_sampling.max_retry_attempts=5 \
      self_play.reward_dynamic_sampling.enable=False \
      self_play.extraction_failure.strategy=reuse \
      self_play.extraction_failure.reuse_success_rate_threshold=1.0 \
      self_play.extraction_failure.pool_clear_interval=10 \
      self_play.extraction_failure.keep_ratio=0 \
      self_play.combine_update=False \
      self_play.mini_epochs=1 \
      self_play.validate_config=True
"
build_runtime_cmd "${TRAINER_SCRIPT}"
"${RUNTIME_CMD[@]}" 2>&1 | tee "${TRAIN_LOG}"

cat > "${SUMMARY_MD}" <<EOF
# EoG Training Run Summary

- Job ID: local
- Launcher: ${KGQA_LAUNCHER} (container: ${KGQA_CONTAINER})
- Head node: ${HEAD_NODE}
- Worker nodes: ${WORKER_NODES[*]}
- KG backend: ${KGQA_BACKEND}
- KG endpoint: ${KG_ENDPOINT_DESCRIPTION}
- KG profile: ${KGQA_PROFILE_PATH}
- Model: ${MODEL_PATH}
- Experiment config: ${EXPERIMENT_CONFIG}
- Run dir: ${RUN_DIR}
- Train log: ${TRAIN_LOG}
- Tool config: ${TOOL_CONFIG_LOCAL}
EOF
