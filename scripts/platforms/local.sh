#!/usr/bin/env bash

# Single-machine entry point using the current Python environment.
#
# Uses the current Python environment (see README "Setup") and an already
# running Freebase SPARQL endpoint, e.g. one set up following ChatKBQA:
#
#   FREEBASE_SPARQL_ENDPOINT=http://127.0.0.1:3001/sparql \
#   MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
#   TRAIN_PARQUET=... GENERATION_DATA_PATH=... VAL_PARQUET=... \
#     bash scripts/platforms/local.sh
#
# Embedded backends (KGQA_BACKEND=gtsqa / car) need no endpoint.

set -euo pipefail

PLATFORM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd -- "${PLATFORM_DIR}/../.." && pwd)}"
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"

export KGQA_LAUNCHER=local
export KGQA_CONTAINER="${KGQA_CONTAINER:-none}"
export NNODES=1
if [[ -z "${NUM_GPUS_PER_NODE:-}" ]]; then
  NUM_GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  [[ "${NUM_GPUS_PER_NODE}" -gt 0 ]] || NUM_GPUS_PER_NODE=8
fi
export NUM_GPUS_PER_NODE
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-$(nproc)}"
export BASH_BIN="${BASH_BIN:-$(command -v bash)}"
export HOSTNAME_BIN="${HOSTNAME_BIN:-$(command -v hostname)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/eog}"
export EXPERIMENT_CONFIG="${KGQA_EXPERIMENT_CONFIG:-${REPO_ROOT}/scripts/experiments/baseline/.env.example}"

exec "${BASH_BIN}" "${REPO_ROOT}/scripts/train_eog.sh" "$@"
