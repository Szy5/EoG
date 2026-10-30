#!/usr/bin/env bash
# Serve the question verifier used by PROPOSER_RAG_FILTER: an OpenAI-compatible
# SGLang server for Qwen2.5-32B-Instruct. The trainer reaches it through
# JUDGE_BASE_URL / JUDGE_MODEL (defaults: http://127.0.0.1:5000/v1,
# Qwen2.5-32B-Instruct).
#
# Run it on GPUs that training does not use (another node, or a disjoint
# CUDA_VISIBLE_DEVICES set), e.g.
#
#   CUDA_VISIBLE_DEVICES=4,5,6,7 \
#   VERIFIER_MODEL_PATH=/path/to/Qwen2.5-32B-Instruct \
#     bash scripts/start_verifier_server.sh
#
# then point the trainer at it from the training host:
#
#   JUDGE_BASE_URL=http://<verifier_host>:5000/v1

set -euo pipefail

VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:?Set VERIFIER_MODEL_PATH to a local Qwen2.5-32B-Instruct checkpoint}"
VERIFIER_SERVED_NAME="${VERIFIER_SERVED_NAME:-Qwen2.5-32B-Instruct}"
VERIFIER_HOST="${VERIFIER_HOST:-0.0.0.0}"
VERIFIER_PORT="${VERIFIER_PORT:-5000}"
VERIFIER_TP="${VERIFIER_TP:-4}"
VERIFIER_MEM_FRACTION="${VERIFIER_MEM_FRACTION:-0.85}"
VERIFIER_CONTEXT_LENGTH="${VERIFIER_CONTEXT_LENGTH:-16384}"

exec python3 -m sglang.launch_server \
  --model-path "${VERIFIER_MODEL_PATH}" \
  --served-model-name "${VERIFIER_SERVED_NAME}" \
  --host "${VERIFIER_HOST}" \
  --port "${VERIFIER_PORT}" \
  --tp-size "${VERIFIER_TP}" \
  --mem-fraction-static "${VERIFIER_MEM_FRACTION}" \
  --context-length "${VERIFIER_CONTEXT_LENGTH}" \
  --trust-remote-code \
  "$@"
