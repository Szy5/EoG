#!/usr/bin/env bash
# Convert SVTS task JSONL into the Proposer seed JSONL + Parquet consumed by
# the self-play trainer (TRAIN_PARQUET / GENERATION_DATA_PATH).
#
#   INPUT_JSONL=data/svts/freebase_tasks.jsonl \
#   OUTPUT_DIR=data/train \
#     bash scripts/prepare_kgqa_data.sh


set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
INPUT_JSONL=${INPUT_JSONL:-"${REPO_ROOT}/data/svts/freebase_tasks.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/data/train"}
OUTPUT_NAME=${OUTPUT_NAME:-svts_freebase}
OUTPUT_JSONL="${OUTPUT_DIR}/${OUTPUT_NAME}.jsonl"
OUTPUT_PARQUET="${OUTPUT_DIR}/${OUTPUT_NAME}.parquet"
# data_source names the KG the tasks were synthesized on (kgqa_svts_<kg>).
DATA_SOURCE=${DATA_SOURCE:-kgqa_svts_freebase}
DATASET_TYPE=${DATASET_TYPE:-kgqa}
# Small smoke inputs are repeated up to this many rows so one batch can be formed.
MIN_SAMPLES=${MIN_SAMPLES:-8}

mkdir -p "${OUTPUT_DIR}"

python "${REPO_ROOT}/training/examples/data_preprocess/convert_svts_to_kg_ssp.py" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --output-parquet "${OUTPUT_PARQUET}" \
  --data-source "${DATA_SOURCE}" \
  --dataset-type "${DATASET_TYPE}" \
  --sys-template "${REPO_ROOT}/training/quarl/template/kgqa_ssp_proposer_sys.jinja" \
  --user-template "${REPO_ROOT}/training/quarl/template/kgqa_ssp_proposer_user.jinja"

python - "${OUTPUT_JSONL}" "${OUTPUT_PARQUET}" "${MIN_SAMPLES}" <<'PY'
import itertools
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq

jsonl_path, parquet_path, min_samples_raw = sys.argv[1:4]
min_samples = int(min_samples_raw)

with open(jsonl_path, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

if not rows:
    raise SystemExit("Converted JSONL is empty; check INPUT_JSONL.")

if len(rows) < min_samples:
    expanded_rows = list(itertools.islice(itertools.cycle(rows), min_samples))
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in expanded_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    table = pq.read_table(parquet_path)
    if table.num_rows == 0:
        raise SystemExit("Converted parquet is empty; cannot expand dataset.")

    repeat_times = (min_samples + table.num_rows - 1) // table.num_rows
    expanded_table = pa.concat_tables([table] * repeat_times).slice(0, min_samples)
    pq.write_table(expanded_table, parquet_path)
PY

echo "Train JSONL  (GENERATION_DATA_PATH): ${OUTPUT_JSONL}"
echo "Train Parquet (TRAIN_PARQUET):       ${OUTPUT_PARQUET}"
