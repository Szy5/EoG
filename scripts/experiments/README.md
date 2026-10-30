# Experiment configurations

Each `.env` file only sets experiment variables. Launch logic (Slurm, Ray,
Virtuoso, containers) lives in `scripts/train_eog.sh` and is selected by a
platform entry point:

| Entry point | Use |
| --- | --- |
| `scripts/platforms/local.sh` | One machine, no Slurm, current Python env, external Freebase endpoint |
| `scripts/platforms/roihu_sbatch.sh` | Reference Slurm + Apptainer cluster (adjust `#SBATCH` lines and paths) |

Every assignment of the form `VAR="${VAR:-default}"` can be overridden from the
command line. Select a file with `KGQA_EXPERIMENT_CONFIG`.

## Training

| File | KG | Backbone | Notes |
| --- | --- | --- | --- |
| `baseline.env` | Freebase | Qwen2.5-7B-Instruct | Full EoG (Proposer + Solver updates); default config |
| `baseline_model_qwen25_7b.env` | Freebase | Qwen2.5-7B-Instruct | Same as baseline with an explicit `MODEL_PATH` slot |
| `baseline_model_llama31_8b_instruct.env` | Freebase | Llama-3.1-8B-Instruct | `MODEL_PATH` must be set |
| `gtsqa_v9.env`, `gtsqa_v9_qwen25_3b.env`, `gtsqa_v9_llama31_8b.env` | Wikidata (GTSQA index) | 7B / 3B / 8B | Embedded mmap backend, no Freebase |
| `car_v1.env` | Industrial car KG | Qwen2.5-7B-Instruct | Embedded mmap backend, no Freebase |
| `recovery_v1.env` | Freebase | — | Alias of `baseline.env` kept for old entry points |

Role ablations reuse any training file. `PROPOSER_ENABLE` / `SOLVER_ENABLE`
switch parameter updates, not rollouts: Proposer-only still needs Solver
rollouts for the Proposer reward, and Solver-only still needs Proposer rollouts
to produce questions.

```bash
# Solver-only ablation on one machine
PROPOSER_ENABLE=False \
FREEBASE_SPARQL_ENDPOINT=http://127.0.0.1:3001/sparql \
KGQA_EXPERIMENT_CONFIG="$PWD/scripts/experiments/baseline.env" \
  bash scripts/platforms/local.sh
```

Expand the resolved configuration without starting anything:

```bash
KGQA_CONFIG_ONLY=1 \
KGQA_EXPERIMENT_CONFIG="$PWD/scripts/experiments/baseline.env" \
  bash scripts/platforms/local.sh
```

`KGQA_PREFLIGHT_ONLY=1` additionally checks the inputs, materializes the KG
profile and validates the backend, then exits before Ray and training start.

## Evaluation (open models)

`val_open_model_roihu.env` and the six `val_<model>_<dataset>_roihu.env` files
configure Solver-only evaluation of a local model served by SGLang on GTSQA or
WikiWebQuestions. They do not start or query Freebase.

```bash
KGQA_VALIDATION_CONFIG="$PWD/scripts/experiments/val_qwen25_7b_gtsqa_roihu.env" \
LIMIT=1 \
  sbatch scripts/platforms/roihu_val_open_model_sbatch.sh

# all six model x dataset combinations as separate jobs
SUBMIT_DRY_RUN=1 bash scripts/submit_open_model_validation_roihu.sh
```

Outputs go to `output/kgqa_validation/<model>/<dataset>/` (per-model files) or
`output/open_model_kgqa/` (generic file). `#SBATCH` resources are read before
the script runs, so change them in the submitted file, not in the `.env`.
