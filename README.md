<div align="center">

<h1><img src="assets/imgs/logo.png" width="64" alt="EoG logo" align="absmiddle"> Evolve on Graph</h1>

### Self-Evolving Knowledge Graph Reasoning without Question Annotations

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2610.35598)
[![Project Page](https://img.shields.io/badge/Project-Page-4c8bf5)](https://szy5.github.io/EoG/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/Bobzong/EoG)

**English** | [简体中文](README_zh.md)

</div>

---

## 🔥 News
[2026-10-27] 📄 The EoG paper is available on [arXiv](https://arxiv.org/abs/2610.35598).

---

Most knowledge graphs (KGs) in real-world domains such as finance and industry have no question–answer (QA) annotations. **Evolve on Graph (EoG)** lets a large language model improve its multi-hop reasoning on such a KG **without any QA annotations**. A single LLM self-plays two roles on the graph: a **Proposer** that poses questions and a **Solver** that answers them through multi-turn graph exploration. The tasks, gold answers and difficulty signal all come from the graph itself.

## 🧭 KG Reasoning Paradigms

<p align="center">
  <img src="docs/static/img/related_work.png" width="100%" alt="Comparison of KG reasoning paradigms">
</p>

- **Prompt-based agents** (e.g. ToG, PoG) explore the graph with a frozen LLM. They need no annotations, but their exploration is fixed by the prompt and cannot improve through interaction.
- **RL-based agents** (e.g. KG-R1, GraphWalker) learn to explore with reinforcement learning, but train on fixed QA data written by humans or synthesized by closed-source models, with a topology mix set before training.
- **EoG** proposes and solves its own KG-verified tasks: answers come from query execution on the graph, and task difficulty follows the Solver. With Qwen2.5-7B-Instruct, it outperforms all shown baselines on three Freebase benchmarks.

## 🧠 Method

<p align="center">
  <img src="docs/static/img/main_methods.png" width="100%" alt="EoG method">
</p>

EoG has three components:

- **Graph Self-Play.** The Proposer reads a reasoning skeleton in which only the anchor entities are real (intermediate and answer entities are placeholders such as `Mid1@` / `Ans@`), explores the graph from the anchors with `get_relations` / `get_triples`, and writes a natural-language question. The Solver answers it by exploring the same graph and is trained with GRPO on an answer-F1 plus format reward.
- **Symbolically Verifiable Topology Synthesis (SVTS).** Six query patterns (`1p`, `2p`, `3p`, `2i`, `ip`, `pi`) are grounded on real facts, compiled into executable queries and run on the KG, giving each task its complete gold answer set.
- **Boundary-Aware Graph Curriculum.** The Proposer is trained with REINFORCE on a Gaussian reward over the variance of the Solver's scores, which peaks at 1/4, where the Solver's GRPO learning signal is largest.

## 🏆 Main Results

Hit@1 / F1 (%) on five public KGQA benchmarks. All methods except EoG are annotation-free baselines; Graph Reranker is the strongest of them on average. The full table, including prompting, agent and fully supervised baselines, is in the paper.

| Method | LLM | WebQSP | CWQ | GrailQA | GTSQA | WikiWebQuestions | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph Reranker | Qwen2.5-7B | 68.27 / 48.89 | 37.69 / 33.28 | 56.28 / 48.99 | 37.85 / 36.65 | 70.37 / 53.48 | 54.09 / 44.26 |
| EoG | Qwen2.5-3B | 63.57 / 55.22 | 38.14 / 36.84 | 56.17 / 55.05 | 30.95 / 28.76 | 41.26 / 34.30 | 46.02 / 42.03 |
| **EoG** | **Qwen2.5-7B** | 70.13 / **62.10** | **50.58 / 49.27** | **65.38 / 63.01** | **48.39 / 46.36** | **73.28** / 60.82 | **61.55 / 56.31** |
| EoG | Llama-3.1-8B | **71.82** / 61.51 | 50.36 / 48.25 | 63.87 / 62.59 | 36.37 / 34.67 | 70.53 / **62.55** | 58.59 / 53.91 |

With Qwen2.5-7B-Instruct, EoG exceeds the strongest annotation-free baseline by **7.46 Hit@1** on average. On IndustryCarBench, a 500-question benchmark over an industrial KG that has no training set, it exceeds the strongest baseline (ToG) by **17.80 Hit@1**.

## ⚙️ Setup

### Requirements

- Linux with NVIDIA GPUs.
- Python with CUDA-enabled PyTorch (the reference environment uses Python 3.13, CUDA 12.8 and PyTorch 2.9).
- A Freebase SPARQL endpoint (see [Freebase](#freebase)).

### Python environment

```bash
git clone https://github.com/Szy5/EoG.git
cd EoG

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the PyTorch build matching your CUDA version first.
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.9.*" torchvision torchaudio

pip install -r training/requirements.txt
```

The training stack (veRL with SGLang multi-turn rollout) follows [Search Self-Play (SSP)](https://github.com/Alibaba-Quark/SSP).

### Freebase

Set up Virtuoso with the Freebase dump following [ChatKBQA](https://github.com/LHRLAB/ChatKBQA) and start the SPARQL service. Check that the endpoint answers:

```bash
python scripts/check_freebase_endpoint.py \
  --endpoint http://127.0.0.1:3001/sparql \
  --query 'SELECT ?p ?o WHERE { <http://rdf.freebase.com/ns/m.0hqs1x_> ?p ?o . } LIMIT 1' \
  --require-bindings
```

### Check the installation

These CPU-only tests need no GPU or KG:

```bash
PYTHONPATH=.:training:training/verl python -m pytest \
  tests/kg_ssp/unit/test_data_contract.py \
  tests/kg_ssp/unit/test_webqsp_cwq_val_convert.py \
  tests/kg_platform/test_config_cli.py
```

## ⚡ Quick Start (Freebase)

### 1. Get the SVTS tasks

Put the Freebase tasks at `data/svts/freebase_tasks.jsonl`. Each line is one task, a grounded query graph with its executed answer set, and contains no natural-language question:

```json
{"qid": 600001000161, "task_id": "iso-1_00161", "template_id": "Iso-1",
 "question_entity": [{"name": "Jenna Dewan", "id": "m.0ck1l0"}],
 "answer_entity": [{"name": "London", "id": "m.04jpl"}],
 "path": ["Jenna Dewan --people.person.children--> Mid1@ --location.location.people_born_here--> Ans@"],
 "triples": [["Jenna Dewan", "people.person.children", "entity_1"],
             ["entity_1", "location.location.people_born_here", "Ans"]],
 "num_edge": 2, "domains": "people",
 "sparql_query": "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?ans WHERE { ... }"}
```

### 2. Build the Proposer seed data

```bash
INPUT_JSONL=data/svts/freebase_tasks.jsonl \
OUTPUT_DIR=data/train \
  bash scripts/prepare_kgqa_data.sh
```

This writes `data/train/svts_freebase.{jsonl,parquet}` with `data_source=kgqa_svts_freebase`. These are the default `GENERATION_DATA_PATH` and `TRAIN_PARQUET`.

### 3. Build the validation set

Validation runs the Solver alone on WebQSP / CWQ test questions. Convert the raw test files into Solver-format Parquet:

```bash
KGQA_SPARQL_ENDPOINT=http://127.0.0.1:3001/sparql \
python training/examples/data_preprocess/webqsp_cwq_val/convert.py \
  --webqsp-path data/raw_datasets/WebQSP.test.json \
  --cwq-path data/raw_datasets/ComplexWebQuestions_test.question_entities.jsonl \
  --webqsp-output-parquet data/val/val_webqsp.parquet \
  --cwq-output-parquet data/val/val_cwq.parquet
```

`data/val/val_webqsp.parquet` is the default `VAL_PARQUET`.

### 4. Check the resolved configuration

```bash
KGQA_CONFIG_ONLY=1 \
FREEBASE_SPARQL_ENDPOINT=http://127.0.0.1:3001/sparql \
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
  bash scripts/platforms/local.sh
```

With `KGQA_PREFLIGHT_ONLY=1` instead, the launcher also checks every input file, probes the endpoint and materializes the KG profile, then exits before starting Ray.

### 5. Train

**One machine** (no Slurm, current Python environment, your running Freebase endpoint and verifier service):

```bash
FREEBASE_SPARQL_ENDPOINT=http://127.0.0.1:3001/sparql \
JUDGE_BASE_URL=http://<verifier_host>:5000/v1 \
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
KGQA_EXPERIMENT_CONFIG="$PWD/scripts/experiments/baseline.env" \
  bash scripts/platforms/local.sh
```

The number of GPUs is detected from `nvidia-smi` (override with `NUM_GPUS_PER_NODE`). Rollout tensor parallelism defaults to `ROLLOUT_TP=4`.

Outputs (checkpoints, rollouts, validation dumps, TensorBoard logs and a run summary) go to `OUTPUT_ROOT/<run_id>/`. All experiment variables are listed in [`scripts/experiments/README.md`](scripts/experiments/README.md). Other backbones and KGs use the other files there, for example `baseline_model_llama31_8b_instruct.env` or `gtsqa_v9.env` (Wikidata, no Freebase needed).

## 🎯 Evaluation

**Freebase benchmarks (WebQSP, CWQ, GrailQA).** Run the Solver of a checkpoint in validation-only mode:

```bash
VAL_BEFORE_TRAIN=True VAL_ONLY=True PROPOSER_RAG_FILTER=False \
MODEL_PATH=/path/to/run/ckpt/global_step_N/actor/huggingface \
VAL_PARQUET=data/val/val_cwq.parquet \
FREEBASE_SPARQL_ENDPOINT=http://127.0.0.1:3001/sparql \
  bash scripts/platforms/local.sh
```

WebQSP and CWQ Parquet files come from step 3. Per-question predictions are written to `OUTPUT_ROOT/<run_id>/val/`, and Hit@1 / F1 to the console and TensorBoard.

## 🏗️ Repository Layout

```text
EoG/
├── configs/                  # KG profiles and evaluation dataset bindings
├── docs/                     # Project page
├── kg_platform/              # Unified KG backends (Freebase SPARQL, Wikidata/car mmap) and tools
├── scripts/
│   ├── platforms/            # Entry points: local.sh, roihu_sbatch.sh, ...
│   ├── experiments/          # Experiment .env files
│   ├── train_eog.sh          # Shared training runtime
│   └── prepare_kgqa_data.sh  # SVTS tasks -> Proposer seed data
├── training/
│   ├── quarl/                # Self-play trainer, rewards, tools and prompts
│   └── verl/                 # Vendored veRL
└── tests/
```

## 🙏 Acknowledgements

EoG builds on [veRL](https://github.com/volcengine/verl), [SGLang](https://github.com/sgl-project/sglang), [Search Self-Play](https://github.com/Alibaba-Quark/SSP) and [ChatKBQA](https://github.com/LHRLAB/ChatKBQA) (Freebase setup). We thank their authors.

## 📚 Citation

```bibtex
@article{sun2026evolve,
  title   = {Evolve on Graph: Self-Evolving Knowledge Graph Reasoning without Question Annotations},
  author  = {Sun, Zongyuan and Zhao, Shuai and Shu, Ke and Zhu, Ziyue and Wu, Shangyang and Lin, Yuanlin and Wang, Tao and Li, Shiyue and Du, Yingjun and Luu, Anh Tuan and Luo, Haoran},
  journal = {arXiv preprint arXiv:2610.35598},
  year    = {2026}
}
```

## 📄 License

The project license will be added before the public release. The vendored components under `training/verl/` keep their original licenses.
