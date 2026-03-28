# Reproducing SkillGraph Experiments

This document summarizes how to rebuild the main experiments from this repository.

## 1. Environment

Recommended:

- Python 3.10+
- PyTorch 2.x
- CUDA optional but helpful for GNN-related scripts

Install dependencies:

```bash
cd skillgraph
pip install -r requirements.txt
```

## 2. Data

The GitHub repo excludes raw and processed datasets because of size and redistribution concerns.

Expected local layout:

```text
skillgraph/data/
├── raw/
│   ├── toolllama_G123_dfs_train.json
│   ├── toolllama_G123_dfs_eval.json
│   └── API-Bank/
└── processed/
```

Relevant sources:

- ToolBench / ToolLLM trajectories
- API-Bank level-3 benchmark

The helper script `skillgraph/data/download.sh` is a starting point, but in practice you may still need to place some files manually depending on access method and local environment.

## 3. Core Experiment Commands

Run from `skillgraph/`.

### Main ToolBench comparison

```bash
python src/final_comparison.py
```

Main outputs:

- `results/final_comparison.csv`
- `results/final_by_length.csv`
- `results/bootstrap_significance.csv`

### API-Bank cross-dataset validation

```bash
python src/run_apibank_eval.py
```

Outputs:

- `results/apibank_eval.csv`
- `results/apibank_by_length.csv`
- `results/apibank_stats.json`

### Feature ablation

```bash
python src/run_feature_ablation.py
```

### LLM Stage-2 reranker comparison

```bash
python src/run_llm_stage2.py
```

### Figure generation

```bash
python src/generate_figures.py
```

Outputs go to:

- `outputs/figures/`

## 4. Code Map

Important files in `skillgraph/src/`:

- `two_stage_pipeline.py`: integrates Stage 1 and Stage 2
- `learned_reranker.py`: pairwise MLP and inference logic
- `graph_build.py`: graph construction from trajectories
- `graph_search.py`: graph-guided retrieval utilities
- `gnn_transition.py`: GNN-based transition scoring baselines
- `metrics_clean_eval.py`: cleaned evaluation helpers

## 5. What Is Not Included in GitHub

By default, the repo ignores:

- `skillgraph/data/raw/`
- `skillgraph/data/processed/`
- `skillgraph/models/`
- `skillgraph/outputs/`

This keeps the repository lightweight and GitHub-safe. If you want a full replication bundle, package those directories separately and distribute them as an external archive.

The paper source is also intentionally kept out of the public GitHub repo until the preprint is released.
