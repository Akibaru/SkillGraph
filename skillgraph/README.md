# SkillGraph Codebase

This directory contains the implementation used for the paper:

**SkillGraph: Graph Foundation Priors for LLM Agent Tool Sequence Recommendation**

## Scope

The code covers:

- SkillGraph construction from LLM agent trajectories
- Stage-1 Graph-Semantic Hybrid Retrieval (GS-Hybrid)
- Stage-2 learned pairwise reranking
- ToolBench evaluation
- API-Bank cross-dataset validation
- ablation, LLM baseline, and figure-generation scripts

## Directory Layout

```text
skillgraph/
├── data/
│   ├── download.sh
│   ├── raw/                # local-only, ignored by git
│   └── processed/          # local-only, ignored by git
├── models/                 # local-only checkpoints, ignored by git
├── outputs/                # generated plots / reports, ignored by git
├── results/                # CSV/TXT experiment summaries kept in repo
├── src/
│   ├── final_comparison.py
│   ├── run_apibank_eval.py
│   ├── learned_reranker.py
│   ├── two_stage_pipeline.py
│   ├── run_feature_ablation.py
│   ├── run_llm_stage2.py
│   └── generate_figures.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Common Commands

### ToolBench main comparison

```bash
python src/final_comparison.py
```

### API-Bank evaluation

```bash
python src/run_apibank_eval.py
```

### Feature ablation

```bash
python src/run_feature_ablation.py
```

### Generate paper figures

```bash
python src/generate_figures.py
```

## Data Note

Raw and processed datasets are intentionally excluded from GitHub because they are large. Put local copies under:

- `data/raw/`
- `data/processed/`

## Paper Source

The paper itself lives in:

- `../Paper/`

For a higher-level repository overview and reproduction notes, see:

- `../README.md`
- `../docs/REPRODUCE.md`
