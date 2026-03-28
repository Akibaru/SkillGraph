# SkillGraph

Code for **SkillGraph: Graph Foundation Priors for LLM Agent Tool Sequence Recommendation**.

SkillGraph models tool-order dependencies for LLM agents as a directed, weighted execution-transition graph mined from successful trajectories. On top of this graph prior, the project implements a two-stage framework:

- **Stage 1**: Graph-Semantic Hybrid Retrieval (GS-Hybrid) for candidate tool selection
- **Stage 2**: a learned pairwise reranker that orders the fixed candidate set with SkillGraph-derived features

## Highlights

- ToolBench evaluation pipeline for large-scale tool recommendation
- API-Bank cross-dataset validation
- learned reranker, ablation, and LLM-reranker comparisons
- figure generation scripts from experiment CSVs

## Repository Layout

```text
SkillGraph/
├── docs/                   # supplementary documentation
├── skillgraph/             # main codebase
│   ├── src/                # experiment scripts
│   ├── data/               # download helper + local dataset location
│   ├── results/            # experiment CSV / TXT outputs
│   ├── outputs/            # generated figures
│   ├── models/             # checkpoints / caches
│   ├── requirements.txt
│   └── README.md
├── .gitignore
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
cd skillgraph
pip install -r requirements.txt
```

### 2. Prepare data

- ToolBench trajectories should be placed under `skillgraph/data/raw/`
- API-Bank data should be placed under `skillgraph/data/raw/API-Bank/`
- `skillgraph/data/download.sh` provides a starting point for dataset download

### 3. Run the main experiments

```bash
cd skillgraph
python src/final_comparison.py
python src/run_apibank_eval.py
python src/run_feature_ablation.py
python src/generate_figures.py
```

## Main Scripts

- `skillgraph/src/final_comparison.py`: main ToolBench comparison table
- `skillgraph/src/run_apibank_eval.py`: leave-one-out API-Bank evaluation
- `skillgraph/src/learned_reranker.py`: Stage-2 pairwise MLP reranker
- `skillgraph/src/two_stage_pipeline.py`: integrated two-stage pipeline
- `skillgraph/src/run_feature_ablation.py`: feature ablation experiments
- `skillgraph/src/run_llm_stage2.py`: LLM-based Stage-2 reranker comparison
- `skillgraph/src/generate_figures.py`: figure generation from result CSVs

## Documentation

- Reproduction notes: [`docs/REPRODUCE.md`](./docs/REPRODUCE.md)
- Codebase notes: [`skillgraph/README.md`](./skillgraph/README.md)

## Citation

If you use this code, please cite the repository. Citation metadata is provided in [`CITATION.cff`](./CITATION.cff).
