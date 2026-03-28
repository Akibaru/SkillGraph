"""
src/run_fixed_k_eval.py  —  Fixed-K Robustness Experiment
==========================================================
Addresses the Oracle-K limitation: the main evaluation uses
K = max(|S*|, 3) (oracle K) for semantic retrieval.  This script
re-evaluates the two key methods under *fixed* K values, showing that
the two-stage advantage is not an artefact of knowing K in advance.

Methods compared at each K in {3, 5, 8}:
  1. Semantic Only (fixed K)     — top-K semantic retrieval + sim-sort
  2. TS-Hybrid + LR (fixed K)   — hybrid stage-1 truncated to K, then LR rerank

Also reports Oracle-K results for Semantic Only and TS-Hybrid+LR
as an upper-bound reference.

Output: results/fixed_k_eval.csv
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau as scipy_kendalltau

warnings.filterwarnings("ignore")

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

ROOT        = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

from graph_search import ToolSequencePlanner
from evaluate import (
    load_trajectories,
    make_train_test_split,
    batch_encode_queries,
    _ordered_precision,
    _transition_accuracy,
    _first_tool_accuracy,
    _lcs_length,
)
from two_stage_pipeline import _build_tp_lookup
from learned_reranker import (
    build_position_stats,
    load_learned_reranker,
    order_learned_rerank,
    CHECKPOINT as LR_CHECKPOINT,
)
from final_comparison import get_hybrid_stage1

# -----------------------------------------------------------------------
# Fixed K values to evaluate
# -----------------------------------------------------------------------
FIXED_K_VALUES = [3, 5, 8]

# -----------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------

def _kendalltau(pred: list, gt: list) -> float:
    common = [t for t in pred if t in set(gt)]
    if len(common) < 2:
        return 0.0
    gt_rank   = {t: i for i, t in enumerate(gt)}
    pred_rank = {t: i for i, t in enumerate(common)}
    tau, _    = scipy_kendalltau([pred_rank[t] for t in common],
                                  [gt_rank[t]   for t in common])
    return float(tau) if not np.isnan(tau) else 0.0


def compute_metrics(pred: list[str], gt: list[str]) -> dict:
    pred_set = set(pred)
    gt_set   = set(gt)
    hits     = len(pred_set & gt_set)
    sp       = hits / len(pred_set)  if pred_set  else 0.0
    sr       = hits / len(gt_set)    if gt_set    else 0.0
    sf1      = 2*sp*sr/(sp+sr)       if sp+sr > 0 else 0.0
    pred_in  = [t for t in pred if t in gt_set]
    lcs_r    = _lcs_length(pred_in, gt) / max(len(gt), 1)
    return {
        "set_f1":           round(sf1, 6),
        "ord_prec":         round(_ordered_precision(pred, gt), 6),
        "lcs_r":            round(lcs_r, 6),
        "kendall_tau":      round(_kendalltau(pred, gt), 6),
        "trans_acc":        round(_transition_accuracy(pred, gt), 6),
        "first_tool_acc":   round(_first_tool_accuracy(pred, gt), 6),
    }


# -----------------------------------------------------------------------
# Stage-1 helpers
# -----------------------------------------------------------------------

def get_sem_fixed_k(
    planner:   ToolSequencePlanner,
    query_vec: np.ndarray,
    K:         int,
) -> list[tuple[str, float]]:
    """Semantic top-K (no oracle), sorted by sim descending."""
    return planner._top_entry_tools(query_vec, k=K)


def get_hybrid_fixed_k(
    planner:       ToolSequencePlanner,
    query_vec:     np.ndarray,
    hybrid_params: dict,
    K:             int,
) -> list[tuple[str, float]]:
    """
    Hybrid stage-1 output truncated/padded to exactly K tools.
    - If hybrid returns ≥ K tools: take top-K sorted by sim score.
    - If hybrid returns < K tools: pad with semantic top-K (deduplicated).
    """
    tools_sims = get_hybrid_stage1(planner, query_vec, hybrid_params, max_steps=8)

    if len(tools_sims) >= K:
        # Sort by sim desc, take top K
        tools_sims_sorted = sorted(tools_sims, key=lambda x: -x[1])
        return tools_sims_sorted[:K]

    # Pad with semantic top-K (avoid duplicates)
    existing = {t for t, _ in tools_sims}
    extra_k  = K - len(tools_sims)
    sem_all  = planner._top_entry_tools(query_vec, k=K + len(existing))
    extras   = [(t, s) for t, s in sem_all if t not in existing][:extra_k]
    return tools_sims + extras


# -----------------------------------------------------------------------
# Evaluation at fixed K
# -----------------------------------------------------------------------

def eval_fixed_k(
    test_recs:      list[dict],
    planner:        ToolSequencePlanner,
    test_vecs:      np.ndarray,
    hybrid_params:  dict,
    tp_lookup:      dict,
    position_stats: dict,
    model,
    K:              int,
) -> list[dict]:
    rows = []

    for method in ("sem_only", "ts_hybrid_lr"):
        t0 = time.time()
        mets = []
        for i, rec in enumerate(test_recs):
            gt = rec["tool_sequence"]

            if method == "sem_only":
                tools_sims = get_sem_fixed_k(planner, test_vecs[i], K)
                pred = [t for t, _ in tools_sims]

            else:  # ts_hybrid_lr
                tools_sims = get_hybrid_fixed_k(
                    planner, test_vecs[i], hybrid_params, K
                )
                if tools_sims:
                    pred = order_learned_rerank(
                        tools_sims, tp_lookup, position_stats, model
                    )
                else:
                    pred = []

            mets.append(compute_metrics(pred, gt))

        agg = {k: round(float(np.mean([m[k] for m in mets])), 4) for k in mets[0]}
        elapsed = time.time() - t0
        row = {"method": method, "K": K, "n": len(mets), **agg}
        rows.append(row)
        print(f"  [{method}]  K={K}"
              f"  Set-F1={agg['set_f1']:.4f}"
              f"  Ord.Prec={agg['ord_prec']:.4f}"
              f"  Kendall-τ={agg['kendall_tau']:.4f}"
              f"  ({elapsed:.0f}s)")

    return rows


# -----------------------------------------------------------------------
# Oracle-K baseline (for reference column)
# -----------------------------------------------------------------------

def eval_oracle_k(
    test_recs:      list[dict],
    planner:        ToolSequencePlanner,
    test_vecs:      np.ndarray,
    hybrid_params:  dict,
    tp_lookup:      dict,
    position_stats: dict,
    model,
) -> list[dict]:
    rows = []
    for method in ("sem_only", "ts_hybrid_lr"):
        t0 = time.time()
        mets = []
        for i, rec in enumerate(test_recs):
            gt       = rec["tool_sequence"]
            K_oracle = max(len(gt), 3)

            if method == "sem_only":
                tools_sims = get_sem_fixed_k(planner, test_vecs[i], K_oracle)
                pred = [t for t, _ in tools_sims]
            else:
                tools_sims = get_hybrid_fixed_k(
                    planner, test_vecs[i], hybrid_params, K_oracle
                )
                if tools_sims:
                    pred = order_learned_rerank(
                        tools_sims, tp_lookup, position_stats, model
                    )
                else:
                    pred = []

            mets.append(compute_metrics(pred, gt))

        agg = {k: round(float(np.mean([m[k] for m in mets])), 4) for k in mets[0]}
        elapsed = time.time() - t0
        row = {"method": method, "K": "oracle", "n": len(mets), **agg}
        rows.append(row)
        print(f"  [{method}]  K=oracle"
              f"  Set-F1={agg['set_f1']:.4f}"
              f"  Ord.Prec={agg['ord_prec']:.4f}"
              f"  Kendall-τ={agg['kendall_tau']:.4f}"
              f"  ({elapsed:.0f}s)")
    return rows


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    DATA_DIR  = ROOT / "data" / "processed"
    traj_path = DATA_DIR / "successful_trajectories.jsonl"

    print("Loading data...")
    all_records = load_trajectories(traj_path)
    train_recs, test_recs = make_train_test_split(all_records, seed=42)
    print(f"  train={len(train_recs):,}  test={len(test_recs):,}")

    planner = ToolSequencePlanner()
    hybrid_params = {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}

    print("Encoding test queries...")
    test_vecs = batch_encode_queries([r["task_description"] for r in test_recs])

    tp_lookup      = _build_tp_lookup(planner)
    position_stats = build_position_stats(train_recs)

    print(f"Loading LR model from {LR_CHECKPOINT}...")
    model = load_learned_reranker(LR_CHECKPOINT)
    model.eval()

    print("\n" + "=" * 60)
    print("  Fixed-K evaluation  (n=9,965)")
    print("=" * 60)

    all_rows = []

    # Fixed K experiments
    for K in FIXED_K_VALUES:
        print(f"\n── K={K} ──")
        rows = eval_fixed_k(
            test_recs, planner, test_vecs, hybrid_params,
            tp_lookup, position_stats, model, K
        )
        all_rows.extend(rows)

    # Oracle-K reference
    print("\n── Oracle-K (reference) ──")
    all_rows.extend(
        eval_oracle_k(
            test_recs, planner, test_vecs, hybrid_params,
            tp_lookup, position_stats, model
        )
    )

    df = pd.DataFrame(all_rows)
    out = RESULTS_DIR / "fixed_k_eval.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    # Pretty summary
    print("\n" + "=" * 80)
    print(f"{'Method':<18} {'K':>6}  {'Set-F1':>7}  {'Ord.Prec':>8}  "
          f"{'Kendall-τ':>9}  {'Trans.Acc':>9}  {'1st.Acc':>7}")
    print("-" * 80)
    prev_k = None
    for _, row in df.iterrows():
        if prev_k is not None and row["K"] != prev_k:
            print()
        print(f"  {row['method']:<16} {str(row['K']):>6}  "
              f"{row['set_f1']:>7.4f}  {row['ord_prec']:>8.4f}  "
              f"{row['kendall_tau']:>9.4f}  {row['trans_acc']:>9.4f}  "
              f"{row['first_tool_acc']:>7.4f}")
        prev_k = row["K"]
    print("=" * 80)


if __name__ == "__main__":
    main()
