"""
src/paper_analysis.py  —  Pareto Analysis + Random Baseline + Paper Figures Data
==================================================================================
Analyses:
  1. Random Ordering Baseline  — 100-shuffle average for TS-Hybrid tool sets
  2. Pareto Frontier Data      — (Set-F1, Ord.Prec) with dominance labels
  3. Alpha Sensitivity         — sweep alpha 0.0..1.0 on full test set
  4. First-Tool Accuracy       — 1st.Acc for all methods (already computed, aggregated)
  5. Error Analysis            — 20 qualitative samples (10 HR better, 10 HR worse)

Usage
-----
  python src/paper_analysis.py               # full test (9,965)
  python src/paper_analysis.py --sample 500  # quick check
"""

from __future__ import annotations

import argparse
import pathlib
import random
import statistics
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import kendalltau as scipy_kendalltau

warnings.filterwarnings("ignore")

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from graph_search import ToolSequencePlanner
from evaluate import (
    load_trajectories,
    make_train_test_split,
    batch_encode_queries,
    SemanticOnlyBaseline,
    _plan_with_vec,
    _lcs_length,
    _ordered_precision,
    _transition_accuracy,
    _first_tool_accuracy,
)
from gnn_transition import load_score_matrix
from learned_reranker import (
    build_position_stats,
    load_learned_reranker,
    order_learned_rerank,
    CHECKPOINT as LR_CHECKPOINT,
)
from two_stage_pipeline import (
    _build_tp_lookup,
    order_semantic_sort,
    order_optimal_perm,
    order_hybrid_rerank,
    _bucket_label,
    LENGTH_BUCKETS,
)
from final_comparison import (
    compute_set_metrics,
    compute_order_metrics,
    get_hybrid_stage1,
    get_sem_stage1,
)

ROOT        = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_SEM    = 0.2   # from final_comparison.py
ALPHA_HYBRID = 0.4   # from final_comparison.py
HYBRID_PARAMS = {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}

ALL_METHODS_PARETO = [
    "semantic_only",
    "beam",
    "hybrid",
    "ts_sem_semsort",
    "ts_sem_hybrid_rerank",
    "ts_hybrid_semsort",
    "ts_hybrid_hybrid_rerank",
    "ts_hybrid_learned_rerank",
]

METHOD_LABELS = {
    "semantic_only":              "Semantic Only",
    "beam":                       "Beam Search",
    "hybrid":                     "Hybrid Sem-Graph",
    "ts_sem_semsort":             "TS-Sem + Sem-Sort",
    "ts_sem_hybrid_rerank":       "TS-Sem + Hybrid-Rerank",
    "ts_hybrid_semsort":          "TS-Hybrid + Sem-Sort",
    "ts_hybrid_hybrid_rerank":    "TS-Hybrid + Hybrid-Rerank",
    "ts_hybrid_learned_rerank":   "TS-Hybrid + Learned-Rerank",
}


# ============================================================================
# Infrastructure setup (shared)
# ============================================================================

def setup(args) -> dict:
    """Load all shared data; return context dict."""
    print("Loading trajectories ...")
    records = load_trajectories()
    train_records, test_records = make_train_test_split(records, seed=args.seed)

    avg_train    = sum(r["num_steps"] for r in train_records) / len(train_records)
    median_train = float(statistics.median(r["num_steps"] for r in train_records))

    rng = random.Random(args.seed)
    if args.sample > 0:
        rng.shuffle(test_records)
        test_records = test_records[: args.sample]

    print(f"  {len(test_records):,} test samples")

    print("Loading ToolSequencePlanner ...")
    planner = ToolSequencePlanner()
    planner._avg_traj_len    = avg_train
    planner._median_traj_len = median_train

    tool_pos_lists: dict[str, list[float]] = defaultdict(list)
    for rec in train_records:
        seq = rec["tool_sequence"]
        n   = len(seq)
        for i, t in enumerate(seq):
            if n:
                tool_pos_lists[t].append(i / n)
    planner._tool_position_stats = {
        t: float(np.mean(v)) for t, v in tool_pos_lists.items()
    }

    score_matrix, _, t2i = load_score_matrix(args.encoder)
    tp_lookup             = _build_tp_lookup(planner)
    sem_baseline          = SemanticOnlyBaseline(planner)

    print("Batch-encoding test queries ...")
    test_vecs = batch_encode_queries(
        [r["task_description"] for r in test_records]
    )

    # Pre-build hybrid tool-set cache for all test samples (reused across analyses)
    print("Pre-caching Hybrid Sem-Graph stage-1 selections ...")
    hybrid_cache: list[list[tuple[str, float]]] = []
    sem_cache:    list[list[tuple[str, float]]] = []
    for i, rec in enumerate(test_records):
        vec      = test_vecs[i]
        K_oracle = max(len(rec["tool_sequence"]), 3)
        hybrid_cache.append(get_hybrid_stage1(planner, vec, HYBRID_PARAMS))
        sem_cache.append(get_sem_stage1(planner, vec, K_oracle))
    print(f"  Cached {len(hybrid_cache):,} samples")

    # Also cache beam and hybrid single-stage predictions
    bp = {"w1": 0.4, "w2": 0.4, "w3": 0.2, "beam_width": 5}
    beam_preds:   list[list[str]] = []
    hybrid_preds: list[list[str]] = []
    sem_preds:    list[list[str]] = []
    print("Pre-caching single-stage predictions ...")
    for i, rec in enumerate(test_records):
        vec      = test_vecs[i]
        K_oracle = max(len(rec["tool_sequence"]), 3)
        sem_preds.append(sem_baseline.predict(vec, K=K_oracle))
        try:
            plans = _plan_with_vec(planner, "beam", vec, max_steps=8, beam_params=bp)
            beam_preds.append(plans[0].tools if plans else [])
        except Exception:
            beam_preds.append([])
        try:
            plans = _plan_with_vec(planner, "hybrid", vec, max_steps=8,
                                   hybrid_params=HYBRID_PARAMS)
            hybrid_preds.append(plans[0].tools if plans else [])
        except Exception:
            hybrid_preds.append([])

    # Load learned reranker if checkpoint exists
    lr_model        = load_learned_reranker(LR_CHECKPOINT) if LR_CHECKPOINT.exists() else None
    position_stats  = build_position_stats(train_records)

    return {
        "test_records":   test_records,
        "planner":        planner,
        "test_vecs":      test_vecs,
        "score_matrix":   score_matrix,
        "t2i":            t2i,
        "tp_lookup":      tp_lookup,
        "sem_baseline":   sem_baseline,
        "hybrid_cache":   hybrid_cache,
        "sem_cache":      sem_cache,
        "beam_preds":     beam_preds,
        "hybrid_preds":   hybrid_preds,
        "sem_preds":      sem_preds,
        "lr_model":       lr_model,
        "position_stats": position_stats,
    }


# ============================================================================
# Analysis 1: Random Ordering Baseline
# ============================================================================

def analysis_random_baseline(ctx: dict, n_shuffles: int = 100) -> pd.DataFrame:
    """
    For each test sample, shuffle the TS-Hybrid tool set 100 times.
    Average ordering metrics across shuffles and samples.
    """
    print(f"\n{'='*60}")
    print(f"  Analysis 1: Random Ordering Baseline  ({n_shuffles} shuffles/sample)")
    print(f"{'='*60}")

    hybrid_cache  = ctx["hybrid_cache"]
    test_records  = ctx["test_records"]
    rng           = np.random.default_rng(42)
    ORDER_M       = ["ordered_precision", "lcs_r", "kendall_tau", "transition_acc"]

    # Also collect deterministic baselines for comparison
    semsort_vals: dict[str, list[float]] = defaultdict(list)
    hr_vals:      dict[str, list[float]] = defaultdict(list)
    rand_vals:    dict[str, list[float]] = defaultdict(list)

    for i, rec in enumerate(test_records):
        gt    = rec["tool_sequence"]
        ts    = hybrid_cache[i]
        tools = [t for t, _ in ts]
        K     = len(tools)
        if K == 0:
            continue

        # Deterministic orderings
        semsort = order_semantic_sort(ts)
        hr_pred = order_hybrid_rerank(ts, ctx["tp_lookup"], ALPHA_HYBRID)

        for m, pred in [("semsort", semsort), ("hybrid_rerank", hr_pred)]:
            vals = (semsort_vals if m == "semsort" else hr_vals)
            om   = compute_order_metrics(pred, gt)
            for k in ORDER_M:
                vals[k].append(om[k])

        # Random shuffles
        shuffle_metrics: dict[str, list[float]] = defaultdict(list)
        for _ in range(n_shuffles):
            shuffled = tools[:]
            rng.shuffle(shuffled)
            om = compute_order_metrics(shuffled, gt)
            for k in ORDER_M:
                shuffle_metrics[k].append(om[k])
        for k in ORDER_M:
            rand_vals[k].append(float(np.mean(shuffle_metrics[k])))

    rows = []
    print(f"\n  {'Method':<30}  " + "  ".join(f"{m:>16}" for m in ORDER_M))
    print("  " + "-" * (30 + 19 * len(ORDER_M)))

    for label, d in [
        ("TS-Hybrid + Sem-Sort",       semsort_vals),
        ("TS-Hybrid + Hybrid-Rerank",  hr_vals),
        ("TS-Hybrid + Random (mean)",  rand_vals),
    ]:
        means = {k: float(np.mean(v)) for k, v in d.items()}
        row   = {"method": label, **means}
        rows.append(row)
        vals_str = "  ".join(f"{means[m]:>16.4f}" for m in ORDER_M)
        print(f"  {label:<30}  {vals_str}")

    # Compute "improvement over random"
    print(f"\n  Improvement over random baseline:")
    hr_means   = {k: float(np.mean(v)) for k, v in hr_vals.items()}
    rand_means = {k: float(np.mean(v)) for k, v in rand_vals.items()}
    for k in ORDER_M:
        pct = 100 * (hr_means[k] - rand_means[k]) / (rand_means[k] + 1e-9)
        abs_d = hr_means[k] - rand_means[k]
        print(f"    {k:>22}:  rand={rand_means[k]:.4f}  HR={hr_means[k]:.4f}"
              f"  abs_delta={abs_d:+.4f}  rel={pct:+.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "random_baseline.csv", index=False)
    print(f"\n  -> {RESULTS_DIR / 'random_baseline.csv'}")
    return df


# ============================================================================
# Analysis 2: Pareto Frontier
# ============================================================================

def analysis_pareto(ctx: dict) -> pd.DataFrame:
    """
    Compute (Set-F1, Ord.Prec) per method and label Pareto-dominant methods.
    """
    print(f"\n{'='*60}")
    print("  Analysis 2: Pareto Frontier  (Set-F1 vs Ord.Prec)")
    print(f"{'='*60}")

    test_records = ctx["test_records"]
    hybrid_cache = ctx["hybrid_cache"]
    sem_cache    = ctx["sem_cache"]
    tp_lookup    = ctx["tp_lookup"]
    bp           = {"w1": 0.4, "w2": 0.4, "w3": 0.2, "beam_width": 5}

    method_preds: dict[str, list[list[str]]] = {}

    # Collect predictions for all Pareto methods
    method_preds["semantic_only"]           = ctx["sem_preds"]
    method_preds["beam"]                    = ctx["beam_preds"]
    method_preds["hybrid"]                  = ctx["hybrid_preds"]
    method_preds["ts_sem_semsort"]          = [order_semantic_sort(ts) for ts in sem_cache]
    method_preds["ts_sem_hybrid_rerank"]    = [
        order_hybrid_rerank(ts, tp_lookup, ALPHA_SEM) for ts in sem_cache
    ]
    method_preds["ts_hybrid_semsort"]       = [order_semantic_sort(ts) for ts in hybrid_cache]
    method_preds["ts_hybrid_hybrid_rerank"] = [
        order_hybrid_rerank(ts, tp_lookup, ALPHA_HYBRID) for ts in hybrid_cache
    ]
    lr_model       = ctx.get("lr_model")
    position_stats = ctx.get("position_stats", {})
    if lr_model is not None:
        method_preds["ts_hybrid_learned_rerank"] = [
            order_learned_rerank(ts, tp_lookup, position_stats, lr_model)
            for ts in hybrid_cache
        ]
    else:
        method_preds["ts_hybrid_learned_rerank"] = [
            order_hybrid_rerank(ts, tp_lookup, ALPHA_HYBRID) for ts in hybrid_cache
        ]

    rows = []
    for m in ALL_METHODS_PARETO:
        preds = method_preds[m]
        sf1_list, op_list = [], []
        for i, rec in enumerate(test_records):
            gt  = rec["tool_sequence"]
            sm  = compute_set_metrics(preds[i], gt)
            om  = compute_order_metrics(preds[i], gt)
            sf1_list.append(sm["set_f1"])
            op_list.append(om["ordered_precision"])
        rows.append({
            "method":   m,
            "label":    METHOD_LABELS[m],
            "set_f1":   round(float(np.mean(sf1_list)), 4),
            "ord_prec": round(float(np.mean(op_list)),  4),
        })

    df = pd.DataFrame(rows)

    # Pareto dominance: method A dominates B if A.set_f1 >= B.set_f1
    #                                          AND A.ord_prec >= B.ord_prec
    #                                          AND at least one is strict
    is_dominated = []
    for i, row_i in df.iterrows():
        dominated = False
        for j, row_j in df.iterrows():
            if i == j:
                continue
            if (row_j["set_f1"]  >= row_i["set_f1"]  and
                row_j["ord_prec"] >= row_i["ord_prec"] and
                (row_j["set_f1"]  > row_i["set_f1"]  or
                 row_j["ord_prec"] > row_i["ord_prec"])):
                dominated = True
                break
        is_dominated.append(dominated)
    df["pareto_frontier"] = [not d for d in is_dominated]

    print(f"\n  {'Method':<38}  {'Set-F1':>8}  {'Ord.Prec':>9}  Frontier?")
    print("  " + "-" * 65)
    for _, row in df.iterrows():
        star = "  [PARETO]" if row["pareto_frontier"] else ""
        print(f"  {row['label']:<38}  {row['set_f1']:>8.4f}  "
              f"{row['ord_prec']:>9.4f}{star}")

    df.to_csv(RESULTS_DIR / "pareto_data.csv", index=False)
    print(f"\n  -> {RESULTS_DIR / 'pareto_data.csv'}")
    return df


# ============================================================================
# Analysis 3: Alpha Sensitivity
# ============================================================================

def analysis_alpha_sensitivity(ctx: dict) -> pd.DataFrame:
    """Sweep alpha 0.0..1.0 on full test set for TS-Hybrid + Hybrid-Rerank."""
    print(f"\n{'='*60}")
    print("  Analysis 3: Alpha Sensitivity  (TS-Hybrid + Hybrid-Rerank)")
    print(f"{'='*60}")

    test_records = ctx["test_records"]
    hybrid_cache = ctx["hybrid_cache"]
    tp_lookup    = ctx["tp_lookup"]
    alphas       = [round(a, 1) for a in np.arange(0.0, 1.1, 0.1)]

    rows = []
    print(f"\n  {'alpha':>6}  {'Ord.Prec':>10}  {'KendallTau':>11}  {'Trans.Acc':>10}")
    print("  " + "-" * 45)

    best_op, best_kt, best_alpha_op, best_alpha_kt = -1, -1, 0, 0

    for alpha in alphas:
        op_list, kt_list, ta_list = [], [], []
        for i, rec in enumerate(test_records):
            gt   = rec["tool_sequence"]
            ts   = hybrid_cache[i]
            pred = order_hybrid_rerank(ts, tp_lookup, alpha)
            om   = compute_order_metrics(pred, gt)
            op_list.append(om["ordered_precision"])
            kt_list.append(om["kendall_tau"])
            ta_list.append(om["transition_acc"])

        mean_op = float(np.mean(op_list))
        mean_kt = float(np.mean(kt_list))
        mean_ta = float(np.mean(ta_list))

        mark_op = " <-- best Ord.Prec" if mean_op > best_op else ""
        mark_kt = " <-- best KendallTau" if mean_kt > best_kt else ""
        if mean_op > best_op:
            best_op, best_alpha_op = mean_op, alpha
        if mean_kt > best_kt:
            best_kt, best_alpha_kt = mean_kt, alpha

        print(f"  {alpha:>6.1f}  {mean_op:>10.4f}  {mean_kt:>11.4f}  {mean_ta:>10.4f}"
              f"{mark_op}{mark_kt}")
        rows.append({
            "alpha":     alpha,
            "ord_prec":  round(mean_op, 4),
            "kendall_tau": round(mean_kt, 4),
            "transition_acc": round(mean_ta, 4),
        })

    print(f"\n  Best alpha by Ord.Prec   : alpha={best_alpha_op:.1f}  "
          f"(currently using {ALPHA_HYBRID:.1f})")
    print(f"  Best alpha by KendallTau : alpha={best_alpha_kt:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "alpha_sensitivity.csv", index=False)
    print(f"\n  -> {RESULTS_DIR / 'alpha_sensitivity.csv'}")
    return df


# ============================================================================
# Analysis 4: First-Tool Accuracy summary
# ============================================================================

def analysis_first_tool_accuracy(ctx: dict) -> pd.DataFrame:
    """Compute 1st.Acc for all Pareto methods."""
    print(f"\n{'='*60}")
    print("  Analysis 4: First-Tool Accuracy (all methods)")
    print(f"{'='*60}")

    test_records = ctx["test_records"]
    hybrid_cache = ctx["hybrid_cache"]
    sem_cache    = ctx["sem_cache"]
    tp_lookup    = ctx["tp_lookup"]

    method_preds = {
        "semantic_only":           ctx["sem_preds"],
        "beam":                    ctx["beam_preds"],
        "hybrid":                  ctx["hybrid_preds"],
        "ts_sem_semsort":          [order_semantic_sort(ts) for ts in sem_cache],
        "ts_sem_hybrid_rerank":    [order_hybrid_rerank(ts, tp_lookup, ALPHA_SEM)
                                    for ts in sem_cache],
        "ts_hybrid_semsort":       [order_semantic_sort(ts) for ts in hybrid_cache],
        "ts_hybrid_optimal_perm":  [order_optimal_perm(ts, tp_lookup)
                                    for ts in hybrid_cache],
        "ts_hybrid_hybrid_rerank": [order_hybrid_rerank(ts, tp_lookup, ALPHA_HYBRID)
                                    for ts in hybrid_cache],
    }

    rows = []
    print(f"\n  {'Method':<38}  {'1st.Acc':>8}  {'Ord.Prec':>9}  {'Set-F1':>7}")
    print("  " + "-" * 65)

    for m, preds in method_preds.items():
        fa_list, op_list, sf1_list = [], [], []
        for i, rec in enumerate(test_records):
            gt   = rec["tool_sequence"]
            pred = preds[i]
            fa_list.append(_first_tool_accuracy(pred, gt))
            op_list.append(compute_order_metrics(pred, gt)["ordered_precision"])
            sf1_list.append(compute_set_metrics(pred, gt)["set_f1"])
        mean_fa  = float(np.mean(fa_list))
        mean_op  = float(np.mean(op_list))
        mean_sf1 = float(np.mean(sf1_list))
        label    = METHOD_LABELS.get(m, m)
        print(f"  {label:<38}  {mean_fa:>8.4f}  {mean_op:>9.4f}  {mean_sf1:>7.4f}")
        rows.append({
            "method":    m,
            "label":     label,
            "first_tool_acc": round(mean_fa,  4),
            "ord_prec":       round(mean_op,  4),
            "set_f1":         round(mean_sf1, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "first_tool_accuracy.csv", index=False)
    print(f"\n  -> {RESULTS_DIR / 'first_tool_accuracy.csv'}")
    return df


# ============================================================================
# Analysis 5: Error Analysis
# ============================================================================

def analysis_error(ctx: dict, n_each: int = 10) -> None:
    """
    Find n_each samples where HR beats SemSort, and n_each where SemSort beats HR.
    Print query / GT / predictions / diff for each.
    """
    print(f"\n{'='*60}")
    print(f"  Analysis 5: Error Analysis  ({n_each} HR-better + {n_each} HR-worse)")
    print(f"{'='*60}")

    test_records = ctx["test_records"]
    hybrid_cache = ctx["hybrid_cache"]
    tp_lookup    = ctx["tp_lookup"]

    # Compute per-sample ordered_precision delta: HR - SemSort
    deltas = []
    ss_preds = []
    hr_preds = []
    for i, rec in enumerate(test_records):
        gt   = rec["tool_sequence"]
        ts   = hybrid_cache[i]
        ss   = order_semantic_sort(ts)
        hr   = order_hybrid_rerank(ts, tp_lookup, ALPHA_HYBRID)
        op_ss = compute_order_metrics(ss, gt)["ordered_precision"]
        op_hr = compute_order_metrics(hr, gt)["ordered_precision"]
        deltas.append(op_hr - op_ss)
        ss_preds.append(ss)
        hr_preds.append(hr)

    deltas_arr = np.array(deltas)

    # Sort by delta: largest positive = HR most helpful
    sorted_pos = np.argsort(deltas_arr)[::-1]          # HR best
    sorted_neg = np.argsort(deltas_arr)                  # HR worst

    # Select top-n_each from each end, ensuring non-zero delta
    pos_idx = [i for i in sorted_pos if deltas_arr[i] > 0.0][:n_each]
    neg_idx = [i for i in sorted_neg if deltas_arr[i] < 0.0][:n_each]

    out_path = RESULTS_DIR / "error_analysis.txt"
    lines    = []

    def _fmt_block(title: str, idx_list: list[int]) -> None:
        lines.append("=" * 78)
        lines.append(title)
        lines.append("=" * 78)
        for rank, i in enumerate(idx_list, 1):
            rec   = test_records[i]
            gt    = rec["tool_sequence"]
            query = rec["task_description"][:100]
            ss    = ss_preds[i]
            hr    = hr_preds[i]
            d     = deltas_arr[i]

            lines.append(f"\n[Sample {rank}]  idx={i}  delta_OrdPrec={d:+.4f}")
            lines.append(f"  Query : {query}")
            lines.append(f"  GT    : {' -> '.join(gt)}")
            lines.append(f"  SemSort: {' -> '.join(ss)}")
            lines.append(f"  HR     : {' -> '.join(hr)}")

            # Show diff annotation
            changed_pos = [
                j for j in range(min(len(ss), len(hr)))
                if j < len(ss) and j < len(hr) and ss[j] != hr[j]
            ]
            if changed_pos:
                lines.append(f"  Diff  : positions changed = {changed_pos}")
                for j in changed_pos:
                    ss_t = ss[j] if j < len(ss) else "(missing)"
                    hr_t = hr[j] if j < len(hr) else "(missing)"
                    in_gt = lambda t: "[IN GT]" if t in set(gt) else "[not GT]"
                    lines.append(
                        f"    pos {j}: SemSort='{ss_t}' {in_gt(ss_t)}"
                        f"  ->  HR='{hr_t}' {in_gt(hr_t)}"
                    )
            else:
                lines.append(f"  Diff  : same tool set, different order")
            lines.append("")

    _fmt_block(
        f"  SECTION A: HR HELPS  (HR Ord.Prec > SemSort Ord.Prec, top {len(pos_idx)} samples)",
        pos_idx,
    )
    _fmt_block(
        f"  SECTION B: HR HURTS  (HR Ord.Prec < SemSort Ord.Prec, top {len(neg_idx)} samples)",
        neg_idx,
    )

    # Summary stats
    pos_deltas = deltas_arr[deltas_arr > 0]
    neg_deltas = deltas_arr[deltas_arr < 0]
    zero_count = int((deltas_arr == 0).sum())
    lines.insert(0, "=" * 78)
    lines.insert(1, "  Error Analysis: TS-Hybrid + Hybrid-Rerank vs TS-Hybrid + Sem-Sort")
    lines.insert(2, "  Metric: per-sample delta in Ordered Precision  (HR - SemSort)")
    lines.insert(3, "=" * 78)
    lines.insert(4, f"  HR better (delta > 0): {len(pos_deltas):,} samples  "
                    f"mean_delta={pos_deltas.mean():.4f}")
    lines.insert(5, f"  HR worse  (delta < 0): {len(neg_deltas):,} samples  "
                    f"mean_delta={neg_deltas.mean():.4f}")
    lines.insert(6, f"  Equal     (delta = 0): {zero_count:,} samples")
    lines.insert(7, f"  Overall mean delta    : {deltas_arr.mean():+.4f}")
    lines.insert(8, "")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n  HR better: {len(pos_deltas):,} samples  mean={pos_deltas.mean():.4f}")
    print(f"  HR worse:  {len(neg_deltas):,} samples  mean={neg_deltas.mean():.4f}")
    print(f"  Equal:     {zero_count:,} samples")
    print(f"  Overall mean delta: {deltas_arr.mean():+.4f}")
    print(f"\n  -> {out_path}")


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper Analysis: Pareto + Random Baseline + Alpha Sensitivity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sample",    type=int, default=0)
    p.add_argument("--encoder",   type=str, default="sage",
                   choices=["gcn", "gat", "sage"])
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--n-shuffles",type=int, default=100,
                   help="Random shuffles per sample for random baseline")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    ctx = setup(args)

    analysis_random_baseline(ctx, n_shuffles=args.n_shuffles)
    analysis_pareto(ctx)
    analysis_alpha_sensitivity(ctx)
    analysis_first_tool_accuracy(ctx)
    analysis_error(ctx)

    print("\n" + "=" * 60)
    print("  All analyses complete.")
    print("  Results saved:")
    for fname in ["random_baseline.csv", "pareto_data.csv",
                  "alpha_sensitivity.csv", "first_tool_accuracy.csv",
                  "error_analysis.txt"]:
        print(f"    {RESULTS_DIR / fname}")
    print("=" * 60)


if __name__ == "__main__":
    main()
