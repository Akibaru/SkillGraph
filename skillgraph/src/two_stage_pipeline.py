"""
src/two_stage_pipeline.py  —  Two-Stage Decoupled Pipeline  (v2, fixed Stage 1)
================================================================================
Root-cause of v1 regression: fixed K=3 in Stage 1 < oracle length for many
trajectories, so set quality was lower than Semantic Only.

Fix: Stage 1 (TS-Matched) mirrors SemanticOnlyBaseline exactly:
    K_oracle = max(len(ground_truth), 3)   (oracle K, same as sem_only)
Stage 2 reorders the *same* tools — F1@K cannot regress below Semantic Only.

Validation invariant:
    TS-Matched + Sem-Sort  F1@K  ==  Semantic Only  F1@K  (within rounding)

Methods evaluated (10 total)
------------------------------
  Reference methods (identical call paths to evaluate.py / gnn_comparison.py):
    1. semantic_only        — SemanticOnlyBaseline with oracle K
    2. beam                 — original Beam Search
    3. hybrid               — original Hybrid Sem-Graph

  TS-Fixed (K=3, v1 reference — kept for comparison):
    4. ts_fixed_sem_sort    — fixed K=3 + semantic sort
    5. ts_fixed_hybrid      — fixed K=3 + Hybrid-Rerank (v1 best)

  TS-Matched (Stage 1 == Semantic Only selection):
    6. ts_matched_sem_sort      ★ — oracle K + semantic sort  (should == sem_only F1)
    7. ts_matched_greedy_graph  ★ — oracle K + Greedy-Graph
    8. ts_matched_greedy_gnn    ★ — oracle K + Greedy-GNN
    9. ts_matched_optimal_perm  ★ — oracle K + Optimal-Perm (K! enum, fallback K>8)
   10. ts_matched_hybrid_rerank ★ — oracle K + Hybrid-Rerank  (alpha re-tuned)

Outputs:
  results/two_stage_v2_comparison.csv
  results/two_stage_v2_by_length.csv

Usage
-----
  python src/two_stage_pipeline.py               # full test (9,965)
  python src/two_stage_pipeline.py --sample 500  # quick check
  python src/two_stage_pipeline.py --encoder gcn # different GNN encoder
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import statistics
import sys
import time
import warnings
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from graph_search import ToolSequencePlanner
from evaluate import (
    METRIC_COLS,
    load_trajectories,
    make_train_test_split,
    batch_encode_queries,
    compute_metrics,
    SemanticOnlyBaseline,
    _plan_with_vec,
)
from gnn_transition import (
    load_score_matrix,
    precompute_full_score_matrix,
    save_score_matrix,
    load_transition_model,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Maximum K for full permutation enumeration (K! feasibility check)
OPT_PERM_LIMIT = 8  # 8! = 40,320; for typical K≤5 this is trivially fast

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
ALL_METHODS = [
    "semantic_only",
    "beam",
    "hybrid",
    "ts_fixed_sem_sort",
    "ts_fixed_hybrid",
    "ts_matched_sem_sort",
    "ts_matched_greedy_graph",
    "ts_matched_greedy_gnn",
    "ts_matched_optimal_perm",
    "ts_matched_hybrid_rerank",
]

METHOD_LABELS = {
    "semantic_only":            "Semantic Only (BL)",
    "beam":                     "Beam Search",
    "hybrid":                   "Hybrid Sem-Graph",
    "ts_fixed_sem_sort":        "TS-Fixed + Sem-Sort  [v1, K=3]",
    "ts_fixed_hybrid":          "TS-Fixed + Hybrid-Rerank [v1]",
    "ts_matched_sem_sort":      "TS-Matched + Sem-Sort ★",
    "ts_matched_greedy_graph":  "TS-Matched + Greedy-Graph ★",
    "ts_matched_greedy_gnn":    "TS-Matched + Greedy-GNN ★",
    "ts_matched_optimal_perm":  "TS-Matched + Optimal-Perm ★",
    "ts_matched_hybrid_rerank": "TS-Matched + Hybrid-Rerank ★",
}

REPORT_METRICS = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "ordered_precision",
    "lcs_r",
    "transition_accuracy",
    "first_tool_accuracy",
    "latency_ms",
]

LENGTH_BUCKETS = [
    ("1-2",  1, 2),
    ("3-4",  3, 4),
    ("5+",   5, 999),
]

FIXED_K   = 3       # v1 reference: ts_fixed_* always use this K
FIXED_ALPHA_V1 = 0.2  # v1 best alpha (Hybrid-Rerank with K=3)


# ============================================================================
# Graph transition-probability lookup
# ============================================================================

def _build_tp_lookup(planner: ToolSequencePlanner) -> dict[tuple[str, str], float]:
    """O(1) lookup: (u, v) -> graph transition_prob."""
    tp: dict[tuple[str, str], float] = {}
    for u, nbrs in planner._adj.items():
        for v, _w, tprob in nbrs:
            tp[(u, v)] = tprob
    return tp


# ============================================================================
# Stage 1: tool selection
# ============================================================================

def select_fixed(
    planner:   ToolSequencePlanner,
    query_vec: np.ndarray,
    k:         int = FIXED_K,
) -> list[tuple[str, float]]:
    """Fixed-K selection: top-k by cosine similarity."""
    return planner._top_entry_tools(query_vec, k=k)


def select_matched(
    planner:   ToolSequencePlanner,
    query_vec: np.ndarray,
    K_oracle:  int,
) -> list[tuple[str, float]]:
    """
    Oracle-K selection: mirrors SemanticOnlyBaseline.predict(vec, K_oracle).

    Verified equivalent:
        SemanticOnlyBaseline.predict  →  np.argsort(emb @ q)[-K:][::-1]
        _top_entry_tools(q, k=K)      →  same argsort, returns (name, sim) pairs

    Both use planner._embeddings and planner._active_tools.
    """
    return planner._top_entry_tools(query_vec, k=K_oracle)


# ============================================================================
# Stage 2: ordering strategies
# ============================================================================

def order_semantic_sort(tools_sims: list[tuple[str, float]]) -> list[str]:
    """Trivial: preserve semantic-similarity order (no reordering)."""
    return [t for t, _ in tools_sims]


def order_greedy_graph(
    tools_sims: list[tuple[str, float]],
    tp_lookup:  dict[tuple[str, str], float],
) -> list[str]:
    """
    Greedy ordering by graph transition probability.
    Start = highest semantic similarity tool.
    Each step = argmax tp(current, remaining); break ties by semantic rank.
    """
    remaining_sim = dict(tools_sims)
    sem_rank      = {t: i for i, (t, _) in enumerate(tools_sims)}
    start         = tools_sims[0][0]
    sequence      = [start]
    del remaining_sim[start]

    while remaining_sim:
        current   = sequence[-1]
        best_next = max(
            remaining_sim,
            key=lambda t: (tp_lookup.get((current, t), 0.0), -sem_rank[t]),
        )
        sequence.append(best_next)
        del remaining_sim[best_next]

    return sequence


def order_greedy_gnn(
    tools_sims:   list[tuple[str, float]],
    score_matrix: np.ndarray,
    t2i:          dict[str, int],
) -> list[str]:
    """
    Greedy ordering by GNN transition score (full N×N matrix).
    Start = highest semantic similarity tool.
    Fallback to semantic sim if current tool not in matrix.
    """
    remaining_sim = dict(tools_sims)
    start         = tools_sims[0][0]
    sequence      = [start]
    del remaining_sim[start]

    while remaining_sim:
        current  = sequence[-1]
        cur_idx  = t2i.get(current)
        rem_list = list(remaining_sim)

        if cur_idx is not None:
            best_next = max(
                rem_list,
                key=lambda t: float(score_matrix[cur_idx, t2i[t]]) if t in t2i else 0.0,
            )
        else:
            best_next = max(rem_list, key=lambda t: remaining_sim[t])

        sequence.append(best_next)
        del remaining_sim[best_next]

    return sequence


def _perm_log_tp(perm: list[str], tp_lookup: dict, eps: float = 1e-9) -> float:
    """Sum of log(tp + eps) along consecutive pairs — proxy for log-product."""
    return sum(
        math.log(tp_lookup.get((perm[i], perm[i + 1]), 0.0) + eps)
        for i in range(len(perm) - 1)
    )


def order_optimal_perm(
    tools_sims: list[tuple[str, float]],
    tp_lookup:  dict[tuple[str, str], float],
    perm_limit: int = OPT_PERM_LIMIT,
) -> list[str]:
    """
    Enumerate all K! permutations, pick max log-tp-sum (proxy for product).
    Falls back to Greedy-Graph when K > perm_limit.
    """
    tools = [t for t, _ in tools_sims]
    if len(tools) > perm_limit:
        return order_greedy_graph(tools_sims, tp_lookup)

    best_seq   = tools
    best_score = float("-inf")
    for perm in permutations(tools):
        score = _perm_log_tp(list(perm), tp_lookup)
        if score > best_score:
            best_score = score
            best_seq   = list(perm)
    return best_seq


def _perm_hybrid_score(
    perm:      list[str],
    tp_lookup: dict,
    sim_dict:  dict[str, float],
    alpha:     float,
) -> float:
    """
    Hybrid score:
        alpha × Σ tp(t_i, t_{i+1})  +  (1-alpha) × Σ sim(t_i) / (i+1)

    The second term is position-dependent: high-sim tools at early positions
    score highest (weight 1.0 at i=0, 0.5 at i=1, ...).
    This is the only term that differs between permutations when tp is sparse.
    """
    K      = len(perm)
    tp_sum = sum(tp_lookup.get((perm[i], perm[i + 1]), 0.0) for i in range(K - 1))
    sem_sum = sum(sim_dict.get(perm[i], 0.0) / (i + 1) for i in range(K))
    return alpha * tp_sum + (1.0 - alpha) * sem_sum


def order_hybrid_rerank(
    tools_sims: list[tuple[str, float]],
    tp_lookup:  dict[tuple[str, str], float],
    alpha:      float,
    perm_limit: int = OPT_PERM_LIMIT,
) -> list[str]:
    """
    Hybrid-Rerank: enumerate all permutations for K ≤ perm_limit; else greedy.
    Scoring: alpha × Σ tp  +  (1-alpha) × Σ sim/(i+1)
    """
    tools    = [t for t, _ in tools_sims]
    sim_dict = dict(tools_sims)

    if len(tools) <= perm_limit:
        best_seq   = tools
        best_score = float("-inf")
        for perm in permutations(tools):
            score = _perm_hybrid_score(list(perm), tp_lookup, sim_dict, alpha)
            if score > best_score:
                best_score = score
                best_seq   = list(perm)
        return best_seq

    # Greedy fallback for K > perm_limit
    remaining = dict(tools_sims)
    start     = tools_sims[0][0]
    sequence  = [start]
    del remaining[start]

    while remaining:
        current   = sequence[-1]
        pos       = len(sequence)
        best_next = max(
            remaining,
            key=lambda t: (
                alpha * tp_lookup.get((current, t), 0.0)
                + (1.0 - alpha) * remaining[t] / (pos + 1)
            ),
        )
        sequence.append(best_next)
        del remaining[best_next]

    return sequence


# ============================================================================
# Hyperparameter searches
# ============================================================================

def find_best_rerank_alpha(
    planner:     ToolSequencePlanner,
    val_records: list[dict],
    val_vecs:    np.ndarray,
    tp_lookup:   dict,
    mode:        str = "matched",   # "fixed" or "matched"
    fixed_k:     int = FIXED_K,
    alphas:      list[float] | None = None,
) -> float:
    """
    Grid-search alpha for Hybrid-Rerank on val set.
    mode="matched"  → use oracle K per sample (mirrors TS-Matched)
    mode="fixed"    → use fixed_k for all samples
    Optimises Ordered Precision (the metric Stage 2 is meant to improve).
    """
    if alphas is None:
        alphas = [round(a, 1) for a in np.arange(0.0, 1.1, 0.1)]

    print(f"\n[alpha search / {mode}] Val={len(val_records)}  "
          f"candidates={alphas}")

    best_alpha = 0.5
    best_ord   = -1.0

    for alpha in alphas:
        ord_list = []
        for i, rec in enumerate(val_records):
            gt  = rec["tool_sequence"]
            vec = val_vecs[i]
            K_oracle = max(len(gt), 3)
            K_sel    = K_oracle if mode == "matched" else fixed_k
            try:
                ts  = planner._top_entry_tools(vec, k=K_sel)
                seq = order_hybrid_rerank(ts, tp_lookup, alpha)
                m   = compute_metrics(seq, gt)
                ord_list.append(m["ordered_precision"])
            except Exception:
                ord_list.append(0.0)

        mean_ord = float(np.mean(ord_list))
        marker   = "  ←" if mean_ord > best_ord else ""
        print(f"  alpha={alpha:.1f}  val_OrdPrec={mean_ord:.4f}{marker}")

        if mean_ord > best_ord:
            best_ord   = mean_ord
            best_alpha = alpha

    print(f"[alpha search / {mode}] Best alpha={best_alpha:.1f}  "
          f"val_OrdPrec={best_ord:.4f}\n")
    return best_alpha


# ============================================================================
# Evaluation loop
# ============================================================================

def _bucket_label(gt_len: int) -> str:
    for label, lo, hi in LENGTH_BUCKETS:
        if lo <= gt_len <= hi:
            return label
    return "5+"


def evaluate_all(
    test_records:     list[dict],
    planner:          ToolSequencePlanner,
    query_vecs:       np.ndarray,
    score_matrix:     np.ndarray,
    t2i:              dict[str, int],
    tp_lookup:        dict,
    sem_baseline:     SemanticOnlyBaseline,
    alpha_fixed:      float,
    alpha_matched:    float,
    max_steps:        int   = 8,
    beam_params:      dict | None = None,
    hybrid_params:    dict | None = None,
    dijkstra_beta:    float = 0.3,
) -> list[dict]:
    """Run all 10 methods; return flat list of per-trajectory metric rows."""
    bp = beam_params   or {"w1": 0.4, "w2": 0.4, "w3": 0.2, "beam_width": 5}
    hp = hybrid_params or {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}

    # Stage 1 cache (fixed and matched) — avoid redundant top-K calls
    fixed_cache:   dict[int, list] = {}
    matched_cache: dict[int, list] = {}

    def _stage1_fixed(i: int) -> list[tuple[str, float]]:
        if i not in fixed_cache:
            fixed_cache[i] = select_fixed(planner, query_vecs[i], FIXED_K)
        return fixed_cache[i]

    def _stage1_matched(i: int, K_oracle: int) -> list[tuple[str, float]]:
        if i not in matched_cache:
            matched_cache[i] = select_matched(planner, query_vecs[i], K_oracle)
        return matched_cache[i]

    all_rows: list[dict] = []

    for method in ALL_METHODS:
        print(f"\n  [{method}]  evaluating {len(test_records):,} trajectories …")
        t_start = time.time()

        for i, rec in enumerate(test_records):
            gt        = rec["tool_sequence"]
            vec       = query_vecs[i]
            K_gt      = len(gt)
            K_oracle  = max(K_gt, 3)
            t0        = time.time()

            try:
                # ── Reference methods ─────────────────────────────────────
                if method == "semantic_only":
                    predicted = sem_baseline.predict(vec, K=K_oracle)

                elif method in ("beam", "hybrid"):
                    plans     = _plan_with_vec(
                        planner, method, vec, max_steps=max_steps,
                        beta=dijkstra_beta,
                        beam_params=bp, hybrid_params=hp,
                    )
                    predicted = plans[0].tools if plans else []

                # ── TS-Fixed (K=3, v1 reference) ──────────────────────────
                elif method == "ts_fixed_sem_sort":
                    ts        = _stage1_fixed(i)
                    predicted = order_semantic_sort(ts)

                elif method == "ts_fixed_hybrid":
                    ts        = _stage1_fixed(i)
                    predicted = order_hybrid_rerank(ts, tp_lookup, alpha_fixed)

                # ── TS-Matched (oracle K = Semantic Only) ─────────────────
                elif method == "ts_matched_sem_sort":
                    ts        = _stage1_matched(i, K_oracle)
                    predicted = order_semantic_sort(ts)

                elif method == "ts_matched_greedy_graph":
                    ts        = _stage1_matched(i, K_oracle)
                    predicted = order_greedy_graph(ts, tp_lookup)

                elif method == "ts_matched_greedy_gnn":
                    ts        = _stage1_matched(i, K_oracle)
                    predicted = order_greedy_gnn(ts, score_matrix, t2i)

                elif method == "ts_matched_optimal_perm":
                    ts        = _stage1_matched(i, K_oracle)
                    predicted = order_optimal_perm(ts, tp_lookup)

                elif method == "ts_matched_hybrid_rerank":
                    ts        = _stage1_matched(i, K_oracle)
                    predicted = order_hybrid_rerank(ts, tp_lookup, alpha_matched)

                else:
                    predicted = []

            except Exception as exc:
                print(f"    [warn] {method} row {i}: {exc}")
                predicted = []

            latency = time.time() - t0
            row     = compute_metrics(predicted, gt, latency_s=latency)
            row["method"] = method
            row["bucket"] = _bucket_label(K_gt)
            all_rows.append(row)

        elapsed     = time.time() - t_start
        method_rows = [r for r in all_rows if r["method"] == method]
        avg_len     = sum(r["pred_len"] for r in method_rows) / max(1, len(method_rows))
        print(f"  [{method}] done in {elapsed:.1f}s  avg_pred_len={avg_len:.2f}")

    return all_rows


# ============================================================================
# Validation check
# ============================================================================

def validate_stage1_parity(rows: list[dict]) -> None:
    """
    Assert that TS-Matched + Sem-Sort  F1@K == Semantic Only F1@K.
    Prints a clear PASS / FAIL message.
    """
    df        = pd.DataFrame(rows)
    sem_f1    = df[df["method"] == "semantic_only"]["f1_at_k"].mean()
    matched_f1 = df[df["method"] == "ts_matched_sem_sort"]["f1_at_k"].mean()
    diff      = abs(sem_f1 - matched_f1)

    print("\n" + "=" * 60)
    print("  Stage 1 Parity Validation")
    print("=" * 60)
    print(f"  Semantic Only F1@K      : {sem_f1:.6f}")
    print(f"  TS-Matched+Sem-Sort F1@K: {matched_f1:.6f}")
    print(f"  Absolute difference     : {diff:.6f}")
    if diff < 1e-4:
        print("  STATUS: PASS [OK]  (Stage 1 matches Semantic Only exactly)")
    else:
        print("  STATUS: FAIL [!!]  (Stage 1 diverges — check _top_entry_tools vs predict)")
    print("=" * 60)


# ============================================================================
# Output helpers
# ============================================================================

def make_summary(rows: list[dict]) -> pd.DataFrame:
    df  = pd.DataFrame(rows)
    agg = df.groupby("method")[REPORT_METRICS].mean().round(4).reset_index()
    agg["_order"] = agg["method"].map({m: i for i, m in enumerate(ALL_METHODS)})
    agg = agg.sort_values("_order").drop(columns="_order")
    agg.insert(1, "label",
               agg["method"].map(METHOD_LABELS).fillna(agg["method"]))
    return agg


def make_by_length(rows: list[dict]) -> pd.DataFrame:
    df  = pd.DataFrame(rows)
    agg = (
        df.groupby(["method", "bucket"])[REPORT_METRICS]
        .mean().round(4).reset_index()
    )
    bucket_order = [b[0] for b in LENGTH_BUCKETS]
    agg["_morder"] = agg["method"].map({m: i for i, m in enumerate(ALL_METHODS)})
    agg["_border"] = agg["bucket"].map({b: i for i, b in enumerate(bucket_order)})
    agg = agg.sort_values(["_morder", "_border"]).drop(columns=["_morder", "_border"])
    agg.insert(2, "label",
               agg["method"].map(METHOD_LABELS).fillna(agg["method"]))
    return agg


def print_main_table(
    summary:       pd.DataFrame,
    alpha_fixed:   float,
    alpha_matched: float,
) -> None:
    cols    = ["label", "precision_at_k", "recall_at_k", "f1_at_k",
               "ordered_precision", "lcs_r", "transition_accuracy",
               "first_tool_accuracy", "latency_ms"]
    headers = ["Method", "Prec@K", "Rec@K", "F1@K",
               "Ord.Prec", "LCS-R", "Trans.Acc", "1st.Acc", "Lat(ms)"]

    sub = summary[cols].copy()
    sub.columns = headers

    col_w = [
        max(len(h), max(
            len(f"{v:.4f}") if isinstance(v, float) else len(str(v))
            for v in sub[h]
        )) + 2
        for h in headers
    ]
    sep    = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_w)) + " |"

    REF_METHODS   = {"semantic_only", "beam", "hybrid"}
    FIXED_METHODS = {"ts_fixed_sem_sort", "ts_fixed_hybrid"}

    print(sep); print(header); print(sep)
    for idx, (_, row) in enumerate(sub.iterrows()):
        m_id = ALL_METHODS[idx]
        if m_id == "ts_fixed_sem_sort":    print(sep)
        if m_id == "ts_matched_sem_sort":  print(sep)
        cells = [
            (f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(w)
            for v, w in zip(row, col_w)
        ]
        print("| " + " | ".join(cells) + " |")
    print(sep)
    print(f"  TS-Fixed alpha={alpha_fixed:.1f}  |  "
          f"TS-Matched alpha={alpha_matched:.1f}")


def print_ordering_delta(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)

    # Baseline: ts_matched_sem_sort (identical set to semantic_only, sem-sorted)
    before = df[df["method"] == "ts_matched_sem_sort"]["ordered_precision"].mean()
    sem_f1 = df[df["method"] == "semantic_only"]["f1_at_k"].mean()

    print("\n" + "=" * 72)
    print("  Stage 2 Ordering Improvement  (vs TS-Matched + Sem-Sort)")
    print("  Metric: Ordered Precision  (F1 is fixed — same tool set)")
    print("=" * 72)
    print(f"  Semantic Only F1@K = {sem_f1:.4f}  "
          f"(guaranteed lower bound for TS-Matched methods)")
    print(f"  {'Method':<34}  {'After':>8}  {'Before':>8}  {'Δ':>8}")
    print("  " + "-" * 62)

    for m in ["ts_matched_greedy_graph",
              "ts_matched_greedy_gnn",
              "ts_matched_optimal_perm",
              "ts_matched_hybrid_rerank"]:
        sub   = df[df["method"] == m]
        if sub.empty:
            continue
        after = sub["ordered_precision"].mean()
        delta = after - before
        sign  = "+" if delta >= 0 else ""
        label = METHOD_LABELS.get(m, m)
        print(f"  {label:<34}  {after:8.4f}  {before:8.4f}  {sign}{delta:.4f}")
    print()


def print_by_length_table(by_length: pd.DataFrame) -> None:
    buckets = [b[0] for b in LENGTH_BUCKETS]

    print("=" * 88)
    print("  Bucket Analysis by Ground Truth Trajectory Length  (F1@K / Ord.Prec)")
    print("=" * 88)

    header = f"  {'Method':<36}" + "".join(f"  {('[' + b + ']'):>14}" for b in buckets)
    print(header)
    print("  " + "-" * (36 + 16 * len(buckets)))

    for m in ALL_METHODS:
        label   = METHOD_LABELS.get(m, m)
        row_str = f"  {label:<36}"
        for b in buckets:
            sub = by_length[(by_length["method"] == m) & (by_length["bucket"] == b)]
            if sub.empty:
                row_str += f"  {'  ---/---':>14}"
            else:
                f1  = sub["f1_at_k"].iloc[0]
                ord = sub["ordered_precision"].iloc[0]
                row_str += f"  {f'{f1:.3f}/{ord:.3f}':>14}"
        print(row_str)
    print()


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-Stage Decoupled Pipeline v2 (fixed Stage 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sample",    type=int,  default=0)
    p.add_argument("--max-steps", type=int,  default=8)
    p.add_argument("--encoder",   type=str,  default="sage",
                   choices=["gcn", "gat", "sage"])
    p.add_argument("--val-n",     type=int,  default=500)
    p.add_argument("--seed",      type=int,  default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng  = random.Random(args.seed)
    np.random.seed(args.seed)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("Loading trajectories …")
    records = load_trajectories()
    train_records, test_records = make_train_test_split(records, seed=args.seed)

    avg_train_len    = sum(r["num_steps"] for r in train_records) / len(train_records)
    median_train_len = float(statistics.median(
        r["num_steps"] for r in train_records
    ))

    if args.sample > 0:
        rng.shuffle(test_records)
        test_records = test_records[: args.sample]

    bucket_counts = defaultdict(int)
    for rec in test_records:
        bucket_counts[_bucket_label(len(rec["tool_sequence"]))] += 1
    print(f"  {len(test_records):,} test trajectories  |  "
          f"buckets: " +
          "  ".join(f"[{b}]={bucket_counts[b]}" for b, *_ in LENGTH_BUCKETS))

    # ── 2. Load planner ───────────────────────────────────────────────────
    print("\nLoading ToolSequencePlanner …")
    planner = ToolSequencePlanner()
    planner._avg_traj_len    = avg_train_len
    planner._median_traj_len = median_train_len

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

    # ── 3. Print Semantic Only selection mechanism ────────────────────────
    print("\n" + "=" * 60)
    print("  Semantic Only Selection Mechanism (inspect)")
    print("=" * 60)
    print("  Class : SemanticOnlyBaseline.predict(query_vec, K)")
    print("  K call: K = max(len(ground_truth), 3)  [oracle K, per-sample]")
    print("  Step 1: sims = planner._embeddings @ query_vec   (4988-dim dot)")
    print("  Step 2: top_K_idx = np.argsort(sims)[-K:][::-1]")
    print("  Step 3: return [_active_tools[i] for i in top_K_idx]")
    print(f"  K distribution in test set:")
    k_vals = [max(len(r["tool_sequence"]), 3) for r in test_records]
    k_arr  = np.array(k_vals)
    for kv in sorted(set(k_vals))[:10]:
        cnt = int((k_arr == kv).sum())
        print(f"    K={kv:2d}  → {cnt:5,} samples  ({100*cnt/len(k_vals):.1f}%)")
    if len(set(k_vals)) > 10:
        print(f"    ... ({len(set(k_vals))} distinct K values)")
    print("=" * 60)

    # ── 4. Load GNN score matrix ─────────────────────────────────────────
    try:
        score_matrix, active_tools, t2i = load_score_matrix(args.encoder)
    except FileNotFoundError:
        print("\n[info] Score matrix not found — computing …")
        model, _, _ = load_transition_model(args.encoder, device_str="auto")
        score_matrix, active_tools = precompute_full_score_matrix(
            model, device_str="auto"
        )
        save_score_matrix(score_matrix, active_tools, encoder_type=args.encoder)
        t2i = {name: i for i, name in enumerate(active_tools)}

    # ── 5. Build transition-probability lookup ───────────────────────────
    print("\nBuilding graph TP lookup …")
    tp_lookup = _build_tp_lookup(planner)
    print(f"  {len(tp_lookup):,} (u,v) pairs with non-zero transition probability")

    # ── 6. Baselines ─────────────────────────────────────────────────────
    sem_baseline = SemanticOnlyBaseline(planner)

    # ── 7. Encode queries ─────────────────────────────────────────────────
    print("\nBatch-encoding test queries …")
    test_queries = [r["task_description"] for r in test_records]
    test_vecs    = batch_encode_queries(test_queries)

    # ── 8. Val set for alpha searches ─────────────────────────────────────
    val_idx     = list(range(len(train_records)))
    rng.shuffle(val_idx)
    val_records = [train_records[i] for i in val_idx[: args.val_n]]
    val_vecs    = batch_encode_queries(
        [r["task_description"] for r in val_records]
    )

    # ── 9. Alpha searches ─────────────────────────────────────────────────
    alpha_fixed   = find_best_rerank_alpha(
        planner, val_records, val_vecs, tp_lookup, mode="fixed"
    )
    alpha_matched = find_best_rerank_alpha(
        planner, val_records, val_vecs, tp_lookup, mode="matched"
    )

    # ── 10. Full evaluation ───────────────────────────────────────────────
    print(f"\nRunning {len(test_records):,} × {len(ALL_METHODS)} methods …")
    rows = evaluate_all(
        test_records, planner, test_vecs,
        score_matrix, t2i, tp_lookup,
        sem_baseline,
        alpha_fixed=alpha_fixed,
        alpha_matched=alpha_matched,
        max_steps=args.max_steps,
    )

    # ── 11. Validate Stage 1 parity ───────────────────────────────────────
    validate_stage1_parity(rows)

    # ── 12. Save and display ──────────────────────────────────────────────
    summary   = make_summary(rows)
    by_length = make_by_length(rows)

    csv_main   = RESULTS_DIR / "two_stage_v2_comparison.csv"
    csv_bucket = RESULTS_DIR / "two_stage_v2_by_length.csv"
    summary.to_csv(csv_main,   index=False)
    by_length.to_csv(csv_bucket, index=False)

    print("\n" + "=" * 80)
    print("  Two-Stage Pipeline v2  —  Final Results")
    print(f"  Test: {len(test_records):,}  |  Encoder: {args.encoder.upper()}  |  "
          f"alpha_fixed={alpha_fixed:.1f}  alpha_matched={alpha_matched:.1f}")
    print("=" * 80 + "\n")

    print_main_table(summary, alpha_fixed, alpha_matched)
    print_ordering_delta(rows)
    print_by_length_table(by_length)

    print(f"CSV (summary)   → {csv_main}")
    print(f"CSV (by length) → {csv_bucket}")


if __name__ == "__main__":
    main()
