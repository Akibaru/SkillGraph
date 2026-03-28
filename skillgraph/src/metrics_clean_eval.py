"""
src/metrics_clean_eval.py  —  Clean Metrics Evaluation
=======================================================
Root-cause analysis of F1@K inconsistency + clean metric redesign.

Root cause
----------
compute_metrics() does:   top_K = predicted[:K]   where K = len(ground_truth)
When K_oracle = max(K_gt, 3) > K_gt  (i.e. K_gt < 3, the [1-2] bucket),
predicted has 3 tools but only the FIRST K_gt tools are evaluated.
→ F1@K becomes order-dependent even for pure-reordering methods.
→ 3,379 / 9,965 (33.9%) of test samples are affected.

Clean metric design
-------------------
Selection quality  (set-based, order-independent)
  set_precision : |pred_set ∩ gt_set| / |pred_set|
  set_recall    : |pred_set ∩ gt_set| / |gt_set|
  set_f1        : harmonic mean — guaranteed equal for all TS-Matched methods

Ordering quality  (order-dependent, evaluated on common tools pred ∩ GT)
  ordered_precision : fraction of GT pairs preserved in predicted order
  lcs_r             : longest common subsequence / len(GT)
  kendall_tau       : rank correlation of common tools (pred order vs GT order)
  transition_acc    : fraction of consecutive GT transitions reproduced

Invariant to validate
---------------------
  All TS-Matched methods must have identical set_f1  (same tool set, different order)

Methods evaluated (8)
---------------------
  1.  semantic_only
  2.  beam
  3.  hybrid
  4.  ts_matched_sem_sort      ★
  5.  ts_matched_greedy_graph  ★
  6.  ts_matched_greedy_gnn    ★
  7.  ts_matched_optimal_perm  ★
  8.  ts_matched_hybrid_rerank ★

Outputs
-------
  results/metrics_clean_comparison.csv
  results/metrics_clean_by_length.csv

Usage
-----
  python src/metrics_clean_eval.py               # full test (9,965)
  python src/metrics_clean_eval.py --sample 500  # quick check
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
)
from gnn_transition import load_score_matrix, precompute_full_score_matrix, save_score_matrix, load_transition_model
from two_stage_pipeline import (
    _build_tp_lookup,
    select_matched,
    order_semantic_sort,
    order_greedy_graph,
    order_greedy_gnn,
    order_optimal_perm,
    order_hybrid_rerank,
    find_best_rerank_alpha,
    _bucket_label,
    LENGTH_BUCKETS,
    OPT_PERM_LIMIT,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
ALL_METHODS = [
    "semantic_only",
    "beam",
    "hybrid",
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
    "ts_matched_sem_sort":      "TS-Matched + Sem-Sort",
    "ts_matched_greedy_graph":  "TS-Matched + Greedy-Graph",
    "ts_matched_greedy_gnn":    "TS-Matched + Greedy-GNN",
    "ts_matched_optimal_perm":  "TS-Matched + Optimal-Perm",
    "ts_matched_hybrid_rerank": "TS-Matched + Hybrid-Rerank",
}

TS_MATCHED = frozenset({
    "ts_matched_sem_sort", "ts_matched_greedy_graph",
    "ts_matched_greedy_gnn", "ts_matched_optimal_perm",
    "ts_matched_hybrid_rerank",
})

# ============================================================================
# Clean metrics
# ============================================================================

def compute_set_metrics(predicted: list[str], ground_truth: list[str]) -> dict:
    """
    Selection quality: pure set-overlap, NO truncation.
    Uses the full predicted list as a set — order-independent.
    """
    pred_set = set(predicted)
    gt_set   = set(ground_truth)
    if not pred_set or not gt_set:
        return {"set_precision": 0.0, "set_recall": 0.0, "set_f1": 0.0}

    hits      = len(pred_set & gt_set)
    set_prec  = hits / len(pred_set)
    set_rec   = hits / len(gt_set)
    set_f1    = (2 * set_prec * set_rec / (set_prec + set_rec)
                 if (set_prec + set_rec) > 0 else 0.0)
    return {"set_precision": set_prec, "set_recall": set_rec, "set_f1": set_f1}


def compute_order_metrics(predicted: list[str], ground_truth: list[str]) -> dict:
    """
    Ordering quality on the full predicted list vs ground truth.
    All four metrics naturally operate on correct-set elements only:
      - ordered_precision: fraction of GT ordered pairs preserved
      - lcs_r            : LCS / len(GT) — only matched elements count
      - kendall_tau      : rank corr on pred ∩ GT (already filtered)
      - transition_acc   : consecutive GT transitions present in pred
    """
    gt_set  = set(ground_truth)
    K       = len(ground_truth)

    ord_prec  = _ordered_precision(predicted, ground_truth)
    trans_acc = _transition_accuracy(predicted, ground_truth)

    lcs_val   = _lcs_length(predicted, ground_truth)
    lcs_r     = lcs_val / K if K else 0.0

    # Kendall tau on common tools (already filtered in evaluate.py)
    common = [t for t in predicted if t in gt_set]
    if len(common) >= 2:
        gt_rank          = {t: i for i, t in enumerate(ground_truth)}
        pred_positions   = list(range(len(common)))
        actual_positions = [gt_rank[t] for t in common]
        tau_val, _       = scipy_kendalltau(pred_positions, actual_positions)
        tau = float(tau_val) if not np.isnan(tau_val) else 0.0
    else:
        tau = 0.0

    return {
        "ordered_precision": ord_prec,
        "lcs_r":             lcs_r,
        "kendall_tau":       tau,
        "transition_acc":    trans_acc,
    }


# ============================================================================
# Diagnostic
# ============================================================================

def run_diagnostic(test_records: list[dict]) -> None:
    """
    Print root-cause analysis of F1@K order-dependency.
    """
    n_total    = len(test_records)
    n_affected = sum(1 for r in test_records
                     if max(len(r["tool_sequence"]), 3) > len(r["tool_sequence"]))
    k_gt_dist  = defaultdict(int)
    for r in test_records:
        k_gt_dist[len(r["tool_sequence"])] += 1

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC: F1@K Order-Dependency Analysis")
    print("=" * 70)
    print("""
  compute_metrics() computes F1 as:
      K      = len(ground_truth)
      top_K  = predicted[:K]          ← TRUNCATION
      hits   = |set(top_K) & gt_set|
      F1@K   = 2*hits/(K + |GT|)

  For TS-Matched methods:
      K_oracle = max(K_gt, 3)   → predicted has K_oracle tools
      top_K    = predicted[:K_gt]

  When K_oracle > K_gt  (i.e. K_gt < 3):
      → only the FIRST K_gt tools are evaluated
      → different orderings put different tools first
      → F1@K becomes ORDER-DEPENDENT for a pure-reordering method
""")
    print(f"  K_gt distribution in test set ({n_total:,} samples):")
    for k in sorted(k_gt_dist):
        k_oracle   = max(k, 3)
        affected   = "ORDER-DEPENDENT" if k_oracle > k else "order-independent"
        cnt        = k_gt_dist[k]
        print(f"    K_gt={k}  K_oracle={k_oracle}  affected={cnt:5,} "
              f"({100*cnt/n_total:.1f}%)  [{affected}]")
    print(f"\n  Total affected samples : {n_affected:,} / {n_total:,} "
          f"({100*n_affected/n_total:.1f}%)")
    print("""
  Fix: Set-F1 uses the FULL predicted set (no truncation).
       Guaranteed order-independent — equal for all TS-Matched methods.
""")
    print("=" * 70)


# ============================================================================
# Stage 2 set-integrity check
# ============================================================================

def check_set_integrity(
    rows_by_method: dict[str, list[list[str]]],
) -> None:
    """
    Verify that all TS-Matched methods predict the same TOOL SET per sample.
    Prints number of samples where the set differs from ts_matched_sem_sort.
    """
    base = rows_by_method.get("ts_matched_sem_sort")
    if base is None:
        print("[set integrity] ts_matched_sem_sort not found — skipping")
        return

    print("\n" + "=" * 60)
    print("  Stage 2 Set-Integrity Check")
    print("  (verified: Stage 2 only reorders, never adds/removes tools)")
    print("=" * 60)

    ts_methods = [m for m in ALL_METHODS if m in TS_MATCHED and m != "ts_matched_sem_sort"]
    all_ok = True
    for m in ts_methods:
        preds = rows_by_method.get(m, [])
        mismatch = sum(
            1 for ref, pred in zip(base, preds)
            if set(ref) != set(pred)
        )
        status = "OK  (no set change)" if mismatch == 0 else f"FAIL  ({mismatch} mismatches)"
        label  = METHOD_LABELS.get(m, m)
        print(f"  {label:<38}  {status}")
        if mismatch > 0:
            all_ok = False
    print(f"\n  Overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)


# ============================================================================
# Evaluation loop
# ============================================================================

def evaluate_all(
    test_records:  list[dict],
    planner:       ToolSequencePlanner,
    query_vecs:    np.ndarray,
    score_matrix:  np.ndarray,
    t2i:           dict[str, int],
    tp_lookup:     dict,
    sem_baseline:  SemanticOnlyBaseline,
    alpha_matched: float,
    max_steps:     int   = 8,
    beam_params:   dict | None = None,
    hybrid_params: dict | None = None,
    dijkstra_beta: float = 0.3,
) -> tuple[list[dict], dict[str, list[list[str]]]]:
    """
    Returns (flat_metric_rows, predictions_by_method).
    flat_metric_rows has both set_* and order_* metrics.
    predictions_by_method maps method -> list of predicted sequences.
    """
    bp = beam_params   or {"w1": 0.4, "w2": 0.4, "w3": 0.2, "beam_width": 5}
    hp = hybrid_params or {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}

    matched_cache: dict[int, list] = {}

    def _stage1(i: int, K_oracle: int) -> list[tuple[str, float]]:
        if i not in matched_cache:
            matched_cache[i] = select_matched(planner, query_vecs[i], K_oracle)
        return matched_cache[i]

    all_rows:    list[dict]                 = []
    preds_by_m:  dict[str, list[list[str]]] = {m: [] for m in ALL_METHODS}

    for method in ALL_METHODS:
        print(f"\n  [{method}]  evaluating {len(test_records):,} trajectories ...")
        t_start = time.time()

        for i, rec in enumerate(test_records):
            gt       = rec["tool_sequence"]
            vec      = query_vecs[i]
            K_gt     = len(gt)
            K_oracle = max(K_gt, 3)
            t0       = time.time()

            try:
                if method == "semantic_only":
                    predicted = sem_baseline.predict(vec, K=K_oracle)

                elif method in ("beam", "hybrid"):
                    plans     = _plan_with_vec(
                        planner, method, vec, max_steps=max_steps,
                        beta=dijkstra_beta,
                        beam_params=bp, hybrid_params=hp,
                    )
                    predicted = plans[0].tools if plans else []

                elif method == "ts_matched_sem_sort":
                    ts        = _stage1(i, K_oracle)
                    predicted = order_semantic_sort(ts)

                elif method == "ts_matched_greedy_graph":
                    ts        = _stage1(i, K_oracle)
                    predicted = order_greedy_graph(ts, tp_lookup)

                elif method == "ts_matched_greedy_gnn":
                    ts        = _stage1(i, K_oracle)
                    predicted = order_greedy_gnn(ts, score_matrix, t2i)

                elif method == "ts_matched_optimal_perm":
                    ts        = _stage1(i, K_oracle)
                    predicted = order_optimal_perm(ts, tp_lookup)

                elif method == "ts_matched_hybrid_rerank":
                    ts        = _stage1(i, K_oracle)
                    predicted = order_hybrid_rerank(ts, tp_lookup, alpha_matched)

                else:
                    predicted = []

            except Exception as exc:
                print(f"    [warn] {method} row {i}: {exc}")
                predicted = []

            latency   = time.time() - t0
            set_m     = compute_set_metrics(predicted, gt)
            order_m   = compute_order_metrics(predicted, gt)
            row       = {**set_m, **order_m,
                         "latency_ms": latency * 1000.0,
                         "pred_len":   len(predicted),
                         "gt_len":     K_gt,
                         "method":     method,
                         "bucket":     _bucket_label(K_gt)}
            all_rows.append(row)
            preds_by_m[method].append(list(predicted))

        elapsed = time.time() - t_start
        n_done  = sum(1 for r in all_rows if r["method"] == method)
        avg_len = sum(r["pred_len"] for r in all_rows if r["method"] == method) / max(1, n_done)
        print(f"  [{method}] done in {elapsed:.1f}s  avg_pred_len={avg_len:.2f}")

    return all_rows, preds_by_m


# ============================================================================
# Invariant validation
# ============================================================================

def validate_set_f1_invariant(rows: list[dict]) -> None:
    """All TS-Matched methods must have identical mean set_f1."""
    df = pd.DataFrame(rows)

    base_m  = "ts_matched_sem_sort"
    base_f1 = df[df["method"] == base_m]["set_f1"].mean()

    print("\n" + "=" * 65)
    print("  Set-F1 Invariant Validation")
    print("  (all TS-Matched methods must share the same tool set)")
    print("=" * 65)
    print(f"  {'Method':<38}  {'Set-F1':>8}  {'Delta':>8}  Status")
    print("  " + "-" * 60)

    all_ok = True
    for m in ALL_METHODS:
        if m not in TS_MATCHED:
            continue
        f1   = df[df["method"] == m]["set_f1"].mean()
        diff = abs(f1 - base_f1)
        ok   = diff < 1e-6
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        sign = "+" if (f1 - base_f1) >= 0 else ""
        print(f"  {METHOD_LABELS.get(m, m):<38}  {f1:.6f}  "
              f"{sign}{f1-base_f1:.6f}  {mark}")

    print(f"\n  Overall: {'PASS [OK]  Set-F1 is order-independent' if all_ok else 'FAIL [!!]  Debug needed'}")
    print("=" * 65)


# ============================================================================
# Output helpers
# ============================================================================

SET_METRICS   = ["set_precision", "set_recall", "set_f1"]
ORDER_METRICS = ["ordered_precision", "lcs_r", "kendall_tau", "transition_acc"]


def make_summaries(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df  = pd.DataFrame(rows)
    ord = {m: i for i, m in enumerate(ALL_METHODS)}

    def _agg(cols):
        agg = df.groupby("method")[cols].mean().round(4).reset_index()
        agg["_o"] = agg["method"].map(ord)
        agg = agg.sort_values("_o").drop(columns="_o")
        agg.insert(1, "label", agg["method"].map(METHOD_LABELS).fillna(agg["method"]))
        return agg

    return _agg(SET_METRICS), _agg(ORDER_METRICS)


def make_by_length_summaries(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df  = pd.DataFrame(rows)
    mordmap  = {m: i for i, m in enumerate(ALL_METHODS)}
    bmap = {b[0]: i for i, b in enumerate(LENGTH_BUCKETS)}

    def _agg(cols):
        agg = df.groupby(["method", "bucket"])[cols].mean().round(4).reset_index()
        agg["_m"] = agg["method"].map(mordmap)
        agg["_b"] = agg["bucket"].map(bmap)
        agg = agg.sort_values(["_m", "_b"]).drop(columns=["_m", "_b"])
        agg.insert(2, "label", agg["method"].map(METHOD_LABELS).fillna(agg["method"]))
        return agg

    return _agg(SET_METRICS), _agg(ORDER_METRICS)


def _table(df: pd.DataFrame, value_cols: list[str], headers: list[str],
           title: str, note: str = "") -> None:
    sub = df[["label"] + value_cols].copy()
    sub.columns = ["Method"] + headers

    col_w = [
        max(len(h), max(
            len(f"{v:.4f}") if isinstance(v, float) else len(str(v))
            for v in sub[h]
        )) + 2
        for h in ["Method"] + headers
    ]
    sep    = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(["Method"] + headers, col_w)) + " |"

    print(f"\n  {title}")
    print(sep); print(header); print(sep)
    prev_ts = False
    for idx, (_, row) in enumerate(sub.iterrows()):
        m_id = ALL_METHODS[idx]
        is_ts = m_id in TS_MATCHED
        if is_ts and not prev_ts:
            print(sep)
        cells = [(f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(w)
                 for v, w in zip(row, col_w)]
        print("| " + " | ".join(cells) + " |")
        prev_ts = is_ts
    print(sep)
    if note:
        print(f"  {note}")


def print_bucket_tables(
    by_len_set:   pd.DataFrame,
    by_len_order: pd.DataFrame,
) -> None:
    buckets = [b[0] for b in LENGTH_BUCKETS]

    for title, df, cols in [
        ("Table A (by length): Selection  (Set-F1)",
         by_len_set,   ["set_f1"]),
        ("Table B (by length): Ordering   (Ordered-Prec / Kendall-Tau)",
         by_len_order, ["ordered_precision", "kendall_tau"]),
    ]:
        print(f"\n  {title}")
        hdr = f"  {'Method':<38}" + "".join(
            f"  {('[' + b + ']'):>16}" for b in buckets
        )
        print(hdr)
        print("  " + "-" * (38 + 19 * len(buckets)))
        for m in ALL_METHODS:
            label = METHOD_LABELS.get(m, m)
            row_s = f"  {label:<38}"
            for b in buckets:
                sub = df[(df["method"] == m) & (df["bucket"] == b)]
                if sub.empty:
                    row_s += f"  {'---':>16}"
                else:
                    vals = "  /  ".join(f"{sub[c].iloc[0]:.3f}" for c in cols)
                    row_s += f"  {vals:>16}"
            print(row_s)
        print()


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean Metrics Evaluation for Two-Stage Pipeline",
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
    print("Loading trajectories ...")
    records = load_trajectories()
    train_records, test_records = make_train_test_split(records, seed=args.seed)

    avg_train_len    = sum(r["num_steps"] for r in train_records) / len(train_records)
    median_train_len = float(statistics.median(
        r["num_steps"] for r in train_records
    ))

    if args.sample > 0:
        rng.shuffle(test_records)
        test_records = test_records[: args.sample]

    # ── 2. Root-cause diagnostic ──────────────────────────────────────────
    run_diagnostic(test_records)

    # ── 3. Load planner ───────────────────────────────────────────────────
    print("Loading ToolSequencePlanner ...")
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

    # ── 4. Load GNN score matrix ─────────────────────────────────────────
    try:
        score_matrix, active_tools, t2i = load_score_matrix(args.encoder)
    except FileNotFoundError:
        print("\n[info] Score matrix not found — computing ...")
        model, _, _ = load_transition_model(args.encoder, device_str="auto")
        score_matrix, active_tools = precompute_full_score_matrix(
            model, device_str="auto"
        )
        save_score_matrix(score_matrix, active_tools, encoder_type=args.encoder)
        t2i = {name: i for i, name in enumerate(active_tools)}

    # ── 5. Build TP lookup + baselines ───────────────────────────────────
    tp_lookup    = _build_tp_lookup(planner)
    sem_baseline = SemanticOnlyBaseline(planner)

    # ── 6. Encode queries ─────────────────────────────────────────────────
    print("\nBatch-encoding test queries ...")
    test_vecs = batch_encode_queries(
        [r["task_description"] for r in test_records]
    )

    # ── 7. Val set + alpha search ─────────────────────────────────────────
    val_idx     = list(range(len(train_records)))
    rng.shuffle(val_idx)
    val_records = [train_records[i] for i in val_idx[: args.val_n]]
    val_vecs    = batch_encode_queries(
        [r["task_description"] for r in val_records]
    )
    alpha = find_best_rerank_alpha(
        planner, val_records, val_vecs, tp_lookup, mode="matched"
    )

    # ── 8. Evaluate ───────────────────────────────────────────────────────
    print(f"\nRunning {len(test_records):,} x {len(ALL_METHODS)} methods ...")
    rows, preds_by_m = evaluate_all(
        test_records, planner, test_vecs,
        score_matrix, t2i, tp_lookup,
        sem_baseline,
        alpha_matched=alpha,
        max_steps=args.max_steps,
    )

    # ── 9. Validate set integrity and F1 invariant ────────────────────────
    check_set_integrity(preds_by_m)
    validate_set_f1_invariant(rows)

    # ── 10. Build summaries ───────────────────────────────────────────────
    sum_set, sum_ord         = make_summaries(rows)
    bl_set,  bl_ord          = make_by_length_summaries(rows)

    # Merge for CSV
    merged = sum_set.merge(sum_ord, on=["method", "label"])
    merged.to_csv(RESULTS_DIR / "metrics_clean_comparison.csv", index=False)
    bl_set.merge(bl_ord, on=["method", "bucket", "label"]).to_csv(
        RESULTS_DIR / "metrics_clean_by_length.csv", index=False
    )

    # ── 11. Print tables ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  Clean Metrics Evaluation  |  {len(test_records):,} test trajectories  |  "
          f"Encoder: {args.encoder.upper()}  |  alpha={alpha:.1f}")
    print("=" * 80)

    _table(
        sum_set, SET_METRICS,
        ["Set-Prec", "Set-Recall", "Set-F1"],
        "Table A: Selection Quality (set-based, order-independent)",
        note="All TS-Matched rows must have identical Set-F1 = Semantic Only Set-F1",
    )

    _table(
        sum_ord, ORDER_METRICS,
        ["Ord.Prec", "LCS-R", "Kendall-Tau", "Trans.Acc"],
        "Table B: Ordering Quality (order-dependent, evaluated on pred ∩ GT)",
        note="Higher = better ordering of the correctly-selected tools",
    )

    print_bucket_tables(bl_set, bl_ord)

    print(f"  CSV (summary)   -> {RESULTS_DIR / 'metrics_clean_comparison.csv'}")
    print(f"  CSV (by length) -> {RESULTS_DIR / 'metrics_clean_by_length.csv'}")


if __name__ == "__main__":
    main()
