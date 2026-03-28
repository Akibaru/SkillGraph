"""
src/final_comparison.py  —  Optimal Combination + Paper Results
===============================================================
Combines the best Stage 1 (Hybrid Sem-Graph's tool set) with the best
Stage 2 ordering strategies, producing the definitive result table.

Stage 1 options
---------------
  Sem  : semantic top-K (oracle K = max(K_gt, 3))
  Hybrid: tools from Hybrid Sem-Graph's greedy output  ← best Set-F1

Stage 2 options applied to each Stage 1 set
--------------------------------------------
  Sem-Sort       : sort by cosine similarity desc  (order-independent baseline)
  Optimal-Perm   : enumerate K! permutations, max log-tp-sum
  Hybrid-Rerank  : alpha * Σ tp + (1-alpha) * Σ sim/(i+1), alpha tuned on val

Methods (8)
-----------
  1. Semantic Only                [single-stage, reference]
  2. Beam Search                  [single-stage, best ordering]
  3. Hybrid Sem-Graph             [single-stage, best selection]
  4. TS-Sem  + Sem-Sort           [= Semantic Only, sanity check]
  5. TS-Sem  + Hybrid-Rerank
  6. TS-Hybrid + Sem-Sort    ★   [Hybrid set + semantic order]
  7. TS-Hybrid + Optimal-Perm ★
  8. TS-Hybrid + Hybrid-Rerank ★  [predicted best combination]

Outputs
-------
  results/final_comparison.csv
  results/final_by_length.csv
  results/bootstrap_significance.csv

Usage
-----
  python src/final_comparison.py               # full test (9,965)
  python src/final_comparison.py --sample 500  # quick check
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
    _first_tool_accuracy,
)
from gnn_transition import (
    load_score_matrix,
    precompute_full_score_matrix,
    save_score_matrix,
    load_transition_model,
)
from two_stage_pipeline import (
    _build_tp_lookup,
    select_matched,
    order_semantic_sort,
    order_greedy_gnn,
    order_optimal_perm,
    order_hybrid_rerank,
    _bucket_label,
    LENGTH_BUCKETS,
)
from learned_reranker import (
    build_position_stats,
    train_learned_reranker,
    load_learned_reranker,
    order_learned_rerank,
    CHECKPOINT as LR_CHECKPOINT,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OPT_PERM_LIMIT = 8

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
ALL_METHODS = [
    "semantic_only",
    "beam",
    "hybrid",
    "ts_sem_semsort",
    "ts_sem_hybrid_rerank",
    "ts_hybrid_semsort",
    "ts_hybrid_optimal_perm",
    "ts_hybrid_hybrid_rerank",
    "ts_hybrid_learned_rerank",
]

METHOD_LABELS = {
    "semantic_only":              "Semantic Only",
    "beam":                       "Beam Search",
    "hybrid":                     "Hybrid Sem-Graph",
    "ts_sem_semsort":             "TS-Sem + Sem-Sort",
    "ts_sem_hybrid_rerank":       "TS-Sem + Hybrid-Rerank",
    "ts_hybrid_semsort":          "TS-Hybrid + Sem-Sort ★",
    "ts_hybrid_optimal_perm":     "TS-Hybrid + Optimal-Perm ★",
    "ts_hybrid_hybrid_rerank":    "TS-Hybrid + Hybrid-Rerank ★",
    "ts_hybrid_learned_rerank":   "TS-Hybrid + Learned-Rerank ★",
}

SEM_METHODS    = frozenset({"semantic_only", "ts_sem_semsort", "ts_sem_hybrid_rerank"})
HYBRID_METHODS = frozenset({"hybrid", "ts_hybrid_semsort", "ts_hybrid_optimal_perm",
                             "ts_hybrid_hybrid_rerank", "ts_hybrid_learned_rerank"})

BUCKET_FOCUS = ["semantic_only", "beam", "hybrid",
                "ts_hybrid_hybrid_rerank", "ts_hybrid_learned_rerank"]

BOOTSTRAP_PAIRS = [
    ("ts_hybrid_hybrid_rerank",  "hybrid",                  "TS-Hybrid+HR vs Hybrid Sem-Graph"),
    ("ts_hybrid_hybrid_rerank",  "beam",                    "TS-Hybrid+HR vs Beam Search"),
    ("ts_sem_hybrid_rerank",     "ts_sem_semsort",          "TS-Sem+HR vs TS-Sem+SemSort"),
    ("ts_hybrid_learned_rerank", "ts_hybrid_hybrid_rerank", "TS-Hybrid+LR vs TS-Hybrid+HR"),
    ("ts_hybrid_learned_rerank", "hybrid",                  "TS-Hybrid+LR vs Hybrid Sem-Graph"),
]

SET_METRICS   = ["set_precision", "set_recall", "set_f1"]
ORDER_METRICS = ["ordered_precision", "lcs_r", "kendall_tau", "transition_acc", "first_tool_acc"]


# ============================================================================
# Metric functions
# ============================================================================

def compute_set_metrics(predicted: list[str], ground_truth: list[str]) -> dict:
    pred_set = set(predicted)
    gt_set   = set(ground_truth)
    if not pred_set or not gt_set:
        return {"set_precision": 0.0, "set_recall": 0.0, "set_f1": 0.0}
    hits     = len(pred_set & gt_set)
    sp       = hits / len(pred_set)
    sr       = hits / len(gt_set)
    sf1      = 2 * sp * sr / (sp + sr) if (sp + sr) > 0 else 0.0
    return {"set_precision": sp, "set_recall": sr, "set_f1": sf1}


def compute_order_metrics(predicted: list[str], ground_truth: list[str]) -> dict:
    gt_set    = set(ground_truth)
    K         = len(ground_truth)
    ord_prec  = _ordered_precision(predicted, ground_truth)
    trans_acc = _transition_accuracy(predicted, ground_truth)
    first_acc = _first_tool_accuracy(predicted, ground_truth)
    lcs_val   = _lcs_length(predicted, ground_truth)
    lcs_r     = lcs_val / K if K else 0.0

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
        "first_tool_acc":    first_acc,
    }


# ============================================================================
# Stage 1 extractors
# ============================================================================

def get_sem_stage1(
    planner:   ToolSequencePlanner,
    query_vec: np.ndarray,
    K_oracle:  int,
) -> list[tuple[str, float]]:
    """Semantic top-K selection, oracle K per sample."""
    return planner._top_entry_tools(query_vec, k=K_oracle)


def get_hybrid_stage1(
    planner:       ToolSequencePlanner,
    query_vec:     np.ndarray,
    hybrid_params: dict,
    max_steps:     int = 8,
) -> list[tuple[str, float]]:
    """
    Extract Hybrid Sem-Graph's output tool set (not its ordering).
    Returns [(tool, sim_score), ...] sorted by sim descending —
    the same TOOLS as Hybrid's output, ready for Stage 2.
    """
    plans = _plan_with_vec(
        planner, "hybrid", query_vec,
        max_steps=max_steps,
        hybrid_params=hybrid_params,
    )
    if not plans or not plans[0].tools:
        return planner._top_entry_tools(query_vec, k=3)

    plan_tools = plans[0].tools   # already truncated to target by hybrid
    tools_sims = sorted(
        [(t, float(planner._tool_sim(t, query_vec))) for t in plan_tools],
        key=lambda x: -x[1],
    )
    return tools_sims


# ============================================================================
# Alpha grid search
# ============================================================================

def find_alpha(
    planner:       ToolSequencePlanner,
    val_records:   list[dict],
    val_vecs:      np.ndarray,
    tp_lookup:     dict,
    stage:         str,            # "sem" or "hybrid"
    hybrid_params: dict,
    max_steps:     int   = 8,
    alphas:        list[float] | None = None,
) -> float:
    if alphas is None:
        alphas = [round(a, 1) for a in np.arange(0.0, 1.1, 0.1)]

    print(f"\n[alpha search / {stage}]  Val={len(val_records)}  "
          f"candidates={alphas}")

    best_alpha, best_ord = 0.5, -1.0

    for alpha in alphas:
        ord_list = []
        for i, rec in enumerate(val_records):
            gt       = rec["tool_sequence"]
            vec      = val_vecs[i]
            K_oracle = max(len(gt), 3)
            try:
                if stage == "sem":
                    ts = get_sem_stage1(planner, vec, K_oracle)
                else:
                    ts = get_hybrid_stage1(planner, vec, hybrid_params, max_steps)
                seq = order_hybrid_rerank(ts, tp_lookup, alpha)
                m   = compute_order_metrics(seq, gt)
                ord_list.append(m["ordered_precision"])
            except Exception:
                ord_list.append(0.0)

        mean_ord = float(np.mean(ord_list))
        marker   = "  <-" if mean_ord > best_ord else ""
        print(f"  alpha={alpha:.1f}  val_OrdPrec={mean_ord:.4f}{marker}")
        if mean_ord > best_ord:
            best_ord, best_alpha = mean_ord, alpha

    print(f"[alpha search / {stage}] Best alpha={best_alpha:.1f}  "
          f"val_OrdPrec={best_ord:.4f}")
    return best_alpha


# ============================================================================
# Evaluation loop
# ============================================================================

def evaluate_all(
    test_records:    list[dict],
    planner:         ToolSequencePlanner,
    query_vecs:      np.ndarray,
    score_matrix:    np.ndarray,
    t2i:             dict[str, int],
    tp_lookup:       dict,
    sem_baseline:    SemanticOnlyBaseline,
    alpha_sem:       float,
    alpha_hybrid:    float,
    hybrid_params:   dict,
    max_steps:       int   = 8,
    beam_params:     dict | None = None,
    dijkstra_beta:   float = 0.3,
    learned_model           = None,   # PairwiseMLP | None
    position_stats:  dict  = None,
) -> tuple[list[dict], dict[str, list[list[str]]]]:
    """
    Returns (flat_rows, preds_by_method).
    Each row has set_* and order_* metrics + method, bucket, gt_len.
    """
    bp = beam_params or {"w1": 0.4, "w2": 0.4, "w3": 0.2, "beam_width": 5}
    _pos_stats = position_stats or {}

    # Per-sample caches
    sem_cache:    dict[int, list] = {}
    hybrid_cache: dict[int, list] = {}

    def _sem(i: int, K_oracle: int) -> list[tuple[str, float]]:
        if i not in sem_cache:
            sem_cache[i] = get_sem_stage1(planner, query_vecs[i], K_oracle)
        return sem_cache[i]

    def _hyb(i: int) -> list[tuple[str, float]]:
        if i not in hybrid_cache:
            hybrid_cache[i] = get_hybrid_stage1(
                planner, query_vecs[i], hybrid_params, max_steps
            )
        return hybrid_cache[i]

    all_rows:   list[dict]                 = []
    preds_by_m: dict[str, list[list[str]]] = {m: [] for m in ALL_METHODS}

    for method in ALL_METHODS:
        # Skip learned rerank if model not available
        if method == "ts_hybrid_learned_rerank" and learned_model is None:
            print(f"\n  [{method}]  SKIPPED (model not trained)")
            for rec in test_records:
                preds_by_m[method].append([])
                all_rows.append({
                    **compute_set_metrics([], rec["tool_sequence"]),
                    **compute_order_metrics([], rec["tool_sequence"]),
                    "latency_ms": 0.0,
                    "pred_len":   0,
                    "gt_len":     len(rec["tool_sequence"]),
                    "method":     method,
                    "bucket":     _bucket_label(len(rec["tool_sequence"])),
                })
            continue

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

                elif method == "beam":
                    plans     = _plan_with_vec(
                        planner, "beam", vec, max_steps=max_steps,
                        beta=dijkstra_beta, beam_params=bp,
                    )
                    predicted = plans[0].tools if plans else []

                elif method == "hybrid":
                    plans     = _plan_with_vec(
                        planner, "hybrid", vec, max_steps=max_steps,
                        beta=dijkstra_beta, hybrid_params=hybrid_params,
                    )
                    predicted = plans[0].tools if plans else []

                # ── TS-Sem (oracle K semantic selection) ──────────────────
                elif method == "ts_sem_semsort":
                    ts        = _sem(i, K_oracle)
                    predicted = order_semantic_sort(ts)

                elif method == "ts_sem_hybrid_rerank":
                    ts        = _sem(i, K_oracle)
                    predicted = order_hybrid_rerank(ts, tp_lookup, alpha_sem)

                # ── TS-Hybrid (Hybrid's tool set, different orderings) ─────
                elif method == "ts_hybrid_semsort":
                    ts        = _hyb(i)
                    predicted = order_semantic_sort(ts)

                elif method == "ts_hybrid_optimal_perm":
                    ts        = _hyb(i)
                    predicted = order_optimal_perm(ts, tp_lookup)

                elif method == "ts_hybrid_hybrid_rerank":
                    ts        = _hyb(i)
                    predicted = order_hybrid_rerank(ts, tp_lookup, alpha_hybrid)

                elif method == "ts_hybrid_learned_rerank":
                    ts        = _hyb(i)
                    predicted = order_learned_rerank(
                        ts, tp_lookup, _pos_stats, learned_model
                    )

                else:
                    predicted = []

            except Exception as exc:
                print(f"    [warn] {method} row {i}: {exc}")
                predicted = []

            latency = time.time() - t0
            row     = {
                **compute_set_metrics(predicted, gt),
                **compute_order_metrics(predicted, gt),
                "latency_ms": latency * 1000.0,
                "pred_len":   len(predicted),
                "gt_len":     K_gt,
                "method":     method,
                "bucket":     _bucket_label(K_gt),
            }
            all_rows.append(row)
            preds_by_m[method].append(list(predicted))

        elapsed = time.time() - t_start
        n_done  = len([r for r in all_rows if r["method"] == method])
        avg_len = sum(r["pred_len"] for r in all_rows if r["method"] == method) / max(1, n_done)
        print(f"  [{method}] done in {elapsed:.1f}s  avg_pred_len={avg_len:.2f}")

    return all_rows, preds_by_m


# ============================================================================
# Validations
# ============================================================================

def validate_sanity(rows: list[dict], preds_by_m: dict) -> None:
    df = pd.DataFrame(rows)

    print("\n" + "=" * 68)
    print("  Validation Checks")
    print("=" * 68)

    # 1. TS-Sem+SemSort == Semantic Only (all metrics)
    for metric in SET_METRICS + ORDER_METRICS:
        v1 = df[df["method"] == "semantic_only"][metric].mean()
        v2 = df[df["method"] == "ts_sem_semsort"][metric].mean()
        ok = abs(v1 - v2) < 1e-5
        print(f"  TS-Sem+SemSort == Semantic Only [{metric:>20}]: "
              f"{v1:.6f} vs {v2:.6f}  {'PASS' if ok else 'FAIL'}")

    print()

    # 2. All TS-Hybrid methods share the same Set-F1
    ref_set_f1 = df[df["method"] == "ts_hybrid_semsort"]["set_f1"].mean()
    hybrid_set_f1 = df[df["method"] == "hybrid"]["set_f1"].mean()
    print(f"  Hybrid Sem-Graph   Set-F1: {hybrid_set_f1:.6f}")
    for m in ["ts_hybrid_semsort", "ts_hybrid_optimal_perm",
              "ts_hybrid_hybrid_rerank", "ts_hybrid_learned_rerank"]:
        if m not in df["method"].values:
            continue
        v  = df[df["method"] == m]["set_f1"].mean()
        ok = abs(v - ref_set_f1) < 1e-6
        print(f"  {METHOD_LABELS[m]:<42}  Set-F1={v:.6f}  {'PASS' if ok else 'FAIL'}")

    print()

    # 3. TS-Hybrid set integrity (all preds have same tool set per sample)
    base = preds_by_m["ts_hybrid_semsort"]
    for m in ["ts_hybrid_optimal_perm", "ts_hybrid_hybrid_rerank",
              "ts_hybrid_learned_rerank"]:
        preds = preds_by_m[m]
        if all(len(p) == 0 for p in preds):
            print(f"  TS-Hybrid set integrity [{m}]: SKIPPED (no predictions)")
            continue
        mismatch = sum(
            1 for ref, p in zip(base, preds)
            if p and set(ref) != set(p)
        )
        ok = mismatch == 0
        print(f"  TS-Hybrid set integrity [{m}]: "
              f"{'PASS (0 mismatches)' if ok else f'FAIL ({mismatch} mismatches)'}")

    print("=" * 68)


# ============================================================================
# Bootstrap significance
# ============================================================================

def bootstrap_paired_diff(
    a:      np.ndarray,
    b:      np.ndarray,
    n_boot: int   = 10_000,
    seed:   int   = 42,
) -> dict:
    """Bootstrap 95% CI for mean(a) - mean(b) on paired samples."""
    rng      = np.random.default_rng(seed)
    n        = len(a)
    obs_diff = float(a.mean() - b.mean())

    boot = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx     = rng.integers(0, n, size=n)
        boot[k] = a[idx].mean() - b[idx].mean()

    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    p_val        = float(np.mean(boot <= 0))   # one-sided P(diff <= 0)

    return {
        "obs_diff":       obs_diff,
        "ci_lo":          ci_lo,
        "ci_hi":          ci_hi,
        "p_one_sided":    p_val,
        "significant_95": ci_lo > 0,
    }


def run_bootstrap(rows: list[dict], n_boot: int = 10_000,
                  metrics: list[str] | None = None) -> pd.DataFrame:
    """Run bootstrap CI for all configured pairs on specified metrics."""
    if metrics is None:
        metrics = ["ordered_precision"]
    df      = pd.DataFrame(rows)
    records = []

    for metric in metrics:
        print(f"\n[bootstrap]  {n_boot:,} resamples  metric={metric}")
        print(f"  {'Comparison':<42}  {'Δ':>8}  {'95% CI':>20}  {'p':>7}  Sig?")
        print("  " + "-" * 80)

        for m_a, m_b, label in BOOTSTRAP_PAIRS:
            a = df[df["method"] == m_a][metric].values
            b = df[df["method"] == m_b][metric].values
            if len(a) == 0 or len(b) == 0:
                continue
            assert len(a) == len(b), f"length mismatch: {m_a} vs {m_b}"

            res = bootstrap_paired_diff(a, b, n_boot=n_boot)
            sig = "YES" if res["significant_95"] else "no"

            print(f"  {label:<42}  {res['obs_diff']:+.4f}  "
                  f"[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  "
                  f"{res['p_one_sided']:.4f}  {sig}")

            records.append({
                "metric":        metric,
                "comparison":    label,
                "method_a":      m_a,
                "method_b":      m_b,
                "obs_diff":      round(res["obs_diff"],    4),
                "ci_lo":         round(res["ci_lo"],       4),
                "ci_hi":         round(res["ci_hi"],       4),
                "p_one_sided":   round(res["p_one_sided"], 4),
                "significant_95": res["significant_95"],
            })

    return pd.DataFrame(records)


# ============================================================================
# Output helpers
# ============================================================================

def make_summary(rows: list[dict]) -> pd.DataFrame:
    df  = pd.DataFrame(rows)
    ord = {m: i for i, m in enumerate(ALL_METHODS)}
    all_m = SET_METRICS + ORDER_METRICS
    agg = df.groupby("method")[all_m].mean().round(4).reset_index()
    agg["_o"] = agg["method"].map(ord)
    agg = agg.sort_values("_o").drop(columns="_o")
    agg.insert(1, "label", agg["method"].map(METHOD_LABELS).fillna(agg["method"]))
    return agg


def make_by_length(rows: list[dict], focus_methods: list[str]) -> pd.DataFrame:
    df   = pd.DataFrame(rows)
    df   = df[df["method"].isin(focus_methods)]
    mordmap = {m: i for i, m in enumerate(focus_methods)}
    bmap    = {b[0]: i for i, b in enumerate(LENGTH_BUCKETS)}
    all_m   = SET_METRICS + ORDER_METRICS
    agg = df.groupby(["method", "bucket"])[all_m].mean().round(4).reset_index()
    agg["_m"] = agg["method"].map(mordmap)
    agg["_b"] = agg["bucket"].map(bmap)
    agg = agg.sort_values(["_m", "_b"]).drop(columns=["_m", "_b"])
    agg.insert(2, "label", agg["method"].map(METHOD_LABELS).fillna(agg["method"]))
    return agg


def _print_table(summary: pd.DataFrame, cols: list[str], hdrs: list[str],
                 title: str, note: str = "") -> None:
    sub = summary[["label"] + cols].copy()
    sub.columns = ["Method"] + hdrs

    col_w = [
        max(len(h), max(
            len(f"{v:.4f}") if isinstance(v, float) else len(str(v))
            for v in sub[h]
        )) + 2
        for h in ["Method"] + hdrs
    ]
    sep    = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    hdr_ln = "| " + " | ".join(h.ljust(w) for h, w in zip(["Method"] + hdrs, col_w)) + " |"

    print(f"\n  {title}")
    print(sep); print(hdr_ln); print(sep)
    prev_group = None
    for idx, (_, row) in enumerate(sub.iterrows()):
        m_id  = ALL_METHODS[idx]
        group = ("single" if m_id in {"semantic_only", "beam", "hybrid"}
                 else "ts_sem" if m_id in {"ts_sem_semsort", "ts_sem_hybrid_rerank"}
                 else "ts_hybrid")
        if prev_group is not None and group != prev_group:
            print(sep)
        cells = [(f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(w)
                 for v, w in zip(row, col_w)]
        print("| " + " | ".join(cells) + " |")
        prev_group = group
    print(sep)
    if note:
        print(f"  {note}")


def _print_bucket_table(by_length: pd.DataFrame, metric: str, title: str) -> None:
    buckets = [b[0] for b in LENGTH_BUCKETS]
    methods = by_length["method"].unique().tolist()

    print(f"\n  {title}")
    hdr = f"  {'Method':<36}" + "".join(f"  {('[' + b + ']'):>10}" for b in buckets)
    print(hdr)
    print("  " + "-" * (36 + 13 * len(buckets)))
    for m in BUCKET_FOCUS:
        if m not in methods:
            continue
        label = METHOD_LABELS.get(m, m)
        row_s = f"  {label:<36}"
        for b in buckets:
            sub = by_length[(by_length["method"] == m) & (by_length["bucket"] == b)]
            row_s += f"  {sub[metric].iloc[0]:.3f}" if not sub.empty else f"  {'---':>10}"
        print(row_s)
    print()


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Final Comparison: Optimal Stage1+Stage2 Combination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sample",      type=int,  default=0)
    p.add_argument("--max-steps",   type=int,  default=8)
    p.add_argument("--encoder",     type=str,  default="sage",
                   choices=["gcn", "gat", "sage"])
    p.add_argument("--val-n",       type=int,  default=500)
    p.add_argument("--n-boot",      type=int,  default=10_000)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--retrain-lr",  action="store_true",
                   help="Force retraining of the learned reranker even if checkpoint exists")
    return p.parse_args()


def main() -> None:
    args         = _parse_args()
    hybrid_params = {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}
    rng          = random.Random(args.seed)
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

    bucket_counts = defaultdict(int)
    for rec in test_records:
        bucket_counts[_bucket_label(len(rec["tool_sequence"]))] += 1
    print(f"  {len(test_records):,} test samples  |  "
          + "  ".join(f"[{b}]={bucket_counts[b]}" for b, *_ in LENGTH_BUCKETS))

    # ── 2. Load planner ───────────────────────────────────────────────────
    print("\nLoading ToolSequencePlanner ...")
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

    # ── 3. Load GNN matrix + TP lookup ───────────────────────────────────
    try:
        score_matrix, _, t2i = load_score_matrix(args.encoder)
    except FileNotFoundError:
        print("\n[info] Score matrix not found - computing ...")
        model, _, _ = load_transition_model(args.encoder, device_str="auto")
        score_matrix, active_tools = precompute_full_score_matrix(
            model, device_str="auto"
        )
        save_score_matrix(score_matrix, active_tools, encoder_type=args.encoder)
        t2i = {name: i for i, name in enumerate(active_tools)}

    tp_lookup    = _build_tp_lookup(planner)
    sem_baseline = SemanticOnlyBaseline(planner)

    # ── 4. Encode queries ─────────────────────────────────────────────────
    print("\nBatch-encoding test queries ...")
    test_vecs = batch_encode_queries(
        [r["task_description"] for r in test_records]
    )

    # ── 5. Val set ────────────────────────────────────────────────────────
    val_idx     = list(range(len(train_records)))
    rng.shuffle(val_idx)
    val_records = [train_records[i] for i in val_idx[: args.val_n]]
    val_vecs    = batch_encode_queries(
        [r["task_description"] for r in val_records]
    )

    # ── 6. Alpha searches ─────────────────────────────────────────────────
    alpha_sem    = find_alpha(planner, val_records, val_vecs, tp_lookup,
                              stage="sem",    hybrid_params=hybrid_params,
                              max_steps=args.max_steps)
    alpha_hybrid = find_alpha(planner, val_records, val_vecs, tp_lookup,
                              stage="hybrid", hybrid_params=hybrid_params,
                              max_steps=args.max_steps)

    # ── 6b. Train Learned Reranker ────────────────────────────────────────
    position_stats = build_position_stats(train_records)

    if LR_CHECKPOINT.exists() and not args.retrain_lr:
        print(f"\n[LearnedReranker]  Loading from {LR_CHECKPOINT}")
        learned_model = load_learned_reranker(LR_CHECKPOINT)
    else:
        print(f"\n[LearnedReranker]  Training on {len(train_records):,} trajectories ...")
        # Encode ALL training queries (needed for pair features)
        train_vecs = batch_encode_queries(
            [r["task_description"] for r in train_records]
        )
        learned_model = train_learned_reranker(
            train_records   = train_records,
            query_vecs      = train_vecs,
            tp_lookup       = tp_lookup,
            position_stats  = position_stats,
            planner         = planner,
            max_epochs      = 30,
            batch_size      = 2048,
            lr              = 1e-3,
            patience        = 5,
            seed            = args.seed,
            verbose         = True,
            save_path       = LR_CHECKPOINT,
        )

    # ── 7. Full evaluation ────────────────────────────────────────────────
    print(f"\nRunning {len(test_records):,} x {len(ALL_METHODS)} methods ...")
    rows, preds_by_m = evaluate_all(
        test_records, planner, test_vecs,
        score_matrix, t2i, tp_lookup, sem_baseline,
        alpha_sem=alpha_sem,
        alpha_hybrid=alpha_hybrid,
        hybrid_params=hybrid_params,
        max_steps=args.max_steps,
        learned_model=learned_model,
        position_stats=position_stats,
    )

    # ── 8. Validations ────────────────────────────────────────────────────
    validate_sanity(rows, preds_by_m)

    # ── 9. Bootstrap significance ─────────────────────────────────────────
    boot_df = run_bootstrap(rows, n_boot=args.n_boot,
                            metrics=["ordered_precision", "kendall_tau"])

    # ── 10. Summaries ─────────────────────────────────────────────────────
    summary   = make_summary(rows)
    by_length = make_by_length(rows, BUCKET_FOCUS)

    # ── 11. Print tables ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  Final Results  |  {len(test_records):,} test  |  "
          f"Encoder: {args.encoder.upper()}  |  "
          f"alpha_sem={alpha_sem:.1f}  alpha_hybrid={alpha_hybrid:.1f}  "
          f"LR={'trained' if learned_model else 'n/a'}")
    print("=" * 80)

    _print_table(
        summary, SET_METRICS,
        ["Set-Prec", "Set-Recall", "Set-F1"],
        "Table A: Selection Quality (set-based, order-independent)",
        note="TS-Sem rows share Set-F1 with Semantic Only; "
             "TS-Hybrid rows share Set-F1 with Hybrid Sem-Graph",
    )

    _print_table(
        summary, ORDER_METRICS,
        ["Ord.Prec", "LCS-R", "KendallTau", "Trans.Acc", "1st.Acc"],
        "Table B: Ordering Quality (order-dependent)",
        note="Stage 2 adds ordering value; TS-Sem+SemSort == Semantic Only by construction",
    )

    _print_bucket_table(by_length, "set_f1",           "Bucket: Set-F1")
    _print_bucket_table(by_length, "ordered_precision", "Bucket: Ordered Precision")
    _print_bucket_table(by_length, "kendall_tau",       "Bucket: Kendall Tau")

    # ── 12. Save CSVs ─────────────────────────────────────────────────────
    merged = summary.copy()
    merged.to_csv(RESULTS_DIR / "final_comparison.csv", index=False)
    by_length.to_csv(RESULTS_DIR / "final_by_length.csv", index=False)
    boot_df.to_csv(RESULTS_DIR / "bootstrap_significance.csv", index=False)

    print(f"\n  CSV (summary)    -> {RESULTS_DIR / 'final_comparison.csv'}")
    print(f"  CSV (by length)  -> {RESULTS_DIR / 'final_by_length.csv'}")
    print(f"  CSV (bootstrap)  -> {RESULTS_DIR / 'bootstrap_significance.csv'}")


if __name__ == "__main__":
    main()
