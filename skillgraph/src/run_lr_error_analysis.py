"""
src/run_lr_error_analysis.py  —  Error Analysis: TS-Hybrid+LR vs TS-Hybrid+SemSort
====================================================================================
Computes per-sample Ordered Precision delta (LR - SemSort) on the full test set,
then prints / saves the breakdown stats and top-10 examples for each direction.

Usage
-----
  cd skillgraph
  python src/run_lr_error_analysis.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from evaluate import (
    load_trajectories,
    make_train_test_split,
    batch_encode_queries,
)
from final_comparison import (
    compute_set_metrics,
    compute_order_metrics,
    get_hybrid_stage1,
)
from two_stage_pipeline import (
    _build_tp_lookup,
    order_semantic_sort,
)
from learned_reranker import (
    build_position_stats,
    load_learned_reranker,
    order_learned_rerank,
    CHECKPOINT as LR_CHECKPOINT,
)
from graph_search import ToolSequencePlanner

ROOT        = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_HYBRID  = 0.4
HYBRID_PARAMS = {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}


def main() -> None:
    print("Loading trajectories ...")
    records = load_trajectories()
    train_records, test_records = make_train_test_split(records, seed=42)
    print(f"  {len(test_records):,} test samples")

    print("Loading planner ...")
    planner = ToolSequencePlanner()
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

    tp_lookup      = _build_tp_lookup(planner)
    position_stats = build_position_stats(train_records)

    print("Loading Learned Reranker ...")
    if not LR_CHECKPOINT.exists():
        print(f"  ERROR: checkpoint not found at {LR_CHECKPOINT}")
        return
    lr_model = load_learned_reranker(LR_CHECKPOINT)
    print(f"  Loaded from {LR_CHECKPOINT}")

    print("Batch-encoding test queries ...")
    test_vecs = batch_encode_queries(
        [r["task_description"] for r in test_records]
    )

    print("Pre-caching Stage-1 hybrid selections ...")
    hybrid_cache: list[list[tuple[str, float]]] = []
    for i, rec in enumerate(test_records):
        hybrid_cache.append(get_hybrid_stage1(planner, test_vecs[i], HYBRID_PARAMS))
    print(f"  Cached {len(hybrid_cache):,} samples")

    print("Computing LR vs SemSort per-sample delta ...")
    deltas   = []
    lr_preds = []
    ss_preds = []
    for i, rec in enumerate(test_records):
        gt  = rec["tool_sequence"]
        ts  = hybrid_cache[i]
        ss  = order_semantic_sort(ts)
        lr  = order_learned_rerank(ts, tp_lookup, position_stats, lr_model)
        op_ss = compute_order_metrics(ss, gt)["ordered_precision"]
        op_lr = compute_order_metrics(lr, gt)["ordered_precision"]
        deltas.append(op_lr - op_ss)
        ss_preds.append(ss)
        lr_preds.append(lr)

    deltas_arr = np.array(deltas)

    pos_deltas = deltas_arr[deltas_arr > 0]
    neg_deltas = deltas_arr[deltas_arr < 0]
    zero_count = int((deltas_arr == 0).sum())
    n          = len(test_records)

    print(f"\n{'='*70}")
    print("  Error Analysis: TS-Hybrid + LR  vs  TS-Hybrid + Sem-Sort")
    print("  Metric: per-sample delta in Ordered Precision  (LR - SemSort)")
    print(f"{'='*70}")
    print(f"  LR better (delta > 0): {len(pos_deltas):,} samples  "
          f"({100*len(pos_deltas)/n:.1f}%)  mean_delta={pos_deltas.mean():.4f}")
    print(f"  LR worse  (delta < 0): {len(neg_deltas):,} samples  "
          f"({100*len(neg_deltas)/n:.1f}%)  mean_delta={neg_deltas.mean():.4f}")
    print(f"  Equal     (delta = 0): {zero_count:,} samples  "
          f"({100*zero_count/n:.1f}%)")
    print(f"  Overall mean delta    : {deltas_arr.mean():+.4f}")

    # Write detailed file
    sorted_pos = np.argsort(deltas_arr)[::-1]
    sorted_neg = np.argsort(deltas_arr)
    pos_idx = [i for i in sorted_pos if deltas_arr[i] > 0.0][:10]
    neg_idx = [i for i in sorted_neg if deltas_arr[i] < 0.0][:10]

    lines = []
    lines.append("=" * 78)
    lines.append("  Error Analysis: TS-Hybrid + LR  vs  TS-Hybrid + Sem-Sort")
    lines.append("  Metric: per-sample delta in Ordered Precision  (LR - SemSort)")
    lines.append("=" * 78)
    lines.append(f"  LR better (delta > 0): {len(pos_deltas):,} samples  "
                 f"({100*len(pos_deltas)/n:.1f}%)  mean_delta={pos_deltas.mean():.4f}")
    lines.append(f"  LR worse  (delta < 0): {len(neg_deltas):,} samples  "
                 f"({100*len(neg_deltas)/n:.1f}%)  mean_delta={neg_deltas.mean():.4f}")
    lines.append(f"  Equal     (delta = 0): {zero_count:,} samples  "
                 f"({100*zero_count/n:.1f}%)")
    lines.append(f"  Overall mean delta    : {deltas_arr.mean():+.4f}")
    lines.append("")

    def _fmt_block(title: str, idx_list: list[int]) -> None:
        lines.append("=" * 78)
        lines.append(title)
        lines.append("=" * 78)
        for rank, i in enumerate(idx_list, 1):
            rec   = test_records[i]
            gt    = rec["tool_sequence"]
            query = rec["task_description"][:100]
            ss    = ss_preds[i]
            lr    = lr_preds[i]
            d     = deltas_arr[i]
            lines.append(f"\n[Sample {rank}]  idx={i}  delta_OrdPrec={d:+.4f}")
            lines.append(f"  Query : {query}")
            lines.append(f"  GT    : {' -> '.join(gt)}")
            lines.append(f"  SemSort: {' -> '.join(ss)}")
            lines.append(f"  LR     : {' -> '.join(lr)}")
            changed = [j for j in range(min(len(ss), len(lr))) if ss[j] != lr[j]]
            if changed:
                lines.append(f"  Diff  : positions changed = {changed}")
                for j in changed:
                    ss_t = ss[j] if j < len(ss) else "(missing)"
                    lr_t = lr[j] if j < len(lr) else "(missing)"
                    in_gt = lambda t: "[IN GT]" if t in set(gt) else "[not GT]"
                    lines.append(
                        f"    pos {j}: SemSort='{ss_t}' {in_gt(ss_t)}"
                        f"  ->  LR='{lr_t}' {in_gt(lr_t)}"
                    )
            lines.append("")

    _fmt_block(f"  SECTION A: LR HELPS  (top {len(pos_idx)} samples)", pos_idx)
    _fmt_block(f"  SECTION B: LR HURTS  (top {len(neg_idx)} samples)", neg_idx)

    out_path = RESULTS_DIR / "lr_error_analysis.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
