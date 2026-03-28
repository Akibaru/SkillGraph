"""
src/run_bm25_toolbench.py  —  BM25 baseline on ToolBench (9,965 test)
=====================================================================
Evaluates BM25 keyword retrieval as a single-stage baseline, using the
same Oracle-K protocol and metrics as final_comparison.py.

Usage
-----
  python src/run_bm25_toolbench.py

Output
------
  results/bm25_toolbench.csv
"""
from __future__ import annotations

import math
import pathlib
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import kendalltau as scipy_kendalltau

warnings.filterwarnings("ignore")

_SRC = pathlib.Path(__file__).resolve().parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))

from evaluate import (
    load_trajectories,
    make_train_test_split,
    _lcs_length,
    _ordered_precision,
    _transition_accuracy,
    _first_tool_accuracy,
)

# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.tools = list(docs.keys())
        self.k1 = k1
        self.b = b
        tokenized = {t: docs[t].lower().split() for t in self.tools}
        dl = {t: len(toks) for t, toks in tokenized.items()}
        avgdl = sum(dl.values()) / max(1, len(dl))
        n = len(self.tools)
        df: dict[str, int] = defaultdict(int)
        for toks in tokenized.values():
            for w in set(toks):
                df[w] += 1
        self.idf = {w: math.log((n - f + 0.5) / (f + 0.5) + 1)
                    for w, f in df.items()}
        self.tf_norm: dict[str, dict[str, float]] = {}
        for tool, toks in tokenized.items():
            cnt = Counter(toks)
            d = dl[tool]
            self.tf_norm[tool] = {
                w: c * (k1 + 1) / (c + k1 * (1 - b + b * d / avgdl))
                for w, c in cnt.items()
            }

    def score(self, query: str) -> dict[str, float]:
        qtoks = query.lower().split()
        scores: dict[str, float] = {}
        for tool in self.tools:
            s = sum(
                self.idf.get(w, 0) * self.tf_norm[tool].get(w, 0)
                for w in qtoks
            )
            scores[tool] = s
        return scores

    def top_k(self, query: str, k: int) -> list[str]:
        scores = self.score(query)
        return sorted(scores, key=lambda t: -scores[t])[:k]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def kendall_tau(pred: list[str], gt: list[str]) -> float:
    common = [t for t in pred if t in gt]
    if len(common) < 2:
        return 0.0
    gt_rank = {t: i for i, t in enumerate(gt)}
    common_sorted = sorted(common, key=lambda t: gt_rank[t])
    pred_rank = {t: i for i, t in enumerate(common)}
    n = len(common_sorted)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = common_sorted[i], common_sorted[j]
            if pred_rank[ti] < pred_rank[tj]:
                concordant += 1
            else:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else 0.0


def evaluate_sequence(pred: list[str], gt: list[str]) -> dict[str, float]:
    gt_set = set(gt)
    pred_set = set(pred)
    tp = len(gt_set & pred_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    ord_prec = _ordered_precision(pred, gt)
    lcs = _lcs_length(pred, gt) / max(len(gt), 1)
    tau = kendall_tau(pred, gt)
    trans = _transition_accuracy(pred, gt)
    first = _first_tool_accuracy(pred, gt)

    return dict(
        set_prec=prec, set_rec=rec, set_f1=f1,
        ordered_precision=ord_prec, lcs_r=lcs,
        kendall_tau=tau, transition_acc=trans, first_tool_acc=first,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading ToolBench data...")
    trajs = load_trajectories(_ROOT / "data/processed/successful_trajectories.jsonl")
    train, test = make_train_test_split(trajs)

    # Build global tool description index from all trajectories
    print("Building tool description index...")
    tool_desc: dict[str, str] = {}
    for t in trajs:
        for td in t.get("tool_details", []):
            name = td.get("name", "")
            desc = td.get("description", "") or ""
            if name and name not in tool_desc:
                tool_desc[name] = f"{name.replace('_', ' ')} {desc}"

    print(f"  {len(tool_desc)} tools indexed")

    print("Building BM25 index (this may take ~30s)...")
    bm25 = BM25(tool_desc)
    print("  BM25 index built.")

    results = []
    for i, traj in enumerate(test):
        if i % 1000 == 0:
            print(f"  [{i}/{len(test)}]")
        query = traj.get("task_description", "")
        gt = traj.get("tool_sequence", [])
        if not gt:
            continue
        K = max(len(gt), 3)
        pred = bm25.top_k(query, K)
        # Order: by BM25 score (already sorted)
        metrics = evaluate_sequence(pred, gt)
        metrics["K"] = K
        metrics["gt_len"] = len(gt)
        results.append(metrics)

    df = pd.DataFrame(results)
    means = df[[
        "set_prec", "set_rec", "set_f1",
        "ordered_precision", "lcs_r", "kendall_tau",
        "transition_acc", "first_tool_acc"
    ]].mean()

    print("\n=== BM25 ToolBench Results ===")
    print(f"  set_precision : {means['set_prec']:.4f}")
    print(f"  set_recall    : {means['set_rec']:.4f}")
    print(f"  set_f1        : {means['set_f1']:.4f}")
    print(f"  ordered_prec  : {means['ordered_precision']:.4f}")
    print(f"  lcs_r         : {means['lcs_r']:.4f}")
    print(f"  kendall_tau   : {means['kendall_tau']:.4f}")
    print(f"  transition_acc: {means['transition_acc']:.4f}")
    print(f"  first_tool_acc: {means['first_tool_acc']:.4f}")
    print(f"  n             : {len(results)}")

    out = _ROOT / "results/bm25_toolbench.csv"
    means_df = means.to_frame().T
    means_df.insert(0, "method", "bm25")
    means_df.insert(1, "n", len(results))
    means_df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
