"""
src/run_apibank_eval.py  —  Cross-Dataset Evaluation on API-Bank
================================================================
Evaluates the two-stage tool recommendation framework on the API-Bank
level-3 dataset using leave-one-out cross-validation (LOO-CV).

Dataset
-------
  50 multi-step trajectories, 21 unique tools, 10 task templates × 5 variants
  Source: data/raw/API-Bank/test-data/level-3.json

Evaluation Protocol
-------------------
  LOO-CV: for each test entry i (i=1..50),
    - fit co-occurrence graph on the other 49 entries
    - run all methods on query i
    - compare to ground-truth tool sequence

Methods Evaluated
-----------------
  1. bm25              BM25 keyword match between query and tool descriptions
  2. semantic_only     Top-K by embedding cosine similarity
  3. hybrid_sem_graph  Hybrid Sem-Graph (Stage 1 single-stage, ToolBench-trained)
  4. ts_hybrid_semsort TS-Hybrid + Semantic Sort
  5. ts_hybrid_hr      TS-Hybrid + Hybrid-Rerank (alpha from ToolBench)
  6. ts_hybrid_lr      TS-Hybrid + Learned Reranker (zero-shot transfer)

Outputs
-------
  results/apibank_eval.csv
  results/apibank_by_length.csv
  results/apibank_stats.json

Usage
-----
  python src/run_apibank_eval.py
  python src/run_apibank_eval.py --no-lr   # skip learned reranker
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

ROOT        = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Metric functions (self-contained, mirrors evaluate.py)
# ---------------------------------------------------------------------------

def _f1_at_k(predicted: list[str], ground_truth: list[str]) -> tuple[float, float, float]:
    pred_set = set(predicted)
    gt_set   = set(ground_truth)
    tp = len(pred_set & gt_set)
    prec = tp / max(1, len(pred_set))
    rec  = tp / max(1, len(gt_set))
    f1   = 2 * prec * rec / max(1e-9, prec + rec)
    return prec, rec, f1


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def _ordered_precision(predicted: list[str], ground_truth: list[str]) -> float:
    gt_set  = set(ground_truth)
    gt_rank = {t: i for i, t in enumerate(ground_truth)}
    common  = [(t, pos) for pos, t in enumerate(predicted) if t in gt_set]
    if len(common) < 2:
        return 0.0
    mp = tp = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            tp += 1
            if gt_rank[common[i][0]] < gt_rank[common[j][0]]:
                mp += 1
    return mp / tp if tp else 0.0


def _transition_accuracy(predicted: list[str], ground_truth: list[str]) -> float:
    if len(ground_truth) < 2:
        return 0.0
    gt_pairs = [(ground_truth[i], ground_truth[i+1]) for i in range(len(ground_truth)-1)]
    pos_map  = {t: i for i, t in enumerate(predicted)}
    matched  = 0
    for a, b in gt_pairs:
        if a in pos_map and b in pos_map and 0 < pos_map[b] - pos_map[a] <= 2:
            matched += 1
    return matched / len(gt_pairs)


def _kendall_tau(predicted: list[str], ground_truth: list[str]) -> float:
    common = [t for t in predicted if t in set(ground_truth)]
    if len(common) < 2:
        return 0.0
    gt_rank = {t: i for i, t in enumerate(ground_truth)}
    pred_ranks = [gt_rank[t] for t in common]
    n = len(pred_ranks)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = (pred_ranks[i] - pred_ranks[j])
            if diff < 0:
                concordant += 1
            elif diff > 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom else 0.0


def _first_tool_accuracy(predicted: list[str], ground_truth: list[str]) -> float:
    if not predicted or not ground_truth:
        return 0.0
    return 1.0 if predicted[0] == ground_truth[0] else 0.0


def compute_metrics(predicted: list[str], ground_truth: list[str]) -> dict:
    prec, rec, f1 = _f1_at_k(predicted, ground_truth)
    lcs = _lcs_length(predicted, ground_truth)
    return {
        "set_f1":            f1,
        "set_prec":          prec,
        "set_rec":           rec,
        "lcs_r":             lcs / max(1, len(ground_truth)),
        "ordered_precision": _ordered_precision(predicted, ground_truth),
        "transition_acc":    _transition_accuracy(predicted, ground_truth),
        "first_tool_acc":    _first_tool_accuracy(predicted, ground_truth),
        "kendall_tau":       _kendall_tau(predicted, ground_truth),
        "pred_len":          len(predicted),
        "gt_len":            len(ground_truth),
    }


# ---------------------------------------------------------------------------
# BM25 (lightweight, no external library)
# ---------------------------------------------------------------------------

class BM25:
    """Simple BM25 over tool descriptions."""

    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.tools = list(docs.keys())
        self.k1 = k1
        self.b  = b
        tokenized = {t: docs[t].lower().split() for t in self.tools}
        dl   = {t: len(toks) for t, toks in tokenized.items()}
        avgdl = sum(dl.values()) / max(1, len(dl))

        from collections import Counter
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

    def score(self, query: str) -> list[tuple[str, float]]:
        q_terms = query.lower().split()
        scores  = []
        for tool in self.tools:
            s = sum(self.idf.get(w, 0.0) * self.tf_norm[tool].get(w, 0.0)
                    for w in q_terms)
            scores.append((tool, s))
        return sorted(scores, key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Tool co-occurrence transition probability lookup
# ---------------------------------------------------------------------------

def build_tp_lookup(records: list[dict]) -> dict[tuple[str, str], float]:
    """Transition probabilities from training trajectories."""
    counts: dict[str, int] = defaultdict(int)
    co:     dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        seq = r["tool_sequence"]
        for i, t in enumerate(seq):
            counts[t] += 1
            for t2 in seq[i+1:]:
                co[(t, t2)] += 1
    tp: dict[tuple[str, str], float] = {}
    for (a, b), c in co.items():
        tp[(a, b)] = c / max(1, counts[a])
    return tp


# ---------------------------------------------------------------------------
# Stage 1: select K candidate tools
# ---------------------------------------------------------------------------

def select_semantic(
    query_vec:  np.ndarray,
    tool_vecs:  np.ndarray,
    tool_names: list[str],
    k:          int,
) -> list[tuple[str, float]]:
    """Return top-K tools by cosine similarity."""
    sims = tool_vecs @ query_vec  # (N,) — vecs already L2-normalised
    idxs = np.argsort(sims)[::-1][:k]
    return [(tool_names[i], float(sims[i])) for i in idxs]


def select_hybrid(
    query_vec:  np.ndarray,
    tool_vecs:  np.ndarray,
    tool_names: list[str],
    tp_lookup:  dict,
    k:          int,
    alpha:      float = 0.5,
) -> list[tuple[str, float]]:
    """
    Greedy hybrid selection: at each step, add tool with highest
    alpha * max_tp_from_selected + (1-alpha) * sim.
    """
    sims = {name: float(tool_vecs[i] @ query_vec)
            for i, name in enumerate(tool_names)}
    remaining = set(tool_names)
    selected:  list[tuple[str, float]] = []

    for _ in range(k):
        if not remaining:
            break
        best_tool  = None
        best_score = -1e9
        for t in remaining:
            graph_sc = max(
                (tp_lookup.get((s, t), 0.0) for s, _ in selected),
                default=0.0,
            ) if selected else 0.0
            score = alpha * graph_sc + (1 - alpha) * sims[t]
            if score > best_score:
                best_score = score
                best_tool  = t
        if best_tool is None:
            break
        selected.append((best_tool, sims[best_tool]))
        remaining.remove(best_tool)
    return selected


# ---------------------------------------------------------------------------
# Stage 2: ordering strategies
# ---------------------------------------------------------------------------

def order_semsort(tools_sims: list[tuple[str, float]]) -> list[str]:
    return [t for t, _ in sorted(tools_sims, key=lambda x: -x[1])]


def order_hybrid_rerank(
    tools_sims: list[tuple[str, float]],
    tp_lookup:  dict,
    alpha:      float = 0.7,
) -> list[str]:
    """Greedy hybrid rerank using transition probs."""
    remaining = list(tools_sims)
    ordered:   list[str] = []
    while remaining:
        best_t = best_sc = None, -1e9
        for t, sim in remaining:
            tp_sc = max(
                (tp_lookup.get((prev, t), 0.0) for prev in ordered),
                default=0.0,
            ) if ordered else sim
            score = alpha * tp_sc + (1 - alpha) * sim
            if score > best_sc[1]:
                best_t, best_sc = t, (t, score)
        ordered.append(best_t)
        remaining = [(t, s) for t, s in remaining if t != best_t]
    return ordered


def order_lr(
    tools_sims:     list[tuple[str, float]],
    tp_lookup:      dict,
    position_stats: dict,
    model,
    tool_names_all: list[str],
) -> list[str]:
    """Learned Reranker (zero-shot transfer from ToolBench)."""
    # Import here to avoid circular deps
    from learned_reranker import order_learned_rerank
    # Adapt tool_names to include all tool library entries
    # (position_stats uses ToolBench tool names; API-Bank tools won't be in it)
    return order_learned_rerank(tools_sims, tp_lookup, position_stats, model)


# ---------------------------------------------------------------------------
# Main evaluation (LOO-CV)
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "bm25":           "BM25",
    "semantic":       "Semantic Only",
    "hybrid":         "Hybrid Sem-Graph",
    "ts_semsort":     "TS-Hybrid + SemSort",
    "ts_hr":          "TS-Hybrid + HybridRerank",
    "ts_lr":          "TS-Hybrid + LR (zero-shot)",
}


def run_eval(
    records:        list[dict],
    tool_descs:     dict[str, str],
    lr_model=None,
    position_stats: dict | None = None,
    hr_alpha:       float = 0.7,
) -> pd.DataFrame:
    """
    LOO-CV evaluation.
    Returns a DataFrame with one row per (method, test_index).
    """
    from sentence_transformers import SentenceTransformer

    print("Loading sentence encoder ...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    tool_names = sorted(tool_descs.keys())
    tool_desc_list = [tool_descs[t] for t in tool_names]

    print(f"Encoding {len(tool_names)} tool descriptions ...")
    tool_vecs = encoder.encode(
        tool_desc_list, normalize_embeddings=True, show_progress_bar=False
    )  # (N, D)

    # Pre-encode all queries
    queries = [r["task_description"] for r in records]
    print(f"Encoding {len(queries)} queries ...")
    query_vecs = encoder.encode(
        queries, normalize_embeddings=True, show_progress_bar=False
    )  # (50, D)

    rows: list[dict] = []
    n = len(records)

    for i, rec in enumerate(records):
        gt = rec["tool_sequence"]
        k  = max(len(gt), 2)   # oracle K (min 2 for multi-step tasks)

        # Build tp_lookup from LOO training data
        loo_train = [records[j] for j in range(n) if j != i]
        tp_lookup = build_tp_lookup(loo_train)

        qvec = query_vecs[i]

        # ---- BM25 ----
        bm25 = BM25(tool_descs)
        bm25_ranked = [t for t, _ in bm25.score(rec["task_description"])[:k]]
        rows.append({"method": "bm25", "idx": i,
                     **compute_metrics(bm25_ranked, gt)})

        # ---- Semantic Only ----
        sem_sel = select_semantic(qvec, tool_vecs, tool_names, k)
        sem_pred = order_semsort(sem_sel)
        rows.append({"method": "semantic", "idx": i,
                     **compute_metrics(sem_pred, gt)})

        # ---- Hybrid Sem-Graph (single stage, no TS) ----
        hybrid_sel = select_hybrid(qvec, tool_vecs, tool_names, tp_lookup, k)
        hybrid_pred = order_semsort(hybrid_sel)
        rows.append({"method": "hybrid", "idx": i,
                     **compute_metrics(hybrid_pred, gt)})

        # ---- TS-Hybrid + SemSort ----
        ts_pred_ss = order_semsort(hybrid_sel)
        rows.append({"method": "ts_semsort", "idx": i,
                     **compute_metrics(ts_pred_ss, gt)})

        # ---- TS-Hybrid + HybridRerank ----
        ts_pred_hr = order_hybrid_rerank(hybrid_sel, tp_lookup, alpha=hr_alpha)
        rows.append({"method": "ts_hr", "idx": i,
                     **compute_metrics(ts_pred_hr, gt)})

        # ---- TS-Hybrid + LR (zero-shot) ----
        if lr_model is not None and position_stats is not None:
            ts_pred_lr = order_lr(hybrid_sel, tp_lookup, position_stats,
                                   lr_model, tool_names)
            rows.append({"method": "ts_lr", "idx": i,
                         **compute_metrics(ts_pred_lr, gt)})

        if (i + 1) % 10 == 0:
            print(f"  LOO progress: {i+1}/{n}")

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per method, sorted by ordered_precision descending."""
    metric_cols = [
        "set_f1", "set_prec", "set_rec", "lcs_r",
        "ordered_precision", "transition_acc", "first_tool_acc", "kendall_tau",
    ]
    cols = [c for c in metric_cols if c in df.columns]
    agg = df.groupby("method")[cols].mean().round(4)
    agg.index = [METHOD_LABELS.get(m, m) for m in agg.index]
    return agg.sort_values("ordered_precision", ascending=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="API-Bank LOO-CV evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-lr", action="store_true",
                   help="Skip learned reranker (faster)")
    p.add_argument("--hr-alpha", type=float, default=0.7,
                   help="Alpha for Hybrid-Rerank Stage 2")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ----- Load API-Bank test data -----
    lv3_path = ROOT / "data" / "raw" / "API-Bank" / "test-data" / "level-3.json"
    with open(lv3_path, encoding="utf-8") as f:
        lv3 = json.load(f)

    # Extract records and tool descriptions
    tool_descs: dict[str, str] = {}
    records: list[dict] = []

    for item in lv3:
        req = item.get("requirement", "").strip()
        if not req:
            continue
        apis = item.get("apis", [])
        for a in apis:
            if a.get("api_name") == "ToolSearcher":
                out = a.get("output", {})
                if isinstance(out, dict):
                    inner = out.get("output", out)
                    if isinstance(inner, dict) and "name" in inner:
                        name = inner["name"]
                        desc = inner.get("description", "")
                        if name and desc:
                            tool_descs[name] = desc

        tools = [a["api_name"] for a in apis if a.get("api_name") and
                 a["api_name"] != "ToolSearcher"]
        deduped: list[str] = []
        for t in tools:
            if not deduped or t != deduped[-1]:
                deduped.append(t)
        if deduped:
            records.append({
                "task_description": req,
                "tool_sequence":    deduped,
                "num_steps":        len(deduped),
            })

    # Fill missing descriptions with tool name
    for name in set(t for r in records for t in r["tool_sequence"]):
        if name not in tool_descs:
            tool_descs[name] = name.replace("_", " ")

    print(f"API-Bank: {len(records)} test entries, {len(tool_descs)} tools")

    # ----- Load LR model (optional) -----
    lr_model = None
    position_stats = None
    if not args.no_lr:
        try:
            from learned_reranker import (
                load_learned_reranker, build_position_stats,
                CHECKPOINT as LR_CHECKPOINT
            )
            # Load ToolBench training records for position stats
            tb_train_path = ROOT / "data" / "processed" / "successful_trajectories.jsonl"
            if tb_train_path.exists() and LR_CHECKPOINT.exists():
                print("Loading learned reranker (ToolBench-trained) ...")
                tb_records: list[dict] = []
                with open(tb_train_path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            tb_records.append(json.loads(line))
                tb_records = tb_records[:50_000]   # cap for speed
                position_stats = build_position_stats(tb_records)
                lr_model = load_learned_reranker(LR_CHECKPOINT)
                print(f"  LR model loaded, position_stats: {len(position_stats)} tools")
            else:
                print("  LR checkpoint not found — skipping LR method")
        except Exception as e:
            print(f"  LR load failed: {e}  — skipping")

    # ----- Run evaluation -----
    print("\nRunning LOO-CV ...")
    df = run_eval(records, tool_descs,
                  lr_model=lr_model,
                  position_stats=position_stats,
                  hr_alpha=args.hr_alpha)

    # ----- Save raw results -----
    raw_path = RESULTS_DIR / "apibank_eval_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results -> {raw_path}")

    # ----- Aggregate -----
    agg = aggregate(df)
    agg_path = RESULTS_DIR / "apibank_eval.csv"
    agg.to_csv(agg_path)
    print(f"Aggregated  -> {agg_path}")

    # ----- Print table -----
    print("\n" + "=" * 70)
    print("  API-Bank Evaluation Results (LOO-CV, n=50)")
    print("=" * 70)
    print(agg.to_string())
    print("=" * 70)

    # ----- Length breakdown -----
    df["bucket"] = df["gt_len"].apply(
        lambda x: "short(2)" if x <= 2 else "medium(3+)"
    )
    bucket_agg = df.groupby(["method", "bucket"])[
        ["set_f1", "ordered_precision", "kendall_tau"]
    ].mean().round(4)
    bucket_path = RESULTS_DIR / "apibank_by_length.csv"
    bucket_agg.to_csv(bucket_path)
    print(f"Length breakdown -> {bucket_path}")

    # ----- Stats -----
    from collections import Counter
    lens = [r["num_steps"] for r in records]
    stats = {
        "n_records":     len(records),
        "n_tools":       len(tool_descs),
        "gt_len_dist":   dict(Counter(lens)),
        "mean_gt_len":   round(sum(lens) / len(lens), 2),
    }
    stats_path = RESULTS_DIR / "apibank_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()
