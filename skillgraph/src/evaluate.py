"""
src/evaluate.py  —  SkillGraph Offline Evaluation Pipeline
===========================================================
Comprehensive evaluation of all planning algorithms against held-out
successful ToolBench trajectories.

Pipeline
--------
  1. Load successful_trajectories.jsonl
  2. 80/20 stratified split by trajectory length (short/medium/long)
  3. Batch-encode all test queries with SentenceTransformer
  4. For each method × test trajectory: get predicted tool sequence, compute metrics
  5. Aggregate, save tables + figures
  6. If prior results exist, print OLD vs NEW comparison side-by-side

Methods evaluated
-----------------
  dijkstra       Semantic-Guided Dijkstra (our method)
  beam           Probabilistic Beam Search (our method)
  hierarchical   Community-Hierarchical Search (our method)
  hybrid         Hybrid Semantic-Graph Planning (key contribution)
  random_graph   Dijkstra on a random same-density graph (structural ablation)
  semantic_only  Pure cosine-sim ranking, no graph (embedding ablation)
  frequency      Always return top-K most frequent training tools (trivial baseline)

Usage
-----
  python src/evaluate.py                         # full evaluation
  python src/evaluate.py --sample 500            # quick run on 500 test samples
  python src/evaluate.py --tune-first            # tune hyperparameters then evaluate
  python src/evaluate.py --use-tuned-params      # load best_hyperparams.json
  python src/evaluate.py --tune-first --sample 500 --use-tuned-params
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import time
import warnings
from collections import Counter, defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau as scipy_kendalltau
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from graph_search import ToolSequencePlanner, ToolPlan, _fallback_plan, _lcs_length

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = pathlib.Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT / "data" / "processed"
OUT_DIR    = ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
FIGS_DIR   = OUT_DIR / "figures"
SPLIT_FILE = PROC_DIR / "train_test_split.json"
TRAJ_FILE  = PROC_DIR / "successful_trajectories.jsonl"

for _d in (TABLES_DIR, FIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
METHODS = [
    "dijkstra",
    "beam",
    "hierarchical",
    "hybrid",
    "random_graph",
    "semantic_only",
    "frequency",
]

METHOD_LABELS: dict[str, str] = {
    "dijkstra":      "Semantic Dijkstra",
    "beam":          "Beam Search",
    "hierarchical":  "Community-Hier.",
    "hybrid":        "Hybrid Sem-Graph",
    "random_graph":  "Random Graph (BL)",
    "semantic_only": "Semantic Only (BL)",
    "frequency":     "Frequency (BL)",
}

_METHOD_COLORS = [
    "#4C72B0",   # dijkstra     — blue
    "#55A868",   # beam         — green
    "#C44E52",   # hierarchical — red
    "#E07B39",   # hybrid       — orange (key contribution)
    "#8172B2",   # random_graph — purple
    "#CCB974",   # semantic_only — sand
    "#64B5CD",   # frequency    — light blue
]

# All columns stored per row
METRIC_COLS = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "hit_rate",
    "jaccard",
    "lcs_r",
    "ordered_overlap",
    "ordered_precision",
    "transition_accuracy",
    "first_tool_accuracy",
    "category_lcs_r",
    "kendall_tau",
    "pred_len",
    "gt_len",
    "latency_ms",
]

# Columns shown in the main summary table
KEY_METRICS = [
    "f1_at_k",
    "lcs_r",
    "ordered_precision",
    "transition_accuracy",
    "first_tool_accuracy",
]


# ---------------------------------------------------------------------------
# Data loading & train/test split
# ---------------------------------------------------------------------------

def load_trajectories(path: pathlib.Path = TRAJ_FILE) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records):,} trajectories from {path.name}")
    return records


def _length_bucket(n: int) -> str:
    if n <= 3:
        return "short"
    if n <= 6:
        return "medium"
    return "long"


def make_train_test_split(
    records:   list[dict],
    test_frac: float        = 0.20,
    seed:      int          = 42,
    save_path: pathlib.Path = SPLIT_FILE,
) -> tuple[list[dict], list[dict]]:
    """80/20 stratified split by trajectory-length bucket. Returns (train, test)."""
    if save_path.exists():
        print(f"Loading existing split from {save_path.name}")
        split     = json.loads(save_path.read_text(encoding="utf-8"))
        train_idx = set(split["train_idx"])
        test_idx  = set(split["test_idx"])
        train = [records[i] for i in range(len(records)) if i in train_idx]
        test  = [records[i] for i in range(len(records)) if i in test_idx]
        print(f"  Train: {len(train):,}  Test: {len(test):,}")
        return train, test

    rng = random.Random(seed)
    bucket_indices: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        bucket_indices[_length_bucket(rec["num_steps"])].append(i)

    train_idx_list: list[int] = []
    test_idx_list:  list[int] = []
    for bucket in sorted(bucket_indices):
        idxs = bucket_indices[bucket]
        rng.shuffle(idxs)
        n_test = max(1, int(len(idxs) * test_frac))
        test_idx_list.extend(idxs[:n_test])
        train_idx_list.extend(idxs[n_test:])

    save_path.write_text(
        json.dumps(
            {"train_idx": sorted(train_idx_list), "test_idx": sorted(test_idx_list)},
            indent=2,
        ),
        encoding="utf-8",
    )
    train = [records[i] for i in train_idx_list]
    test  = [records[i] for i in test_idx_list]
    print(f"  Train: {len(train):,}  Test: {len(test):,}  (saved → {save_path.name})")
    return train, test


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def _ordered_precision(predicted: list[str], ground_truth: list[str]) -> float:
    """
    Pairwise order accuracy over tools that appear in BOTH sequences.
    Returns: matching_pairs / total_pairs.
    """
    gt_set  = set(ground_truth)
    gt_rank = {t: i for i, t in enumerate(ground_truth)}

    common = [(t, pos) for pos, t in enumerate(predicted) if t in gt_set]
    if len(common) < 2:
        return 0.0

    matching_pairs = 0
    total_pairs    = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            t_i, _ = common[i]
            t_j, _ = common[j]
            total_pairs += 1
            if gt_rank[t_i] < gt_rank[t_j]:
                matching_pairs += 1

    return matching_pairs / total_pairs if total_pairs > 0 else 0.0


def _lcs_r(predicted: list[str], ground_truth: list[str]) -> float:
    """LCS Recall Ratio: LCS / len(ground_truth)."""
    if not ground_truth:
        return 0.0
    return _lcs_length(predicted, ground_truth) / len(ground_truth)


def _transition_accuracy(predicted: list[str], ground_truth: list[str]) -> float:
    """
    For each consecutive pair (A→B) in ground truth, check whether B appears
    within 1-2 positions after A in the predicted sequence.
    """
    if len(ground_truth) < 2:
        return 0.0

    gt_transitions = [
        (ground_truth[i], ground_truth[i + 1])
        for i in range(len(ground_truth) - 1)
    ]
    pred_pos = {t: i for i, t in enumerate(predicted)}

    matching = 0
    for a, b in gt_transitions:
        if a in pred_pos and b in pred_pos:
            diff = pred_pos[b] - pred_pos[a]
            if 0 < diff <= 2:
                matching += 1

    return matching / len(gt_transitions)


def _first_tool_accuracy(predicted: list[str], ground_truth: list[str]) -> float:
    """1.0 if the predicted sequence starts with the same tool as ground truth."""
    if not predicted or not ground_truth:
        return 0.0
    return float(predicted[0] == ground_truth[0])


def _category_lcs_r(predicted: list[str], ground_truth: list[str]) -> float:
    """LCS-R computed at the API-category level (suffix after last _for_)."""
    def to_cat(t: str) -> str:
        return t.rsplit("_for_", 1)[1] if "_for_" in t else t

    pred_cats = [to_cat(t) for t in predicted]
    gt_cats   = [to_cat(t) for t in ground_truth]
    if not gt_cats:
        return 0.0
    return _lcs_length(pred_cats, gt_cats) / len(gt_cats)


def compute_metrics(
    predicted:    list[str],
    ground_truth: list[str],
    latency_s:    float = 0.0,
) -> dict:
    """Compute the full metric suite comparing predicted to ground_truth."""
    gt_set   = set(ground_truth)
    pred_set = set(predicted)
    K        = len(ground_truth)
    top_K    = predicted[:K]
    top_K_set = set(top_K)

    hits   = len(top_K_set & gt_set)
    p_at_k = hits / K if K else 0.0
    r_at_k = hits / len(gt_set) if gt_set else 0.0
    f1     = (2 * p_at_k * r_at_k / (p_at_k + r_at_k)
              if (p_at_k + r_at_k) > 0 else 0.0)
    hit_rate = float(gt_set.issubset(top_K_set))

    union   = len(pred_set | gt_set)
    jaccard = len(pred_set & gt_set) / union if union else 0.0

    lcs_val         = _lcs_length(predicted, ground_truth)
    denom_oo        = max(len(predicted), len(ground_truth))
    ordered_overlap = lcs_val / denom_oo if denom_oo else 0.0
    lcs_r_val       = lcs_val / K if K else 0.0

    ord_prec  = _ordered_precision(predicted, ground_truth)
    trans_acc = _transition_accuracy(predicted, ground_truth)
    first_acc = _first_tool_accuracy(predicted, ground_truth)
    cat_lcs_r = _category_lcs_r(predicted, ground_truth)

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
        "precision_at_k":      p_at_k,
        "recall_at_k":         r_at_k,
        "f1_at_k":             f1,
        "hit_rate":            hit_rate,
        "jaccard":             jaccard,
        "lcs_r":               lcs_r_val,
        "ordered_overlap":     ordered_overlap,
        "ordered_precision":   ord_prec,
        "transition_accuracy": trans_acc,
        "first_tool_accuracy": first_acc,
        "category_lcs_r":      cat_lcs_r,
        "kendall_tau":         tau,
        "pred_len":            len(predicted),
        "gt_len":              K,
        "latency_ms":          latency_s * 1000.0,
    }


# ---------------------------------------------------------------------------
# Batch query encoding
# ---------------------------------------------------------------------------

def batch_encode_queries(
    queries:    list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Encode all queries at once; returns (N, 384) float32 L2-normalised array."""
    from sentence_transformers import SentenceTransformer
    print(f"\nBatch-encoding {len(queries):,} queries with SentenceTransformer …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs  = model.encode(
        queries,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Planner injection helper
# ---------------------------------------------------------------------------

def _plan_with_vec(
    planner:      ToolSequencePlanner,
    method:       str,
    vec:          np.ndarray,
    max_steps:    int   = 8,
    beta:         float = 0.3,
    n_entries:    int   = 5,
    beam_params:  Optional[dict] = None,
    hybrid_params: Optional[dict] = None,
) -> list[ToolPlan]:
    """
    Run planner using a pre-computed L2-normalised query vector,
    bypassing per-call SentenceTransformer encoding.
    """
    had_override = "_encode_query" in planner.__dict__
    orig         = planner.__dict__.get("_encode_query")
    planner._encode_query = lambda text: vec          # type: ignore[method-assign]
    try:
        if method == "dijkstra":
            return [planner.dijkstra(
                "__precomputed__", max_steps=max_steps,
                beta=beta, n_entries=n_entries,
            )]
        elif method == "beam":
            bp = beam_params or {}
            return planner.beam_search(
                "__precomputed__", max_steps=max_steps,
                beam_width=bp.get("beam_width", 5),
                w1=bp.get("w1", 0.4), w2=bp.get("w2", 0.4), w3=bp.get("w3", 0.2),
            )
        elif method == "hybrid":
            hp = hybrid_params or {}
            return [planner.hybrid(
                "__precomputed__", max_steps=max_steps,
                k_multiplier=hp.get("k_multiplier", 3),
                alpha=hp.get("alpha", 0.5),
                gamma=hp.get("gamma", 0.1),
            )]
        else:
            return planner.plan("__precomputed__", method=method, max_steps=max_steps)
    finally:
        if had_override:
            planner._encode_query = orig              # type: ignore[method-assign]
        else:
            del planner._encode_query


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class SemanticOnlyBaseline:
    """Pure cosine-similarity ranking over all tools — no graph structure."""

    def __init__(self, planner: ToolSequencePlanner) -> None:
        self._embeddings   = planner._embeddings
        self._active_tools = planner._active_tools

    def predict(self, query_vec: np.ndarray, K: int) -> list[str]:
        sims    = self._embeddings @ query_vec
        top_idx = np.argsort(sims)[-K:][::-1]
        return [self._active_tools[int(i)] for i in top_idx]


class FrequencyBaseline:
    """Always recommend the K globally most-frequent tools from training data."""

    def __init__(self, train_records: list[dict]) -> None:
        counter: Counter = Counter()
        for rec in train_records:
            counter.update(rec["tool_sequence"])
        self._top_tools: list[str] = [t for t, _ in counter.most_common()]
        print(f"[FrequencyBaseline] {len(counter):,} unique training tools  "
              f"(top: {self._top_tools[0] if self._top_tools else 'n/a'})")

    def predict(self, K: int) -> list[str]:
        return self._top_tools[:K]


class RandomGraphBaseline:
    """
    Semantic-Guided Dijkstra on a random Erdos-Renyi graph with the same
    node set and edge count as the real skill graph.
    """

    def __init__(self, planner: ToolSequencePlanner, seed: int = 99) -> None:
        rng     = np.random.default_rng(seed)
        n       = planner.n_tools
        n_edges = planner.G.number_of_edges()
        tools   = planner._active_tools

        print(f"[RandomGraphBaseline] Building ER graph "
              f"({n:,} nodes, {n_edges:,} random edges, seed={seed}) …")

        rand_adj: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        srcs    = rng.integers(0, n, size=n_edges)
        dsts    = rng.integers(0, n, size=n_edges)
        weights = rng.random(n_edges).astype(np.float32)
        for s, d, w in zip(srcs, dsts, weights):
            if s != d:
                w_f = float(w)
                rand_adj[tools[s]].append((tools[d], w_f, w_f))

        self._rand_adj: dict[str, list[tuple[str, float, float]]] = {
            k: sorted(v, key=lambda x: -x[1]) for k, v in rand_adj.items()
        }
        self._planner = planner

    def predict(
        self,
        query_vec:      np.ndarray,
        max_steps:      int   = 8,
        beta:           float = 0.3,
        max_expansions: int   = 300,
    ) -> list[str]:
        planner  = self._planner
        orig_adj = planner._adj
        planner._adj = self._rand_adj
        try:
            entry_tools = planner._top_entry_tools(query_vec, k=5)
            best_plan: Optional[ToolPlan] = None
            for entry, _ in entry_tools:
                if entry not in planner.G:
                    continue
                plan = planner._dijkstra(
                    entry, query_vec, max_steps=max_steps,
                    beta=beta, max_expansions=max_expansions,
                )
                if plan and (best_plan is None or plan.total_score > best_plan.total_score):
                    best_plan = plan
        finally:
            planner._adj = orig_adj

        if best_plan:
            return best_plan.tools
        return [t for t, _ in planner._top_entry_tools(query_vec, k=max_steps)]


# ---------------------------------------------------------------------------
# Graph-level quality metrics
# ---------------------------------------------------------------------------

def graph_level_metrics(planner: ToolSequencePlanner) -> dict:
    import networkx as nx

    G           = planner.G
    communities = planner.communities

    G_und = G.to_undirected()
    comm_sets_d: dict[int, set] = defaultdict(set)
    for node, cid in communities.items():
        if node in G_und:
            comm_sets_d[cid].add(node)
    comm_sets = [s for s in comm_sets_d.values() if s]

    try:
        modularity = float(nx.community.modularity(G_und, comm_sets, weight="weight"))
    except Exception:
        modularity = float("nan")

    purity_scores: list[float] = []
    purity_rows:   list[dict]  = []
    for cid, members in comm_sets_d.items():
        cats = [
            m.rsplit("_for_", 1)[1] if "_for_" in m else "unknown"
            for m in members
        ]
        if not cats:
            continue
        top_cat, top_count = Counter(cats).most_common(1)[0]
        pur = top_count / len(cats)
        purity_scores.append(pur)
        purity_rows.append({
            "community_id":      cid,
            "size":              len(members),
            "purity":            round(pur, 4),
            "dominant_category": top_cat,
            "n_categories":      len(set(cats)),
        })

    purity_df = (pd.DataFrame(purity_rows)
                   .sort_values("purity", ascending=False)
                   .reset_index(drop=True))

    nodes_with_cat = [(n, c) for n, c in communities.items() if "_for_" in n and n in G]
    if len(nodes_with_cat) >= 2:
        pred_labels = [c for _, c in nodes_with_cat]
        true_cats   = [n.rsplit("_for_", 1)[1] for n, _ in nodes_with_cat]
        unique_cats = {v: i for i, v in enumerate(set(true_cats))}
        true_int    = [unique_cats[c] for c in true_cats]
        nmi         = float(normalized_mutual_info_score(true_int, pred_labels))
    else:
        nmi = float("nan")

    try:
        sample_nodes = list(G.nodes())[:2000]
        avg_clust    = float(nx.average_clustering(G.subgraph(sample_nodes), weight="weight"))
    except Exception:
        avg_clust = float("nan")

    return {
        "graph_metrics": {
            "n_nodes":               G.number_of_nodes(),
            "n_edges":               G.number_of_edges(),
            "n_communities":         len(comm_sets_d),
            "modularity":            round(modularity, 4),
            "mean_community_purity": round(float(np.mean(purity_scores)), 4)
                                     if purity_scores else 0.0,
            "nmi_vs_api_category":   round(nmi, 4),
            "avg_clustering_coeff":  round(avg_clust, 4),
        },
        "purity_df": purity_df,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all_methods(
    test_records:    list[dict],
    planner:         ToolSequencePlanner,
    query_vecs:      np.ndarray,
    sem_baseline:    SemanticOnlyBaseline,
    freq_baseline:   FrequencyBaseline,
    rand_baseline:   RandomGraphBaseline,
    max_steps:       int   = 8,
    dijkstra_beta:   float = 0.3,
    dijkstra_n_entries: int = 5,
    beam_params:     Optional[dict] = None,
    hybrid_params:   Optional[dict] = None,
) -> tuple[list[dict], dict[str, list[int]]]:
    """
    Iterate over all methods × test trajectories.
    Returns (rows, path_lens).
    """
    all_rows:  list[dict]           = []
    path_lens: dict[str, list[int]] = {m: [] for m in METHODS}

    for method in METHODS:
        print(f"\n  [{method}]  evaluating {len(test_records):,} trajectories …")
        t_start = time.time()

        for i, rec in enumerate(tqdm(test_records, desc=method, unit="traj")):
            gt     = rec["tool_sequence"]
            vec    = query_vecs[i]
            K      = len(gt)
            bucket = _length_bucket(K)

            gt_cats   = {t.rsplit("_for_", 1)[1] if "_for_" in t else "?" for t in gt}
            n_gt_cats = len(gt_cats)

            t0 = time.time()
            try:
                if method in ("dijkstra", "beam", "hierarchical", "hybrid"):
                    plans = _plan_with_vec(
                        planner, method, vec, max_steps=max_steps,
                        beta=dijkstra_beta, n_entries=dijkstra_n_entries,
                        beam_params=beam_params, hybrid_params=hybrid_params,
                    )
                    predicted = plans[0].tools if plans else []

                elif method == "random_graph":
                    predicted = rand_baseline.predict(
                        vec, max_steps=max_steps, beta=dijkstra_beta,
                    )

                elif method == "semantic_only":
                    predicted = sem_baseline.predict(vec, K=max(K, 3))

                elif method == "frequency":
                    predicted = freq_baseline.predict(K=max(K, 3))

                else:
                    predicted = []

            except Exception:
                predicted = []

            latency = time.time() - t0
            path_lens[method].append(len(predicted))

            row = compute_metrics(predicted, gt, latency_s=latency)
            row["method"]    = method
            row["bucket"]    = bucket
            row["n_gt_cats"] = n_gt_cats
            all_rows.append(row)

        elapsed  = time.time() - t_start
        avg_plen = (sum(path_lens[method]) / len(path_lens[method])
                    if path_lens[method] else 0.0)
        print(f"  [{method}] done in {elapsed:.1f}s  avg_path_len={avg_plen:.2f}")

    return all_rows, path_lens


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols)[METRIC_COLS].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def pretty_table(df_main: pd.DataFrame) -> str:
    """ASCII table of KEY_METRICS per method, sorted by F1@K desc."""
    display_cols = (["method"]
                    + [f"{m}_mean" for m in KEY_METRICS]
                    + ["latency_ms_mean"])
    sub = df_main[display_cols].copy()
    sub["method"] = sub["method"].map(METHOD_LABELS).fillna(sub["method"])
    sub = sub.sort_values("f1_at_k_mean", ascending=False).reset_index(drop=True)
    sub.columns = ["Method", "F1@K", "LCS-R", "Ord.Prec", "Trans.Acc", "First.Acc", "Lat(ms)"]

    col_widths = [
        max(len(str(col)),
            max((len(f"{v:.4f}") if isinstance(v, float) else len(str(v)))
                for v in sub[col])) + 2
        for col in sub.columns
    ]

    sep    = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header = ("| "
              + " | ".join(str(c).ljust(w) for c, w in zip(sub.columns, col_widths))
              + " |")

    lines = [sep, header, sep]
    for _, row in sub.iterrows():
        cells = []
        for v, w in zip(row, col_widths):
            s = f"{v:.4f}" if isinstance(v, float) else str(v)
            cells.append(s.ljust(w))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Before/After comparison
# ---------------------------------------------------------------------------

def print_comparison(df_old: pd.DataFrame, df_new: pd.DataFrame) -> None:
    """Print side-by-side BEFORE vs AFTER for key metrics."""
    print("\n" + "=" * 70)
    print("  OLD vs NEW: key metric comparison")
    print("=" * 70)

    compare_metrics: list[str] = []
    for m in ["f1_at_k_mean", "lcs_r_mean", "ordered_precision_mean"]:
        if m in df_old.columns and m in df_new.columns:
            compare_metrics.append(m)

    if not compare_metrics:
        print("  (no common metric columns to compare)")
        return

    header = f"  {'Method':<22}" + "".join(
        f"  {'OLD ' + m.replace('_mean',''):>13}  {'NEW ' + m.replace('_mean',''):>13}  {'Δ':>8}"
        for m in compare_metrics
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for method in METHODS:
        old_row = df_old[df_old["method"] == method]
        new_row = df_new[df_new["method"] == method]
        if old_row.empty or new_row.empty:
            continue
        label = METHOD_LABELS.get(method, method)
        line  = f"  {label:<22}"
        for col in compare_metrics:
            old_v = float(old_row[col].iloc[0])
            new_v = float(new_row[col].iloc[0])
            delta = new_v - old_v
            sign  = "+" if delta >= 0 else ""
            line += f"  {old_v:>13.4f}  {new_v:>13.4f}  {sign}{delta:>7.4f}"
        print(line)

    print()


def save_before_after_csv(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    path:   pathlib.Path,
) -> None:
    """Save BEFORE vs AFTER delta table as CSV."""
    compare_metrics = [
        m for m in ["f1_at_k_mean", "lcs_r_mean",
                    "ordered_precision_mean", "transition_accuracy_mean",
                    "first_tool_accuracy_mean"]
        if m in df_old.columns and m in df_new.columns
    ]

    rows = []
    for method in METHODS:
        old_row = df_old[df_old["method"] == method]
        new_row = df_new[df_new["method"] == method]
        row = {"method": method, "label": METHOD_LABELS.get(method, method)}
        for col in compare_metrics:
            short = col.replace("_mean", "")
            if not old_row.empty and not new_row.empty:
                old_v = float(old_row[col].iloc[0])
                new_v = float(new_row[col].iloc[0])
                row[f"old_{short}"] = round(old_v, 4)
                row[f"new_{short}"] = round(new_v, 4)
                row[f"delta_{short}"] = round(new_v - old_v, 4)
            elif not new_row.empty:
                new_v = float(new_row[col].iloc[0])
                row[f"old_{short}"] = None
                row[f"new_{short}"] = round(new_v, 4)
                row[f"delta_{short}"] = None
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  CSV  → {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(df_main: pd.DataFrame, df_length: pd.DataFrame) -> None:
    """Save publication-quality figures."""

    # ── 1. Grouped bar chart: key metrics per method ──────────────────────
    n_metrics = len(KEY_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 5))

    for ax, metric in zip(axes, KEY_METRICS):
        means, stds, labels, colors = [], [], [], []
        for ci, (m_id, m_label) in enumerate(METHOD_LABELS.items()):
            row = df_main[df_main["method"] == m_id]
            if row.empty:
                continue
            means.append(float(row[f"{metric}_mean"].iloc[0]))
            stds.append(float(row[f"{metric}_std"].iloc[0]))
            labels.append(m_label)
            colors.append(_METHOD_COLORS[ci % len(_METHOD_COLORS)])

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=3,
               color=colors, edgecolor="white", width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(metric.replace("_", " "), fontsize=9)
        top = max(means) * 1.35 + 0.02 if means else 0.1
        ax.set_ylim(0, min(1.05, top))
        ax.set_title(metric.replace("_", " ").title(), fontsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.suptitle("SkillGraph: Method Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGS_DIR / "method_comparison_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure → {out}")

    # ── 2. F1@K and LCS-R vs trajectory length bucket ─────────────────────
    bucket_order = ["short", "medium", "long"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, metric in zip(axes, ["f1_at_k_mean", "lcs_r_mean"]):
        for ci, (m_id, m_label) in enumerate(METHOD_LABELS.items()):
            sub = (df_length[df_length["method"] == m_id]
                   .set_index("bucket")
                   .reindex(bucket_order))
            ys = sub[metric].tolist()
            ax.plot(bucket_order, ys, marker="o", label=m_label,
                    color=_METHOD_COLORS[ci % len(_METHOD_COLORS)],
                    linewidth=1.8, markersize=5)
        ax.set_xlabel("Trajectory length bucket (short≤3, medium≤6, long≥7)")
        ax.set_ylabel(metric.replace("_mean", "").replace("_", " "))
        ax.set_title(metric.replace("_mean", "").replace("_", " ").upper())
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    out = FIGS_DIR / "length_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure → {out}")

    # ── 3. Latency comparison ─────────────────────────────────────────────
    lats, lat_labels, lat_colors = [], [], []
    for ci, (m_id, m_label) in enumerate(METHOD_LABELS.items()):
        row = df_main[df_main["method"] == m_id]
        if row.empty:
            continue
        lats.append(float(row["latency_ms_mean"].iloc[0]))
        lat_labels.append(m_label)
        lat_colors.append(_METHOD_COLORS[ci % len(_METHOD_COLORS)])

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(lat_labels))
    ax.bar(x, lats, color=lat_colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(lat_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Avg latency (ms)")
    ax.set_title("Planning Latency per Query")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    out = FIGS_DIR / "latency_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure → {out}")

    # ── 4. Order-aware metrics grouped bar ────────────────────────────────
    order_metrics = ["lcs_r_mean", "ordered_precision_mean",
                     "transition_accuracy_mean", "first_tool_accuracy_mean"]
    fig, ax = plt.subplots(figsize=(11, 4.5))

    n_m     = len(order_metrics)
    x_base  = np.arange(n_m)
    n_meth  = len(METHOD_LABELS)
    width   = 0.10
    offsets = np.linspace(-(n_meth - 1) * width / 2, (n_meth - 1) * width / 2, n_meth)

    for ci, (m_id, m_label) in enumerate(METHOD_LABELS.items()):
        row = df_main[df_main["method"] == m_id]
        if row.empty:
            continue
        vals = [float(row[col].iloc[0]) if col in row.columns else 0.0
                for col in order_metrics]
        ax.bar(x_base + offsets[ci], vals, width=width,
               label=m_label, color=_METHOD_COLORS[ci % len(_METHOD_COLORS)],
               edgecolor="white")

    ax.set_xticks(x_base)
    ax.set_xticklabels([c.replace("_mean", "").replace("_", " ") for c in order_metrics],
                       fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Order-Aware Metrics Comparison")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    out = FIGS_DIR / "order_aware_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure → {out}")


def make_final_figures(df_main: pd.DataFrame, df_length: pd.DataFrame) -> None:
    """Save final paper-quality figures (includes all methods + hybrid)."""

    # ── final_method_comparison.png ───────────────────────────────────────
    n_metrics = len(KEY_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 5))

    for ax, metric in zip(axes, KEY_METRICS):
        means, stds, labels, colors = [], [], [], []
        for ci, (m_id, m_label) in enumerate(METHOD_LABELS.items()):
            row = df_main[df_main["method"] == m_id]
            if row.empty:
                continue
            means.append(float(row[f"{metric}_mean"].iloc[0]))
            stds.append(float(row[f"{metric}_std"].iloc[0]))
            labels.append(m_label)
            colors.append(_METHOD_COLORS[ci % len(_METHOD_COLORS)])

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=3,
                      color=colors, edgecolor="white", width=0.65)
        # Highlight hybrid bar with a bold edge
        for bar, lbl in zip(bars, labels):
            if "Hybrid" in lbl:
                bar.set_edgecolor("#333333")
                bar.set_linewidth(2.0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(metric.replace("_", " "), fontsize=9)
        top = max(means) * 1.35 + 0.02 if means else 0.1
        ax.set_ylim(0, min(1.05, top))
        ax.set_title(metric.replace("_", " ").title(), fontsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.suptitle("SkillGraph — Final Method Comparison (with Hybrid)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGS_DIR / "final_method_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure → {out}")

    # ── order_vs_recall_scatter.png ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    xs, ys = [], []
    for ci, (m_id, m_label) in enumerate(METHOD_LABELS.items()):
        row = df_main[df_main["method"] == m_id]
        if row.empty:
            continue
        x = float(row["f1_at_k_mean"].iloc[0])
        y = float(row["ordered_precision_mean"].iloc[0])
        xs.append(x)
        ys.append(y)
        color  = _METHOD_COLORS[ci % len(_METHOD_COLORS)]
        ms     = 180 if "Hybrid" in m_label else 120
        zorder = 6  if "Hybrid" in m_label else 5
        ax.scatter(x, y, s=ms, color=color, zorder=zorder,
                   edgecolors="#333333" if "Hybrid" in m_label else "none",
                   linewidths=1.5, label=m_label)
        offset = (6, 6)
        ax.annotate(m_label, (x, y), textcoords="offset points",
                    xytext=offset, fontsize=8,
                    fontweight="bold" if "Hybrid" in m_label else "normal")

    # Quadrant reference lines (at data midpoints)
    if xs and ys:
        xmid = (min(xs) + max(xs)) / 2
        ymid = (min(ys) + max(ys)) / 2
        ax.axvline(x=xmid, color="gray", linestyle="--", alpha=0.35, linewidth=0.9)
        ax.axhline(y=ymid, color="gray", linestyle="--", alpha=0.35, linewidth=0.9)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(xlim[1] - 0.002, ylim[1] - 0.002,
                "High Recall &\nHigh Order (Ideal)",
                fontsize=7, ha="right", va="top", color="gray", style="italic")

    ax.set_xlabel("F1@K  (Tool Recall)", fontsize=11)
    ax.set_ylabel("Ordered Precision  (Sequence Quality)", fontsize=11)
    ax.set_title("Recall vs. Order Quality Trade-off", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.8)
    ax.grid(alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    out = FIGS_DIR / "order_vs_recall_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure → {out}")


# ---------------------------------------------------------------------------
# Save all outputs
# ---------------------------------------------------------------------------

def save_outputs(
    rows:       list[dict],
    graph_info: dict,
    path_lens:  dict[str, list[int]],
    is_final:   bool = False,
) -> None:
    if not rows:
        print("Warning: no evaluation rows — skipping output.")
        return

    df = pd.DataFrame(rows)

    # Load old results for comparison (before overwriting)
    old_csv = TABLES_DIR / "main_results.csv"
    df_old: Optional[pd.DataFrame] = None
    if old_csv.exists():
        try:
            df_old = pd.read_csv(old_csv)
        except Exception:
            pass

    # ── Main results (per method) ─────────────────────────────────────────
    df_main = aggregate(df, ["method"])
    df_main.to_csv(TABLES_DIR / "main_results.csv", index=False)
    print(f"  CSV  → {TABLES_DIR / 'main_results.csv'}")

    df_length = aggregate(df, ["method", "bucket"])
    df_length.to_csv(TABLES_DIR / "results_by_length.csv", index=False)
    print(f"  CSV  → {TABLES_DIR / 'results_by_length.csv'}")

    df_cats = aggregate(df, ["method", "n_gt_cats"])
    df_cats.to_csv(TABLES_DIR / "results_by_category.csv", index=False)
    print(f"  CSV  → {TABLES_DIR / 'results_by_category.csv'}")

    pretty = pretty_table(df_main)
    (TABLES_DIR / "main_results_pretty.txt").write_text(pretty, encoding="utf-8")
    print(f"  TXT  → {TABLES_DIR / 'main_results_pretty.txt'}")

    graph_info["purity_df"].to_csv(TABLES_DIR / "community_purity.csv", index=False)
    print(f"  CSV  → {TABLES_DIR / 'community_purity.csv'}")

    # ── Final output files (written when is_final=True) ───────────────────
    if is_final:
        df_main.to_csv(TABLES_DIR / "final_results.csv", index=False)
        print(f"  CSV  → {TABLES_DIR / 'final_results.csv'}")

        (TABLES_DIR / "final_results_pretty.txt").write_text(pretty, encoding="utf-8")
        print(f"  TXT  → {TABLES_DIR / 'final_results_pretty.txt'}")

        if df_old is not None:
            save_before_after_csv(
                df_old, df_main,
                TABLES_DIR / "before_after_comparison.csv",
            )

    # ── Print summary ─────────────────────────────────────────────────────
    gm = graph_info["graph_metrics"]
    print("\n" + "=" * 70)
    print("  SkillGraph — Evaluation Results")
    print("=" * 70)

    print("\n  Graph-level metrics:")
    for k, v in gm.items():
        print(f"    {k:35s}: {v}")

    print("\n  Average predicted path lengths:")
    for method in METHODS:
        lens = path_lens.get(method, [])
        if lens:
            avg    = sum(lens) / len(lens)
            gt_avg = df[df["method"] == method]["gt_len"].mean()
            flag   = " [!short path]" if avg <= 1.1 else ""
            print(f"    {METHOD_LABELS.get(method, method):26s}: "
                  f"pred={avg:.2f}  gt={gt_avg:.2f}{flag}")

    print("\n  Planning metrics (all methods):")
    print(pretty)

    # ── OLD vs NEW comparison ─────────────────────────────────────────────
    if df_old is not None:
        print_comparison(df_old, df_main)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    make_figures(df_main, df_length)
    if is_final:
        make_final_figures(df_main, df_length)

    # ── Final ranking table ────────────────────────────────────────────────
    if is_final:
        print("\n" + "=" * 70)
        print("  FINAL RANKING TABLE")
        print("=" * 70)
        print(pretty)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkillGraph offline evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample",    type=int, default=0,
        help="Evaluate on N random test samples (0 = full test set)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=8,
        help="Max planning steps per query",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-tuned-params", action="store_true",
        help="Load best_hyperparams.json if it exists",
    )
    parser.add_argument(
        "--tune-first", action="store_true",
        help="Run hyperparameter tuning on training data before evaluating",
    )
    parser.add_argument(
        "--tune-val-n", type=int, default=500,
        help="Number of training samples to use for tuning (with --tune-first)",
    )
    parser.add_argument(
        "--final", action="store_true",
        help="Save outputs to final_results.csv and generate final figures",
    )
    args = parser.parse_args()

    rng_global = random.Random(args.seed)
    np.random.seed(args.seed)

    # ── 1. Load data ──────────────────────────────────────────────────────
    if not TRAJ_FILE.exists():
        print(f"ERROR: {TRAJ_FILE} not found.  Run extract.py first.")
        return

    records = load_trajectories()
    train_records, test_records = make_train_test_split(records, seed=args.seed)

    avg_train_len    = sum(r["num_steps"] for r in train_records) / len(train_records)
    median_train_len = float(statistics.median(r["num_steps"] for r in train_records))
    print(f"  Avg training trajectory length: {avg_train_len:.2f}")
    print(f"  Median training trajectory length: {median_train_len:.2f}")

    if args.sample > 0:
        rng_global.shuffle(test_records)
        test_records = test_records[: args.sample]
        print(f"  Sampled {len(test_records):,} test trajectories for quick eval")

    # ── 2. Load planner ───────────────────────────────────────────────────
    print("\nLoading ToolSequencePlanner …")
    planner = ToolSequencePlanner()
    planner._avg_traj_len    = avg_train_len
    planner._median_traj_len = median_train_len

    # Compute tool position statistics for Hybrid method
    tool_pos_lists: dict[str, list[float]] = defaultdict(list)
    for rec in train_records:
        seq = rec["tool_sequence"]
        n   = len(seq)
        if n == 0:
            continue
        for i, t in enumerate(seq):
            tool_pos_lists[t].append(i / n)
    planner._tool_position_stats = {
        t: float(np.mean(v)) for t, v in tool_pos_lists.items()
    }
    print(f"  Tool position stats computed for {len(planner._tool_position_stats):,} tools")

    # ── 3. Hyperparameter tuning (optional) ──────────────────────────────
    if args.tune_first:
        print("\nRunning hyperparameter tuning before evaluation …")
        _rng = random.Random(42)
        split_data = json.loads(SPLIT_FILE.read_text(encoding="utf-8")) \
                     if SPLIT_FILE.exists() else None
        if split_data:
            train_idx = split_data["train_idx"]
        else:
            train_idx = list(range(int(len(records) * 0.8)))

        sample_idx = train_idx.copy()
        _rng.shuffle(sample_idx)
        val_records = [records[i] for i in sample_idx[:args.tune_val_n]]

        planner.tune_hyperparameters(val_records, n_trials=100)
        print("  Tuning complete.")

    # ── 4. Load / use tuned hyperparameters ──────────────────────────────
    dijkstra_beta      = 0.3
    dijkstra_n_entries = 5
    beam_params        = {"w1": 0.4, "w2": 0.4, "w3": 0.2, "beam_width": 5}
    hybrid_params      = {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}

    if args.use_tuned_params:
        hp = planner.load_best_hyperparams()
        if hp:
            dij_hp = hp.get("dijkstra", {})
            dijkstra_beta      = dij_hp.get("beta",      dijkstra_beta)
            dijkstra_n_entries = dij_hp.get("n_entries", dijkstra_n_entries)

            bm_hp = hp.get("beam", {})
            for k in beam_params:
                beam_params[k] = bm_hp.get(k, beam_params[k])

            hyb_hp = hp.get("hybrid", {})
            for k in hybrid_params:
                hybrid_params[k] = hyb_hp.get(k, hybrid_params[k])

            print(f"  Loaded tuned params:")
            print(f"    Dijkstra  beta={dijkstra_beta}  n_entries={dijkstra_n_entries}")
            print(f"    Beam      {beam_params}")
            print(f"    Hybrid    {hybrid_params}")
        else:
            print("  No tuned params found — using defaults. "
                  "Run with --tune-first to generate them.")

    # ── 5. Build baselines ────────────────────────────────────────────────
    sem_baseline  = SemanticOnlyBaseline(planner)
    freq_baseline = FrequencyBaseline(train_records)
    rand_baseline = RandomGraphBaseline(planner, seed=99)

    # ── 6. Batch-encode all test queries ──────────────────────────────────
    queries    = [rec["task_description"] for rec in test_records]
    query_vecs = batch_encode_queries(queries)

    # ── 7. Graph-level metrics ────────────────────────────────────────────
    print("\nComputing graph-level metrics …")
    graph_info = graph_level_metrics(planner)
    gm = graph_info["graph_metrics"]
    print(f"  Modularity            : {gm['modularity']}")
    print(f"  Mean community purity : {gm['mean_community_purity']}")
    print(f"  NMI vs API category   : {gm['nmi_vs_api_category']}")

    # ── 8. Planning evaluation ────────────────────────────────────────────
    n_total = len(test_records) * len(METHODS)
    print(f"\nRunning {len(test_records):,} trajectories x {len(METHODS)} methods "
          f"= {n_total:,} evaluations …")

    rows, path_lens = evaluate_all_methods(
        test_records, planner, query_vecs,
        sem_baseline, freq_baseline, rand_baseline,
        max_steps=args.max_steps,
        dijkstra_beta=dijkstra_beta,
        dijkstra_n_entries=dijkstra_n_entries,
        beam_params=beam_params,
        hybrid_params=hybrid_params,
    )

    # ── 9. Save + compare ────────────────────────────────────────────────
    is_final = args.final or args.tune_first or args.use_tuned_params
    print("\nSaving outputs …")
    save_outputs(rows, graph_info, path_lens, is_final=is_final)

    print("\nDone.")


if __name__ == "__main__":
    main()
