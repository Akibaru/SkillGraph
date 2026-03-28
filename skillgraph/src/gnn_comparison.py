"""
src/gnn_comparison.py  —  GNN Integration Evaluation  (v2, fixed)
=================================================================
Fixes applied vs v1:
  Fix 1  Score coverage   : uses full N×N score matrix (all 24.9M pairs),
                            not just the 39K graph edges.  Every (i,j) pair
                            gets a meaningful score from learned embeddings.
  Fix 2  Beam formula     : beam_gnn = 0.4*gnn + 0.4*sim + 0.2*diversity
                            (was: score = gnn only; now only replaces tp term)
  Fix 3  Score normalise  : per-step min-max normalisation of GNN scores and
                            semantic scores before alpha-weighted fusion
  Fix 4  Alpha re-search  : grid-search alpha ∈ {0.1,…,0.9} with new scores

Original methods (frequency / random_graph / semantic_only / dijkstra /
hierarchical / beam / hybrid) call the exact same code paths as evaluate.py
— their results are identical to a standalone evaluate.py run.

Usage
-----
  python src/gnn_comparison.py               # full test set (~10K trajectories)
  python src/gnn_comparison.py --sample 500  # quick sanity check
  python src/gnn_comparison.py --encoder gcn # use GCN instead of SAGE
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import random
import statistics
import sys
import time
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from graph_search import ToolSequencePlanner, ToolPlan, _fallback_plan
from evaluate import (
    METRIC_COLS,
    load_trajectories,
    make_train_test_split,
    batch_encode_queries,
    compute_metrics,
    SemanticOnlyBaseline,
    FrequencyBaseline,
    RandomGraphBaseline,
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
FIGS_DIR    = ROOT / "outputs" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
ALL_METHODS = [
    "frequency",       # 1. frequency baseline
    "random_graph",    # 2. random-graph dijkstra
    "semantic_only",   # 3. semantic-only baseline
    "dijkstra",        # 4. semantic-guided Dijkstra
    "hierarchical",    # 5. community-hierarchical
    "beam",            # 6. beam search (original)
    "beam_gnn",        # 7. beam search (GNN) ← new
    "hybrid",          # 8. hybrid sem-graph (original)
    "hybrid_gnn",      # 9. hybrid (GNN) ← new
    "gnn_only",        # 10. GNN-only ← new
]

METHOD_LABELS = {
    "frequency":    "Frequency (BL)",
    "random_graph": "Random Graph (BL)",
    "semantic_only":"Semantic Only (BL)",
    "dijkstra":     "Semantic Dijkstra",
    "hierarchical": "Community-Hier.",
    "beam":         "Beam Search",
    "beam_gnn":     "Beam (GNN)",
    "hybrid":       "Hybrid Sem-Graph",
    "hybrid_gnn":   "Hybrid (GNN)",
    "gnn_only":     "GNN Only",
}

ORIG_METHODS = frozenset({
    "frequency", "random_graph", "semantic_only",
    "dijkstra", "hierarchical", "beam", "hybrid",
})


# ============================================================================
# Score-access helpers
# ============================================================================

def get_score(
    tool_i: str,
    tool_j: str,
    score_matrix: np.ndarray,
    t2i: dict[str, int],
) -> float:
    """O(1) look-up from the full N×N score matrix."""
    i = t2i.get(tool_i)
    j = t2i.get(tool_j)
    if i is None or j is None:
        return 0.0
    return float(score_matrix[i, j])


def _minmax(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns 0.5 everywhere if flat."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < eps:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


# ============================================================================
# Fix 2+3: beam_search_gnn
# ============================================================================

def beam_search_gnn(
    planner:      ToolSequencePlanner,
    query_vec:    np.ndarray,
    score_matrix: np.ndarray,
    t2i:          dict[str, int],
    beam_width:   int   = 5,
    n_entries:    int   = 5,
    max_steps:    int   = 8,
    w1:           float = 0.4,   # GNN transition weight   (replaces tp)
    w2:           float = 0.4,   # semantic similarity weight
    w3:           float = 0.2,   # community-diversity weight
) -> list[ToolPlan]:
    """
    Beam Search with GNN transition score replacing the statistical
    transition probability term.  Semantic similarity and community
    diversity components are preserved from the original beam_search.

    Scoring (Fix 2):
        step_score = w1 * norm_gnn(curr→nbr)
                   + w2 * norm_sim(nbr, query)
                   + w3 * community_diversity

    Both gnn and sim components are min-max normalised across all
    valid neighbours at each expansion step (Fix 3).
    """
    entry_tools = planner._top_entry_tools(query_vec, k=n_entries)
    target      = max(2, int(round(planner._median_traj_len)))

    # BeamState: (neg_cum_score, path, visited_comms, step_scores)
    initial_beams: list = []
    for entry, entry_sim in entry_tools:
        if entry not in planner.G:
            continue
        cid = planner.communities.get(entry, -1)
        initial_beams.append((-entry_sim, [entry], frozenset([cid]), [entry_sim]))

    if not initial_beams:
        return [_fallback_plan("", entry_tools, "beam_gnn")]

    initial_beams.sort(key=lambda x: x[0])
    beams     = initial_beams[:beam_width]
    completed: list[ToolPlan] = []

    for _step in range(max_steps - 1):
        if not beams:
            break
        candidates: list = []

        for neg_score, path, visited_comms, step_scores in beams:
            if len(path) >= target:
                completed.append(ToolPlan(
                    tools=path, scores=step_scores,
                    total_score=(-neg_score) / len(path),
                    path_length=len(path), method="beam_gnn",
                    entry_tool=path[0],
                ))
                continue

            current   = path[-1]
            cum_score = -neg_score
            neighbors = [
                (nbr, we, tp)
                for nbr, we, tp in planner._adj.get(current, [])[:30]
                if nbr not in path
            ]

            if not neighbors:
                completed.append(ToolPlan(
                    tools=path, scores=step_scores,
                    total_score=cum_score / len(path),
                    path_length=len(path), method="beam_gnn",
                    entry_tool=path[0],
                ))
                continue

            # Collect raw gnn + sim scores for all valid neighbors
            nbr_names  = [n for n, _, _ in neighbors]
            gnn_raw    = np.array([
                score_matrix[t2i[current], t2i[n]]
                if current in t2i and n in t2i else 0.0
                for n in nbr_names
            ], dtype=np.float32)
            sim_raw    = np.array([
                planner._tool_sim(n, query_vec) for n in nbr_names
            ], dtype=np.float32)

            # Fix 3: per-step min-max normalisation
            gnn_norm = _minmax(gnn_raw)
            sim_norm = _minmax(sim_raw)

            for idx, nbr in enumerate(nbr_names):
                nbr_cid   = planner.communities.get(nbr, -1)
                same_comm = sum(
                    1 for t in path if planner.communities.get(t) == nbr_cid
                )
                diversity  = 1.0 - same_comm / len(path)
                step_score = (w1 * float(gnn_norm[idx])
                              + w2 * float(sim_norm[idx])
                              + w3 * diversity)
                new_comms  = visited_comms | {nbr_cid}
                candidates.append((
                    -(cum_score + step_score),
                    path + [nbr],
                    new_comms,
                    step_scores + [step_score],
                ))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        beams = candidates[:beam_width]

    # Flush remaining active beams
    for neg_score, path, _, step_scores in beams:
        completed.append(ToolPlan(
            tools=path, scores=step_scores,
            total_score=(-neg_score) / len(path),
            path_length=len(path), method="beam_gnn",
            entry_tool=path[0],
        ))

    if not completed:
        return [_fallback_plan("", entry_tools, "beam_gnn")]

    # Truncate all to target_length
    final: list[ToolPlan] = []
    for p in completed:
        t = p.tools[:target]
        s = p.scores[:target]
        final.append(ToolPlan(
            tools=t, scores=s,
            total_score=float(np.mean(s)) if s else 0.0,
            path_length=len(t), method="beam_gnn",
            entry_tool=t[0] if t else "",
        ))

    final.sort(key=lambda p: -p.total_score)
    return final[:3]


# ============================================================================
# Fix 2+3: hybrid_gnn
# ============================================================================

def hybrid_gnn(
    planner:      ToolSequencePlanner,
    query_vec:    np.ndarray,
    score_matrix: np.ndarray,
    t2i:          dict[str, int],
    alpha:        float = 0.5,   # GNN weight; (1-alpha) = semantic
    k_multiplier: int   = 3,
    max_steps:    int   = 8,
) -> ToolPlan:
    """
    Hybrid planning with alpha-weighted fusion of GNN and semantic scores.

    Steps 1+2: same as original hybrid (semantic candidates + bridges).
    Step 3 scoring (Fix 2+3):
        Per greedy step, for all remaining candidates compute:
            gnn_scores = score_matrix[current, candidates]  (full N×N)
            sem_scores = cosine_sim(candidates, query)
            step_score = alpha * minmax(gnn_scores)
                       + (1-alpha) * minmax(sem_scores)
    """
    import networkx as nx

    target = max(2, min(max_steps, int(round(planner._median_traj_len))))
    K = max(target + 2, k_multiplier * target)
    K = min(K, planner.n_tools)

    # Step 1: semantic candidates
    candidates_list = planner._top_entry_tools(query_vec, k=K)
    candidate_set:  set[str]        = {t for t, _ in candidates_list}
    candidate_sims: dict[str, float] = {t: s for t, s in candidates_list}

    # Step 2: induced subgraph + bridges (identical to original hybrid)
    sg = planner.G.subgraph(candidate_set).copy()
    try:
        components = list(nx.weakly_connected_components(sg))
    except Exception:
        components = [candidate_set]

    if len(components) > 1:
        bridge_tools: set[str] = set()
        for ci in range(len(components) - 1):
            comp_a = sorted(
                components[ci],
                key=lambda t: candidate_sims.get(t, 0.0), reverse=True,
            )
            comp_b = sorted(
                components[ci + 1],
                key=lambda t: candidate_sims.get(t, 0.0), reverse=True,
            )
            connected = False
            for ta in comp_a[:3]:
                for tb in comp_b[:3]:
                    try:
                        sp = nx.shortest_path(planner.G, ta, tb, weight=None)
                        for bnode in sp[1:-1]:
                            bridge_tools.add(bnode)
                        connected = True
                        break
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                if connected:
                    break
        for bt in bridge_tools:
            candidate_set.add(bt)
            candidate_sims[bt] = planner._tool_sim(bt, query_vec)

    # Step 3: greedy sequencing with normalised GNN + semantic fusion
    start_tool   = max(candidate_set, key=lambda t: candidate_sims.get(t, 0.0))
    sequence:    list[str]   = [start_tool]
    step_scores: list[float] = [candidate_sims.get(start_tool, 0.0)]
    remaining = candidate_set - {start_tool}

    for _ in range(1, target):
        if not remaining:
            break
        current        = sequence[-1]
        remaining_list = list(remaining)
        cur_idx        = t2i.get(current)

        # Raw GNN scores from full matrix (covers all pairs)
        if cur_idx is not None:
            gnn_raw = np.array([
                float(score_matrix[cur_idx, t2i[n]])
                if n in t2i else 0.0
                for n in remaining_list
            ], dtype=np.float32)
        else:
            gnn_raw = np.zeros(len(remaining_list), dtype=np.float32)

        sem_raw = np.array(
            [candidate_sims.get(n, 0.0) for n in remaining_list],
            dtype=np.float32,
        )

        # Fix 3: normalise within this step's candidate set
        gnn_norm   = _minmax(gnn_raw)
        sem_norm   = _minmax(sem_raw)
        fused      = alpha * gnn_norm + (1.0 - alpha) * sem_norm
        best_idx   = int(np.argmax(fused))
        best_tool  = remaining_list[best_idx]

        sequence.append(best_tool)
        step_scores.append(float(fused[best_idx]))
        remaining.remove(best_tool)

    # Step 4: length calibration
    if len(sequence) < target and remaining:
        extras = sorted(
            remaining, key=lambda t: candidate_sims.get(t, 0.0), reverse=True
        )
        for t in extras:
            sequence.append(t)
            step_scores.append(candidate_sims.get(t, 0.0))
            if len(sequence) >= target:
                break

    sequence    = sequence[:target]
    step_scores = step_scores[:target]
    total       = float(np.mean(step_scores)) if step_scores else 0.0

    return ToolPlan(
        tools=sequence, scores=step_scores,
        total_score=total, path_length=len(sequence),
        method="hybrid_gnn", entry_tool=start_tool,
    )


# ============================================================================
# Fix 1: gnn_only (now uses full matrix → no zero-score fallback)
# ============================================================================

def gnn_only(
    planner:      ToolSequencePlanner,
    query_vec:    np.ndarray,
    score_matrix: np.ndarray,
    t2i:          dict[str, int],
    k_multiplier: int = 3,
    max_steps:    int = 8,
) -> ToolPlan:
    """
    GNN-only greedy planning.

    Candidate pool  : top-K semantic tools (K = k_multiplier × target).
    Sequencing      : at each step, pick next = argmax score_matrix[current, ·]
                      over remaining candidates.
    Fallback        : semantic sim for the first (entry) step only.

    With the full N×N matrix every (current, candidate) pair has a non-zero
    score, so no fallback is needed after the first step.
    """
    target = max(2, min(max_steps, int(round(planner._median_traj_len))))
    K = max(target + 2, k_multiplier * target)
    K = min(K, planner.n_tools)

    candidates_list = planner._top_entry_tools(query_vec, k=K)
    candidate_set:  set[str]        = {t for t, _ in candidates_list}
    candidate_sims: dict[str, float] = {t: s for t, s in candidates_list}

    # Bootstrap: semantic entry
    start_tool   = max(candidate_set, key=lambda t: candidate_sims.get(t, 0.0))
    sequence:    list[str]   = [start_tool]
    step_scores: list[float] = [candidate_sims.get(start_tool, 0.0)]
    remaining = candidate_set - {start_tool}

    for _ in range(1, target):
        if not remaining:
            break
        current  = sequence[-1]
        cur_idx  = t2i.get(current)

        if cur_idx is not None:
            remaining_list = list(remaining)
            gnn_scores = np.array([
                float(score_matrix[cur_idx, t2i[n]])
                if n in t2i else 0.0
                for n in remaining_list
            ], dtype=np.float32)
            best_idx  = int(np.argmax(gnn_scores))
            best_tool = remaining_list[best_idx]
            score_val = float(gnn_scores[best_idx])
        else:
            # tool not in graph (rare) → semantic fallback
            best_tool = max(remaining, key=lambda t: candidate_sims.get(t, 0.0))
            score_val = candidate_sims.get(best_tool, 0.0)

        sequence.append(best_tool)
        step_scores.append(score_val)
        remaining.remove(best_tool)

    # Pad if short
    if len(sequence) < target and remaining:
        extras = sorted(
            remaining, key=lambda t: candidate_sims.get(t, 0.0), reverse=True
        )
        for t in extras:
            sequence.append(t)
            step_scores.append(candidate_sims.get(t, 0.0))
            if len(sequence) >= target:
                break

    sequence    = sequence[:target]
    step_scores = step_scores[:target]
    total       = float(np.mean(step_scores)) if step_scores else 0.0

    return ToolPlan(
        tools=sequence, scores=step_scores,
        total_score=total, path_length=len(sequence),
        method="gnn_only", entry_tool=start_tool,
    )


# ============================================================================
# Fix 4: Alpha grid search for hybrid_gnn (re-run with new full scores)
# ============================================================================

def find_best_hybrid_alpha(
    planner:        ToolSequencePlanner,
    val_records:    list[dict],
    score_matrix:   np.ndarray,
    t2i:            dict[str, int],
    val_query_vecs: np.ndarray,
    max_steps:      int = 8,
) -> float:
    """Grid-search alpha ∈ {0.1,…,0.9} on val set; returns best alpha by F1@K."""
    alphas = [round(a, 1) for a in np.arange(0.1, 1.0, 0.1)]

    print(f"\n[alpha search] Val size={len(val_records)}  "
          f"candidates={alphas}")

    best_alpha = 0.5
    best_f1    = -1.0

    for alpha in alphas:
        f1_list = []
        for i, rec in enumerate(val_records):
            gt  = rec["tool_sequence"]
            vec = val_query_vecs[i]
            try:
                plan = hybrid_gnn(
                    planner, vec, score_matrix, t2i,
                    alpha=alpha, max_steps=max_steps,
                )
                m = compute_metrics(plan.tools, gt)
                f1_list.append(m["f1_at_k"])
            except Exception:
                f1_list.append(0.0)

        mean_f1 = float(np.mean(f1_list))
        marker  = "  ←" if mean_f1 > best_f1 else ""
        print(f"  alpha={alpha:.1f}  val_F1@K={mean_f1:.4f}{marker}")

        if mean_f1 > best_f1:
            best_f1    = mean_f1
            best_alpha = alpha

    print(f"[alpha search] Best alpha={best_alpha:.1f}  val_F1={best_f1:.4f}\n")
    return best_alpha


# ============================================================================
# Score-distribution figure  (Fix 1 diagnostic)
# ============================================================================

def plot_score_distribution(
    score_matrix: np.ndarray,
    planner:      ToolSequencePlanner,
    t2i:          dict[str, int],
    out_path:     pathlib.Path,
    n_sample:     int = 50_000,
) -> None:
    """
    Histogram comparing GNN scores for:
      • existing graph edges  (observed co-occurrences)
      • random non-edge pairs (zero-shot generalization)
    """
    G    = planner.G
    rng  = np.random.default_rng(42)
    N    = score_matrix.shape[0]

    # Scores on existing graph edges
    edge_scores = []
    for u, v in G.edges():
        ui = t2i.get(u)
        vi = t2i.get(v)
        if ui is not None and vi is not None:
            edge_scores.append(float(score_matrix[ui, vi]))
    edge_scores = np.array(edge_scores, dtype=np.float32)

    # Scores on random non-edge pairs (same size sample)
    edge_set = {
        (t2i[u], t2i[v])
        for u, v in G.edges()
        if u in t2i and v in t2i
    }
    n_target = min(n_sample, len(edge_scores))
    non_edge_scores = []
    while len(non_edge_scores) < n_target:
        us = rng.integers(0, N, size=n_target * 3)
        vs = rng.integers(0, N, size=n_target * 3)
        for ui, vi in zip(us.tolist(), vs.tolist()):
            if ui != vi and (ui, vi) not in edge_set:
                non_edge_scores.append(float(score_matrix[ui, vi]))
                if len(non_edge_scores) == n_target:
                    break
    non_edge_scores = np.array(non_edge_scores, dtype=np.float32)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bins = 50
    for ax, scores, label, color in [
        (axes[0], edge_scores,     "Existing edges\n(graph co-occurrences)", "#4C72B0"),
        (axes[1], non_edge_scores, "Random non-edges\n(zero-shot pairs)",    "#C44E52"),
    ]:
        ax.hist(scores, bins=bins, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(float(np.mean(scores)), color="#333333", linestyle="--",
                   linewidth=1.4, label=f"mean={np.mean(scores):.3f}")
        ax.axvline(float(np.median(scores)), color="#666666", linestyle=":",
                   linewidth=1.4, label=f"median={np.median(scores):.3f}")
        ax.set_xlabel("GNN Transition Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"GNN Score Distribution — SAGE encoder\n"
        f"Existing edges: {len(edge_scores):,}   "
        f"Non-edge sample: {len(non_edge_scores):,}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[figure] {out_path}")


# ============================================================================
# Coverage stats
# ============================================================================

def print_coverage_stats(
    score_matrix: np.ndarray,
    planner:      ToolSequencePlanner,
    t2i:          dict[str, int],
) -> None:
    N       = score_matrix.shape[0]
    n_edges = planner.G.number_of_edges()
    n_full  = N * N - N   # exclude self-loops

    print("\n" + "=" * 60)
    print("  GNN Score Coverage Comparison")
    print("=" * 60)
    print(f"  Graph nodes (N)            : {N:>10,}")
    print(f"  Graph edges (existing)     : {n_edges:>10,}")
    print(f"  Total N×N pairs (no self)  : {n_full:>10,}")
    print(f"  v1 cache (graph edges only): {n_edges:>10,}  "
          f"({100*n_edges/n_full:.2f}% coverage)")
    print(f"  v2 matrix (all pairs)      : {n_full:>10,}  "
          f"(100.00% coverage)")
    print(f"  Coverage improvement       :  ×{n_full/n_edges:,.0f}")

    # Score summary by group
    edge_idx = [
        (t2i[u], t2i[v])
        for u, v in planner.G.edges()
        if u in t2i and v in t2i
    ]
    if edge_idx:
        us = [p[0] for p in edge_idx]
        vs = [p[1] for p in edge_idx]
        edge_scores = score_matrix[us, vs]
        print(f"\n  Existing edges  — "
              f"mean={edge_scores.mean():.4f}  "
              f"median={np.median(edge_scores):.4f}  "
              f"p90={np.percentile(edge_scores, 90):.4f}")

    # Random non-edge sample
    rng = np.random.default_rng(42)
    us  = rng.integers(0, N, size=50_000)
    vs  = rng.integers(0, N, size=50_000)
    mask = us != vs
    sample_scores = score_matrix[us[mask], vs[mask]][:50_000]
    print(f"  Non-edge sample — "
          f"mean={sample_scores.mean():.4f}  "
          f"median={np.median(sample_scores):.4f}  "
          f"p90={np.percentile(sample_scores, 90):.4f}")
    print("=" * 60)


# ============================================================================
# Evaluation loop
# ============================================================================

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


def evaluate_all(
    test_records:     list[dict],
    planner:          ToolSequencePlanner,
    query_vecs:       np.ndarray,
    score_matrix:     np.ndarray,
    t2i:              dict[str, int],
    sem_baseline:     SemanticOnlyBaseline,
    freq_baseline:    FrequencyBaseline,
    rand_baseline:    RandomGraphBaseline,
    hybrid_gnn_alpha: float = 0.5,
    max_steps:        int   = 8,
    beam_params:      dict | None = None,
    hybrid_params:    dict | None = None,
    dijkstra_beta:    float = 0.3,
) -> list[dict]:
    """Run all 10 methods and return flat list of per-trajectory metric rows."""
    bp = beam_params   or {"w1": 0.4, "w2": 0.4, "w3": 0.2, "beam_width": 5}
    hp = hybrid_params or {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}
    all_rows: list[dict] = []

    for method in ALL_METHODS:
        print(f"\n  [{method}]  evaluating {len(test_records):,} trajectories …")
        t_start = time.time()

        for i, rec in enumerate(test_records):
            gt  = rec["tool_sequence"]
            vec = query_vecs[i]
            K   = len(gt)

            t0 = time.time()
            try:
                # ── Original methods — IDENTICAL call paths to evaluate.py ──
                if method in ("dijkstra", "beam", "hierarchical", "hybrid"):
                    plans     = _plan_with_vec(
                        planner, method, vec, max_steps=max_steps,
                        beta=dijkstra_beta,
                        beam_params=bp, hybrid_params=hp,
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

                # ── New GNN methods ──────────────────────────────────────────
                elif method == "beam_gnn":
                    plans     = beam_search_gnn(
                        planner, vec, score_matrix, t2i, max_steps=max_steps
                    )
                    predicted = plans[0].tools if plans else []

                elif method == "hybrid_gnn":
                    plan      = hybrid_gnn(
                        planner, vec, score_matrix, t2i,
                        alpha=hybrid_gnn_alpha, max_steps=max_steps,
                    )
                    predicted = plan.tools

                elif method == "gnn_only":
                    plan      = gnn_only(
                        planner, vec, score_matrix, t2i, max_steps=max_steps
                    )
                    predicted = plan.tools

                else:
                    predicted = []

            except Exception as exc:
                print(f"    [warn] {method} row {i}: {exc}")
                predicted = []

            latency = time.time() - t0
            row     = compute_metrics(predicted, gt, latency_s=latency)
            row["method"] = method
            all_rows.append(row)

        elapsed = time.time() - t_start
        method_rows = [r for r in all_rows if r["method"] == method]
        avg_len = sum(r["pred_len"] for r in method_rows) / max(1, len(method_rows))
        print(f"  [{method}] done in {elapsed:.1f}s  avg_pred_len={avg_len:.2f}")

    return all_rows


# ============================================================================
# Output helpers
# ============================================================================

def make_summary(rows: list[dict]) -> pd.DataFrame:
    df  = pd.DataFrame(rows)
    agg = df.groupby("method")[REPORT_METRICS].mean().round(4).reset_index()
    agg["_order"] = agg["method"].map(
        {m: i for i, m in enumerate(ALL_METHODS)}
    )
    agg = agg.sort_values("_order").drop(columns="_order")
    agg.insert(1, "label",
               agg["method"].map(METHOD_LABELS).fillna(agg["method"]))
    return agg


def print_table(summary: pd.DataFrame) -> None:
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

    print(sep); print(header); print(sep)

    gnn_methods = {"beam_gnn", "hybrid_gnn", "gnn_only"}
    prev_was_orig = False
    for idx, (_, row) in enumerate(sub.iterrows()):
        m_id = ALL_METHODS[idx]
        if m_id in gnn_methods and prev_was_orig:
            print(sep)
        cells = [
            (f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(w)
            for v, w in zip(row, col_w)
        ]
        print("| " + " | ".join(cells) + " |")
        prev_was_orig = m_id in ORIG_METHODS

    print(sep)
    print("  * = new GNN-based method (v2: full N×N score matrix)")


def print_delta(summary: pd.DataFrame) -> None:
    pairs        = [("beam", "beam_gnn", "Beam Search"),
                    ("hybrid", "hybrid_gnn", "Hybrid")]
    delta_metrics = ["f1_at_k", "ordered_precision", "lcs_r",
                     "transition_accuracy"]

    print("\n" + "=" * 80)
    print("  Delta: GNN variants vs originals (v2 fixed)")
    print("=" * 80)

    header = f"  {'Pair':<22}" + "".join(
        f"  {'orig→gnn':>14}  {'Δ':>8}" for _ in delta_metrics
    )
    sub_header = f"  {'':22}" + "".join(
        f"  {m:>14}  {'':>8}" for m in delta_metrics
    )
    print(sub_header)
    print("  " + "-" * (len(header) - 2))

    for orig_m, gnn_m, label in pairs:
        orig_row = summary[summary["method"] == orig_m]
        gnn_row  = summary[summary["method"] == gnn_m]
        if orig_row.empty or gnn_row.empty:
            continue
        line = f"  {label:<22}"
        for col in delta_metrics:
            ov = float(orig_row[col].iloc[0])
            gv = float(gnn_row[col].iloc[0])
            d  = gv - ov
            sg = "+" if d >= 0 else ""
            line += f"  {ov:.4f}→{gv:.4f}  {sg}{d:.4f}  "
        print(line)
    print()


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GNN-extended SkillGraph evaluation — v2 (full N×N matrix)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sample",    type=int,  default=0,
                   help="Evaluate on N random test samples (0 = full)")
    p.add_argument("--max-steps", type=int,  default=8)
    p.add_argument("--encoder",   type=str,  default="sage",
                   choices=["gcn", "gat", "sage"])
    p.add_argument("--val-n",     type=int,  default=500,
                   help="Val samples for alpha grid search")
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
        print(f"  Sampled {len(test_records):,} test trajectories")

    # ── 2. Load planner ───────────────────────────────────────────────────
    print("\nLoading ToolSequencePlanner …")
    planner = ToolSequencePlanner()
    planner._avg_traj_len    = avg_train_len
    planner._median_traj_len = median_train_len

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

    # ── 3. Load / generate full N×N score matrix ─────────────────────────
    try:
        score_matrix, active_tools, t2i = load_score_matrix(args.encoder)
    except FileNotFoundError:
        print(f"\n[info] Score matrix not found — computing now …")
        model, _, _ = load_transition_model(args.encoder, device_str="auto")
        score_matrix, active_tools = precompute_full_score_matrix(
            model, device_str="auto"
        )
        save_score_matrix(score_matrix, active_tools, encoder_type=args.encoder)
        t2i = {name: i for i, name in enumerate(active_tools)}

    # ── 4. Coverage stats + histogram ─────────────────────────────────────
    print_coverage_stats(score_matrix, planner, t2i)

    hist_path = FIGS_DIR / "gnn_score_distribution.png"
    plot_score_distribution(score_matrix, planner, t2i, hist_path)

    # ── 5. Build baselines ────────────────────────────────────────────────
    print("\nBuilding baselines …")
    sem_baseline  = SemanticOnlyBaseline(planner)
    freq_baseline = FrequencyBaseline(train_records)
    rand_baseline = RandomGraphBaseline(planner, seed=99)

    # ── 6. Batch-encode test queries ──────────────────────────────────────
    test_queries = [r["task_description"] for r in test_records]
    test_vecs    = batch_encode_queries(test_queries)

    # ── 7. Alpha grid search for hybrid_gnn ──────────────────────────────
    val_idx = list(range(len(train_records)))
    rng.shuffle(val_idx)
    val_records = [train_records[i] for i in val_idx[: args.val_n]]
    val_vecs    = batch_encode_queries(
        [r["task_description"] for r in val_records]
    )

    best_alpha = find_best_hybrid_alpha(
        planner, val_records, score_matrix, t2i, val_vecs,
        max_steps=args.max_steps,
    )

    # ── 8. Full evaluation ────────────────────────────────────────────────
    print(f"\nRunning {len(test_records):,} × {len(ALL_METHODS)} methods …")
    rows = evaluate_all(
        test_records, planner, test_vecs,
        score_matrix, t2i,
        sem_baseline, freq_baseline, rand_baseline,
        hybrid_gnn_alpha=best_alpha,
        max_steps=args.max_steps,
    )

    # ── 9. Save and display ───────────────────────────────────────────────
    df_all = pd.DataFrame(rows)
    df_all.to_csv(RESULTS_DIR / "gnn_comparison_raw.csv", index=False)

    summary  = make_summary(rows)
    csv_path = RESULTS_DIR / "gnn_comparison.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary → {csv_path}")

    print("\n" + "=" * 80)
    print(f"  GNN Comparison v2 — Final Results")
    print(f"  Test: {len(test_records):,} trajectories  |  "
          f"Encoder: {args.encoder.upper()}  |  "
          f"Best hybrid_gnn alpha: {best_alpha:.1f}")
    print("=" * 80 + "\n")
    print_table(summary)
    print_delta(summary)
    print(f"\nFigure → {hist_path}")
    print(f"CSV    → {csv_path}")


if __name__ == "__main__":
    main()
