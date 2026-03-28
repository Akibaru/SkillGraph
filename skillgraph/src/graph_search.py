"""
src/graph_search.py  —  SkillGraph Tool Planning
=================================================
Four graph-based algorithms for tool sequence planning given a task description.

Algorithms
----------
  dijkstra      Semantic-Guided Dijkstra (cost = 1/edge_weight + relevance penalty)
  beam          Probabilistic Beam Search (transition_prob + semantic + community diversity)
  hierarchical  Community-Hierarchical Search (single best community + position ordering)
  hybrid        Hybrid Semantic-Graph (semantic candidates + graph-guided sequencing)

Usage
-----
  python src/graph_search.py --query "Find weather in Tokyo"
  python src/graph_search.py --query "..." --method hybrid --max-steps 6
  python src/graph_search.py --smoke-test
  python src/graph_search.py --tune --val-n 500
"""

from __future__ import annotations

import argparse
import heapq
import json
import pickle
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT / "data" / "processed"
GRAPH_FILE = PROC_DIR / "final_graph.gpickle"
COMM_FILE  = PROC_DIR / "communities.json"
META_FILE  = PROC_DIR / "meta_graph.gpickle"
EMB_FILE   = PROC_DIR / "tool_embeddings.npy"
TOOL_META  = PROC_DIR / "tool_metadata.json"
HYPERPARAM_FILE = PROC_DIR / "best_hyperparams.json"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _lcs_length(a: list, b: list) -> int:
    """Longest Common Subsequence length (O(N) space DP)."""
    m, n = len(a), len(b)
    if not m or not n:
        return 0
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0
        for j in range(n):
            tmp      = dp[j + 1]
            dp[j + 1] = prev + 1 if a[i] == b[j] else max(dp[j + 1], dp[j])
            prev     = tmp
    return dp[n]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolPlan:
    tools:       list[str]
    scores:      list[float]          # per-step relevance score
    total_score: float
    path_length: int
    method:      str
    entry_tool:  str = ""

    def display(self, width: int = 78) -> str:
        lines = [
            f"  Method: {self.method:<14}  score={self.total_score:.4f}  steps={self.path_length}",
            f"  Entry : {self.entry_tool}",
        ]
        for i, (t, s) in enumerate(zip(self.tools, self.scores)):
            clean = t.rsplit("_for_", 1)[0].replace("_", " ")
            cat   = t.rsplit("_for_", 1)[1].replace("_", " ") if "_for_" in t else ""
            lines.append(f"  {i+1:>2}. [{s:.3f}] {clean}  ({cat})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ToolSequencePlanner
# ---------------------------------------------------------------------------

class ToolSequencePlanner:
    """
    Load the skill graph, embeddings, and communities once at init.
    All planning methods run entirely in-memory.
    """

    # Default trajectory stats (overridden after construction from training data)
    _avg_traj_len:        float = 4.2
    _median_traj_len:     float = 3.0
    _tool_position_stats: dict  = {}   # {tool_name: avg_normalised_position}

    def __init__(self) -> None:
        t0 = time.time()
        self._load_artefacts()
        self._precompute_community_embeddings()
        elapsed = time.time() - t0
        print(f"[ToolSequencePlanner] Ready in {elapsed:.1f}s  "
              f"({self.n_tools} tools, {self.G.number_of_edges()} edges, "
              f"{self.n_communities} communities)")

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_artefacts(self) -> None:
        for p in (GRAPH_FILE, COMM_FILE, META_FILE, EMB_FILE, TOOL_META):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Run graph_build.py first."
                )

        with GRAPH_FILE.open("rb") as fh:
            self.G = pickle.load(fh)

        with META_FILE.open("rb") as fh:
            self.MG = pickle.load(fh)

        self.communities: dict[str, int] = json.loads(
            COMM_FILE.read_text(encoding="utf-8")
        )
        self.tool_meta: dict[str, dict] = json.loads(
            TOOL_META.read_text(encoding="utf-8")
        )

        # Embeddings: row i  ↔  sorted(G.nodes())[i]  (same order as graph_build.py)
        self._active_tools: list[str] = sorted(self.G.nodes())
        self._tool_to_row: dict[str, int] = {
            name: i for i, name in enumerate(self._active_tools)
        }
        self._embeddings: np.ndarray = np.load(str(EMB_FILE))   # (n_tools, 384), L2-norm

        self.n_tools        = len(self._active_tools)
        self.n_communities  = len(set(self.communities.values()))

        # Community → list[tool_name]
        self._comm_to_tools: dict[int, list[str]] = defaultdict(list)
        for tool, cid in self.communities.items():
            if tool in self._tool_to_row:
                self._comm_to_tools[cid].append(tool)

        # Pre-build adjacency for O(1) neighbour look-up
        # {tool: [(neighbour, weight, transition_prob), ...]} sorted by weight desc
        self._adj: dict[str, list[tuple[str, float, float]]] = {}
        for node in self.G.nodes():
            nbrs = sorted(
                [
                    (v, d["weight"], d.get("transition_prob", 0.0))
                    for v, d in self.G[node].items()
                ],
                key=lambda x: -x[1],
            )
            self._adj[node] = nbrs

    def _precompute_community_embeddings(self) -> None:
        """
        IDF-weighted mean-pool tool embeddings within each community.

        idf(cat) = log(n_communities / (1 + df(cat)))
          where df(cat) = number of communities containing ≥1 tool of that category.
        """
        cat_df: Counter = Counter()
        for cid, tools in self._comm_to_tools.items():
            cats_in_comm = {
                t.rsplit("_for_", 1)[1] if "_for_" in t else "?"
                for t in tools
            }
            cat_df.update(cats_in_comm)

        n_comm = max(1, len(self._comm_to_tools))

        def idf(cat: str) -> float:
            return float(np.log(n_comm / (1.0 + cat_df.get(cat, 0))))

        self._comm_embeddings: dict[int, np.ndarray] = {}
        for cid, tools in self._comm_to_tools.items():
            rows: list[int] = []
            weights: list[float] = []
            for t in tools:
                if t not in self._tool_to_row:
                    continue
                cat = t.rsplit("_for_", 1)[1] if "_for_" in t else "?"
                rows.append(self._tool_to_row[t])
                weights.append(max(0.01, idf(cat)))

            if not rows:
                continue

            w    = np.array(weights, dtype=np.float32)
            w   /= w.sum()
            vecs = self._embeddings[rows]
            vec  = (vecs * w[:, None]).sum(axis=0)
            norm = np.linalg.norm(vec)
            self._comm_embeddings[cid] = vec / norm if norm > 1e-9 else vec

    # ── Embedding helpers ────────────────────────────────────────────────────

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode one query string; returns L2-normalised (384,) vector."""
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, "_st_model"):
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = self._st_model.encode([text], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def _tool_sim(self, tool: str, query_vec: np.ndarray) -> float:
        """Cosine similarity between a tool's embedding and a query vector."""
        row = self._tool_to_row.get(tool)
        if row is None:
            return 0.0
        return float(self._embeddings[row] @ query_vec)

    def _top_entry_tools(
        self, query_vec: np.ndarray, k: int = 5
    ) -> list[tuple[str, float]]:
        """Top-k tools by cosine similarity to query vector."""
        sims    = self._embeddings @ query_vec              # (n_tools,)
        top_idx = np.argsort(sims)[-k:][::-1]
        return [(self._active_tools[int(i)], float(sims[i])) for i in top_idx]

    def _position_score(self, tool: str, position_norm: float) -> float:
        """
        Reward a tool appearing at its typical training position.
        position_norm ∈ [0, 1]; 0 = first step, 1 = last step.
        """
        expected = self._tool_position_stats.get(tool, 0.5)
        return max(0.0, 1.0 - abs(expected - position_norm))

    # ── Algorithm 1: Semantic-Guided Dijkstra ────────────────────────────────

    def _dijkstra(
        self,
        entry_tool:         str,
        query_vec:          np.ndarray,
        max_steps:          int   = 8,
        beta:               float = 0.3,
        eps:                float = 1e-6,
        low_sim_threshold:  float = 0.05,
        low_sim_window:     int   = 3,
        max_expansions:     int   = 1500,
    ) -> Optional[ToolPlan]:
        """
        Modified Dijkstra from entry_tool.

        cost(i→j) = 1/(w_edge + ε) + β*(1 - sim(j, query))
          — used only for heap ORDERING (exploring promising paths first).

        Path SELECTION uses mean(sims) of the best path found at each length.
        Target length = self._avg_traj_len (default 4.2 ≈ 4 steps).
        """
        if entry_tool not in self.G:
            return None

        target_len = max(2, min(max_steps, int(round(self._avg_traj_len))))

        entry_sim = self._tool_sim(entry_tool, query_vec)
        heap      = [(0.0, (entry_tool,), (entry_sim,))]
        visited   = set()

        # paths_by_length[L] = (best_mean_sim, path_tuple, sims_tuple)
        paths_by_length: dict[int, tuple[float, tuple, tuple]] = {}
        n_expanded = 0

        while heap and n_expanded < max_expansions:
            cum_cost, path, sims = heapq.heappop(heap)

            current = path[-1]
            if current in visited:
                continue
            visited.add(current)
            n_expanded += 1

            path_len   = len(path)
            path_score = float(np.mean(sims)) if sims else 0.0
            if path_len not in paths_by_length or path_score > paths_by_length[path_len][0]:
                paths_by_length[path_len] = (path_score, path, sims)

            if path_len >= max_steps:
                continue

            if len(sims) >= low_sim_window:
                if all(s < low_sim_threshold for s in sims[-low_sim_window:]):
                    continue

            for nbr, w_edge, _ in self._adj.get(current, []):
                if nbr in visited:
                    continue
                nbr_sim   = self._tool_sim(nbr, query_vec)
                edge_cost = 1.0 / (w_edge + eps)
                rel_cost  = beta * (1.0 - nbr_sim)
                new_cost  = cum_cost + edge_cost + rel_cost
                heapq.heappush(heap, (new_cost, path + (nbr,), sims + (nbr_sim,)))

        if not paths_by_length:
            return None

        for candidate_len in [target_len,
                               target_len + 1, target_len - 1,
                               target_len + 2, target_len - 2,
                               target_len + 3]:
            if candidate_len in paths_by_length and candidate_len >= 1:
                _, path, sims = paths_by_length[candidate_len]
                path_list = list(path)
                sims_list = list(sims)
                return ToolPlan(
                    tools=path_list, scores=sims_list,
                    total_score=float(np.mean(sims_list)),
                    path_length=len(path_list), method="dijkstra",
                    entry_tool=entry_tool,
                )

        best_len = max(paths_by_length, key=lambda l: paths_by_length[l][0])
        _, path, sims = paths_by_length[best_len]
        path_list, sims_list = list(path), list(sims)
        return ToolPlan(
            tools=path_list, scores=sims_list,
            total_score=float(np.mean(sims_list)),
            path_length=len(path_list), method="dijkstra",
            entry_tool=entry_tool,
        )

    def dijkstra(
        self,
        task_description: str,
        max_steps:  int   = 8,
        n_entries:  int   = 5,
        beta:       float = 0.3,
        verbose:    bool  = False,
    ) -> ToolPlan:
        """Run Semantic-Guided Dijkstra; return best plan across all entry tools."""
        query_vec   = self._encode_query(task_description)
        entry_tools = self._top_entry_tools(query_vec, k=n_entries)

        plans: list[ToolPlan] = []
        for entry, _ in entry_tools:
            plan = self._dijkstra(entry, query_vec, max_steps=max_steps, beta=beta)
            if plan:
                plans.append(plan)

        best_plan = max(plans, key=lambda p: p.total_score) if plans else None

        if verbose and plans:
            avg_len = sum(p.path_length for p in plans) / len(plans)
            print(f"  [dijkstra debug] avg path length across entry tools: {avg_len:.2f}")

        return best_plan or _fallback_plan(task_description, entry_tools, "dijkstra")

    # ── Algorithm 2: Probabilistic Beam Search ───────────────────────────────

    def beam_search(
        self,
        task_description: str,
        max_steps:     int   = 8,
        beam_width:    int   = 5,
        n_entries:     int   = 5,
        w1:            float = 0.4,    # transition_prob weight
        w2:            float = 0.4,    # semantic similarity weight
        w3:            float = 0.2,    # community-diversity weight
        score_floor:   float = 0.05,
        target_length: Optional[int] = None,
    ) -> list[ToolPlan]:
        """
        Probabilistic Beam Search with target-length control.

        target_length controls when to stop expansion:
          - None → use self._median_traj_len (set from training data)
          - int  → explicit target

        All returned paths are truncated to target_length.
        Returns top-3 plans ranked by mean per-step score.
        """
        query_vec   = self._encode_query(task_description)
        entry_tools = self._top_entry_tools(query_vec, k=n_entries)

        tgt = target_length if target_length is not None else max(2, int(round(self._median_traj_len)))

        BeamState = tuple   # (neg_cum_score, path, visited_comms, step_scores)

        initial_beams: list[BeamState] = []
        for entry, entry_sim in entry_tools:
            if entry not in self.G:
                continue
            cid = self.communities.get(entry, -1)
            initial_beams.append((
                -entry_sim,
                [entry],
                frozenset([cid]),
                [entry_sim],
            ))

        if not initial_beams:
            return [_fallback_plan(task_description, entry_tools, "beam")]

        initial_beams.sort(key=lambda x: x[0])
        beams: list[BeamState] = initial_beams[:beam_width]
        completed: list[ToolPlan] = []

        for _step in range(max_steps - 1):
            if not beams:
                break
            candidates: list[BeamState] = []

            for neg_score, path, visited_comms, step_scores in beams:
                # Target length reached: complete this beam
                if len(path) >= tgt:
                    completed.append(ToolPlan(
                        tools=path, scores=step_scores,
                        total_score=(-neg_score) / len(path),
                        path_length=len(path), method="beam",
                        entry_tool=path[0],
                    ))
                    continue

                current   = path[-1]
                cum_score = -neg_score

                all_low        = True
                best_step_score = 0.0

                for nbr, w_edge, tp in self._adj.get(current, [])[:30]:
                    if nbr in path:
                        continue
                    nbr_sim = self._tool_sim(nbr, query_vec)
                    nbr_cid = self.communities.get(nbr, -1)

                    same_comm = sum(1 for t in path if self.communities.get(t) == nbr_cid)
                    overlap   = same_comm / len(path)
                    diversity = 1.0 - overlap

                    step_score = w1 * tp + w2 * nbr_sim + w3 * diversity
                    if step_score >= score_floor:
                        all_low = False
                    if step_score > best_step_score:
                        best_step_score = step_score

                    new_cum   = cum_score + step_score
                    new_comms = visited_comms | {nbr_cid}
                    candidates.append((
                        -new_cum,
                        path + [nbr],
                        new_comms,
                        step_scores + [step_score],
                    ))

                # Diminishing returns or exhausted neighbors: complete beam
                if all_low or not self._adj.get(current) or best_step_score < 0.01:
                    completed.append(ToolPlan(
                        tools=path, scores=step_scores,
                        total_score=cum_score / len(path),
                        path_length=len(path), method="beam",
                        entry_tool=path[0],
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
                path_length=len(path), method="beam",
                entry_tool=path[0],
            ))

        if not completed:
            return [_fallback_plan(task_description, entry_tools, "beam")]

        # Truncate all plans to target_length
        final: list[ToolPlan] = []
        for p in completed:
            if len(p.tools) > tgt:
                tools_t  = p.tools[:tgt]
                scores_t = p.scores[:tgt]
                final.append(ToolPlan(
                    tools=tools_t, scores=scores_t,
                    total_score=float(np.mean(scores_t)) if scores_t else 0.0,
                    path_length=len(tools_t), method="beam",
                    entry_tool=tools_t[0] if tools_t else "",
                ))
            else:
                final.append(p)

        final.sort(key=lambda p: -p.total_score)
        return final[:3]

    # ── Algorithm 3: Community-Hierarchical Search ───────────────────────────

    def hierarchical(
        self,
        task_description: str,
        max_steps: int = 8,
    ) -> ToolPlan:
        """
        Community-Hierarchical Search (revised).

        Steps:
          1. Find the single most relevant community (IDF-weighted embedding similarity)
          2. Rank tools within that community by cosine similarity to task
          3. Take top target_length tools; if the community is too small, expand to
             neighboring communities (via meta-graph, fallback: next-best by similarity)
          4. Order selected tools using avg training position stats (topological sort
             on induced subgraph if acyclic; otherwise by position stats)
        """
        import networkx as nx

        query_vec = self._encode_query(task_description)
        target = max(2, min(max_steps, int(round(self._median_traj_len))))

        # ── Step 1: Find best community ───────────────────────────────────────
        comm_ids    = list(self._comm_embeddings.keys())
        comm_vecs   = np.stack([self._comm_embeddings[c] for c in comm_ids])
        comm_sims_a = (comm_vecs @ query_vec).astype(float)        # (n_comms,)
        best_idx    = int(np.argmax(comm_sims_a))
        best_cid    = comm_ids[best_idx]
        comm_sim_dict = {comm_ids[i]: float(comm_sims_a[i]) for i in range(len(comm_ids))}

        # ── Step 2 + 3: Collect tools with expansion ─────────────────────────
        selected_tools: list[tuple[str, float]] = []
        seen_tools: set[str] = set()

        def add_from_community(cid: int, budget: int) -> None:
            members = self._comm_to_tools.get(cid, [])
            rows    = [self._tool_to_row[t] for t in members if t in self._tool_to_row]
            if not rows:
                return
            sims  = self._embeddings[rows] @ query_vec
            order = np.argsort(sims)[::-1]
            added = 0
            for i in order:
                t = members[i]
                if t not in seen_tools:
                    seen_tools.add(t)
                    selected_tools.append((t, float(sims[i])))
                    added += 1
                if added >= budget:
                    break

        add_from_community(best_cid, target)

        if len(selected_tools) < target:
            # Try meta-graph neighbors first
            neighbor_comms: list[int] = []
            try:
                if best_cid in self.MG:
                    neighbor_comms = list(self.MG.neighbors(best_cid))
                    try:
                        neighbor_comms += list(self.MG.predecessors(best_cid))
                    except AttributeError:
                        pass
                    neighbor_comms = list(set(neighbor_comms) - {best_cid})
                    neighbor_comms.sort(
                        key=lambda c: comm_sim_dict.get(c, 0.0), reverse=True
                    )
            except Exception:
                pass

            # Fallback: all communities sorted by similarity
            if not neighbor_comms:
                sorted_idx     = np.argsort(comm_sims_a)[::-1]
                neighbor_comms = [comm_ids[i] for i in sorted_idx
                                  if comm_ids[i] != best_cid]

            for ncid in neighbor_comms:
                if len(selected_tools) >= target:
                    break
                add_from_community(ncid, target - len(selected_tools))

        # Last-resort fallback
        if len(selected_tools) < 2:
            for t, s in self._top_entry_tools(query_vec, k=target):
                if t not in seen_tools:
                    seen_tools.add(t)
                    selected_tools.append((t, s))
                if len(selected_tools) >= target:
                    break

        # ── Step 4: Order tools ───────────────────────────────────────────────
        sel_tools    = [t for t, _ in selected_tools[:target]]
        sel_sim_dict = {t: s for t, s in selected_tools[:target]}

        def position_key(t: str) -> float:
            return self._tool_position_stats.get(t, 0.5)

        ordered: list[str] = []
        if len(sel_tools) >= 2:
            try:
                sg = self.G.subgraph(sel_tools)
                if nx.is_directed_acyclic_graph(sg):
                    topo = list(nx.topological_sort(sg))
                    # Re-rank within topological groups by position stats
                    ordered = sorted(topo, key=position_key)
                else:
                    ordered = sorted(sel_tools, key=position_key)
            except Exception:
                ordered = sorted(sel_tools, key=position_key)
        else:
            ordered = sel_tools

        scores = [sel_sim_dict.get(t, 0.0) for t in ordered]
        total  = float(np.mean(scores)) if scores else 0.0

        return ToolPlan(
            tools=ordered, scores=scores,
            total_score=total, path_length=len(ordered),
            method="hierarchical",
            entry_tool=f"comm#{best_cid}",
        )

    # ── Algorithm 4: Hybrid Semantic-Graph Planning ──────────────────────────

    def hybrid(
        self,
        task_description: str,
        max_steps:    int   = 8,
        k_multiplier: int   = 3,     # K = k_multiplier * target  (candidate pool)
        alpha:        float = 0.5,   # edge_weight vs semantic balance
        gamma:        float = 0.1,   # position-bonus weight
        target_length: Optional[int] = None,
    ) -> ToolPlan:
        """
        Hybrid Semantic-Graph Planning.

        Step 1 — Semantic Candidate Selection:
            Encode task, retrieve top-K tools by cosine similarity
            (K = k_multiplier * target_length).

        Step 2 — Graph-Guided Subgraph + Bridges:
            Build induced subgraph from candidates.
            Connect disconnected components by adding minimum bridge tools
            (shortest path in full graph).

        Step 3 — Greedy Sequencing:
            Start from the highest-similarity candidate, then at each step:
              next = argmax( alpha * edge_weight(curr, next)
                           + (1-alpha) * cosine_sim(next, task)
                           + gamma * position_bonus(next) )

        Step 4 — Length Calibration:
            Pad with highest-scoring unused candidates (if short) or truncate.
        """
        import networkx as nx

        query_vec = self._encode_query(task_description)
        target    = target_length if target_length is not None else max(
            2, min(max_steps, int(round(self._median_traj_len)))
        )

        # ── Step 1: Semantic candidates ───────────────────────────────────────
        K = max(target + 2, k_multiplier * target)
        K = min(K, self.n_tools)

        candidates_list = self._top_entry_tools(query_vec, k=K)
        candidate_set: set[str]    = {t for t, _ in candidates_list}
        candidate_sims: dict[str, float] = {t: s for t, s in candidates_list}

        # ── Step 2: Induced subgraph + bridges ───────────────────────────────
        sg = self.G.subgraph(candidate_set).copy()

        try:
            components = list(nx.weakly_connected_components(sg))
        except Exception:
            components = [candidate_set]   # treat as fully connected

        if len(components) > 1:
            bridge_tools: set[str] = set()
            for ci in range(len(components) - 1):
                comp_a = sorted(
                    components[ci],
                    key=lambda t: candidate_sims.get(t, 0.0), reverse=True
                )
                comp_b = sorted(
                    components[ci + 1],
                    key=lambda t: candidate_sims.get(t, 0.0), reverse=True
                )
                connected = False
                for ta in comp_a[:3]:
                    for tb in comp_b[:3]:
                        try:
                            sp = nx.shortest_path(self.G, ta, tb, weight=None)
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
                candidate_sims[bt] = self._tool_sim(bt, query_vec)

            sg = self.G.subgraph(candidate_set).copy()

        # ── Step 3: Greedy sequencing ─────────────────────────────────────────
        start_tool = max(candidate_set, key=lambda t: candidate_sims.get(t, 0.0))

        sequence:    list[str]   = [start_tool]
        step_scores: list[float] = [candidate_sims.get(start_tool, 0.0)]
        remaining = candidate_set - {start_tool}

        for step_idx in range(1, target):
            if not remaining:
                break
            current  = sequence[-1]
            pos_norm = step_idx / max(1, target - 1)

            best_score = -float("inf")
            best_tool  = None

            for nbr in remaining:
                # Edge weight (both directions; reverse edge gets half weight)
                if sg.has_edge(current, nbr):
                    edge_w = float(sg[current][nbr].get("weight", 0.0))
                elif sg.has_edge(nbr, current):
                    edge_w = float(sg[nbr][current].get("weight", 0.0)) * 0.5
                else:
                    edge_w = 0.0

                sem_s     = candidate_sims.get(nbr, 0.0)
                pos_bonus = self._position_score(nbr, pos_norm)
                score     = alpha * edge_w + (1.0 - alpha) * sem_s + gamma * pos_bonus

                if score > best_score:
                    best_score = score
                    best_tool  = nbr

            if best_tool is None:
                break
            sequence.append(best_tool)
            step_scores.append(best_score)
            remaining.remove(best_tool)

        # ── Step 4: Length calibration ─────────────────────────────────────────
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

        total = float(np.mean(step_scores)) if step_scores else 0.0
        return ToolPlan(
            tools=sequence, scores=step_scores,
            total_score=total, path_length=len(sequence),
            method="hybrid",
            entry_tool=start_tool,
        )

    # ── Unified entry point ───────────────────────────────────────────────────

    def plan(
        self,
        task_description: str,
        method:    str = "dijkstra",
        max_steps: int = 8,
    ) -> list[ToolPlan]:
        """Returns a list of ToolPlan objects (beam returns ≤3; others return 1)."""
        method = method.lower()
        if method == "dijkstra":
            return [self.dijkstra(task_description, max_steps=max_steps)]
        elif method == "beam":
            return self.beam_search(task_description, max_steps=max_steps)
        elif method in ("hierarchical", "hier"):
            return [self.hierarchical(task_description, max_steps=max_steps)]
        elif method == "hybrid":
            return [self.hybrid(task_description, max_steps=max_steps)]
        else:
            raise ValueError(
                f"Unknown method: {method!r}. "
                f"Choose dijkstra / beam / hierarchical / hybrid"
            )

    # ── Hyperparameter tuning ─────────────────────────────────────────────────

    def tune_hyperparameters(
        self,
        val_trajectories: list[dict],
        n_trials:    int          = 100,
        output_path: Optional[Path] = None,
    ) -> dict:
        """
        Random-search hyperparameters on a held-out validation set.

        Objective: F1@K + 0.5 * LCS-R + 0.3 * ordered_precision

        Dijkstra: beta ∈ [0.1,0.2,0.3,0.4,0.5]  ×  n_entries ∈ [3,5,8,10]
        Beam    : w1 ∈ [0.2,0.3,0.4,0.5], w2 ∈ [0.3,0.4,0.5,0.6], w3≥0.05,
                  beam_width ∈ [3,5,8,10]
        Hybrid  : k_multiplier ∈ [2,3,4], alpha ∈ [0.3,0.4,0.5,0.6,0.7],
                  gamma ∈ [0.0,0.05,0.1,0.15,0.2]

        Randomly samples up to n_trials combos per method if grid is larger.
        Saves to data/processed/best_hyperparams.json.
        """
        from tqdm import tqdm as _tqdm

        if output_path is None:
            output_path = HYPERPARAM_FILE

        # ── Batch-encode val queries ─────────────────────────────────────────
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, "_st_model"):
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")

        print(f"\n[tune] Encoding {len(val_trajectories)} val queries …")
        val_queries = [r["task_description"] for r in val_trajectories]
        val_vecs    = self._st_model.encode(
            val_queries, normalize_embeddings=True,
            batch_size=256, show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

        rng = np.random.default_rng(42)

        def _objective(predicted: list[str], ground_truth: list[str]) -> float:
            """F1@K + 0.5 * LCS-R + 0.3 * ordered_precision."""
            K = len(ground_truth)
            if K == 0:
                return 0.0
            gt_set = set(ground_truth)
            hits   = len(set(predicted[:K]) & gt_set)
            f1     = hits / K
            lcs_r  = _lcs_length(predicted, ground_truth) / K

            gt_rank = {t: i for i, t in enumerate(ground_truth)}
            common  = [(t, pos) for pos, t in enumerate(predicted) if t in gt_set]
            if len(common) >= 2:
                n_common    = len(common)
                total_pairs = n_common * (n_common - 1) / 2
                matching    = sum(
                    1
                    for i in range(n_common)
                    for j in range(i + 1, n_common)
                    if gt_rank[common[i][0]] < gt_rank[common[j][0]]
                )
                ord_prec = matching / total_pairs
            else:
                ord_prec = 0.0

            return f1 + 0.5 * lcs_r + 0.3 * ord_prec

        # ── Dijkstra sweep ───────────────────────────────────────────────────
        beta_values    = [0.1, 0.2, 0.3, 0.4, 0.5]
        n_entry_values = [3, 5, 8, 10]
        dij_combos: list[tuple[float, int]] = [
            (b, n) for b in beta_values for n in n_entry_values
        ]
        if len(dij_combos) > n_trials:
            idx       = rng.choice(len(dij_combos), size=n_trials, replace=False)
            dij_combos = [dij_combos[int(i)] for i in idx]

        best_dij_score    = -1.0
        best_dij_beta     = 0.3
        best_dij_n_entries = 5

        print(f"\n[tune] Dijkstra: {len(dij_combos)} combos (beta × n_entries) …")
        for beta, n_ent in _tqdm(dij_combos, desc="Dijkstra"):
            scores = []
            for i, rec in enumerate(val_trajectories):
                vec = val_vecs[i]
                gt  = rec["tool_sequence"]
                entry_tools = self._top_entry_tools(vec, k=n_ent)
                plans: list[ToolPlan] = []
                for entry, _ in entry_tools:
                    p = self._dijkstra(entry, vec, max_steps=8, beta=beta)
                    if p:
                        plans.append(p)
                predicted = max(plans, key=lambda p: p.total_score).tools if plans else []
                scores.append(_objective(predicted, gt))
            avg = float(np.mean(scores))
            if avg > best_dij_score:
                best_dij_score     = avg
                best_dij_beta      = beta
                best_dij_n_entries = n_ent

        print(f"  Best Dijkstra beta={best_dij_beta}  n_entries={best_dij_n_entries}  "
              f"(val={best_dij_score:.4f})")

        # ── Beam sweep ───────────────────────────────────────────────────────
        beam_combos: list[tuple[float, float, float, int]] = []
        for _w1 in [0.2, 0.3, 0.4, 0.5]:
            for _w2 in [0.3, 0.4, 0.5, 0.6]:
                _w3 = round(1.0 - _w1 - _w2, 3)
                if _w3 >= 0.05:
                    for bw in [3, 5, 8, 10]:
                        beam_combos.append((_w1, _w2, _w3, bw))

        if len(beam_combos) > n_trials:
            idx         = rng.choice(len(beam_combos), size=n_trials, replace=False)
            beam_combos = [beam_combos[int(i)] for i in idx]

        best_beam_score  = -1.0
        best_beam_params = {"w1": 0.4, "w2": 0.4, "w3": 0.2, "beam_width": 5}

        print(f"\n[tune] Beam: {len(beam_combos)} combos …")
        for w1, w2, w3, bw in _tqdm(beam_combos, desc="Beam"):
            scores = []
            for i, rec in enumerate(val_trajectories):
                vec = val_vecs[i]
                gt  = rec["tool_sequence"]
                self._encode_query = lambda text, _v=vec: _v   # type: ignore
                try:
                    plans = self.beam_search(
                        "__precomputed__", max_steps=8, beam_width=bw,
                        n_entries=5, w1=w1, w2=w2, w3=w3,
                    )
                finally:
                    del self._encode_query
                predicted = plans[0].tools if plans else []
                scores.append(_objective(predicted, gt))
            avg = float(np.mean(scores))
            if avg > best_beam_score:
                best_beam_score  = avg
                best_beam_params = {"w1": w1, "w2": w2, "w3": w3, "beam_width": bw}

        print(f"  Best Beam w1={best_beam_params['w1']}/w2={best_beam_params['w2']}/"
              f"w3={best_beam_params['w3']} width={best_beam_params['beam_width']}  "
              f"(val={best_beam_score:.4f})")

        # ── Hybrid sweep ─────────────────────────────────────────────────────
        hybrid_combos: list[tuple[int, float, float]] = [
            (km, al, ga)
            for km in [2, 3, 4]
            for al in [0.3, 0.4, 0.5, 0.6, 0.7]
            for ga in [0.0, 0.05, 0.1, 0.15, 0.2]
        ]
        if len(hybrid_combos) > n_trials:
            idx           = rng.choice(len(hybrid_combos), size=n_trials, replace=False)
            hybrid_combos = [hybrid_combos[int(i)] for i in idx]

        best_hyb_score  = -1.0
        best_hyb_params = {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}

        print(f"\n[tune] Hybrid: {len(hybrid_combos)} combos …")
        for km, al, ga in _tqdm(hybrid_combos, desc="Hybrid"):
            scores = []
            for i, rec in enumerate(val_trajectories):
                vec = val_vecs[i]
                gt  = rec["tool_sequence"]
                self._encode_query = lambda text, _v=vec: _v   # type: ignore
                try:
                    plan = self.hybrid(
                        "__precomputed__", max_steps=8,
                        k_multiplier=km, alpha=al, gamma=ga,
                    )
                finally:
                    del self._encode_query
                predicted = plan.tools if plan else []
                scores.append(_objective(predicted, gt))
            avg = float(np.mean(scores))
            if avg > best_hyb_score:
                best_hyb_score  = avg
                best_hyb_params = {"k_multiplier": km, "alpha": al, "gamma": ga}

        print(f"  Best Hybrid k_mult={best_hyb_params['k_multiplier']} "
              f"alpha={best_hyb_params['alpha']} gamma={best_hyb_params['gamma']}  "
              f"(val={best_hyb_score:.4f})")

        # ── Save ─────────────────────────────────────────────────────────────
        result = {
            "dijkstra": {
                "beta":      best_dij_beta,
                "n_entries": best_dij_n_entries,
                "val_score": round(best_dij_score, 4),
            },
            "beam": {
                **best_beam_params,
                "val_score": round(best_beam_score, 4),
            },
            "hybrid": {
                **best_hyb_params,
                "val_score": round(best_hyb_score, 4),
            },
        }
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  Saved best hyperparameters → {output_path}")
        return result

    def load_best_hyperparams(self) -> dict:
        """Load previously tuned hyperparameters (if they exist)."""
        if HYPERPARAM_FILE.exists():
            return json.loads(HYPERPARAM_FILE.read_text(encoding="utf-8"))
        return {}


# ---------------------------------------------------------------------------
# Fallback plan (disconnected / no valid graph path)
# ---------------------------------------------------------------------------

def _fallback_plan(
    task_description: str,
    entry_tools: list[tuple[str, float]],
    method: str,
) -> ToolPlan:
    tools  = [t for t, _ in entry_tools]
    scores = [s for _, s in entry_tools]
    return ToolPlan(
        tools=tools, scores=scores,
        total_score=float(np.mean(scores)) if scores else 0.0,
        path_length=len(tools), method=f"{method}(fallback)",
        entry_tool=tools[0] if tools else "?",
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_header(title: str, width: int = 78) -> None:
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def run_single(
    planner:   ToolSequencePlanner,
    query:     str,
    method:    str,
    max_steps: int,
) -> None:
    _print_header(f"Query: {query[:70]}")
    t0      = time.time()
    plans   = planner.plan(query, method=method, max_steps=max_steps)
    elapsed = time.time() - t0
    print(f"  [{method}]  {elapsed*1000:.0f} ms\n")
    for i, p in enumerate(plans, 1):
        if len(plans) > 1:
            print(f"  — Plan {i} —")
        print(p.display())
        print()


def smoke_test(planner: ToolSequencePlanner) -> None:
    queries = [
        "Find the current weather in Tokyo and convert the temperature to Fahrenheit",
        "Search for recent AI papers and summarize the top 3 results",
        "Download a CSV file, calculate the average of column B, and plot a bar chart",
        "Translate this English text to French and then to German",
        "Find restaurants near me, check their ratings, and make a reservation",
    ]
    methods = ["dijkstra", "beam", "hierarchical", "hybrid"]

    path_lens: dict[str, list[int]] = {m: [] for m in methods}

    for query in queries:
        _print_header(f"QUERY: {query}", width=90)
        for method in methods:
            t0      = time.time()
            plans   = planner.plan(query, method=method, max_steps=6)
            elapsed = time.time() - t0

            for p in plans:
                path_lens[method].append(p.path_length)

            print(f"\n  ┌─ {method.upper()} ({elapsed*1000:.0f} ms)")
            for rank, plan in enumerate(plans, 1):
                prefix    = f"  │  {'Plan ' + str(rank) + ': ' if len(plans) > 1 else ''}"
                tools_str = " → ".join(
                    t.rsplit("_for_", 1)[0].replace("_", " ")[:20]
                    for t in plan.tools
                )
                print(f"{prefix}[score={plan.total_score:.3f}  len={plan.path_length}] "
                      f"{tools_str}")
                if rank == 1:
                    for step, (t, s) in enumerate(zip(plan.tools, plan.scores), 1):
                        cat = t.rsplit("_for_", 1)[1] if "_for_" in t else "?"
                        print(f"  │    {step}. [{s:.3f}] {t}  [{cat}]")
            print(f"  └{'─'*60}")

    print("\n" + "═" * 90)
    print("  Average path lengths:")
    for m, lens in path_lens.items():
        if lens:
            print(f"    {m:15s}: {sum(lens)/len(lens):.2f}")
    print("  Smoke test complete.")
    print("═" * 90)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkillGraph Tool Sequence Planner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query",      type=str,  default=None)
    parser.add_argument("--method",     type=str,  default="dijkstra",
                        choices=["dijkstra", "beam", "hierarchical", "hybrid", "all"])
    parser.add_argument("--max-steps",  type=int,  default=8)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--tune",       action="store_true",
                        help="Run hyperparameter tuning on training val split")
    parser.add_argument("--val-n",      type=int,  default=500,
                        help="Number of training samples to use for tuning")
    args = parser.parse_args()

    planner = ToolSequencePlanner()

    if args.tune:
        traj_file  = PROC_DIR / "successful_trajectories.jsonl"
        split_file = PROC_DIR / "train_test_split.json"
        if not traj_file.exists():
            print("ERROR: successful_trajectories.jsonl not found. Run extract.py first.")
            return

        import json as _json, random, statistics
        from collections import defaultdict as _dd

        all_records = []
        with traj_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_records.append(_json.loads(line))

        if split_file.exists():
            split     = _json.loads(split_file.read_text(encoding="utf-8"))
            train_idx = split["train_idx"]
        else:
            train_idx = list(range(int(len(all_records) * 0.8)))

        # Set trajectory stats from training data
        train_records = [all_records[i] for i in train_idx]
        planner._avg_traj_len    = sum(r["num_steps"] for r in train_records) / len(train_records)
        planner._median_traj_len = float(statistics.median(r["num_steps"] for r in train_records))

        tool_pos_lists: dict[str, list[float]] = _dd(list)
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

        # Use first val_n training records (shuffled) as validation
        _rng = random.Random(42)
        sample_idx = train_idx.copy()
        _rng.shuffle(sample_idx)
        val_records = [all_records[i] for i in sample_idx[:args.val_n]]

        result = planner.tune_hyperparameters(val_records, n_trials=100)
        print("\nTuning complete. Best hyperparameters:")
        print(json.dumps(result, indent=2))

    elif args.smoke_test:
        smoke_test(planner)

    elif args.query:
        methods = ["dijkstra", "beam", "hierarchical", "hybrid"] if args.method == "all" \
                  else [args.method]
        for m in methods:
            run_single(planner, args.query, m, args.max_steps)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
