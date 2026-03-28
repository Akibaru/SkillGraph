"""
src/graph_build.py  —  SkillGraph pipeline
==========================================
Phase 1: Statistical co-occurrence graph  (bigrams + PMI weighting)
Phase 2: Semantic-enhanced graph          (embedding fusion + Louvain communities)

Usage
-----
  python src/graph_build.py --phase 1          # statistical graph only
  python src/graph_build.py --phase 2          # semantic fusion (requires phase 1 outputs)
  python src/graph_build.py --phase all        # run both end-to-end
  python src/graph_build.py --phase 2 --alpha 0.6 --topk 15 --embedder openai
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import pickle
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = pathlib.Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT / "data" / "processed"
FIG_DIR   = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Phase 1 inputs / outputs
TRAJ_FILE   = PROC_DIR / "successful_trajectories.jsonl"
META_FILE   = PROC_DIR / "tool_metadata.json"
OUT_ADJ     = PROC_DIR / "adjacency_matrix.npz"
OUT_EDGES   = PROC_DIR / "edge_list.csv"
OUT_INDEX   = PROC_DIR / "tool_index.json"

# Phase 2 outputs
OUT_FINAL_GRAPH    = PROC_DIR / "final_graph.gpickle"
OUT_FINAL_EDGES    = PROC_DIR / "final_edge_list.csv"
OUT_COMMUNITIES    = PROC_DIR / "communities.json"
OUT_META_GRAPH     = PROC_DIR / "meta_graph.gpickle"
OUT_EMBED_CACHE    = PROC_DIR / "tool_embeddings.npy"

# Figures
FIG_PYVIS          = ROOT / "outputs" / "figures" / "skill_graph_full.html"
FIG_META           = FIG_DIR / "skill_graph_meta.png"
FIG_DEGREE         = FIG_DIR / "degree_distribution.png"
FIG_WEIGHT         = FIG_DIR / "weight_distribution.png"

# ---------------------------------------------------------------------------
# Phase 1 hyperparameters
# ---------------------------------------------------------------------------
SKIP_WEIGHT = 0.5
MIN_COUNT   = 3
MIN_WEIGHT  = 0.01


# ============================================================================
#  PHASE 1 — Statistical Graph
# ============================================================================

def load_trajectories(path: pathlib.Path) -> list[list[str]]:
    sequences: list[list[str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                seq = rec.get("tool_sequence", [])
                if seq:
                    sequences.append(seq)
    return sequences


def build_counts(
    sequences: list[list[str]],
    skip_weight: float = SKIP_WEIGHT,
) -> tuple[dict[str, int], dict[tuple[int, int], float]]:
    vocab: dict[str, int] = {}
    for seq in sequences:
        for t in seq:
            if t not in vocab:
                vocab[t] = len(vocab)

    edge_counts: dict[tuple[int, int], float] = defaultdict(float)
    for seq in tqdm(sequences, desc="Building counts", unit="traj"):
        ids = [vocab[t] for t in seq]
        n   = len(ids)
        for pos in range(n):
            i = ids[pos]
            if pos + 1 < n:
                j = ids[pos + 1]
                if i != j:
                    edge_counts[(i, j)] += 1.0
            if pos + 2 < n:
                j2 = ids[pos + 2]
                if i != j2:
                    edge_counts[(i, j2)] += skip_weight

    return vocab, edge_counts


def compute_transition_probs(
    edge_counts: dict[tuple[int, int], float],
) -> dict[int, float]:
    outgoing: dict[int, float] = defaultdict(float)
    for (i, _j), w in edge_counts.items():
        outgoing[i] += w
    return outgoing


def compute_pmi(
    edge_counts: dict[tuple[int, int], float],
    outgoing: dict[int, float],
) -> dict[tuple[int, int], float]:
    Z = sum(edge_counts.values())
    if Z == 0:
        return {}
    incoming: dict[int, float] = defaultdict(float)
    for (_i, j), w in edge_counts.items():
        incoming[j] += w

    pmi_map: dict[tuple[int, int], float] = {}
    for (i, j), w in edge_counts.items():
        p_ij = w / Z
        p_i  = outgoing[i] / Z
        p_j  = incoming[j] / Z
        pmi_map[(i, j)] = math.log(p_ij / (p_i * p_j)) if p_i > 0 and p_j > 0 else 0.0
    return pmi_map


def compute_final_weights(
    edge_counts: dict[tuple[int, int], float],
    outgoing: dict[int, float],
    pmi_map: dict[tuple[int, int], float],
) -> dict[tuple[int, int], dict]:
    result: dict[tuple[int, int], dict] = {}
    for (i, j), count in edge_counts.items():
        tp  = count / outgoing[i] if outgoing[i] > 0 else 0.0
        pmi = pmi_map.get((i, j), 0.0)
        w   = tp * max(0.0, pmi)
        result[(i, j)] = {"weight": w, "count": count, "pmi": pmi, "transition_prob": tp}
    return result


def prune(
    edges: dict[tuple[int, int], dict],
    min_count: float = MIN_COUNT,
    min_weight: float = MIN_WEIGHT,
) -> dict[tuple[int, int], dict]:
    before  = len(edges)
    pruned  = {k: v for k, v in edges.items()
               if v["count"] >= min_count and v["weight"] >= min_weight}
    removed = before - len(pruned)
    print(f"  Pruning: {before:,} → {len(pruned):,} edges  (removed {removed:,})")
    return pruned


def save_phase1(
    edges: dict[tuple[int, int], dict],
    vocab: dict[str, int],
) -> None:
    n           = len(vocab)
    idx_to_name = {v: k for k, v in vocab.items()}

    rows, cols, data = [], [], []
    for (i, j), v in edges.items():
        rows.append(i); cols.append(j); data.append(v["weight"])

    mat = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    sp.save_npz(str(OUT_ADJ), mat)

    with OUT_EDGES.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "target", "weight", "count", "pmi", "transition_prob"])
        for (i, j), v in sorted(edges.items(), key=lambda x: -x[1]["weight"]):
            w.writerow([idx_to_name[i], idx_to_name[j],
                        f"{v['weight']:.6f}", f"{v['count']:.2f}",
                        f"{v['pmi']:.4f}",   f"{v['transition_prob']:.4f}"])

    OUT_INDEX.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Adjacency matrix → {OUT_ADJ}  shape=({n},{n})  nnz={mat.nnz}")
    print(f"  Edge list        → {OUT_EDGES}  ({len(edges):,} edges)")
    print(f"  Tool index       → {OUT_INDEX}  ({n:,} tools)")


def phase1() -> None:
    print("\n" + "=" * 60)
    print("PHASE 1 — Statistical Co-occurrence Graph")
    print("=" * 60)

    sequences = load_trajectories(TRAJ_FILE)
    print(f"  Loaded {len(sequences):,} trajectories")

    vocab, edge_counts = build_counts(sequences)
    print(f"  Vocab: {len(vocab):,} tools  |  Raw edges: {len(edge_counts):,}")

    outgoing = compute_transition_probs(edge_counts)
    pmi_map  = compute_pmi(edge_counts, outgoing)
    edges    = compute_final_weights(edge_counts, outgoing, pmi_map)
    edges    = prune(edges)

    save_phase1(edges, vocab)

    # Quick stats
    active = set(i for i, j in edges) | set(j for i, j in edges)
    print(f"\n  Active nodes: {len(active):,}  |  Edges: {len(edges):,}")
    print("  Phase 1 complete.")


# ============================================================================
#  PHASE 2 — Semantic Fusion + Community Detection
# ============================================================================

# ── 2a. Text encoding ────────────────────────────────────────────────────────

def build_tool_texts(
    active_tools: list[str],
    meta: dict[str, dict],
) -> list[str]:
    """Concatenate name + category + description for each tool."""
    texts = []
    for name in active_tools:
        info = meta.get(name, {})
        cat  = info.get("category", "").replace("_", " ")
        desc = info.get("description", "")
        # Normalise tool name: replace underscores, strip _for_CATEGORY suffix
        clean = name.rsplit("_for_", 1)[0].replace("_", " ")
        text  = f"{clean}. Category: {cat}. {desc}".strip()
        texts.append(text)
    return texts


def embed_sentence_transformers(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print("  Loading all-MiniLM-L6-v2 …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Encoding {len(texts):,} tool descriptions …")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors → dot product == cosine sim
    )
    return embeddings.astype(np.float32)


def embed_openai(texts: list[str]) -> np.ndarray:
    import os
    from openai import OpenAI
    client    = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_id  = "text-embedding-3-small"
    batch_sz  = 100
    vecs: list[list[float]] = []

    for start in tqdm(range(0, len(texts), batch_sz), desc="OpenAI embed"):
        batch    = texts[start: start + batch_sz]
        response = client.embeddings.create(model=model_id, input=batch)
        vecs.extend([d.embedding for d in response.data])

    mat = np.array(vecs, dtype=np.float32)
    # L2-normalise
    norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(1e-9)
    return (mat / norms).astype(np.float32)


def get_embeddings(
    active_tools: list[str],
    meta: dict[str, dict],
    embedder: str = "local",
    cache: bool = True,
) -> np.ndarray:
    if cache and OUT_EMBED_CACHE.exists():
        print(f"  Loading cached embeddings from {OUT_EMBED_CACHE.name} …")
        return np.load(str(OUT_EMBED_CACHE))

    texts      = build_tool_texts(active_tools, meta)
    t0         = time.time()

    if embedder == "openai":
        emb = embed_openai(texts)
    else:
        emb = embed_sentence_transformers(texts)

    print(f"  Embedding done in {time.time() - t0:.1f}s  shape={emb.shape}")
    if cache:
        np.save(str(OUT_EMBED_CACHE), emb)
        print(f"  Embeddings cached → {OUT_EMBED_CACHE}")
    return emb


# ── 2b. Sparse top-K cosine similarity ───────────────────────────────────────

def topk_cosine_sparse(
    embeddings: np.ndarray,
    active_tools: list[str],
    tool_to_local: dict[str, int],
    top_k: int = 20,
    batch_size: int = 512,
) -> dict[tuple[int, int], float]:
    """
    Compute cosine similarity between all pairs, keep only top-K per row.
    Returns {(local_i, local_j): sim}  in global tool indices.

    Embeddings are assumed to be L2-normalised → cosine = dot product.
    """
    n       = len(active_tools)
    sim_map: dict[tuple[int, int], float] = {}

    for start in tqdm(range(0, n, batch_size), desc="Cosine sim (batched)"):
        end      = min(start + batch_size, n)
        block    = embeddings[start:end]           # (B, D)
        # Full dot-product row against all tools
        scores   = block @ embeddings.T            # (B, n)
        # Zero out self-similarity
        for bi in range(end - start):
            scores[bi, start + bi] = -1.0

        # Take top-K indices per row
        k_actual = min(top_k, n - 1)
        topk_idx = np.argpartition(scores, -k_actual, axis=1)[:, -k_actual:]

        for bi in range(end - start):
            gi = start + bi                        # global index in active_tools
            for gj in topk_idx[bi]:
                s = float(scores[bi, gj])
                if s > 0.0:
                    sim_map[(gi, gj)] = s

    return sim_map


# ── 2c. Graph fusion ──────────────────────────────────────────────────────────

def fuse_graphs(
    stat_edges:  dict[tuple[int, int], dict],   # (vocab_i, vocab_j) → attrs
    sim_map:     dict[tuple[int, int], float],  # (local_i, local_j) → cosine sim
    active_tools: list[str],
    vocab:       dict[str, int],
    alpha:       float = 0.7,
    sem_threshold: float = 0.6,
) -> dict[tuple[int, int], dict]:
    """
    Fuse statistical + semantic edges.
    All indices are in the global vocabulary (vocab[tool_name] → int).

    active_tools[local_i]  →  vocab index via vocab[active_tools[local_i]]
    """
    # Map local active-tool index → global vocab index
    local_to_vocab = {li: vocab[name] for li, name in enumerate(active_tools)}

    # Normalise statistical weights to [0, 1]
    stat_vals = np.array([v["weight"] for v in stat_edges.values()], dtype=np.float32)
    w_min, w_max = float(stat_vals.min()), float(stat_vals.max())
    w_range = w_max - w_min if w_max > w_min else 1.0

    def norm_stat(w: float) -> float:
        return (w - w_min) / w_range

    # Build lookup: (vocab_i, vocab_j) → normalised stat weight
    stat_norm: dict[tuple[int, int], float] = {
        k: norm_stat(v["weight"]) for k, v in stat_edges.items()
    }

    # Collect all candidate edges
    candidate_edges: dict[tuple[int, int], dict] = {}

    # -- Existing statistical edges --
    for (vi, vj), attrs in stat_edges.items():
        w_s = stat_norm[(vi, vj)]
        # Semantic component: look up via local indices
        li  = next((l for l, v in local_to_vocab.items() if v == vi), None)
        lj  = next((l for l, v in local_to_vocab.items() if v == vj), None)
        sem = float(sim_map.get((li, lj), 0.0)) if li is not None and lj is not None else 0.0
        w_f = alpha * w_s + (1.0 - alpha) * sem
        candidate_edges[(vi, vj)] = {
            **attrs,
            "w_stat":  float(attrs["weight"]),
            "w_sem":   sem,
            "w_final": w_f,
            "source":  "statistical",
        }

    # -- Zero-shot semantic-only edges (not in statistical graph) --
    zero_shot = 0
    for (li, lj), sem in sim_map.items():
        if sem < sem_threshold:
            continue
        vi = local_to_vocab[li]
        vj = local_to_vocab[lj]
        if (vi, vj) in candidate_edges:
            # Already exists statistically — update semantic component
            old = candidate_edges[(vi, vj)]
            old["w_sem"]   = max(old["w_sem"], sem)
            old["w_final"] = alpha * norm_stat(old["weight"]) + (1.0 - alpha) * old["w_sem"]
        else:
            # Pure semantic edge
            w_f = (1.0 - alpha) * sem
            candidate_edges[(vi, vj)] = {
                "weight": 0.0, "count": 0.0, "pmi": 0.0, "transition_prob": 0.0,
                "w_stat":  0.0,
                "w_sem":   sem,
                "w_final": w_f,
                "source":  "semantic",
            }
            zero_shot += 1

    print(f"  Statistical edges   : {len(stat_edges):,}")
    print(f"  Zero-shot sem edges : {zero_shot:,}")
    print(f"  Total candidate     : {len(candidate_edges):,}")
    return candidate_edges


# ── 2d. NetworkX graph construction ──────────────────────────────────────────

def build_nx_graph(
    edges:       dict[tuple[int, int], dict],
    vocab:       dict[str, int],
    meta:        dict[str, dict],
    freq:        dict[str, int],
) -> "nx.DiGraph":
    import networkx as nx
    idx_to_name = {v: k for k, v in vocab.items()}

    G = nx.DiGraph()

    # Add nodes
    for name, idx in vocab.items():
        # Only add nodes that appear in at least one edge
        G.add_node(
            name,
            idx=idx,
            category=meta.get(name, {}).get("category", ""),
            description=meta.get(name, {}).get("description", ""),
            frequency=freq.get(name, 0),
            label=name,
        )

    # Add edges
    for (vi, vj), attrs in edges.items():
        src = idx_to_name[vi]
        dst = idx_to_name[vj]
        G.add_edge(
            src, dst,
            weight=attrs["w_final"],
            w_stat=attrs["w_stat"],
            w_sem=attrs["w_sem"],
            count=attrs["count"],
            pmi=attrs["pmi"],
            transition_prob=attrs["transition_prob"],
            source=attrs["source"],
        )

    # Remove isolated nodes (no edges)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    print(f"  NetworkX graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ── 2e. Louvain community detection ──────────────────────────────────────────

def detect_communities(G: "nx.DiGraph") -> dict[str, int]:
    """
    Louvain on undirected projection of the DiGraph.
    Returns {tool_name: community_id}.
    """
    import community as community_louvain
    import networkx as nx

    UG = G.to_undirected()
    # Use w_final as weight
    partition = community_louvain.best_partition(UG, weight="weight", random_state=42)
    n_comm = len(set(partition.values()))
    print(f"  Louvain communities detected: {n_comm}")

    # Attach community to graph nodes
    for node, cid in partition.items():
        G.nodes[node]["community"] = cid

    return partition


def build_meta_graph(
    G: "nx.DiGraph",
    partition: dict[str, int],
) -> "nx.Graph":
    """
    Build community-level meta-graph.
    Each node = community_id, edge weight = sum of inter-community edges.
    """
    import networkx as nx

    # Community sizes
    comm_sizes: dict[int, int] = defaultdict(int)
    for cid in partition.values():
        comm_sizes[cid] += 1

    MG = nx.Graph()
    for cid, sz in comm_sizes.items():
        # Gather member names for label
        members = [n for n, c in partition.items() if c == cid]
        # Pick most-frequent tool as label
        label = max(members, key=lambda n: G.nodes[n].get("frequency", 0))
        MG.add_node(cid, size=sz, label=label)

    # Aggregate inter-community edge weights
    inter: dict[tuple[int, int], float] = defaultdict(float)
    for u, v, d in G.edges(data=True):
        cu, cv = partition.get(u, -1), partition.get(v, -1)
        if cu != cv:
            key = tuple(sorted([cu, cv]))
            inter[key] += d.get("weight", 0.0)

    for (ca, cb), w in inter.items():
        MG.add_edge(ca, cb, weight=w)

    print(f"  Meta-graph: {MG.number_of_nodes()} communities, "
          f"{MG.number_of_edges()} inter-community edges")
    return MG


# ── 2f. Save outputs ──────────────────────────────────────────────────────────

def save_phase2(
    G:         "nx.DiGraph",
    MG:        "nx.Graph",
    partition: dict[str, int],
    edges:     dict[tuple[int, int], dict],
    vocab:     dict[str, int],
) -> None:
    import pickle
    idx_to_name = {v: k for k, v in vocab.items()}

    # final_graph.gpickle
    with OUT_FINAL_GRAPH.open("wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Final graph   → {OUT_FINAL_GRAPH}")

    # meta_graph.gpickle
    with OUT_META_GRAPH.open("wb") as fh:
        pickle.dump(MG, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Meta-graph    → {OUT_META_GRAPH}")

    # communities.json
    OUT_COMMUNITIES.write_text(
        json.dumps(partition, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Communities   → {OUT_COMMUNITIES}  ({len(set(partition.values()))} clusters)")

    # final_edge_list.csv
    with OUT_FINAL_EDGES.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "target", "w_final", "w_stat", "w_sem",
                    "count", "pmi", "transition_prob", "edge_source"])
        for (vi, vj), attrs in sorted(edges.items(), key=lambda x: -x[1]["w_final"]):
            w.writerow([
                idx_to_name[vi], idx_to_name[vj],
                f"{attrs['w_final']:.6f}", f"{attrs['w_stat']:.6f}", f"{attrs['w_sem']:.4f}",
                f"{attrs['count']:.2f}", f"{attrs['pmi']:.4f}", f"{attrs['transition_prob']:.4f}",
                attrs["source"],
            ])
    print(f"  Edge list     → {OUT_FINAL_EDGES}  ({len(edges):,} edges)")


# ── 2g. Visualisations ───────────────────────────────────────────────────────

# --- Community colour palette ------------------------------------------------

def _community_colors(n_communities: int) -> list[str]:
    cmap = plt.cm.get_cmap("tab20", min(n_communities, 20))
    colors = [mcolors.to_hex(cmap(i % 20)) for i in range(n_communities)]
    return colors


# --- Pyvis interactive HTML --------------------------------------------------

def make_pyvis(
    G:         "nx.DiGraph",
    partition: dict[str, int],
    max_nodes: int = 600,
) -> None:
    from pyvis.network import Network

    # Sub-select: top nodes by frequency to keep the HTML snappy
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(
            G.nodes(), key=lambda n: G.nodes[n].get("frequency", 0), reverse=True
        )[:max_nodes]
        H = G.subgraph(top_nodes).copy()
        print(f"  Pyvis: showing top {max_nodes} nodes by frequency "
              f"(of {G.number_of_nodes():,} total)")
    else:
        H = G

    n_comm   = len(set(partition.values()))
    palette  = _community_colors(n_comm)
    comm_col = {cid: palette[cid % len(palette)] for cid in range(n_comm)}

    net = Network(
        height="900px", width="100%",
        bgcolor="#0f0f0f", font_color="#e0e0e0",
        directed=True,
        notebook=False,
    )
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.003,
          "springLength": 120,
          "springConstant": 0.05,
          "damping": 0.6
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 150 }
      },
      "edges": {
        "smooth": { "type": "continuous", "forceDirection": "none" },
        "color": { "opacity": 0.55 },
        "scaling": { "min": 0.5, "max": 6 }
      },
      "nodes": {
        "shape": "dot",
        "scaling": { "min": 5, "max": 30 },
        "font": { "size": 10, "face": "Inter, Arial" }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": true,
        "navigationButtons": true
      }
    }
    """)

    freq_vals = [H.nodes[n].get("frequency", 1) for n in H.nodes()]
    max_freq  = max(freq_vals) if freq_vals else 1

    for node in H.nodes(data=True):
        name = node[0]
        d    = node[1]
        cid  = partition.get(name, 0)
        freq = d.get("frequency", 1)
        size = 5 + 25 * (freq / max_freq) ** 0.5

        tooltip = (
            f"<b>{name}</b><br>"
            f"Category: {d.get('category','?')}<br>"
            f"Community: #{cid}<br>"
            f"Frequency: {freq}<br>"
            f"{d.get('description','')[:100]}"
        )
        net.add_node(
            name,
            label=name.rsplit("_for_", 1)[0].replace("_", " ")[:28],
            title=tooltip,
            color=comm_col.get(cid, "#888888"),
            size=size,
            borderWidth=0,
        )

    w_vals   = [d.get("weight", 0) for _, _, d in H.edges(data=True)]
    max_w    = max(w_vals) if w_vals else 1.0
    EDGE_CAP = 0.5       # show only edges above this normalised weight threshold

    shown = 0
    for u, v, d in H.edges(data=True):
        w   = d.get("weight", 0)
        nw  = w / max_w
        if nw < EDGE_CAP / 10:
            continue
        src   = "stat" if d.get("source") == "statistical" else "sem"
        color = "#4da6ff" if src == "stat" else "#ff9f43"
        net.add_edge(
            u, v,
            width=0.5 + 5 * nw,
            color={"color": color, "opacity": 0.45 + 0.5 * nw},
            title=f"w={w:.3f}  [{src}]",
        )
        shown += 1

    net.write_html(str(FIG_PYVIS))
    print(f"  Pyvis HTML    → {FIG_PYVIS}  ({shown} edges rendered)")


# --- Meta-graph matplotlib ---------------------------------------------------

def plot_meta_graph(MG: "nx.Graph", partition: dict[str, int]) -> None:
    import networkx as nx

    # Prune meta-graph: keep only top communities by size
    top_comms = sorted(MG.nodes(), key=lambda n: MG.nodes[n].get("size", 0), reverse=True)[:40]
    H = MG.subgraph(top_comms).copy()

    pos = nx.spring_layout(H, seed=42, weight="weight", k=2.0)

    sizes     = [H.nodes[n]["size"] * 12 for n in H.nodes()]
    edge_w    = np.array([d["weight"] for _, _, d in H.edges(data=True)])
    edge_w_n  = (edge_w - edge_w.min()) / (np.ptp(edge_w) + 1e-9)
    n_comm    = len(set(partition.values()))
    palette   = _community_colors(n_comm)
    node_col  = [palette[n % len(palette)] for n in H.nodes()]

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#111111")
    ax.set_facecolor("#111111")

    nx.draw_networkx_edges(
        H, pos, ax=ax,
        width=0.5 + 4 * edge_w_n,
        alpha=0.4,
        edge_color=[plt.cm.Blues(0.3 + 0.7 * w) for w in edge_w_n],
    )
    nx.draw_networkx_nodes(
        H, pos, ax=ax,
        node_size=sizes,
        node_color=node_col,
        alpha=0.92,
        linewidths=0,
    )
    labels = {n: f"#{n}\n{H.nodes[n]['label'].rsplit('_for_',1)[0][:16]}\n(n={H.nodes[n]['size']})"
              for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=5.5,
                            font_color="#eeeeee", ax=ax)
    ax.set_title("SkillGraph — Community Meta-Graph (top 40 communities)",
                 color="white", fontsize=13, pad=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_META, dpi=160, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Meta-graph    → {FIG_META}")


# --- Degree distribution -----------------------------------------------------

def plot_degree_distribution(G: "nx.DiGraph") -> None:
    from collections import Counter

    out_degs = sorted(d for _, d in G.out_degree())
    in_degs  = sorted(d for _, d in G.in_degree())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, degs, title, color in zip(
        axes,
        [out_degs, in_degs],
        ["Out-degree distribution", "In-degree distribution"],
        ["#4C72B0", "#DD8452"],
    ):
        counts = Counter(degs)
        xs = sorted(counts)
        ys = [counts[x] for x in xs]
        ax.scatter(xs, ys, s=12, alpha=0.7, color=color)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Degree"); ax.set_ylabel("Number of nodes")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    fig.suptitle("SkillGraph — Degree Distributions (log-log)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DEGREE, dpi=150)
    plt.close(fig)
    print(f"  Degree dist   → {FIG_DEGREE}")


# --- Weight distribution -----------------------------------------------------

def plot_weight_distribution(G: "nx.DiGraph") -> None:
    w_final = [d["weight"]  for _, _, d in G.edges(data=True)]
    w_stat  = [d.get("w_stat", 0) for _, _, d in G.edges(data=True)]
    w_sem   = [d.get("w_sem", 0)  for _, _, d in G.edges(data=True)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        (w_final, "Final weight  w(i,j)", "#2ecc71"),
        (w_stat,  "Stat component (norm)", "#4C72B0"),
        (w_sem,   "Semantic component",    "#e74c3c"),
    ]
    for ax, (vals, title, color) in zip(axes, specs):
        ax.hist(vals, bins=60, color=color, edgecolor="none", log=True, alpha=0.85)
        ax.set_xlabel("Weight"); ax.set_ylabel("Count (log)")
        ax.set_title(title); ax.grid(True, alpha=0.2)

    fig.suptitle("SkillGraph — Edge Weight Distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_WEIGHT, dpi=150)
    plt.close(fig)
    print(f"  Weight dist   → {FIG_WEIGHT}")


# ── 2h. Statistics ───────────────────────────────────────────────────────────

def print_phase2_stats(
    G:         "nx.DiGraph",
    partition: dict[str, int],
    edges:     dict[tuple[int, int], dict],
) -> None:
    import networkx as nx
    from collections import Counter

    n_comm  = len(set(partition.values()))
    comm_sz = Counter(partition.values())
    out_deg = [d for _, d in G.out_degree()]
    in_deg  = [d for _, d in G.in_degree()]
    density = nx.density(G)

    stat_edges = sum(1 for v in edges.values() if v["source"] == "statistical")
    sem_edges  = len(edges) - stat_edges

    print("\n" + "=" * 60)
    print("PHASE 2 — Fused Graph Statistics")
    print("=" * 60)
    print(f"  Nodes                    : {G.number_of_nodes():,}")
    print(f"  Edges (total)            : {G.number_of_edges():,}")
    print(f"    └─ Statistical         : {stat_edges:,}")
    print(f"    └─ Zero-shot semantic  : {sem_edges:,}")
    print(f"  Graph density            : {density:.6f}")
    print(f"  Avg out-degree           : {sum(out_deg)/len(out_deg):.2f}")
    print(f"  Avg in-degree            : {sum(in_deg)/len(in_deg):.2f}")
    print(f"  Max out-degree           : {max(out_deg)}")
    print(f"  Max in-degree            : {max(in_deg)}")
    print(f"  Communities (Louvain)    : {n_comm}")
    print(f"  Largest community        : {comm_sz.most_common(1)[0][1]} tools")
    print(f"  Smallest community       : {comm_sz.most_common()[-1][1]} tools")
    print(f"  Median community size    : {sorted(comm_sz.values())[n_comm//2]}")

    print("\n  Top 20 edges by final weight:")
    header = f"  {'w_final':>8}  {'w_stat':>7}  {'w_sem':>7}  source → target"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Reconstruct name mapping from G
    for u, v, d in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"])[:20]:
        print(f"  {d['weight']:8.4f}  {d.get('w_stat',0):7.4f}  {d.get('w_sem',0):7.4f}"
              f"  {u} → {v}")


# ── Phase 2 entry ─────────────────────────────────────────────────────────────

def phase2(
    alpha:         float = 0.7,
    top_k:         int   = 20,
    sem_threshold: float = 0.6,
    embedder:      str   = "local",
    no_cache:      bool  = False,
) -> None:
    print("\n" + "=" * 60)
    print("PHASE 2 — Semantic Fusion + Community Detection")
    print("=" * 60)

    # ── Load Phase 1 outputs
    for f in (OUT_ADJ, OUT_EDGES, OUT_INDEX, META_FILE):
        if not f.exists():
            print(f"ERROR: {f} not found — run phase 1 first.")
            return

    print("\n[1/7] Loading Phase 1 artefacts …")
    vocab    = json.loads(OUT_INDEX.read_text(encoding="utf-8"))
    meta     = json.loads(META_FILE.read_text(encoding="utf-8"))

    # Rebuild stat edges from CSV
    stat_edges: dict[tuple[int, int], dict] = {}
    with OUT_EDGES.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vi = vocab[row["source"]]
            vj = vocab[row["target"]]
            stat_edges[(vi, vj)] = {
                "weight":          float(row["weight"]),
                "count":           float(row["count"]),
                "pmi":             float(row["pmi"]),
                "transition_prob": float(row["transition_prob"]),
            }

    # Tool frequency from metadata
    freq: dict[str, int] = {name: info.get("frequency", 0) for name, info in meta.items()}

    # Active tools (those appearing in stat edges)
    active_set   = {vocab_name for vocab_name, idx in vocab.items()
                    if any(idx in (vi, vj) for vi, vj in stat_edges)}
    active_tools = sorted(active_set)        # deterministic order
    print(f"  Stat edges loaded  : {len(stat_edges):,}")
    print(f"  Active tools       : {len(active_tools):,}")

    # ── Embeddings
    print("\n[2/7] Generating tool embeddings …")
    embeddings = get_embeddings(
        active_tools, meta,
        embedder=embedder,
        cache=not no_cache,
    )

    # ── Top-K cosine similarity
    print(f"\n[3/7] Computing top-{top_k} cosine similarity per tool …")
    tool_to_local = {name: i for i, name in enumerate(active_tools)}
    sim_map = topk_cosine_sparse(embeddings, active_tools, tool_to_local, top_k=top_k)
    print(f"  Semantic pairs kept: {len(sim_map):,}")

    # ── Graph fusion
    print(f"\n[4/7] Fusing graphs  (α={alpha}) …")
    fused_edges = fuse_graphs(
        stat_edges, sim_map, active_tools, vocab,
        alpha=alpha, sem_threshold=sem_threshold,
    )

    # ── Build NetworkX graph
    print("\n[5/7] Building NetworkX DiGraph …")
    G = build_nx_graph(fused_edges, vocab, meta, freq)

    # ── Community detection
    print("\n[6/7] Louvain community detection …")
    partition = detect_communities(G)
    MG        = build_meta_graph(G, partition)

    # ── Save
    print("\n[7/7] Saving outputs …")
    save_phase2(G, MG, partition, fused_edges, vocab)

    # ── Statistics
    print_phase2_stats(G, partition, fused_edges)

    # ── Visualisations
    print("\nGenerating visualisations …")
    make_pyvis(G, partition)
    plot_meta_graph(MG, partition)
    plot_degree_distribution(G)
    plot_weight_distribution(G)

    print("\nPhase 2 complete.")


# ============================================================================
#  Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkillGraph — Statistical + Semantic Graph Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase",   choices=["1", "2", "all"], default="all")
    parser.add_argument("--alpha",   type=float, default=0.7,
                        help="Fusion weight for statistical component (0–1)")
    parser.add_argument("--topk",    type=int,   default=20,
                        help="Top-K semantic neighbours per tool")
    parser.add_argument("--sem-threshold", type=float, default=0.6,
                        help="Min cosine sim for zero-shot semantic edges")
    parser.add_argument("--embedder", choices=["local", "openai"], default="local",
                        help="Embedding backend")
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute embeddings even if cache exists")
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        phase1()

    if args.phase in ("2", "all"):
        phase2(
            alpha=args.alpha,
            top_k=args.topk,
            sem_threshold=args.sem_threshold,
            embedder=args.embedder,
            no_cache=args.no_cache,
        )


if __name__ == "__main__":
    main()
