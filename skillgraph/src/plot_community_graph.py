"""
Plot a clean, publication-ready community meta-graph (top-10 communities).
White background, readable labels, suitable for IEEE journal inclusion.
"""
import pickle, collections, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# ── Times New Roman throughout ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",   # STIX matches Times New Roman for math
})

ROOT    = pathlib.Path(__file__).resolve().parent.parent
OUT     = ROOT / "outputs" / "figures" / "fig_community_graph.png"

# ── Human-readable community names (from category analysis) ──────────────────
COMM_LABELS = {
    26: "Sports",
     5: "Currency\n& Crypto",
     8: "Travel\n& Places",
    28: "News\n& Media",
    18: "Geography",
    22: "Social\nMedia",
    38: "E-commerce",
     1: "Finance",
     2: "Entertain-\nment",
     4: "Language\n& NLP",
}

COMM_COLORS = [
    "#e74c3c",  # red      — Sports
    "#f39c12",  # orange   — Currency
    "#3498db",  # blue     — Travel
    "#9b59b6",  # purple   — News
    "#1abc9c",  # teal     — Geography
    "#e67e22",  # dark-org — Social
    "#2ecc71",  # green    — E-commerce
    "#2980b9",  # dark-blue— Finance
    "#e91e63",  # pink     — Entertainment
    "#00bcd4",  # cyan     — Language
]

TOP10_IDS = [26, 5, 8, 28, 18, 22, 38, 1, 2, 4]

def main():
    with open(ROOT / "data/processed/final_graph.gpickle", "rb") as f:
        G = pickle.load(f)

    # Community sizes
    comm_size = collections.Counter(d["community"] for _, d in G.nodes(data=True))

    # Build inter-community directed weighted graph
    MG = nx.DiGraph()
    for cid in TOP10_IDS:
        MG.add_node(cid, size=comm_size[cid], label=COMM_LABELS[cid])

    inter = collections.defaultdict(float)
    for u, v, d in G.edges(data=True):
        cu = G.nodes[u]["community"]
        cv = G.nodes[v]["community"]
        if cu in TOP10_IDS and cv in TOP10_IDS and cu != cv:
            inter[(cu, cv)] += d.get("weight", 1.0)

    # Only keep edges with meaningful weight (threshold = 1.0)
    for (a, b), w in inter.items():
        if w >= 1.0:
            MG.add_edge(a, b, weight=w)

    # Layout — fixed seed for reproducibility
    pos = nx.spring_layout(MG, seed=7, weight="weight", k=4.5)

    # ── Figure ──────────────────────────────────────────────────────────────
    # 3.5 × 3.1 inches = single IEEE column width; fonts specified at print size
    fig = plt.figure(figsize=(3.5, 3.1), facecolor="white")
    ax  = fig.add_axes([0.01, 0.16, 0.98, 0.80])   # [left, bottom, width, height]
    ax.set_facecolor("white")

    # Edge widths and colours — darker palette, stronger contrast
    edges     = list(MG.edges(data=True))
    weights   = np.array([d["weight"] for _, _, d in edges])
    w_norm    = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    edge_widths = 0.4 + 2.5 * w_norm
    edge_alphas = 0.45 + 0.50 * w_norm

    for (u, v, d), lw, alpha in zip(edges, edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            MG, pos, edgelist=[(u, v)], ax=ax,
            width=lw, alpha=float(alpha),
            edge_color="#333333",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=8,
            connectionstyle="arc3,rad=0.12",
            node_size=[MG.nodes[n]["size"] * 1.2 for n in MG.nodes()],
        )

    # Nodes — scale down to fit single-column width
    node_sizes  = [MG.nodes[n]["size"] * 1.2 for n in MG.nodes()]
    node_colors = [COMM_COLORS[TOP10_IDS.index(n)] for n in MG.nodes()]
    nx.draw_networkx_nodes(
        MG, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.90,
        linewidths=0.6,
        edgecolors="white",
    )

    # Labels — outside nodes; anchor the text edge (not centre) at the offset point
    cx = np.mean([p[0] for p in pos.values()])
    cy = np.mean([p[1] for p in pos.values()])
    for node in MG.nodes():
        x, y = pos[node]
        dx, dy = x - cx, y - cy
        norm = max((dx**2 + dy**2) ** 0.5, 0.01)
        ux, uy = dx / norm, dy / norm   # unit vector away from centroid

        # Offset = node visual radius (in data coords) + clearance gap
        node_r = (MG.nodes[node]["size"] * 1.2) ** 0.5 / 110  # ≈ node radius in data units
        gap    = 0.06                                           # extra clearance
        lx = x + ux * (node_r + gap)
        ly = y + uy * (node_r + gap)

        # Align text anchor so the near edge faces the node, not the centre
        ha = "left"  if ux >  0.3 else ("right" if ux < -0.3 else "center")
        va = "bottom" if uy >  0.3 else ("top"   if uy < -0.3 else "center")

        ax.text(lx, ly, COMM_LABELS[node],
                ha=ha, va=va,
                fontsize=8, fontweight="bold",
                color="#111111",
                multialignment="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))

    # ── Legend — three circles in one row below the axes ─────────────────────
    legend_handles = []
    for disp_s, label in [(60, "350 tools"), (35, "200 tools"), (15, "100 tools")]:
        legend_handles.append(
            plt.scatter([], [], s=disp_s, c="#999999", alpha=0.80,
                        edgecolors="#555555", linewidths=0.5, label=label)
        )
    fig.legend(
        handles=legend_handles,
        title="Community size",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=8, title_fontsize=8,
        framealpha=0.92, edgecolor="#cccccc",
        scatterpoints=1,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    # Expand axis limits so peripheral labels aren't clipped
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 0.38
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_title("SkillGraph — Top-10 Tool Communities\n"
                 "(node size ~ community size; arrow weight ~ transition probability)",
                 fontsize=8, pad=6)
    ax.axis("off")
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {OUT}")

if __name__ == "__main__":
    main()
