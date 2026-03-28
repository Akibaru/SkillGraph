"""
src/generate_figures.py  —  Generate all paper figures from result CSVs
=======================================================================
Reads pre-computed CSVs in results/ and writes publication-quality figures
to outputs/figures/.

Figures generated
-----------------
  fig1_pareto.pdf/.png          — Pareto scatter: Set-F1 vs Ord.Prec
  fig2_method_bar.pdf/.png      — Bar chart: all methods × key metrics
  fig3_bucket_bar.pdf/.png      — Bucket (length) analysis bar chart
  fig4_alpha_sensitivity.pdf/.png — Alpha sensitivity curve
  fig5_bootstrap_sig.pdf/.png   — Significance summary heatmap

Usage
-----
  python src/generate_figures.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent
RES_DIR  = ROOT / "results"
FIGS_DIR = ROOT / "outputs" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour / style palette
# ---------------------------------------------------------------------------
# Groups: single-stage baselines, TS-Sem, TS-Hybrid
COLORS = {
    "semantic_only":            "#9e9e9e",   # grey
    "beam":                     "#78909c",   # blue-grey
    "hybrid":                   "#5c7cfa",   # blue
    "ts_sem_semsort":           "#f9a825",   # amber (light)
    "ts_sem_hybrid_rerank":     "#fb8c00",   # amber
    "ts_hybrid_semsort":        "#66bb6a",   # green (light)
    "ts_hybrid_optimal_perm":   "#43a047",   # green
    "ts_hybrid_hybrid_rerank":  "#e53935",   # red
    "ts_hybrid_learned_rerank": "#b71c1c",   # deep red  ← our best
}

MARKERS = {
    "semantic_only":            "o",
    "beam":                     "s",
    "hybrid":                   "D",
    "ts_sem_semsort":           "^",
    "ts_sem_hybrid_rerank":     "v",
    "ts_hybrid_semsort":        "P",
    "ts_hybrid_optimal_perm":   "X",
    "ts_hybrid_hybrid_rerank":  "*",
    "ts_hybrid_learned_rerank": "★",
}

SHORT_LABELS = {
    "semantic_only":            "Sem-Only",
    "beam":                     "Beam",
    "hybrid":                   "Hybrid S-G",
    "ts_sem_semsort":           "TS-Sem+SS",
    "ts_sem_hybrid_rerank":     "TS-Sem+HR",
    "ts_hybrid_semsort":        "TS-Hyb+SS",
    "ts_hybrid_optimal_perm":   "TS-Hyb+OP",
    "ts_hybrid_hybrid_rerank":  "TS-Hyb+HR",
    "ts_hybrid_learned_rerank": "TS-Hyb+LR",
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":   150,
    "savefig.bbox": "tight",
    "savefig.dpi":  300,
})


# ============================================================================
# Figure 1: Pareto scatter
# ============================================================================

def fig_pareto(df: pd.DataFrame) -> None:
    """Set-F1 (x) vs Ord.Prec (y) scatter with Pareto frontier marked."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in df.iterrows():
        m   = row["method"]
        col = COLORS.get(m, "#333")
        mk  = MARKERS.get(m, "o")
        ms  = 14 if row.get("pareto_frontier", False) else 8
        lw  = 2.0 if row.get("pareto_frontier", False) else 0.8
        zord = 5 if row.get("pareto_frontier", False) else 3
        ax.scatter(row["set_f1"], row["ord_prec"],
                   color=col, marker="*" if m == "ts_hybrid_learned_rerank" else mk,
                   s=ms**2, linewidths=lw, edgecolors="k",
                   zorder=zord, label=SHORT_LABELS.get(m, m))

    # Draw dashed Pareto frontier line
    frontier = df[df["pareto_frontier"] == True].sort_values("set_f1")
    if len(frontier) > 1:
        ax.plot(frontier["set_f1"], frontier["ord_prec"],
                "k--", lw=1.0, alpha=0.5, zorder=2, label="_nolegend_")

    ax.set_xlabel("Set-F1  (selection quality, order-independent)")
    ax.set_ylabel("Ordered Precision  (ranking quality)")
    ax.set_title("Pareto Frontier: Selection vs. Ranking Quality")
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Annotate Pareto methods
    for _, row in df[df["pareto_frontier"] == True].iterrows():
        ax.annotate(
            SHORT_LABELS.get(row["method"], row["method"]),
            xy=(row["set_f1"], row["ord_prec"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, color=COLORS.get(row["method"], "#333"),
        )

    _save(fig, "fig1_pareto")
    print("  fig1_pareto saved")


# ============================================================================
# Figure 2: Method comparison bar chart
# ============================================================================

def fig_method_bar(df: pd.DataFrame) -> None:
    """
    Grouped bar chart: Ord.Prec, Kendall-Tau, Trans.Acc for every method.
    Methods on x-axis; three metric groups side-by-side.
    """
    METRICS  = ["ordered_precision", "kendall_tau", "transition_acc"]
    M_LABELS = ["Ordered Precision", "Kendall-Tau", "Transition Acc"]
    M_COLORS = ["#e53935", "#fb8c00", "#43a047"]

    # Order methods as in the paper
    order = [
        "semantic_only", "beam", "hybrid",
        "ts_sem_semsort", "ts_sem_hybrid_rerank",
        "ts_hybrid_semsort", "ts_hybrid_optimal_perm",
        "ts_hybrid_hybrid_rerank", "ts_hybrid_learned_rerank",
    ]
    df_plot = df.set_index("method").reindex(order).reset_index()
    xlabels = [SHORT_LABELS.get(m, m) for m in df_plot["method"]]

    n_m = len(METRICS)
    n_x = len(df_plot)
    x   = np.arange(n_x)
    w   = 0.25

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, (met, mlbl, mc) in enumerate(zip(METRICS, M_LABELS, M_COLORS)):
        vals = df_plot[met].values.astype(float)
        bars = ax.bar(x + (k - 1) * w, vals, width=w,
                      color=mc, alpha=0.85, label=mlbl, edgecolor="white", lw=0.5)

    # Vertical separators between method groups
    for sep in [2.5, 4.5]:
        ax.axvline(sep, color="grey", lw=0.8, ls="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Ordering Quality Across Methods")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)

    # Group annotations
    for txt, xpos in [("Single-stage", 1.0), ("TS-Sem", 3.5), ("TS-Hybrid", 6.5)]:
        ax.text(xpos, ax.get_ylim()[1] * 1.04, txt,
                ha="center", fontsize=8.5, color="grey", style="italic")

    _save(fig, "fig2_method_bar")
    print("  fig2_method_bar saved")


# ============================================================================
# Figure 3: Bucket / length breakdown
# ============================================================================

def fig_bucket_bar(df_len: pd.DataFrame) -> None:
    """
    Ord.Prec by sequence-length bucket for key methods.
    """
    FOCUS = [
        "semantic_only",
        "beam",
        "hybrid",
        "ts_hybrid_hybrid_rerank",
        "ts_hybrid_learned_rerank",
    ]
    BUCKET_ORDER = ["1-2", "3-4", "5+"]

    df_f = df_len[df_len["method"].isin(FOCUS)].copy()
    df_f["bucket"] = pd.Categorical(df_f["bucket"], categories=BUCKET_ORDER, ordered=True)
    df_f = df_f.sort_values(["method", "bucket"])

    n_m = len(FOCUS)
    n_b = len(BUCKET_ORDER)
    x   = np.arange(n_b)
    w   = 0.14
    offsets = np.linspace(-(n_m - 1) / 2 * w, (n_m - 1) / 2 * w, n_m)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, metric, ylabel, title in [
        (axes[0], "ordered_precision", "Ordered Precision", "Ordered Precision by Sequence Length"),
        (axes[1], "kendall_tau",       "Kendall-Tau",       "Kendall-Tau by Sequence Length"),
    ]:
        for k, m in enumerate(FOCUS):
            sub = df_f[df_f["method"] == m].sort_values("bucket")
            vals = [sub[sub["bucket"] == b][metric].values[0]
                    if len(sub[sub["bucket"] == b]) > 0 else 0.0
                    for b in BUCKET_ORDER]
            ax.bar(x + offsets[k], vals, width=w,
                   color=COLORS.get(m, "#aaa"), label=SHORT_LABELS.get(m, m),
                   alpha=0.85, edgecolor="white", lw=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"[{b}]" for b in BUCKET_ORDER])
        ax.set_xlabel("GT Sequence Length Bucket")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig3_bucket_bar")
    print("  fig3_bucket_bar saved")


# ============================================================================
# Figure 4: Alpha sensitivity curve
# ============================================================================

def fig_alpha_sensitivity(df_alpha: pd.DataFrame) -> None:
    """Alpha sensitivity for TS-Hybrid + Hybrid-Rerank."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(df_alpha["alpha"], df_alpha["ord_prec"],
            "o-", color="#e53935", lw=2, ms=6, label="Ordered Precision")
    ax.plot(df_alpha["alpha"], df_alpha["kendall_tau"],
            "s--", color="#fb8c00", lw=2, ms=6, label="Kendall-Tau")
    ax.plot(df_alpha["alpha"], df_alpha["transition_acc"],
            "^:", color="#43a047", lw=2, ms=6, label="Transition Acc")

    # Shade plateau region
    best_op = df_alpha.loc[df_alpha["ord_prec"].idxmax(), "alpha"]
    ax.axvspan(0.3, 0.7, alpha=0.08, color="grey", label="Plateau (0.3–0.7)")
    ax.axvline(best_op, color="#e53935", lw=1.0, ls="--", alpha=0.6)

    ax.set_xlabel("α  (weight on transition probability)")
    ax.set_ylabel("Score")
    ax.set_title("Alpha Sensitivity: TS-Hybrid + Hybrid-Rerank")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    _save(fig, "fig4_alpha_sensitivity")
    print("  fig4_alpha_sensitivity saved")


# ============================================================================
# Figure 5: Significance summary
# ============================================================================

def fig_significance(df_boot: pd.DataFrame) -> None:
    """
    Heatmap-style table showing p-values and significance for all pairs × metrics.
    """
    metrics = df_boot["metric"].unique().tolist()
    pairs   = df_boot["comparison"].unique().tolist()

    # Build matrix of p-values
    p_mat   = np.ones((len(pairs), len(metrics)))
    sig_mat = np.zeros((len(pairs), len(metrics)), dtype=bool)
    diff_mat= np.zeros((len(pairs), len(metrics)))

    for i, pair in enumerate(pairs):
        for j, met in enumerate(metrics):
            row = df_boot[(df_boot["comparison"] == pair) & (df_boot["metric"] == met)]
            if not row.empty:
                p_mat[i, j]    = row["p_one_sided"].values[0]
                sig_mat[i, j]  = row["significant_95"].values[0]
                diff_mat[i, j] = row["obs_diff"].values[0]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(p_mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p.replace(" vs ", "\nvs ") for p in pairs], fontsize=8)

    # Annotate cells
    for i in range(len(pairs)):
        for j in range(len(metrics)):
            txt   = f"p={p_mat[i,j]:.3f}\nΔ{diff_mat[i,j]:+.4f}"
            star  = "✓" if sig_mat[i, j] else ""
            color = "white" if p_mat[i, j] < 0.05 else "black"
            ax.text(j, i, f"{star}\n{txt}", ha="center", va="center",
                    fontsize=7.5, color=color)

    plt.colorbar(im, ax=ax, label="p-value (one-sided)")
    ax.set_title("Bootstrap Significance (10k resamples, paired)")
    fig.tight_layout()

    _save(fig, "fig5_bootstrap_sig")
    print("  fig5_bootstrap_sig saved")


# ============================================================================
# Helpers
# ============================================================================

def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIGS_DIR / f"{name}.{ext}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Loading CSVs ...")
    df_main   = pd.read_csv(RES_DIR / "final_comparison.csv")
    df_len    = pd.read_csv(RES_DIR / "final_by_length.csv")
    df_boot   = pd.read_csv(RES_DIR / "bootstrap_significance.csv")
    df_alpha  = pd.read_csv(RES_DIR / "alpha_sensitivity.csv")

    # Load Pareto data if available; otherwise derive from main results
    pareto_path = RES_DIR / "pareto_data.csv"
    if pareto_path.exists():
        df_pareto = pd.read_csv(pareto_path)
        # Merge in LR row if not already present (in case pareto_data.csv is stale)
        if "ts_hybrid_learned_rerank" not in df_pareto["method"].values:
            lr_row = df_main[df_main["method"] == "ts_hybrid_learned_rerank"]
            if not lr_row.empty:
                new_row = pd.DataFrame([{
                    "method":          "ts_hybrid_learned_rerank",
                    "label":           "TS-Hybrid + Learned-Rerank",
                    "set_f1":          float(lr_row["set_f1"].values[0]),
                    "ord_prec":        float(lr_row["ordered_precision"].values[0]),
                    "pareto_frontier": False,
                }])
                df_pareto = pd.concat([df_pareto, new_row], ignore_index=True)
                # Recompute Pareto dominance
                df_pareto = _recompute_pareto(df_pareto)
    else:
        # Build Pareto data from main results
        df_pareto = _build_pareto_from_main(df_main)

    print("Generating figures ...")
    fig_pareto(df_pareto)
    fig_method_bar(df_main)
    fig_bucket_bar(df_len)
    fig_alpha_sensitivity(df_alpha)
    fig_significance(df_boot)

    print(f"\nAll figures saved to {FIGS_DIR}")


def _recompute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    is_dominated = []
    for i, ri in df.iterrows():
        dominated = False
        for j, rj in df.iterrows():
            if i == j:
                continue
            if (rj["set_f1"]  >= ri["set_f1"]  and
                rj["ord_prec"] >= ri["ord_prec"] and
                (rj["set_f1"] > ri["set_f1"] or rj["ord_prec"] > ri["ord_prec"])):
                dominated = True
                break
        is_dominated.append(dominated)
    df = df.copy()
    df["pareto_frontier"] = [not d for d in is_dominated]
    return df


def _build_pareto_from_main(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "method":   row["method"],
            "label":    row["label"],
            "set_f1":   float(row["set_f1"]),
            "ord_prec": float(row["ordered_precision"]),
        })
    df_p = pd.DataFrame(rows)
    return _recompute_pareto(df_p)


if __name__ == "__main__":
    main()
