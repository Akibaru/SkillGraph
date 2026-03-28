"""
src/run_feature_ablation.py  —  LR Feature Group Ablation
==========================================================
Loads the pre-trained PairwiseMLP, runs TS-Hybrid Stage-1 selection,
then evaluates ordering on the full test set with each feature group
zeroed out at inference time.

Feature groups
--------------
  Semantic  (f1,f2 = idx 0,1): sim_query_tool, rank_sem_norm
  Graph     (f3-f6 = idx 2-5): sum/max tp_out, sum/max tp_in
  Positional(f7,f8 = idx 6,7): avg_train_pos, set_size_norm

Output: results/feature_ablation.csv
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau as scipy_kendalltau

warnings.filterwarnings("ignore")

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

ROOT        = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

from graph_search import ToolSequencePlanner
from evaluate import (
    load_trajectories,
    make_train_test_split,
    batch_encode_queries,
    _ordered_precision,
    _transition_accuracy,
    _first_tool_accuracy,
    _lcs_length,
)
from two_stage_pipeline import _build_tp_lookup
from learned_reranker import (
    build_position_stats,
    extract_features,
    load_learned_reranker,
    CHECKPOINT as LR_CHECKPOINT,
    FEAT_DIM,
)
from final_comparison import get_hybrid_stage1

# -----------------------------------------------------------------------
# Feature group masks  (True = keep, False = zero out)
# -----------------------------------------------------------------------
ABLATIONS = {
    "Full LR":                     [True]  * 8,
    "-Semantic (f1,f2)":           [False, False, True,  True,  True,  True,  True,  True],
    "-Graph transitions (f3-f6)":  [True,  True,  False, False, False, False, True,  True],
    "-Positional priors (f7,f8)":  [True,  True,  True,  True,  True,  True,  False, False],
}

# -----------------------------------------------------------------------
# Masked inference
# -----------------------------------------------------------------------
@torch.no_grad()
def order_masked_rerank(
    tools_sims:     list[tuple[str, float]],
    tp_lookup:      dict[tuple[str, str], float],
    position_stats: dict[str, float],
    model,
    mask:           list[bool],
) -> list[str]:
    if len(tools_sims) <= 1:
        return [t for t, _ in tools_sims]

    mask_arr  = np.array(mask, dtype=np.float32)
    tool_set  = [t for t, _ in tools_sims]
    n         = len(tool_set)
    sim_dict  = dict(tools_sims)
    sorted_by_sim = sorted(tool_set, key=lambda t: -sim_dict.get(t, 0.0))
    rank_dict = {t: r for r, t in enumerate(sorted_by_sim)}

    feat_mat = np.stack([
        extract_features(t, sim_dict[t], rank_dict[t],
                         tool_set, tp_lookup, position_stats) * mask_arr
        for t in tool_set
    ], axis=0)

    feat_t  = torch.tensor(feat_mat, dtype=torch.float32)
    diff    = feat_t.unsqueeze(1) - feat_t.unsqueeze(0)   # (n, n, FEAT_DIM)
    logits  = model(diff.view(n * n, FEAT_DIM)).view(n, n)
    scores  = torch.sigmoid(logits).sum(dim=1).numpy()

    return [tool_set[i] for i in np.argsort(-scores)]


# -----------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------
def _kendalltau(pred: list, gt: list) -> float:
    common = [t for t in pred if t in set(gt)]
    if len(common) < 2:
        return 0.0
    gt_rank   = {t: i for i, t in enumerate(gt)}
    pred_rank = {t: i for i, t in enumerate(common)}
    tau, _    = scipy_kendalltau([pred_rank[t] for t in common],
                                  [gt_rank[t]   for t in common])
    return float(tau) if not np.isnan(tau) else 0.0


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    DATA_DIR   = ROOT / "data" / "processed"
    traj_path  = DATA_DIR / "successful_trajectories.jsonl"

    print("Loading data...")
    all_records = load_trajectories(traj_path)
    train_recs, test_recs = make_train_test_split(all_records, seed=42)
    print(f"  train={len(train_recs):,}  test={len(test_recs):,}")

    planner = ToolSequencePlanner()

    print("Encoding test queries...")
    test_vecs = batch_encode_queries([r["task_description"] for r in test_recs])

    tp_lookup      = _build_tp_lookup(planner)
    position_stats = build_position_stats(train_recs)
    hybrid_params  = {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}

    print(f"Loading LR model from {LR_CHECKPOINT}...")
    model = load_learned_reranker(LR_CHECKPOINT)
    model.eval()

    # --- cache Stage-1 hybrid output once for all ablation variants ---
    print("\nBuilding Stage-1 TS-Hybrid cache...")
    stage1_cache: list[list[tuple[str, float]]] = []
    for i, rec in enumerate(test_recs):
        if i % 2000 == 0:
            print(f"  {i}/{len(test_recs)}")
        stage1_cache.append(
            get_hybrid_stage1(planner, test_vecs[i], hybrid_params, max_steps=8)
        )
    print("  Stage-1 cache built.")

    rows = []
    for ablation_name, mask in ABLATIONS.items():
        print(f"\n[{ablation_name}]")
        t0 = time.time()
        ops, lcss, taus, tas, fas = [], [], [], [], []

        for i, rec in enumerate(test_recs):
            gt_seq    = rec["tool_sequence"]
            tools_sim = stage1_cache[i]
            if not tools_sim:
                continue

            pred_seq = order_masked_rerank(
                tools_sim, tp_lookup, position_stats, model, mask
            )

            gt_set   = set(gt_seq)
            pred_in  = [t for t in pred_seq if t in gt_set]
            ops .append(_ordered_precision(pred_seq, gt_seq))
            lcss.append(_lcs_length(pred_in, gt_seq) / max(len(gt_seq), 1))
            taus.append(_kendalltau(pred_seq, gt_seq))
            tas .append(_transition_accuracy(pred_seq, gt_seq))
            fas .append(_first_tool_accuracy(pred_seq, gt_seq))

        row = {
            "method":         ablation_name,
            "n":              len(ops),
            "ord_prec":       round(float(np.mean(ops)),  4),
            "lcs_r":          round(float(np.mean(lcss)), 4),
            "kendall_tau":    round(float(np.mean(taus)), 4),
            "trans_acc":      round(float(np.mean(tas)),  4),
            "first_tool_acc": round(float(np.mean(fas)),  4),
        }
        rows.append(row)
        elapsed = time.time() - t0
        print(f"  Ord.Prec={row['ord_prec']:.4f}  "
              f"Kendall-τ={row['kendall_tau']:.4f}  "
              f"Trans.Acc={row['trans_acc']:.4f}  "
              f"({elapsed:.0f}s)")

    df  = pd.DataFrame(rows)
    out = RESULTS_DIR / "feature_ablation.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
