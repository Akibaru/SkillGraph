"""
src/run_llm_stage2.py  —  LLM as Stage 2 Reranker (Fair Comparison)
=====================================================================
Uses the EXACT same pipeline as final_comparison.py for Stage 1,
then applies LLaMA3.1-8B as Stage 2 ordering (zero-shot and 3-shot).

Direct apples-to-apples comparison:
  TS-Hybrid + HR            (from final_comparison.py)
  TS-Hybrid + LR (ours)     (from final_comparison.py)
  TS-Hybrid + LLaMA 0-shot  (LLM ordering of Stage 1 output)
  TS-Hybrid + LLaMA 3-shot  (LLM ordering of Stage 1 output)

Outputs
-------
  results/llm_stage2_toolbench.csv
  results/llm_stage2_comparison.csv

Usage
-----
  python src/run_llm_stage2.py --sample 300
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

ROOT        = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

# ---------------------------------------------------------------------------
# Metrics (inline — same formulas as final_comparison.py)
# ---------------------------------------------------------------------------

def _lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j],dp[i][j-1])
    return dp[m][n]

def _f1(p, g):
    ps, gs = set(p), set(g)
    tp = len(ps & gs)
    pr = tp/max(1,len(ps)); re = tp/max(1,len(gs))
    return 2*pr*re/max(1e-9,pr+re)

def _op(pred, gt):
    gt_set = set(gt); gt_r = {t:i for i,t in enumerate(gt)}
    com = [(t,i) for i,t in enumerate(pred) if t in gt_set]
    if len(com)<2: return 0.
    mp=tp=0
    for i in range(len(com)):
        for j in range(i+1,len(com)):
            tp+=1
            if gt_r[com[i][0]]<gt_r[com[j][0]]: mp+=1
    return mp/tp if tp else 0.

def _ktau(pred, gt):
    com = [t for t in pred if t in set(gt)]
    if len(com)<2: return 0.
    gt_r={t:i for i,t in enumerate(gt)}; rk=[gt_r[t] for t in com]; n=len(rk)
    c=d=0
    for i in range(n):
        for j in range(i+1,n):
            df=rk[i]-rk[j]
            if df<0: c+=1
            elif df>0: d+=1
    return (c-d)/(n*(n-1)/2)

def _ta(pred, gt):
    if len(gt)<2: return 0.
    pairs=[(gt[i],gt[i+1]) for i in range(len(gt)-1)]
    pos={t:i for i,t in enumerate(pred)}
    return sum(1 for a,b in pairs if a in pos and b in pos and 0<pos[b]-pos[a]<=2)/len(pairs)

def _fa(pred, gt):
    return float(bool(pred) and bool(gt) and pred[0]==gt[0])

def compute_metrics(pred, gt):
    return dict(
        set_f1=round(_f1(pred,gt),4),
        lcs_r=round(_lcs(pred,gt)/max(1,len(gt)),4),
        ordered_precision=round(_op(pred,gt),4),
        kendall_tau=round(_ktau(pred,gt),4),
        transition_acc=round(_ta(pred,gt),4),
        first_tool_acc=round(_fa(pred,gt),4),
        pred_len=len(pred), gt_len=len(gt),
    )

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _call(prompt: str, timeout: int = 90) -> str:
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "top_p": 1.0, "num_predict": 256}
        }, timeout=timeout)
        return r.json().get("response","").strip()
    except Exception:
        return ""

def _parse(raw: str, valid: set[str], k: int) -> list[str]:
    if not raw:
        return []
    # Try arrow separators first
    for sep in ["->", "→", ","]:
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep)]
            break
    else:
        parts = raw.strip().split("\n")

    out: list[str] = []; seen: set[str] = set()
    for p in parts:
        p = re.sub(r"^[\d\.\)\-\s]+", "", p).strip().strip('"\'`[]() ')
        match = p if p in valid else next((t for t in valid if t.lower()==p.lower()), None)
        if match and match not in seen:
            seen.add(match); out.append(match)
        if len(out)==k: break
    return out

# ---------------------------------------------------------------------------
# LLM Stage 2 prompts
# ---------------------------------------------------------------------------

def _make_zs_prompt(query: str, tools_sims: list[tuple[str,float]],
                    tool_descs: dict[str,str], k: int) -> str:
    tool_str = "\n".join(
        f"  {i+1}. {t}: {tool_descs.get(t, t.replace('_',' '))}"
        for i, (t,_) in enumerate(tools_sims)
    )
    return (
        f"You are a tool execution planner. The following {len(tools_sims)} tools are ALL "
        f"needed for the task. Output them in the correct EXECUTION ORDER, "
        f"separated by ' -> '. Output ONLY the tool names. Do not explain.\n\n"
        f"Tools:\n{tool_str}\n\n"
        f"Task: \"{query}\"\n"
        f"Execution order:"
    )

def _make_fs_prompt(query: str, tools_sims: list[tuple[str,float]],
                    tool_descs: dict[str,str], k: int,
                    examples: list[dict]) -> str:
    tool_str = "\n".join(
        f"  {i+1}. {t}: {tool_descs.get(t, t.replace('_',' '))}"
        for i, (t,_) in enumerate(tools_sims)
    )
    ex_block = "\n\n".join(
        f"Task: \"{e['task_description']}\"\n"
        f"Tools available: {', '.join(e['tool_sequence'])}\n"
        f"Execution order: {' -> '.join(e['tool_sequence'])}"
        for e in examples
    )
    return (
        f"You are a tool execution planner. The following tools are ALL needed. "
        f"Output them in correct EXECUTION ORDER separated by ' -> '. "
        f"Output ONLY tool names. Do not explain.\n\n"
        f"Tools:\n{tool_str}\n\n"
        f"Examples:\n{ex_block}\n\n"
        f"Task: \"{query}\"\n"
        f"Execution order:"
    )

# ---------------------------------------------------------------------------
# Main evaluation using the ACTUAL pipeline infrastructure
# ---------------------------------------------------------------------------

def run(sample: int = 300, seed: int = 42, few_k: int = 3) -> pd.DataFrame:
    import warnings
    warnings.filterwarnings("ignore")

    # Import pipeline components
    from graph_search import ToolSequencePlanner
    from evaluate import (
        load_trajectories, make_train_test_split,
        batch_encode_queries, SemanticOnlyBaseline, _plan_with_vec,
    )
    from two_stage_pipeline import _build_tp_lookup
    from final_comparison import (
        get_hybrid_stage1, order_hybrid_rerank, compute_order_metrics,
    )
    from learned_reranker import (
        build_position_stats, load_learned_reranker, order_learned_rerank,
        CHECKPOINT as LR_CKPT,
    )

    # Load data
    proc     = ROOT / "data" / "processed"
    records  = load_trajectories(proc / "successful_trajectories.jsonl")
    train_records, test_records = make_train_test_split(records)

    # Build planner (same as final_comparison.py)
    print("Building planner ...")
    avg_train_len    = sum(r["num_steps"] for r in train_records) / len(train_records)
    median_train_len = float(sorted(r["num_steps"] for r in train_records)[len(train_records)//2])

    planner = ToolSequencePlanner()
    planner._avg_traj_len    = avg_train_len
    planner._median_traj_len = median_train_len

    from collections import defaultdict as _dd
    tool_pos_lists: dict = _dd(list)
    for rec in train_records:
        seq = rec["tool_sequence"]; n = len(seq)
        for i, t in enumerate(seq):
            if n: tool_pos_lists[t].append(i / n)
    planner._tool_position_stats = {t: float(np.mean(v)) for t, v in tool_pos_lists.items()}

    # TP lookup and hybrid params (match final_comparison.py defaults)
    tp_lookup     = _build_tp_lookup(planner)
    hybrid_params = {"k_multiplier": 3, "alpha": 0.5, "gamma": 0.1}
    alpha_hr      = 0.4   # tuned alpha for Hybrid-Rerank

    # Learned Reranker
    lr_model = position_stats = None
    if LR_CKPT.exists():
        position_stats = build_position_stats(train_records)
        lr_model       = load_learned_reranker(LR_CKPT)
        print("LR model loaded.")

    # Tool descriptions for LLM prompts
    tool_descs: dict[str, str] = {}
    for r in records:
        for td in r.get("tool_details", []):
            n = td.get("name",""); d = td.get("description","")
            if n and n not in tool_descs:
                tool_descs[n] = d or n.replace("_"," ")

    # Sample test examples
    rng         = random.Random(seed)
    test_sample = rng.sample(test_records, min(sample, len(test_records)))
    print(f"\nEncoding {len(test_sample)} test queries ...")
    queries  = [r["task_description"] for r in test_sample]
    qvecs    = batch_encode_queries(queries)

    def get_few(gt_len: int) -> list[dict]:
        pool = [r for r in train_records if abs(r["num_steps"]-gt_len)<=1] or train_records
        return rng.sample(pool, min(few_k, len(pool)))

    rows: list[dict] = []
    print(f"Evaluating {len(test_sample)} samples ...")

    for i, rec in enumerate(test_sample):
        gt   = rec["tool_sequence"]
        vec  = qvecs[i]

        # Stage 1: TS-Hybrid (identical to final_comparison.py)
        tools_sims = get_hybrid_stage1(planner, vec, hybrid_params)
        if not tools_sims:
            for method in ["ts_hr","ts_lr","ts_llm_zs","ts_llm_fs"]:
                rows.append({"method": method, "idx": i, **compute_metrics([], gt)})
            continue

        # ---- TS-Hybrid + HR ----
        hr_pred = order_hybrid_rerank(tools_sims, tp_lookup, alpha_hr)
        rows.append({"method":"ts_hr","idx":i,**compute_metrics(hr_pred,gt)})

        # ---- TS-Hybrid + LR ----
        if lr_model and position_stats:
            lr_pred = order_learned_rerank(tools_sims, tp_lookup, position_stats, lr_model)
            rows.append({"method":"ts_lr","idx":i,**compute_metrics(lr_pred,gt)})

        # LLM: build candidate set from Stage 1 output
        cand_tools = [t for t,_ in tools_sims]
        k          = len(cand_tools)

        # ---- TS-Hybrid + LLM Zero-Shot ----
        zs_raw  = _call(_make_zs_prompt(rec["task_description"], tools_sims, tool_descs, k))
        zs_pred = _parse(zs_raw, set(cand_tools), k)
        rows.append({"method":"ts_llm_zs","idx":i,"raw":zs_raw[:100],
                     **compute_metrics(zs_pred, gt)})

        # ---- TS-Hybrid + LLM Few-Shot ----
        examples = get_few(len(gt))
        fs_raw   = _call(_make_fs_prompt(rec["task_description"], tools_sims,
                                         tool_descs, k, examples))
        fs_pred  = _parse(fs_raw, set(cand_tools), k)
        rows.append({"method":"ts_llm_fs","idx":i,"raw":fs_raw[:100],
                     **compute_metrics(fs_pred, gt)})

        if (i+1) % 30 == 0:
            agg = pd.DataFrame(rows)
            out = []
            for m in ["ts_hr","ts_lr","ts_llm_zs","ts_llm_fs"]:
                sub = agg[agg.method==m]
                if len(sub):
                    out.append(f"{m}: F1={sub.set_f1.mean():.3f} "
                               f"OP={sub.ordered_precision.mean():.3f}")
            print(f"  [{i+1}/{len(test_sample)}]  " + "  ".join(out))

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "ts_hr":     "TS-Hybrid + HR",
    "ts_lr":     "TS-Hybrid + LR (ours)",
    "ts_llm_zs": "TS-Hybrid + LLaMA3.1-8B (0-shot)",
    "ts_llm_fs": "TS-Hybrid + LLaMA3.1-8B (3-shot)",
}

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample",  type=int, default=300)
    p.add_argument("--few-k",   type=int, default=3)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()

def main():
    args = _parse_args()

    # Verify Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models",[])]
        if not any(MODEL_NAME in n for n in names):
            print(f"[ERROR] {MODEL_NAME} not found. Run: ollama pull {MODEL_NAME}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Ollama not reachable: {e}"); sys.exit(1)

    df = run(sample=args.sample, seed=args.seed, few_k=args.few_k)

    # Save
    raw_path = RESULTS_DIR / "llm_stage2_toolbench.csv"
    df.to_csv(raw_path, index=False)

    cols = ["set_f1","lcs_r","ordered_precision","kendall_tau",
            "transition_acc","first_tool_acc"]
    agg  = df.groupby("method")[cols].mean().round(4)
    agg.index = [METHOD_LABELS.get(m,m) for m in agg.index]
    agg = agg.sort_values("ordered_precision", ascending=False)

    cmp_path = RESULTS_DIR / "llm_stage2_comparison.csv"
    agg.to_csv(cmp_path)

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {cmp_path}")
    print("\n" + "="*70)
    print(f"  ToolBench Stage-2 Comparison (n={args.sample}, same Stage-1)")
    print("="*70)
    print(agg.to_string())
    print("="*70)

if __name__ == "__main__":
    main()
