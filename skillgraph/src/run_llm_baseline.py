"""
src/run_llm_baseline.py  —  LLM Baseline Evaluation (LLaMA 3.1 8B)
====================================================================
Evaluates LLaMA 3.1 8B (via Ollama) as a zero-shot and few-shot
tool recommendation baseline on both ToolBench and API-Bank test sets.

Protocol
--------
  Zero-shot:  Only tool descriptions in the prompt. Ask model to predict
              the ordered tool sequence needed for the query.
  Few-shot:   Provide 3 examples from training set (same format as test).

Both settings:
  - Oracle K: model is told how many tools to output (= len(GT))
    OR unconstrained version (model decides K)
  - Constrained: tools must come from the provided candidate list

We use two variants:
  zero_shot_constrained   : tool list provided, oracle K told
  few_shot_constrained    : 3 training examples + tool list

Outputs
-------
  results/llm_baseline_toolbench.csv
  results/llm_baseline_apibank.csv
  results/llm_baseline_summary.csv

Usage
-----
  python src/run_llm_baseline.py --dataset apibank
  python src/run_llm_baseline.py --dataset toolbench --sample 500
  python src/run_llm_baseline.py --dataset both --sample 200
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

ROOT        = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "llama3.1:8b"


# ---------------------------------------------------------------------------
# Metrics (self-contained)
# ---------------------------------------------------------------------------

def _f1(pred: list[str], gt: list[str]) -> float:
    ps, gs = set(pred), set(gt)
    tp = len(ps & gs)
    p = tp / max(1, len(ps))
    r = tp / max(1, len(gs))
    return 2 * p * r / max(1e-9, p + r)


def _ordered_precision(pred: list[str], gt: list[str]) -> float:
    gt_set  = set(gt)
    gt_rank = {t: i for i, t in enumerate(gt)}
    common  = [(t, i) for i, t in enumerate(pred) if t in gt_set]
    if len(common) < 2:
        return 0.0
    mp = tp = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            tp += 1
            if gt_rank[common[i][0]] < gt_rank[common[j][0]]:
                mp += 1
    return mp / tp if tp else 0.0


def _kendall_tau(pred: list[str], gt: list[str]) -> float:
    common = [t for t in pred if t in set(gt)]
    if len(common) < 2:
        return 0.0
    gt_rank = {t: i for i, t in enumerate(gt)}
    ranks   = [gt_rank[t] for t in common]
    n = len(ranks)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = ranks[i] - ranks[j]
            if diff < 0:
                c += 1
            elif diff > 0:
                d += 1
    return (c - d) / (n * (n - 1) / 2)


def _transition_acc(pred: list[str], gt: list[str]) -> float:
    if len(gt) < 2:
        return 0.0
    pairs = [(gt[i], gt[i + 1]) for i in range(len(gt) - 1)]
    pos   = {t: i for i, t in enumerate(pred)}
    return sum(
        1 for a, b in pairs
        if a in pos and b in pos and 0 < pos[b] - pos[a] <= 2
    ) / len(pairs)


def _first_acc(pred: list[str], gt: list[str]) -> float:
    return float(bool(pred) and bool(gt) and pred[0] == gt[0])


def compute_metrics(pred: list[str], gt: list[str]) -> dict:
    return {
        "set_f1":            round(_f1(pred, gt), 4),
        "ordered_precision": round(_ordered_precision(pred, gt), 4),
        "kendall_tau":       round(_kendall_tau(pred, gt), 4),
        "transition_acc":    round(_transition_acc(pred, gt), 4),
        "first_tool_acc":    round(_first_acc(pred, gt), 4),
        "pred_len":          len(pred),
        "gt_len":            len(gt),
    }


# ---------------------------------------------------------------------------
# Ollama interface
# ---------------------------------------------------------------------------

def _check_ollama() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(MODEL_NAME in m for m in models):
            print(f"[WARN] Model {MODEL_NAME} not found in Ollama. Available: {models}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Ollama not reachable: {e}")
        return False


def _call_ollama(prompt: str, max_retries: int = 3, timeout: int = 120) -> str:
    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":   0.0,
            "top_p":         1.0,
            "num_predict":   256,
        },
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                return ""
            time.sleep(2)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [WARN] Ollama call failed: {e}")
                return ""
            time.sleep(1)
    return ""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _tool_list_str(tool_descs: dict[str, str]) -> str:
    lines = []
    for i, (name, desc) in enumerate(sorted(tool_descs.items()), 1):
        lines.append(f"  {i}. {name}: {desc}")
    return "\n".join(lines)


def _example_str(record: dict) -> str:
    """Format one training example for few-shot prompt."""
    query = record["task_description"]
    seq   = record["tool_sequence"]
    return f'Task: "{query}"\nAnswer: {" -> ".join(seq)}'


def build_zero_shot_prompt(
    query:      str,
    tool_descs: dict[str, str],
    k:          int,
) -> str:
    tool_list = _tool_list_str(tool_descs)
    return f"""You are a tool-use planner. Given a user task, select and order exactly {k} tools from the list below to complete the task. Output ONLY the tool names in execution order, separated by " -> ". Do not explain.

Available tools:
{tool_list}

Task: "{query}"
Answer:"""


def build_few_shot_prompt(
    query:        str,
    tool_descs:   dict[str, str],
    k:            int,
    examples:     list[dict],
) -> str:
    tool_list = _tool_list_str(tool_descs)
    ex_block  = "\n\n".join(_example_str(e) for e in examples)
    return f"""You are a tool-use planner. Given a user task, select and order exactly {k} tools from the list below to complete the task. Output ONLY the tool names in execution order, separated by " -> ". Do not explain.

Available tools:
{tool_list}

Here are {len(examples)} examples:

{ex_block}

Now answer:
Task: "{query}"
Answer:"""


# ---------------------------------------------------------------------------
# Parse LLM output
# ---------------------------------------------------------------------------

def _parse_tool_sequence(
    raw:        str,
    valid_tools: set[str],
    k:           int,
) -> list[str]:
    """
    Extract tool sequence from LLM output.
    Accepts: "ToolA -> ToolB -> ToolC" or numbered lists or comma-separated.
    Returns at most k tools that are in valid_tools.
    """
    if not raw:
        return []

    # Try arrow-separated first
    if "->" in raw:
        parts = [p.strip() for p in raw.split("->")]
    elif "→" in raw:
        parts = [p.strip() for p in raw.split("→")]
    elif "\n" in raw:
        parts = [re.sub(r"^\d+[\.\)]\s*", "", l).strip()
                 for l in raw.strip().split("\n") if l.strip()]
    else:
        parts = [p.strip() for p in re.split(r"[,;]", raw)]

    # Clean each part
    cleaned = []
    for p in parts:
        # Remove numbering, quotes, brackets
        p = re.sub(r"^[\d\.\)\-\s]+", "", p).strip()
        p = p.strip('"\'`[]()').strip()
        if p in valid_tools:
            cleaned.append(p)
        else:
            # Fuzzy match: try case-insensitive
            match = next((t for t in valid_tools if t.lower() == p.lower()), None)
            if match:
                cleaned.append(match)

    # Deduplicate preserving order
    seen: set[str] = set()
    result = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            result.append(t)
        if len(result) == k:
            break

    return result


# ---------------------------------------------------------------------------
# API-Bank evaluation
# ---------------------------------------------------------------------------

def eval_apibank(
    few_shot_k: int = 3,
    seed:       int = 42,
) -> pd.DataFrame:
    """Evaluate on API-Bank level-3 (LOO-CV style, same as run_apibank_eval.py)."""

    # Load data
    lv3_path = ROOT / "data" / "raw" / "API-Bank" / "test-data" / "level-3.json"
    with open(lv3_path, encoding="utf-8") as f:
        lv3 = json.load(f)

    # Extract records and tool descriptions
    tool_descs: dict[str, str] = {}
    records: list[dict] = []
    for item in lv3:
        req  = item.get("requirement", "").strip()
        if not req:
            continue
        apis = item.get("apis", [])
        for a in apis:
            if a.get("api_name") == "ToolSearcher":
                out = a.get("output", {})
                inner = (out.get("output", {}) if isinstance(out, dict) else {})
                if isinstance(inner, dict) and "name" in inner:
                    nm = inner["name"]
                    tool_descs[nm] = inner.get("description", nm)
        tools = [a["api_name"] for a in apis if a.get("api_name") and
                 a["api_name"] != "ToolSearcher"]
        deduped: list[str] = []
        for t in tools:
            if not deduped or t != deduped[-1]:
                deduped.append(t)
        if deduped:
            records.append({"task_description": req, "tool_sequence": deduped})

    valid_tools = set(tool_descs.keys())
    rng = random.Random(seed)

    rows: list[dict] = []
    for i, rec in enumerate(records):
        query = rec["task_description"]
        gt    = rec["tool_sequence"]
        k     = len(gt)

        # LOO training data for few-shot examples
        loo_train = [records[j] for j in range(len(records)) if j != i]
        examples  = rng.sample(loo_train, min(few_shot_k, len(loo_train)))

        # Zero-shot
        zs_prompt = build_zero_shot_prompt(query, tool_descs, k)
        zs_raw    = _call_ollama(zs_prompt)
        zs_pred   = _parse_tool_sequence(zs_raw, valid_tools, k)
        rows.append({"method": "llm_zero_shot", "idx": i,
                     "raw_output": zs_raw,
                     **compute_metrics(zs_pred, gt)})

        # Few-shot
        fs_prompt = build_few_shot_prompt(query, tool_descs, k, examples)
        fs_raw    = _call_ollama(fs_prompt)
        fs_pred   = _parse_tool_sequence(fs_raw, valid_tools, k)
        rows.append({"method": "llm_few_shot", "idx": i,
                     "raw_output": fs_raw,
                     **compute_metrics(fs_pred, gt)})

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(records)}]  last_zs: {zs_pred}  gt: {gt}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ToolBench evaluation (sampled)
# ---------------------------------------------------------------------------

def eval_toolbench(
    sample:     int  = 300,
    few_shot_k: int  = 3,
    seed:       int  = 42,
) -> pd.DataFrame:
    """Evaluate on ToolBench test split (sampled)."""
    import json

    proc_dir   = ROOT / "data" / "processed"
    traj_file  = proc_dir / "successful_trajectories.jsonl"
    split_file = proc_dir / "train_test_split.json"
    meta_file  = proc_dir / "tool_metadata.json"

    # Load all records
    all_records: list[dict] = []
    with open(traj_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    # Load split
    split = json.loads(split_file.read_text(encoding="utf-8"))
    train_idx = set(split["train_idx"])
    test_idx  = set(split["test_idx"])
    train_records = [all_records[i] for i in range(len(all_records)) if i in train_idx]
    test_records  = [all_records[i] for i in range(len(all_records)) if i in test_idx]

    # Sample test
    rng = random.Random(seed)
    test_sample = rng.sample(test_records, min(sample, len(test_records)))

    # Load tool descriptions
    tool_descs: dict[str, str] = {}
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        for t, info in meta.items():
            if isinstance(info, dict):
                tool_descs[t] = info.get("description", t)
            else:
                tool_descs[t] = str(info)
    else:
        # Fall back to extracting from trajectories
        for r in all_records:
            for td in r.get("tool_details", []):
                name = td.get("name", "")
                desc = td.get("description", "")
                if name and name not in tool_descs:
                    tool_descs[name] = desc or name

    # ToolBench has 18k+ tools — too many to list in prompt.
    # Use oracle candidate set: GT tools + 10 negatives
    rows: list[dict] = []

    # Build per-sample few-shot examples from train
    def get_few_shot(tgt_len: int) -> list[dict]:
        """Sample train examples with similar length."""
        similar = [r for r in train_records if abs(r["num_steps"] - tgt_len) <= 1]
        pool    = similar if similar else train_records
        return rng.sample(pool, min(few_shot_k, len(pool)))

    print(f"ToolBench: evaluating {len(test_sample)} test samples ...")
    for i, rec in enumerate(test_sample):
        query = rec["task_description"]
        gt    = rec["tool_sequence"]
        k     = len(gt)

        # Build a small candidate set: GT + 10 random negatives
        all_tools    = list(tool_descs.keys())
        neg_pool     = [t for t in all_tools if t not in set(gt)]
        negatives    = rng.sample(neg_pool, min(10, len(neg_pool)))
        cand_tools   = gt + negatives
        rng.shuffle(cand_tools)
        cand_descs   = {t: tool_descs.get(t, t) for t in cand_tools}

        examples  = get_few_shot(k)

        # Zero-shot
        zs_prompt = build_zero_shot_prompt(query, cand_descs, k)
        zs_raw    = _call_ollama(zs_prompt)
        zs_pred   = _parse_tool_sequence(zs_raw, set(cand_tools), k)
        rows.append({"method": "llm_zero_shot", "idx": i,
                     "raw_output": zs_raw,
                     **compute_metrics(zs_pred, gt)})

        # Few-shot
        fs_prompt = build_few_shot_prompt(query, cand_descs, k, examples)
        fs_raw    = _call_ollama(fs_prompt)
        fs_pred   = _parse_tool_sequence(fs_raw, set(cand_tools), k)
        rows.append({"method": "llm_few_shot", "idx": i,
                     "raw_output": fs_raw,
                     **compute_metrics(fs_pred, gt)})

        if (i + 1) % 20 == 0:
            agg = pd.DataFrame(rows)
            zs_f1 = agg[agg.method=="llm_zero_shot"]["set_f1"].mean()
            fs_f1 = agg[agg.method=="llm_few_shot"]["set_f1"].mean()
            print(f"  [{i+1}/{len(test_sample)}] ZS F1={zs_f1:.3f}  FS F1={fs_f1:.3f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "llm_zero_shot": "LLaMA3.1-8B (Zero-Shot)",
    "llm_few_shot":  "LLaMA3.1-8B (3-Shot)",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM baseline evaluation using LLaMA3.1-8B via Ollama",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",    choices=["apibank", "toolbench", "both"],
                   default="both")
    p.add_argument("--sample",     type=int,  default=300,
                   help="Number of ToolBench test samples")
    p.add_argument("--few-shot-k", type=int,  default=3,
                   help="Number of few-shot examples")
    p.add_argument("--seed",       type=int,  default=42)
    return p.parse_args()


def _print_summary(df: pd.DataFrame, title: str) -> None:
    cols = ["set_f1", "ordered_precision", "kendall_tau",
            "transition_acc", "first_tool_acc"]
    agg  = df.groupby("method")[cols].mean().round(4)
    agg.index = [METHOD_LABELS.get(m, m) for m in agg.index]
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(agg.to_string())
    print(f"{'='*60}")


def main() -> None:
    args = _parse_args()

    if not _check_ollama():
        print("Please start Ollama and ensure llama3.1:8b is pulled.")
        print("Run: ollama pull llama3.1:8b")
        sys.exit(1)

    if args.dataset in ("apibank", "both"):
        print(f"\n=== API-Bank LOO-CV (n=50) ===")
        df_ab = eval_apibank(few_shot_k=args.few_shot_k, seed=args.seed)
        out   = RESULTS_DIR / "llm_baseline_apibank.csv"
        df_ab.to_csv(out, index=False)
        print(f"Saved: {out}")
        _print_summary(df_ab, "API-Bank — LLM Baseline")

    if args.dataset in ("toolbench", "both"):
        print(f"\n=== ToolBench (n={args.sample}) ===")
        df_tb = eval_toolbench(sample=args.sample,
                               few_shot_k=args.few_shot_k,
                               seed=args.seed)
        out   = RESULTS_DIR / "llm_baseline_toolbench.csv"
        df_tb.to_csv(out, index=False)
        print(f"Saved: {out}")
        _print_summary(df_tb, "ToolBench — LLM Baseline")

    # Combined summary
    frames: list[pd.DataFrame] = []
    if args.dataset in ("apibank", "both"):
        df_ab["dataset"] = "apibank"
        frames.append(df_ab)
    if args.dataset in ("toolbench", "both"):
        df_tb["dataset"] = "toolbench"
        frames.append(df_tb)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        cols = ["set_f1", "ordered_precision", "kendall_tau",
                "transition_acc", "first_tool_acc"]
        summary = combined.groupby(["dataset", "method"])[cols].mean().round(4)
        summary_path = RESULTS_DIR / "llm_baseline_summary.csv"
        summary.to_csv(summary_path)
        print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
