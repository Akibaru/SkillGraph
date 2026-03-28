"""
src/extract_apibank.py  —  Extract API-Bank trajectories
=========================================================
Processes the API-Bank dataset (EMNLP 2023) from:
  data/raw/API-Bank/

Source files used
-----------------
  training-data/lv3-api-train.json  — per-step API calls, 907 entries (~529 queries)
  test-data/level-3.json            — complete multi-step trajectories, 50 entries

The lv3 (Plan+Retrieve+Call) level is used because it has multi-step tool
sequences that best match our framework's target task.

Output
------
  data/processed/apibank_train_trajectories.jsonl
  data/processed/apibank_test_trajectories.jsonl
  data/processed/apibank_all_trajectories.jsonl
  data/processed/apibank_tool_index.json
  data/processed/apibank_stats.json

Schema (same as ToolBench trajectories):
  task_id           : str
  task_description  : str
  tool_sequence     : list[str]
  tool_details      : list[{name, category, description, call_success}]
  num_steps         : int
  source_split      : str  (train / test)

Usage
-----
  python src/extract_apibank.py
  python src/extract_apibank.py --no-lv2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict

ROOT    = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "API-Bank"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Parse lv3-api-train.json  (per-step flat format)
# ---------------------------------------------------------------------------

def _parse_query(inp: str) -> str:
    """Extract user query from lv3-api-train input field."""
    # Pattern: "User: <query>\nTIME: ..." or "User: <query>TIME: ..."
    m = re.search(r"User:\s*(.+?)(?:\nTIME:|\nGenerate|TIME:)", inp, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: everything after "User:"
    m2 = re.search(r"User:\s*(.+)", inp)
    return m2.group(1).strip() if m2 else ""


def _parse_api_name(output: str) -> str:
    """Extract API name from output like 'API-Request: [ApiName(key=val)]'."""
    m = re.search(r"\[(\w+)\(", output)
    return m.group(1) if m else ""


def _extract_tool_desc_from_input(inp: str) -> dict[str, str]:
    """
    Some lv3-api-train inputs embed a JSON tool description after
    ToolSearcher output. Try to parse it.
    Returns {tool_name: description}.
    """
    descs: dict[str, str] = {}
    # Look for JSON with "name" and "description" keys
    for m in re.finditer(r'\{"name":\s*"([^"]+)"[^}]*"description":\s*"([^"]+)"', inp):
        descs[m.group(1)] = m.group(2)
    return descs


def extract_lv3_train(path: pathlib.Path) -> tuple[list[dict], dict[str, str]]:
    """
    Extract trajectories from lv3-api-train.json.
    Returns (records, tool_descriptions).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Parse each entry
    parsed: list[tuple[str, str]] = []   # (query, api_name)
    tool_descs: dict[str, str] = {}

    for item in raw:
        query   = _parse_query(item.get("input", ""))
        api     = _parse_api_name(item.get("output", ""))
        if not query or not api:
            continue
        parsed.append((query, api))
        # Try to extract description
        descs = _extract_tool_desc_from_input(item.get("input", ""))
        tool_descs.update(descs)

    # Group by query (preserving order of first occurrence)
    query_steps: dict[str, list[str]] = defaultdict(list)
    seen_order: list[str] = []
    for query, api in parsed:
        if query not in query_steps:
            seen_order.append(query)
        query_steps[query].append(api)

    records: list[dict] = []
    for query in seen_order:
        steps = query_steps[query]
        # Filter ToolSearcher
        tools = [s for s in steps if s != "ToolSearcher"]
        # Deduplicate consecutive
        deduped: list[str] = []
        for t in tools:
            if not deduped or t != deduped[-1]:
                deduped.append(t)
        if not deduped:
            continue
        rec = _build_record(query, deduped, tool_descs, "train")
        records.append(rec)

    return records, tool_descs


# ---------------------------------------------------------------------------
# Parse test-data/level-3.json  (complete trajectory format)
# ---------------------------------------------------------------------------

def extract_lv3_test(path: pathlib.Path) -> tuple[list[dict], dict[str, str]]:
    """
    Extract trajectories from level-3.json.
    Returns (records, tool_descriptions).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    tool_descs: dict[str, str] = {}
    records: list[dict] = []

    for item in raw:
        requirement = item.get("requirement", "").strip()
        if not requirement:
            continue

        apis = item.get("apis", [])

        # Extract tool descriptions from ToolSearcher outputs
        for a in apis:
            if a.get("api_name") == "ToolSearcher":
                out = a.get("output", {})
                if isinstance(out, dict):
                    inner = out.get("output", out)
                    if isinstance(inner, dict):
                        name = inner.get("name", "")
                        desc = inner.get("description", "")
                        if name and desc:
                            tool_descs[name] = desc

        # Build tool sequence (non-ToolSearcher calls)
        tools = [a["api_name"] for a in apis
                 if a.get("api_name") and a["api_name"] != "ToolSearcher"]
        # Deduplicate consecutive
        deduped: list[str] = []
        for t in tools:
            if not deduped or t != deduped[-1]:
                deduped.append(t)
        if not deduped:
            continue

        rec = _build_record(requirement, deduped, tool_descs, "test")
        records.append(rec)

    return records, tool_descs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_record(
    query:      str,
    tool_seq:   list[str],
    tool_descs: dict[str, str],
    split:      str,
) -> dict:
    tool_details = [
        {
            "name":         t,
            "category":     "",          # API-Bank has no category field
            "description":  tool_descs.get(t, ""),
            "call_success": True,
        }
        for t in dict.fromkeys(tool_seq)   # unique, preserve order
    ]
    return {
        "task_id":          query,
        "task_description": query,
        "tool_sequence":    tool_seq,
        "tool_details":     tool_details,
        "num_steps":        len(tool_seq),
        "source_split":     split,
    }


def _save_jsonl(records: list[dict], path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path}  ({len(records):,} records)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract API-Bank Level-3 trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-lv2", action="store_true",
                   help="Skip lv2 data (not currently used)")
    return p.parse_args()


def main() -> None:
    _parse_args()   # reserved for future flags

    lv3_train_path = RAW_DIR / "training-data" / "lv3-api-train.json"
    lv3_test_path  = RAW_DIR / "test-data"      / "level-3.json"

    # --- Training trajectories ---
    train_records: list[dict] = []
    all_tool_descs: dict[str, str] = {}

    if lv3_train_path.exists():
        recs, descs = extract_lv3_train(lv3_train_path)
        train_records.extend(recs)
        all_tool_descs.update(descs)
        print(f"[lv3-train]  {len(recs):,} trajectories extracted")
    else:
        print(f"[lv3-train]  NOT FOUND: {lv3_train_path}")

    # --- Test trajectories ---
    test_records: list[dict] = []
    if lv3_test_path.exists():
        recs, descs = extract_lv3_test(lv3_test_path)
        test_records.extend(recs)
        all_tool_descs.update(descs)
        print(f"[lv3-test]   {len(recs):,} trajectories extracted")
    else:
        print(f"[lv3-test]   NOT FOUND: {lv3_test_path}")

    # --- Save trajectories ---
    _save_jsonl(train_records,
                OUT_DIR / "apibank_train_trajectories.jsonl")
    _save_jsonl(test_records,
                OUT_DIR / "apibank_test_trajectories.jsonl")
    _save_jsonl(train_records + test_records,
                OUT_DIR / "apibank_all_trajectories.jsonl")

    # --- Tool index ---
    # Collect all tool names from both splits
    all_tools: set[str] = set()
    for r in train_records + test_records:
        all_tools.update(r["tool_sequence"])

    tool_index: dict[str, dict] = {}
    for name in sorted(all_tools):
        tool_index[name] = {
            "description": all_tool_descs.get(name, ""),
            "category":    "",
        }

    tool_index_path = OUT_DIR / "apibank_tool_index.json"
    with open(tool_index_path, "w", encoding="utf-8") as f:
        json.dump(tool_index, f, indent=2, ensure_ascii=False)
    print(f"  -> {tool_index_path}  ({len(tool_index):,} tools)")

    # --- Stats ---
    def _stats(records: list[dict]) -> dict:
        lens = [r["num_steps"] for r in records]
        if not lens:
            return {"n": 0}
        unique_tools: set[str] = set()
        for r in records:
            unique_tools.update(r["tool_sequence"])
        return {
            "n_trajectories": len(records),
            "unique_tools":   len(unique_tools),
            "mean_len":       round(sum(lens) / len(lens), 2),
            "min_len":        min(lens),
            "max_len":        max(lens),
        }

    stats = {
        "train": _stats(train_records),
        "test":  _stats(test_records),
        "total": _stats(train_records + test_records),
    }

    stats_path = OUT_DIR / "apibank_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  -> {stats_path}")

    # --- Summary ---
    print("\n" + "=" * 55)
    print("  API-Bank Extraction Summary")
    print("=" * 55)
    for split, s in stats.items():
        if s.get("n_trajectories", 0) == 0:
            continue
        print(f"\n  {split.upper()}:")
        print(f"    trajectories : {s['n_trajectories']:,}")
        print(f"    unique tools : {s['unique_tools']:,}")
        print(f"    seq len      : {s['min_len']}–{s['max_len']}  "
              f"mean={s['mean_len']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
