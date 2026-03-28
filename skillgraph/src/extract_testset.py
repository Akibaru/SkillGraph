"""
src/extract_testset.py  —  Extract ToolBench G1/G2/G3 test trajectories
==========================================================================
Processes the per-query answer files in data/raw/ToolBench_data/data/answer/
(G1_answer, G2_answer, G3_answer) and extracts successful tool-use trajectories,
producing a second evaluation dataset separate from the main training corpus.

Rationale
---------
The main dataset (successful_trajectories.jsonl) came from:
  toolllama_G123_dfs_train.json (187k DFS conversation chains)
  toolllama_G123_dfs_eval.json  (762 entries)

The G1/G2/G3 answer directories contain per-query DFS answer files from
ToolBench's official test evaluation pipeline — these are structurally
different from the train conversations and correspond to the ToolBench
benchmark test queries.

Output
------
  data/processed/test_trajectories_G1.jsonl
  data/processed/test_trajectories_G2.jsonl
  data/processed/test_trajectories_G3.jsonl
  data/processed/test_trajectories_all.jsonl   (concatenation of G1+G2+G3)
  data/processed/test_split_stats.json

Same schema as successful_trajectories.jsonl:
  task_id           : str  (query text used as ID)
  task_description  : str  (query text)
  tool_sequence     : list[str]
  tool_details      : list[{name, category, description, call_success}]
  num_steps         : int
  source_split      : str  (G1 / G2 / G3)

Usage
-----
  python src/extract_testset.py
  python src/extract_testset.py --splits G3        # only G3
  python src/extract_testset.py --max-per-split 5000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

ROOT     = pathlib.Path(__file__).resolve().parent.parent
ANS_BASE = ROOT / "data" / "raw" / "ToolBench_data" / "data" / "answer"
OUT_DIR  = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DIRS = {"G1": ANS_BASE / "G1_answer",
              "G2": ANS_BASE / "G2_answer",
              "G3": ANS_BASE / "G3_answer"}


# ---------------------------------------------------------------------------
# Tool-name normalisation (mirrors extract.py logic)
# ---------------------------------------------------------------------------

def _tool_name(func_name: str) -> str:
    """Return the function name as-is (ToolBench already uses snake_case)."""
    return func_name.strip()


def _category(func_name: str) -> str:
    """Extract category from 'endpoint_for_category' names."""
    parts = func_name.rsplit("_for_", 1)
    return parts[1] if len(parts) == 2 else ""


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------

def _extract_one(path: pathlib.Path) -> dict | None:
    """
    Return a trajectory dict if the file represents a successful trajectory,
    else None.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    ag = data.get("answer_generation", {})
    if not ag.get("valid_data", False):
        return None
    if ag.get("finish_type") != "give_answer":
        return None

    query_raw = ag.get("query", "")
    query = query_raw.strip() if isinstance(query_raw, str) else ""
    if not query:
        return None

    # Extract tool call sequence
    raw_funcs = ag.get("function", [])
    tool_seq: list[str] = []
    for func in raw_funcs:
        name = func.get("name", "")
        if not name or name == "Finish":
            continue
        tool_seq.append(_tool_name(name))

    # Deduplicate consecutive repeats
    deduped: list[str] = []
    for t in tool_seq:
        if not deduped or t != deduped[-1]:
            deduped.append(t)

    if not deduped:
        return None

    # Build tool_details (best-effort from function records)
    seen: set[str] = set()
    tool_details: list[dict] = []
    for func in raw_funcs:
        name = func.get("name", "")
        if not name or name == "Finish" or name in seen:
            continue
        seen.add(name)
        tool_details.append({
            "name":         _tool_name(name),
            "category":     _category(name),
            "description":  func.get("description", ""),
            "call_success": True,   # we only keep give_answer trajectories
        })

    return {
        "task_id":         query,
        "task_description": query,
        "tool_sequence":   deduped,
        "tool_details":    tool_details,
        "num_steps":       len(deduped),
    }


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_split(
    split:         str,
    ans_dir:       pathlib.Path,
    max_samples:   int = 0,
    existing_queries: set[str] | None = None,
) -> list[dict]:
    """Extract successful trajectories from one answer directory."""
    files = sorted(ans_dir.iterdir())
    print(f"\n[{split}]  {len(files):,} answer files -> extracting ...")

    records: list[dict] = []
    skipped_overlap = 0
    skipped_fail    = 0

    for i, fpath in enumerate(files):
        if max_samples and len(records) >= max_samples:
            break
        if not fpath.suffix == ".json":
            continue

        rec = _extract_one(fpath)
        if rec is None:
            skipped_fail += 1
            continue

        # Overlap check against existing data
        if existing_queries and rec["task_description"] in existing_queries:
            skipped_overlap += 1
            continue

        rec["source_split"] = split
        records.append(rec)

        if (i + 1) % 5000 == 0:
            print(f"  [{split}]  processed {i+1:,} files  extracted={len(records):,}")

    print(f"  [{split}]  extracted={len(records):,}  "
          f"skipped_fail={skipped_fail:,}  skipped_overlap={skipped_overlap:,}")
    return records


# ---------------------------------------------------------------------------
# Tool coverage analysis
# ---------------------------------------------------------------------------

def coverage_report(
    new_records:   list[dict],
    known_tools:   set[str],
) -> dict:
    new_tools  = set(t for r in new_records for t in r["tool_sequence"])
    covered    = new_tools & known_tools
    uncovered  = new_tools - known_tools
    pct        = 100 * len(covered) / max(1, len(new_tools))
    return {
        "new_unique_tools":    len(new_tools),
        "covered_by_existing": len(covered),
        "uncovered":           len(uncovered),
        "coverage_pct":        round(pct, 1),
        "uncovered_examples":  sorted(uncovered)[:10],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract ToolBench G1/G2/G3 test trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--splits",         nargs="+", default=["G1", "G2", "G3"],
                   choices=["G1", "G2", "G3"])
    p.add_argument("--max-per-split",  type=int, default=0,
                   help="Max trajectories per split (0=all)")
    p.add_argument("--no-overlap-check", action="store_true",
                   help="Skip overlap check against existing training data")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load existing query texts for overlap check
    existing_queries: set[str] = set()
    if not args.no_overlap_check:
        main_traj = OUT_DIR / "successful_trajectories.jsonl"
        if main_traj.exists():
            print("Loading existing training queries for overlap check ...")
            with open(main_traj, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    existing_queries.add(rec["task_description"].strip())
            print(f"  Loaded {len(existing_queries):,} existing queries")

    # Load existing tool set for coverage analysis
    tool_index_path = OUT_DIR / "tool_index.json"
    known_tools: set[str] = set()
    if tool_index_path.exists():
        with open(tool_index_path, encoding="utf-8") as f:
            ti = json.load(f)
        known_tools = set(ti.keys()) if isinstance(ti, dict) else set(ti)
        print(f"Existing tool library: {len(known_tools):,} tools")

    # Extract each split
    all_records: list[dict]       = []
    stats: dict[str, dict]        = {}
    per_split: dict[str, list]    = {}

    for split in args.splits:
        ans_dir = SPLIT_DIRS.get(split)
        if ans_dir is None or not ans_dir.exists():
            print(f"[{split}]  directory not found: {ans_dir}  SKIP")
            continue

        recs = extract_split(split, ans_dir,
                             max_samples=args.max_per_split,
                             existing_queries=existing_queries)
        per_split[split] = recs
        all_records.extend(recs)

        # Per-split stats
        lens = [r["num_steps"] for r in recs]
        cov  = coverage_report(recs, known_tools) if known_tools else {}
        stats[split] = {
            "n_trajectories": len(recs),
            "mean_len":       round(sum(lens) / max(1, len(lens)), 2),
            "min_len":        min(lens) if lens else 0,
            "max_len":        max(lens) if lens else 0,
            **cov,
        }

    # Save per-split JSONL
    for split, recs in per_split.items():
        out = OUT_DIR / f"test_trajectories_{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {out}  ({len(recs):,} records)")

    # Save combined JSONL
    combined_out = OUT_DIR / "test_trajectories_all.jsonl"
    with open(combined_out, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {combined_out}  ({len(all_records):,} records total)")

    # Save stats
    stats["combined"] = {
        "n_trajectories": len(all_records),
        "splits":         list(per_split.keys()),
    }
    stats_out = OUT_DIR / "test_split_stats.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  -> {stats_out}")

    # Print summary
    print("\n" + "=" * 60)
    print("  Extraction Summary")
    print("=" * 60)
    for split, s in stats.items():
        if split == "combined":
            continue
        print(f"\n  {split}:")
        print(f"    trajectories  : {s['n_trajectories']:,}")
        print(f"    seq len       : {s['min_len']}–{s['max_len']}  mean={s['mean_len']}")
        if "coverage_pct" in s:
            print(f"    tool coverage : {s['coverage_pct']}%  "
                  f"({s['covered_by_existing']}/{s['new_unique_tools']} known)")
            if s["uncovered_examples"]:
                print(f"    uncovered ex  : {s['uncovered_examples'][:3]}")
    print(f"\n  Total: {len(all_records):,} trajectories")
    print("=" * 60)


if __name__ == "__main__":
    main()
