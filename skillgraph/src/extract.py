"""
src/extract.py  —  SkillGraph pipeline Step 1
==============================================
Extracts successful trajectories from the ToolBench flat train/eval JSON files.

Real data schema (toolllama_G123_dfs_train.json)
-------------------------------------------------
List of dicts, each:
  {
    "id": "Step N: <task description>",
    "conversations": [
      {"from": "system",    "value": "...Specifically, you have access to the following APIs: [...]..."},
      {"from": "user",      "value": "<task query>"},
      {"from": "assistant", "value": "Thought: ...\nAction: <tool_name>\nAction Input: {...}"},
      {"from": "function",  "value": "{\"error\": \"...\", \"response\": \"...\"}"},
      ...
      {"from": "assistant", "value": "...Action: Finish\nAction Input: {\"return_type\": \"give_answer\", ...}"}
    ]
  }

Tool names follow pattern:  <endpoint>_for_<category>
Trajectory success:         last assistant turn contains "give_answer"  (not "give_up_and_restart")
Call success:               function turn's "error" field is empty string

Outputs
-------
  data/processed/successful_trajectories.jsonl   — one JSON record per line
  data/processed/tool_metadata.json              — tool registry
  outputs/figures/seq_length_dist.png            — histogram
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ToolBench_data" / "data"
OUT_DIR  = ROOT / "data" / "processed"
FIG_DIR  = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSON = DATA_DIR / "toolllama_G123_dfs_train.json"
EVAL_JSON  = DATA_DIR / "toolllama_G123_dfs_eval.json"

TRAJ_OUT   = OUT_DIR / "successful_trajectories.jsonl"
META_OUT   = OUT_DIR / "tool_metadata.json"
HIST_OUT   = FIG_DIR / "seq_length_dist.png"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_TOOL_LIST_RE = re.compile(
    r"Specifically, you have access to the following APIs:\s*(\[.+\])",
    re.DOTALL,
)
_ACTION_RE      = re.compile(r"^Action:\s*(.+)$",       re.MULTILINE)
_ACT_INPUT_RE   = re.compile(r"^Action Input:\s*(.+)",  re.MULTILINE | re.DOTALL)


def _parse_tool_registry(system_msg: str) -> dict[str, dict]:
    """
    Parse the Python-literal tool list embedded in the system message.
    Returns {tool_name: {"description": str, "category": str}}.
    Falls back to empty dict on parse error.
    """
    m = _TOOL_LIST_RE.search(system_msg)
    if not m:
        return {}
    raw = m.group(1).strip()
    try:
        tools: list[dict] = ast.literal_eval(raw)
    except Exception:
        return {}

    registry: dict[str, dict] = {}
    for t in tools:
        name = t.get("name", "")
        if not name or name == "Finish":
            continue
        # category: everything after last '_for_'
        parts = name.rsplit("_for_", 1)
        category = parts[1] if len(parts) == 2 else ""
        desc_raw = t.get("description", "")
        # Strip boilerplate prefix from description
        desc = re.sub(
            r'This is the subfunction for tool "[^"]+", you can use this tool\.'
            r'The description of this function is: "?',
            "",
            desc_raw,
        ).strip().rstrip('"')
        registry[name] = {"description": desc, "category": category}
    return registry


def _parse_action(assistant_value: str) -> str | None:
    """
    Extract tool name from an assistant turn.
    Returns None if the action is 'Finish' or not found.
    """
    m = _ACTION_RE.search(assistant_value)
    if not m:
        return None
    name = m.group(1).strip()
    return None if name == "Finish" else name


def _call_succeeded(function_value: str) -> bool:
    """True if the function turn's 'error' field is empty."""
    try:
        parsed = json.loads(function_value)
        return parsed.get("error", "X") == ""
    except Exception:
        return False


def _traj_succeeded(conversations: list[dict]) -> bool:
    """
    True if the last assistant turn ends with 'give_answer'
    (i.e., not 'give_up_and_restart').
    """
    for turn in reversed(conversations):
        if turn.get("from") == "assistant":
            val = turn.get("value", "") or ""
            if "give_answer" in val:
                return True
            if "give_up_and_restart" in val:
                return False
    return False


def _extract_task_description(sample_id: str) -> str:
    """Strip the 'Step N: ' prefix from the id field."""
    return re.sub(r"^Step\s+\d+:\s*", "", sample_id).strip()


def process_sample(sample: dict) -> dict | None:
    """
    Parse one sample. Returns a trajectory record or None if unsuccessful /
    malformed.
    """
    try:
        conversations = sample.get("conversations", [])
        if not conversations:
            return None

        # Trajectory-level success filter
        if not _traj_succeeded(conversations):
            return None

        task_id   = sample.get("id", "")
        task_desc = _extract_task_description(task_id)

        # Tool registry from the system message
        sys_msg  = conversations[0].get("value", "") if conversations[0].get("from") == "system" else ""
        registry = _parse_tool_registry(sys_msg)

        # Walk conversation turns to extract ordered tool calls
        tool_sequence: list[str]   = []
        tool_details:  list[dict]  = []

        i = 0
        while i < len(conversations):
            turn = conversations[i]

            if turn.get("from") == "assistant":
                tool_name = _parse_action(turn.get("value", "") or "")
                if tool_name:
                    # Peek ahead for the matching function result
                    call_ok = False
                    if i + 1 < len(conversations) and conversations[i + 1].get("from") == "function":
                        call_ok = _call_succeeded(conversations[i + 1].get("value", "") or "")

                    tool_info = registry.get(tool_name, {})
                    tool_sequence.append(tool_name)
                    tool_details.append({
                        "name":         tool_name,
                        "category":     tool_info.get("category", ""),
                        "description":  tool_info.get("description", ""),
                        "call_success": call_ok,
                    })
            i += 1

        if not tool_sequence:
            return None

        return {
            "task_id":          task_id,
            "task_description": task_desc,
            "tool_sequence":    tool_sequence,
            "tool_details":     tool_details,
            "num_steps":        len(tool_sequence),
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_file(path: pathlib.Path) -> tuple[list[dict], Counter]:
    """Load one JSON file and return (records, tool_counter)."""
    if not path.exists():
        print(f"  [skip] not found: {path}")
        return [], Counter()

    print(f"\nLoading {path.name}  ({path.stat().st_size / 1e6:.0f} MB) …")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        print(f"  [skip] expected a list, got {type(data)}")
        return [], Counter()

    records: list[dict] = []
    tool_counter: Counter = Counter()

    for sample in tqdm(data, desc=path.stem, unit="traj"):
        rec = process_sample(sample)
        if rec is not None:
            records.append(rec)
            tool_counter.update(rec["tool_sequence"])

    success_rate = len(records) / len(data) * 100 if data else 0
    print(f"  {len(records):,} / {len(data):,} successful  ({success_rate:.1f}%)")
    return records, tool_counter


# ---------------------------------------------------------------------------
# Tool metadata registry builder
# ---------------------------------------------------------------------------

def build_tool_metadata(
    records: list[dict],
    tool_counter: Counter,
) -> dict[str, dict]:
    """
    Merge per-tool info across all trajectories into a unified registry.
    Returns {tool_name: {category, description, frequency}}.
    """
    meta: dict[str, dict] = {}
    for rec in records:
        for td in rec["tool_details"]:
            name = td["name"]
            if name not in meta:
                meta[name] = {
                    "category":    td["category"],
                    "description": td["description"],
                    "frequency":   0,
                }
            # Prefer non-empty description / category
            if not meta[name]["description"] and td["description"]:
                meta[name]["description"] = td["description"]
            if not meta[name]["category"] and td["category"]:
                meta[name]["category"] = td["category"]
    # Attach frequencies
    for name, count in tool_counter.items():
        if name in meta:
            meta[name]["frequency"] = count
    return meta


# ---------------------------------------------------------------------------
# Statistics & figure
# ---------------------------------------------------------------------------

def print_stats(
    records: list[dict],
    tool_meta: dict[str, dict],
    tool_counter: Counter,
) -> None:
    lengths = [r["num_steps"] for r in records]

    print("\n" + "=" * 60)
    print("SkillGraph — Extraction Statistics")
    print("=" * 60)
    print(f"  Successful trajectories : {len(records):,}")
    print(f"  Unique tools            : {len(tool_meta):,}")
    print(f"  Total tool calls        : {sum(lengths):,}")
    print(f"  Avg sequence length     : {mean(lengths):.2f}")
    print(f"  Median sequence length  : {median(lengths):.1f}")
    print(f"  Max sequence length     : {max(lengths)}")
    print(f"  Min sequence length     : {min(lengths)}")

    print(f"\n  Top 20 most-used tools:")
    for tool, cnt in tool_counter.most_common(20):
        cat = tool_meta.get(tool, {}).get("category", "?")
        print(f"    {cnt:6,}  {tool}  [{cat}]")


def save_histogram(records: list[dict]) -> None:
    lengths = [r["num_steps"] for r in records]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=range(1, min(max(lengths) + 2, 51)), color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Tool-call sequence length (# API calls per trajectory)")
    ax.set_ylabel("Number of trajectories")
    ax.set_title(f"ToolBench: Successful Trajectory Length Distribution  (n={len(records):,})")
    ax.axvline(x=sum(lengths) / len(lengths), color="red", linestyle="--",
               label=f"mean = {sum(lengths)/len(lengths):.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HIST_OUT, dpi=150)
    plt.close(fig)
    print(f"\n  Histogram saved → {HIST_OUT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    all_records:     list[dict] = []
    combined_counter: Counter   = Counter()

    for path in [TRAIN_JSON, EVAL_JSON]:
        recs, ctr = process_file(path)
        all_records.extend(recs)
        combined_counter.update(ctr)

    if not all_records:
        print("\nNo records extracted. Check that the data files exist at:")
        print(f"  {TRAIN_JSON}")
        print(f"  {EVAL_JSON}")
        sys.exit(1)

    # Deduplicate by task_id (keep first occurrence)
    seen: set[str] = set()
    deduped: list[dict] = []
    for rec in all_records:
        tid = rec["task_id"]
        if tid not in seen:
            seen.add(tid)
            deduped.append(rec)
    if len(deduped) < len(all_records):
        print(f"\n  Deduplication: {len(all_records):,} → {len(deduped):,} records")
    all_records = deduped

    # Build tool metadata
    tool_meta = build_tool_metadata(all_records, combined_counter)

    # Save trajectories as JSONL
    print(f"\nWriting {len(all_records):,} records → {TRAJ_OUT}")
    with TRAJ_OUT.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save tool metadata
    print(f"Writing {len(tool_meta):,} tools → {META_OUT}")
    META_OUT.write_text(
        json.dumps(tool_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Statistics + figure
    print_stats(all_records, tool_meta, combined_counter)
    save_histogram(all_records)

    print("\nDone.")


if __name__ == "__main__":
    main()
