"""
src/inspect_data.py
====================
Inspect ToolBench trajectory data.

Loads the first N samples from whatever JSONL / JSON files it finds in
data/raw/, pretty-prints structure, counts statistics, and saves a summary
to outputs/data_summary.txt.

Usage:
    python src/inspect_data.py                   # default: first 10 samples
    python src/inspect_data.py --n 50            # first 50 samples
    python src/inspect_data.py --file data/raw/toolbench_default.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import textwrap
from collections import Counter, defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUT_DIR / "data_summary.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _type_label(v: Any) -> str:
    if isinstance(v, dict):
        return f"dict({len(v)} keys)"
    if isinstance(v, list):
        inner = type(v[0]).__name__ if v else "empty"
        return f"list[{inner}] len={len(v)}"
    return type(v).__name__


def describe_structure(obj: Any, indent: int = 0, max_depth: int = 5) -> list[str]:
    """Recursively describe the keys / types of a JSON object."""
    pad = "  " * indent
    lines: list[str] = []

    if indent > max_depth:
        lines.append(f"{pad}... (max depth reached)")
        return lines

    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.append(f"{pad}{k!r}: {_type_label(v)}")
            if isinstance(v, (dict, list)) and indent < max_depth:
                sub = v if isinstance(v, dict) else (v[0] if v else None)
                if sub is not None:
                    lines.extend(describe_structure(sub, indent + 1, max_depth))
    elif isinstance(obj, list):
        if obj:
            lines.append(f"{pad}[0]: {_type_label(obj[0])}")
            lines.extend(describe_structure(obj[0], indent + 1, max_depth))
    else:
        lines.append(f"{pad}value={obj!r}")

    return lines


# ---------------------------------------------------------------------------
# ToolBench-specific extraction
# ---------------------------------------------------------------------------

def _extract_tool_calls(sample: dict) -> list[dict]:
    """
    Walk the known ToolBench schema variants and collect tool/API calls.

    Known layouts:
      A) sample["answer"]["train_messages"]  — list of conversation turns
         Each turn: {"role": "...", "content": "..."}
         Tool calls appear as JSON strings in assistant content prefixed with
         "Action:" / "Action Input:" or as role="tool" / role="function" turns.

      B) sample["conversations"]  — flat list of turns (some HF exports)

      C) sample["process"]  — list of steps with "action", "action_input"
    """
    calls: list[dict] = []

    # Layout C — process list
    if "process" in sample and isinstance(sample["process"], list):
        for step in sample["process"]:
            if isinstance(step, dict) and "action" in step:
                calls.append({
                    "tool_name": step.get("action", ""),
                    "tool_input": step.get("action_input", {}),
                    "observation": step.get("observation", ""),
                })
        return calls

    # Layout A — answer.train_messages  OR  Layout B — conversations
    turns: list[dict] = []
    if "answer" in sample and isinstance(sample.get("answer"), dict):
        msgs = sample["answer"].get("train_messages", [])
        # train_messages is sometimes a list-of-lists (one per attempt)
        if msgs and isinstance(msgs[0], list):
            turns = msgs[0]
        else:
            turns = msgs
    elif "conversations" in sample:
        turns = sample["conversations"]

    i = 0
    while i < len(turns):
        turn = turns[i] if isinstance(turns[i], dict) else {}
        role = turn.get("role", turn.get("from", ""))
        content = turn.get("content", turn.get("value", "")) or ""

        # ReAct-style: assistant emits "Action: tool_name\nAction Input: {...}"
        if role in ("assistant", "gpt") and "Action:" in content:
            lines = content.splitlines()
            tool_name, tool_input = "", {}
            for line in lines:
                if line.startswith("Action:"):
                    tool_name = line.split("Action:", 1)[1].strip()
                elif line.startswith("Action Input:"):
                    raw = line.split("Action Input:", 1)[1].strip()
                    try:
                        tool_input = json.loads(raw)
                    except json.JSONDecodeError:
                        tool_input = {"raw": raw}
            if tool_name:
                # Observation is the next turn
                obs = ""
                if i + 1 < len(turns):
                    next_turn = turns[i + 1] if isinstance(turns[i + 1], dict) else {}
                    obs_role = next_turn.get("role", next_turn.get("from", ""))
                    if obs_role in ("tool", "observation", "function"):
                        obs = next_turn.get("content", next_turn.get("value", "")) or ""
                calls.append({"tool_name": tool_name, "tool_input": tool_input, "observation": obs})

        # OpenAI-function-call style: role="tool" or role="function"
        elif role in ("tool", "function"):
            # The preceding assistant turn should have the call name
            prev = turns[i - 1] if i > 0 and isinstance(turns[i - 1], dict) else {}
            tool_name = prev.get("name", prev.get("function_call", {}).get("name", "unknown"))
            raw_input = prev.get("function_call", {}).get("arguments", "{}")
            try:
                tool_input = json.loads(raw_input) if isinstance(raw_input, str) else raw_input
            except json.JSONDecodeError:
                tool_input = {"raw": raw_input}
            calls.append({
                "tool_name": tool_name,
                "tool_input": tool_input,
                "observation": content,
            })

        i += 1

    return calls


def _trajectory_length(sample: dict) -> int:
    """Number of turns in the trajectory."""
    if "process" in sample and isinstance(sample["process"], list):
        return len(sample["process"])
    if "answer" in sample and isinstance(sample.get("answer"), dict):
        msgs = sample["answer"].get("train_messages", [])
        if msgs and isinstance(msgs[0], list):
            return len(msgs[0])
        return len(msgs)
    if "conversations" in sample:
        return len(sample["conversations"])
    return 0


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_data_files() -> list[pathlib.Path]:
    """Return all JSONL / JSON files in data/raw/, newest first."""
    files = sorted(
        list(RAW_DIR.glob("**/*.jsonl")) + list(RAW_DIR.glob("**/*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


def load_samples(path: pathlib.Path, n: int) -> list[dict]:
    """Load up to n samples from a JSONL or JSON file."""
    samples: list[dict] = []
    suffix = path.suffix.lower()

    with path.open("r", encoding="utf-8") as fh:
        if suffix == ".jsonl":
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(samples) >= n:
                    break
        else:
            # Plain JSON — could be a list or a single object
            data = json.load(fh)
            if isinstance(data, list):
                samples = data[:n]
            elif isinstance(data, dict):
                # Maybe a dict-of-lists or a single sample
                # Try common wrapper keys
                for key in ("data", "samples", "trajectories", "train", "test"):
                    if key in data and isinstance(data[key], list):
                        samples = data[key][:n]
                        break
                else:
                    samples = [data]

    return samples


def count_all_samples(path: pathlib.Path) -> int:
    """Count total records in a file without loading everything into memory."""
    count = 0
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as fh:
        if suffix == ".jsonl":
            for line in fh:
                if line.strip():
                    count += 1
        else:
            data = json.load(fh)
            if isinstance(data, list):
                count = len(data)
            elif isinstance(data, dict):
                for key in ("data", "samples", "trajectories", "train", "test"):
                    if key in data and isinstance(data[key], list):
                        count = len(data[key])
                        break
                else:
                    count = 1
    return count


# ---------------------------------------------------------------------------
# Main inspection logic
# ---------------------------------------------------------------------------

def inspect(path: pathlib.Path, n: int, output_lines: list[str]) -> None:
    def emit(line: str = "") -> None:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode('ascii', errors='replace').decode('ascii'))
        output_lines.append(line)

    emit(f"{'=' * 70}")
    emit(f"FILE: {path}")
    emit(f"SIZE: {path.stat().st_size / 1024:.1f} KB")
    emit(f"{'=' * 70}")

    samples = load_samples(path, n)
    if not samples:
        emit("  [no samples loaded — file may be empty or unrecognised format]")
        return

    # ---- 1. JSON structure of first sample --------------------------------
    emit(f"\n--- Structure of sample[0] (keys / types, max depth=5) ---")
    struct_lines = describe_structure(samples[0])
    for sl in struct_lines[:80]:   # cap output length
        emit(sl)
    if len(struct_lines) > 80:
        emit(f"  ... ({len(struct_lines) - 80} more lines truncated)")

    # ---- 2. Raw first sample (pretty-printed, truncated) ------------------
    emit(f"\n--- sample[0] pretty-print (first 60 lines) ---")
    pretty = json.dumps(samples[0], indent=2, ensure_ascii=False)
    for i, line in enumerate(pretty.splitlines()):
        if i >= 60:
            emit("  ... (truncated)")
            break
        emit(line)

    # ---- 3. Tool call extraction ------------------------------------------
    emit(f"\n--- Tool/API calls extracted from first {len(samples)} samples ---")
    all_tool_names: list[str] = []
    traj_lengths: list[int] = []

    for idx, sample in enumerate(samples):
        calls = _extract_tool_calls(sample)
        length = _trajectory_length(sample)
        traj_lengths.append(length)
        names = [c["tool_name"] for c in calls if c["tool_name"]]
        all_tool_names.extend(names)

        call_summary = ", ".join(names[:6]) + (" ..." if len(names) > 6 else "")
        emit(f"  sample[{idx:2d}]  turns={length:3d}  tool_calls={len(calls):3d}  [{call_summary}]")

    # ---- 4. Statistics across the whole file ------------------------------
    emit(f"\n--- File-level statistics ---")
    total = count_all_samples(path)
    emit(f"  Total trajectories in file : {total:,}")
    emit(f"  Samples inspected          : {len(samples)}")

    if all_tool_names:
        tool_counts = Counter(all_tool_names)
        emit(f"  Unique tools (in sample)   : {len(tool_counts)}")
        emit(f"  Total tool calls (sample)  : {len(all_tool_names)}")
        emit(f"  Avg tool calls / trajectory: {len(all_tool_names)/len(samples):.1f}")
        emit(f"\n  Top 20 most-called tools:")
        for tool, cnt in tool_counts.most_common(20):
            emit(f"    {cnt:5d}  {tool}")
    else:
        emit("  [no tool calls extracted — check schema above and update _extract_tool_calls()]")

    if traj_lengths:
        avg_len = sum(traj_lengths) / len(traj_lengths)
        emit(f"\n  Avg trajectory length (turns): {avg_len:.1f}")
        emit(f"  Min / Max turns              : {min(traj_lengths)} / {max(traj_lengths)}")

    # ---- 5. Key path guidance --------------------------------------------
    emit(f"\n--- Where to find things in the schema ---")
    s0 = samples[0]
    hints: dict[str, str] = {}

    # Tool name
    for path_expr in [
        "process[*].action",
        "answer.train_messages[*].content (Action: ...)",
        "conversations[*].content (Action: ...)",
    ]:
        if "process" in s0 and path_expr.startswith("process"):
            hints["tool_name"] = path_expr
            break
        if "answer" in s0 and "train_messages" in path_expr:
            hints["tool_name"] = path_expr
            break
        if "conversations" in s0 and "conversations" in path_expr:
            hints["tool_name"] = path_expr
            break

    for k, v in hints.items():
        emit(f"  {k:20s} → {v}")

    if "query" in s0:
        emit(f"  {'task_query':20s} → sample['query']")
    if "instruction" in s0:
        emit(f"  {'task_instruction':20s} → sample['instruction']")
    if "answer" in s0 and isinstance(s0.get("answer"), dict):
        emit(f"  {'answer_dict':20s} → sample['answer']  (keys: {list(s0['answer'].keys())[:8]})")

    emit("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ToolBench trajectory data")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to inspect (default: 10)")
    parser.add_argument("--file", type=str, default=None, help="Path to a specific JSONL/JSON file")
    args = parser.parse_args()

    output_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        output_lines.append(line)

    # Determine which files to inspect
    if args.file:
        files = [pathlib.Path(args.file).resolve()]
        if not files[0].exists():
            print(f"ERROR: file not found: {files[0]}")
            sys.exit(1)
    else:
        files = find_data_files()

    if not files:
        emit("No data files found in data/raw/")
        emit("")
        emit("Run one of the following to download ToolBench:")
        emit("  bash data/download.sh                 # auto (HuggingFace preferred)")
        emit("  bash data/download.sh --git           # sparse git clone")
        emit("  bash data/download.sh --manual        # print manual instructions")
        emit("")
        emit("Or manually place JSONL files under data/raw/")
        SUMMARY_FILE.write_text("\n".join(output_lines), encoding="utf-8")
        sys.exit(0)

    emit(f"Found {len(files)} data file(s) in {RAW_DIR}")
    emit(f"Inspecting first {args.n} samples per file")
    emit("")

    for f in files:
        inspect(f, args.n, output_lines)

    # Write summary
    SUMMARY_FILE.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nSummary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
