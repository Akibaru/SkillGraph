#!/usr/bin/env bash
# =============================================================================
# SkillGraph — ToolBench data download script
#
# Strategy (fastest first):
#   1. HuggingFace Hub  → ToolBench/ToolBench  (preferred — single dataset pull)
#   2. Sparse git clone  → only data/ subtree from OpenBMB/ToolBench
#   3. Manual curl fallback for individual JSONL files
#
# Usage:
#   bash data/download.sh              # auto (tries HF first)
#   bash data/download.sh --git        # force git sparse-checkout
#   bash data/download.sh --manual     # print manual curl commands
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/raw"
HF_DATASET="ToolBench/ToolBench"
GH_REPO="https://github.com/OpenBMB/ToolBench.git"

MODE="${1:-auto}"

# ---------------------------------------------------------------------------
# Helper: check command exists
# ---------------------------------------------------------------------------
need() { command -v "$1" &>/dev/null || { echo "ERROR: '$1' not found. Install it first."; exit 1; }; }

# ---------------------------------------------------------------------------
# Option 1 — HuggingFace Hub via Python datasets library
# ---------------------------------------------------------------------------
download_huggingface() {
    echo "=== Downloading ToolBench from HuggingFace Hub ==="
    echo "Dataset: ${HF_DATASET}"
    echo "Destination: ${RAW_DIR}"
    echo

    need python3

    python3 - <<'PYEOF'
import os, sys, pathlib

raw_dir = pathlib.Path(os.environ.get("RAW_DIR", "data/raw"))
raw_dir.mkdir(parents=True, exist_ok=True)

try:
    from datasets import load_dataset, get_dataset_config_names
except ImportError:
    print("ERROR: 'datasets' package not installed.")
    print("  pip install datasets huggingface_hub")
    sys.exit(1)

hf_name = "ToolBench/ToolBench"

# Discover available configs (splits)
try:
    configs = get_dataset_config_names(hf_name)
except Exception as e:
    print(f"WARNING: Could not list configs ({e}). Trying default config.")
    configs = [None]

print(f"Available configs: {configs}\n")

# Download each config and save as JSONL
for cfg in configs:
    label = cfg if cfg else "default"
    out_file = raw_dir / f"toolbench_{label}.jsonl"

    if out_file.exists():
        print(f"  [skip] {out_file.name} already exists")
        continue

    print(f"  Loading config='{label}' ...")
    try:
        ds = load_dataset(hf_name, cfg, trust_remote_code=True)
    except Exception as e:
        print(f"  WARNING: Failed to load config '{label}': {e}")
        continue

    # Combine all splits into one JSONL file
    import json
    count = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for split_name, split_ds in ds.items():
            for row in split_ds:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"  Saved {count:,} records → {out_file}")

print("\nDone.")
PYEOF
}

# ---------------------------------------------------------------------------
# Option 2 — Git sparse checkout (only data/ subtree)
# ---------------------------------------------------------------------------
download_git_sparse() {
    echo "=== Sparse git clone of OpenBMB/ToolBench (data/ subtree) ==="
    need git

    CLONE_DIR="${RAW_DIR}/ToolBench_repo"

    if [[ -d "${CLONE_DIR}/.git" ]]; then
        echo "Repo already cloned at ${CLONE_DIR}. Pulling latest..."
        git -C "${CLONE_DIR}" pull
    else
        echo "Initialising sparse clone into ${CLONE_DIR} ..."
        mkdir -p "${CLONE_DIR}"
        git -C "${CLONE_DIR}" init
        git -C "${CLONE_DIR}" remote add origin "${GH_REPO}"

        # Enable sparse checkout — we only want data/toolenv/data and data/answer
        git -C "${CLONE_DIR}" config core.sparseCheckout true

        mkdir -p "${CLONE_DIR}/.git/info"
        cat > "${CLONE_DIR}/.git/info/sparse-checkout" <<'SPARSE'
data/toolenv/data/
data/answer/
data/instruction/
SPARSE

        echo "Fetching (this may take a few minutes — only checking out data/ subtree) ..."
        git -C "${CLONE_DIR}" fetch --depth=1 origin main
        git -C "${CLONE_DIR}" checkout FETCH_HEAD
    fi

    echo
    echo "Data subtree checked out at: ${CLONE_DIR}/data/"
    echo "JSONL files:"
    find "${CLONE_DIR}/data" -name "*.jsonl" 2>/dev/null | head -30 || echo "  (none found — check repo structure)"
}

# ---------------------------------------------------------------------------
# Option 3 — Manual instructions / curl fallback
# ---------------------------------------------------------------------------
print_manual_instructions() {
    cat <<'MANUAL'
=== Manual download instructions for ToolBench ===

The ToolBench dataset has several mirrors. Try these in order:

--- A. HuggingFace (recommended) ---
  pip install datasets huggingface_hub
  python3 -c "
from datasets import load_dataset
ds = load_dataset('ToolBench/ToolBench', trust_remote_code=True)
print(ds)
"

--- B. HuggingFace Hub CLI ---
  pip install huggingface_hub
  huggingface-cli download ToolBench/ToolBench --repo-type dataset --local-dir data/raw/toolbench_hf

--- C. Specific JSONL files via curl (if you know the direct URLs) ---
  # G1 (tools from a single category)
  curl -L "https://huggingface.co/datasets/ToolBench/ToolBench/resolve/main/data/test_instructions/G1_instruction.json" \
       -o data/raw/G1_instruction.json

  # G2 (tools from same category)
  curl -L "https://huggingface.co/datasets/ToolBench/ToolBench/resolve/main/data/test_instructions/G2_instruction.json" \
       -o data/raw/G2_instruction.json

  # G3 (tools from different categories)
  curl -L "https://huggingface.co/datasets/ToolBench/ToolBench/resolve/main/data/test_instructions/G3_instruction.json" \
       -o data/raw/G3_instruction.json

--- D. GitHub sparse clone (data subtree only) ---
  git clone --filter=blob:none --sparse https://github.com/OpenBMB/ToolBench.git data/raw/ToolBench_repo
  cd data/raw/ToolBench_repo
  git sparse-checkout set data/toolenv/data data/answer data/instruction

MANUAL
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p "${RAW_DIR}"

export RAW_DIR

case "${MODE}" in
    --git)
        download_git_sparse
        ;;
    --manual)
        print_manual_instructions
        ;;
    auto|--auto)
        echo "Attempting HuggingFace download first..."
        if download_huggingface; then
            echo "HuggingFace download complete."
        else
            echo
            echo "HuggingFace download failed. Falling back to sparse git clone..."
            download_git_sparse || {
                echo
                echo "Git clone also failed. Printing manual instructions:"
                print_manual_instructions
                exit 1
            }
        fi
        ;;
    *)
        echo "Usage: $0 [--git | --manual | auto]"
        exit 1
        ;;
esac
