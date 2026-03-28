"""
src/learned_reranker.py  —  Learned Pairwise Reranker (Stage 2)
================================================================
Replaces the hand-tuned alpha-weighted Hybrid-Rerank with a small MLP
trained on pairwise ordering preferences from training trajectories.

Architecture
------------
Per-tool feature vector  (dim = 8):
  f1  sim_query_tool : cosine similarity between query embedding and tool
  f2  rank_sem_norm  : rank by similarity, normalised to [0,1] (0 = best)
  f3  sum_tp_out     : Σ tp(t → t′) for t′ ∈ candidate set
  f4  sum_tp_in      : Σ tp(t′ → t) for t′ ∈ candidate set
  f5  max_tp_out     : max tp(t → t′) for t′ ∈ candidate set
  f6  max_tp_in      : max tp(t′ → t) for t′ ∈ candidate set
  f7  avg_train_pos  : average normalised position in training trajectories
                       (0 = always first, 1 = always last; 0.5 if unseen)
  f8  set_size_norm  : |S| / 10  (context-size signal)

Pairwise MLP  (8 → 64 → 32 → 1, ReLU + Sigmoid):
  input  : feat(A) − feat(B)           (difference vector, antisymmetric)
  output : P(A precedes B) ∈ (0, 1)
  loss   : binary cross-entropy

Inference ranking:
  score(t) = Σ_{t′ ≠ t ∈ S}  model( feat(t) − feat(t′) )
  Sort descending by score → predicted order.

Training protocol
-----------------
  Data   : train_records trajectories (passed in)
  Pairs  : every ordered pair (i < j) from each GT sequence (label = 1)
           reversed pairs are handled by sign-flipping in BCE  (label = 0)
  Epochs : up to max_epochs with early-stop on val-BCE (patience = 5)
  Save   : models/learned_reranker.pt
"""

from __future__ import annotations

import pathlib
import sys
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

ROOT      = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = MODEL_DIR / "learned_reranker.pt"
FEAT_DIM   = 8    # number of per-tool features


# ============================================================================
# Feature extraction
# ============================================================================

def build_position_stats(train_records: list[dict]) -> dict[str, float]:
    """
    Compute mean normalised position for each tool over training trajectories.
    Returns dict: tool_name -> mean(position / seq_len).  Unseen tools get 0.5.
    """
    pos_lists: dict[str, list[float]] = {}
    for rec in train_records:
        seq = rec["tool_sequence"]
        n   = len(seq)
        if n == 0:
            continue
        for i, t in enumerate(seq):
            pos_lists.setdefault(t, []).append(i / n)
    return {t: float(np.mean(v)) for t, v in pos_lists.items()}


def extract_features(
    tool:          str,
    sim:           float,
    rank_in_set:   int,          # 0-based rank by similarity (0 = most similar)
    tool_set:      list[str],    # all tools in candidate set
    tp_lookup:     dict[tuple[str, str], float],
    position_stats: dict[str, float],
) -> np.ndarray:
    """
    Return 1-D feature vector of length FEAT_DIM for a single tool
    in the context of a candidate set.
    """
    n = len(tool_set)

    # f1: semantic similarity
    f1 = sim

    # f2: normalised rank by similarity (0 = best; 1 = worst)
    f2 = rank_in_set / max(n - 1, 1)

    # f3, f5: outgoing transition features
    tp_out = [tp_lookup.get((tool, t2), 0.0) for t2 in tool_set if t2 != tool]
    f3 = float(np.sum(tp_out))    if tp_out else 0.0
    f5 = float(np.max(tp_out))    if tp_out else 0.0

    # f4, f6: incoming transition features
    tp_in  = [tp_lookup.get((t2, tool), 0.0) for t2 in tool_set if t2 != tool]
    f4 = float(np.sum(tp_in))     if tp_in  else 0.0
    f6 = float(np.max(tp_in))     if tp_in  else 0.0

    # f7: average training position (0.5 for unseen tools)
    f7 = position_stats.get(tool, 0.5)

    # f8: set size, normalised
    f8 = n / 10.0

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=np.float32)


# ============================================================================
# MLP model
# ============================================================================

class PairwiseMLP(nn.Module):
    """
    Input  : feat(A) − feat(B)  (FEAT_DIM-dimensional)
    Output : scalar logit → P(A precedes B) after sigmoid
    """

    def __init__(self, input_dim: int = FEAT_DIM, hidden: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (batch,)


# ============================================================================
# Training
# ============================================================================

def _build_pairs(
    train_records:  list[dict],
    query_vecs:     np.ndarray,
    tp_lookup:      dict[tuple[str, str], float],
    position_stats: dict[str, float],
    planner,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each training trajectory build pairwise training examples.
    Returns X (N, FEAT_DIM) and y (N,) where:
      x = feat(A) − feat(B)
      y = 1  (A precedes B in GT)
    Both directions are included (with flipped sign for the y=0 direction).
    """
    X_list: list[np.ndarray] = []
    y_list: list[float]      = []

    for i, rec in enumerate(train_records):
        gt  = rec["tool_sequence"]
        n   = len(gt)
        if n < 2:
            continue

        vec       = query_vecs[i]
        tool_set  = list(dict.fromkeys(gt))   # unique, preserving first-occurrence order

        # Similarity scores for all tools in this set
        sim_dict: dict[str, float] = {
            t: float(planner._tool_sim(t, vec)) for t in tool_set
        }
        sorted_by_sim = sorted(tool_set, key=lambda t: -sim_dict.get(t, 0.0))
        rank_dict     = {t: r for r, t in enumerate(sorted_by_sim)}

        # Per-tool feature cache
        feat_cache: dict[str, np.ndarray] = {}
        for t in tool_set:
            feat_cache[t] = extract_features(
                t, sim_dict.get(t, 0.0), rank_dict.get(t, 0),
                tool_set, tp_lookup, position_stats,
            )

        # All ordered pairs (A, B) where A precedes B in GT
        seen_pairs: set[tuple[str, str]] = set()
        for ia in range(n - 1):
            for ib in range(ia + 1, n):
                a, b = gt[ia], gt[ib]
                if a not in feat_cache or b not in feat_cache:
                    continue
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))

                diff = feat_cache[a] - feat_cache[b]
                X_list.append(diff)
                y_list.append(1.0)

                # Add reversed pair (label = 0)
                X_list.append(-diff)
                y_list.append(0.0)

    if not X_list:
        raise ValueError("No training pairs collected — check train_records content.")

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.float32)


def train_learned_reranker(
    train_records:  list[dict],
    query_vecs:     np.ndarray,
    tp_lookup:      dict[tuple[str, str], float],
    position_stats: dict[str, float],
    planner,
    val_fraction:   float = 0.1,
    max_epochs:     int   = 30,
    batch_size:     int   = 2048,
    lr:             float = 1e-3,
    patience:       int   = 5,
    seed:           int   = 42,
    verbose:        bool  = True,
    save_path:      pathlib.Path = CHECKPOINT,
) -> PairwiseMLP:
    """
    Train pairwise MLP reranker and save checkpoint.
    Returns the trained model (on CPU).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print("\n[LearnedReranker]  Building pairwise training pairs ...")

    X, y = _build_pairs(train_records, query_vecs, tp_lookup, position_stats, planner)

    if verbose:
        print(f"  Total pairs: {len(X):,}  (pos={int(y.sum()):,}, neg={int((1-y).sum()):,})")

    # Train / val split
    n_val  = max(1, int(len(X) * val_fraction))
    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(len(X))
    idx_v  = perm[:n_val]
    idx_t  = perm[n_val:]

    X_tr, y_tr = torch.tensor(X[idx_t]), torch.tensor(y[idx_t])
    X_va, y_va = torch.tensor(X[idx_v]), torch.tensor(y[idx_v])

    tr_ds = TensorDataset(X_tr, y_tr)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PairwiseMLP().to(device)
    opt    = optim.Adam(model.parameters(), lr=lr)
    crit   = nn.BCEWithLogitsLoss()

    X_va_d = X_va.to(device)
    y_va_d = y_va.to(device)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(X_tr)

        model.eval()
        with torch.no_grad():
            va_loss = crit(model(X_va_d), y_va_d).item()
            # pairwise accuracy on val
            probs   = torch.sigmoid(model(X_va_d))
            va_acc  = ((probs > 0.5) == y_va_d.bool()).float().mean().item()

        if verbose:
            marker = "  *" if va_loss < best_val else ""
            print(f"  epoch {epoch:3d}  tr_bce={tr_loss:.4f}  "
                  f"val_bce={va_loss:.4f}  val_acc={va_acc:.4f}{marker}")

        if va_loss < best_val - 1e-5:
            best_val   = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()

    torch.save({"state_dict": model.state_dict(), "feat_dim": FEAT_DIM}, save_path)
    if verbose:
        print(f"  Saved -> {save_path}  (best val_bce={best_val:.4f})")

    return model


def load_learned_reranker(path: pathlib.Path = CHECKPOINT) -> PairwiseMLP:
    ckpt  = torch.load(path, map_location="cpu")
    model = PairwiseMLP()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def order_learned_rerank(
    tools_sims:     list[tuple[str, float]],
    tp_lookup:      dict[tuple[str, str], float],
    position_stats: dict[str, float],
    model:          PairwiseMLP,
) -> list[str]:
    """
    Rank `tools_sims` using pairwise MLP.

    score(t) = Σ_{t′ ≠ t ∈ S}  sigmoid( model(feat(t) − feat(t′)) )
    Sort descending by score.

    Falls back to semantic order if |S| == 1 or model not provided.
    """
    if len(tools_sims) <= 1:
        return [t for t, _ in tools_sims]

    tool_set  = [t for t, _ in tools_sims]
    n         = len(tool_set)
    sim_dict  = dict(tools_sims)
    sorted_by_sim = sorted(tool_set, key=lambda t: -sim_dict.get(t, 0.0))
    rank_dict = {t: r for r, t in enumerate(sorted_by_sim)}

    # Build feature matrix  (n, FEAT_DIM)
    feat_mat = np.stack([
        extract_features(
            t, sim_dict[t], rank_dict[t],
            tool_set, tp_lookup, position_stats,
        )
        for t in tool_set
    ], axis=0)                                      # (n, FEAT_DIM)

    feat_t = torch.tensor(feat_mat, dtype=torch.float32)   # (n, FEAT_DIM)

    # Compute pairwise difference matrix  (n, n, FEAT_DIM)
    diff = feat_t.unsqueeze(1) - feat_t.unsqueeze(0)       # (n, n, FEAT_DIM)
    diff_flat = diff.view(n * n, FEAT_DIM)

    logits = model(diff_flat).view(n, n)                    # (n, n)
    probs  = torch.sigmoid(logits)                          # P(row precedes col)

    # score(t) = sum of win-probabilities over all opponents
    scores = probs.sum(dim=1).numpy()                       # (n,)

    # Sort descending by score
    order = np.argsort(-scores)
    return [tool_set[i] for i in order]
