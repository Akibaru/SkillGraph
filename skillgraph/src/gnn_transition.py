"""
src/gnn_transition.py  —  GNN-based Tool Transition Model
==========================================================
Learns directed transition probabilities between tools via link-prediction
on the SkillGraph.  Three GNN encoders (GCN / GAT / GraphSAGE) share a
common MLP decoder that scores edge (u → v).

Pipeline
--------
  1. TransitionDataset   — extracts (tool_i, tool_j) bigram pairs from
                           trajectories, samples negatives 1:5, splits 80/10/10
  2. GNNEncoder          — 2-layer GCN / GAT / GraphSAGE  (hidden_dim=256)
  3. TransitionDecoder   — MLP on [h_u ‖ h_v ‖ h_u⊙h_v]  (768→256→128→1)
  4. TransitionModel     — encoder + decoder
  5. train_transition_model()  — BCE loss, Adam, early-stop on val-AUC
  6. load_transition_model()   — reload checkpoint, return score function
  7. precompute_transition_scores() — cache all graph-edge scores as dict
  8. get_gnn_transition_score() — single-pair look-up from the cache

Usage
-----
  python src/gnn_transition.py --encoder gcn
  python src/gnn_transition.py --encoder gat  --epochs 200
  python src/gnn_transition.py --encoder sage --eval-only
  python src/gnn_transition.py --encoder gcn  --precompute
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_FILE = PROC_DIR / "final_graph.gpickle"
EMB_FILE   = PROC_DIR / "tool_embeddings.npy"
TRAJ_FILE  = PROC_DIR / "successful_trajectories.jsonl"

# ---------------------------------------------------------------------------
# Global hyper-parameters
# ---------------------------------------------------------------------------
IN_DIM     = 384    # SentenceBERT all-MiniLM-L6-v2 output dim
HIDDEN_DIM = 256
NEG_RATIO  = 5     # negative edges per positive
SEED       = 42


# ============================================================================
# 1. TransitionDataset
# ============================================================================

class TransitionDataset:
    """
    Prepares a PyG Data object for directed link-prediction.

    Positive samples : every consecutive (tool_i, tool_j) bigram found across
                       ALL trajectories, de-duplicated at the pair level.
    Negative samples : random (u, v) pairs that never appear as positives,
                       sampled at neg_ratio× the positive count.
    Split            : 80 / 10 / 10  on positive edges (random shuffle).

    Node features    : (N, 384) L2-normalised SentenceBERT embeddings,
                       row i ↔ sorted(G.nodes())[i]  (same convention as
                       graph_build.py / graph_search.py).
    """

    def __init__(self, neg_ratio: int = NEG_RATIO, seed: int = SEED) -> None:
        self.neg_ratio = neg_ratio
        self.rng = np.random.default_rng(seed)

        self._load_graph()
        self._extract_transitions()
        self._build_splits()
        self._sample_negatives()

    # ── Loading ──────────────────────────────────────────────────────────

    def _load_graph(self) -> None:
        for p in (GRAPH_FILE, EMB_FILE, TRAJ_FILE):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Run graph_build.py and extract.py first."
                )

        with GRAPH_FILE.open("rb") as fh:
            self.G = pickle.load(fh)

        # Canonical node ordering (must match tool_embeddings.npy row order)
        self._active_tools: list[str] = sorted(self.G.nodes())
        self._tool_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self._active_tools)
        }
        self.n_tools = len(self._active_tools)

        # Node feature matrix  (N, 384), L2-normalised
        emb = np.load(str(EMB_FILE))          # (n_tools, 384) float32
        assert emb.shape[0] == self.n_tools, (
            f"Embedding rows ({emb.shape[0]}) ≠ graph nodes ({self.n_tools}). "
            "Re-run graph_build.py phase 2 to regenerate tool_embeddings.npy."
        )
        self.x = torch.tensor(emb, dtype=torch.float32)

        # Build edge_index for message passing from the full graph structure
        rows, cols = [], []
        for u, v in self.G.edges():
            ui = self._tool_to_idx.get(u)
            vi = self._tool_to_idx.get(v)
            if ui is not None and vi is not None:
                rows.append(ui)
                cols.append(vi)
        self.graph_edge_index = torch.tensor([rows, cols], dtype=torch.long)

        print(
            f"[TransitionDataset] Graph loaded: {self.n_tools:,} nodes, "
            f"{self.graph_edge_index.shape[1]:,} edges"
        )

    def _extract_transitions(self) -> None:
        """
        Extract (tool_i_idx, tool_j_idx) bigram pairs from all trajectories.
        Only tools present in the graph are kept; self-loops are dropped.
        Pair counts are recorded (used for analysis; BCE uses binary labels).
        """
        from collections import Counter
        pair_counts: Counter = Counter()

        with open(TRAJ_FILE, encoding="utf-8") as fh:
            for line in fh:
                rec  = json.loads(line)
                seq  = rec.get("tool_sequence", [])
                # Keep only tools that exist in the graph
                seq  = [t for t in seq if t in self._tool_to_idx]
                for k in range(len(seq) - 1):
                    ui, vi = self._tool_to_idx[seq[k]], self._tool_to_idx[seq[k + 1]]
                    if ui != vi:
                        pair_counts[(ui, vi)] += 1

        self._pos_pairs:   np.ndarray = np.array(
            list(pair_counts.keys()), dtype=np.int64
        )                                                        # (E_pos, 2)
        self._pos_weights: np.ndarray = np.array(
            list(pair_counts.values()), dtype=np.float32
        )                                                        # (E_pos,)
        self._pos_set: set[tuple[int, int]] = set(
            map(tuple, self._pos_pairs.tolist())
        )

        print(
            f"[TransitionDataset] Positive pairs: {len(self._pos_pairs):,}  "
            f"(total transition instances: {int(self._pos_weights.sum()):,})"
        )

    def _build_splits(self) -> None:
        """Shuffle and split positive edges 80 / 10 / 10."""
        n   = len(self._pos_pairs)
        idx = self.rng.permutation(n)

        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        # remainder → test

        self._train_pos_idx = idx[:n_train]
        self._val_pos_idx   = idx[n_train : n_train + n_val]
        self._test_pos_idx  = idx[n_train + n_val :]

        print(
            f"[TransitionDataset] Positive split — "
            f"train={len(self._train_pos_idx):,}  "
            f"val={len(self._val_pos_idx):,}  "
            f"test={len(self._test_pos_idx):,}"
        )

    def _sample_negatives(self) -> None:
        """
        For each split sample neg_ratio × |pos| pairs that are NOT in pos_set.
        Rejection sampling; very fast because pos_set << total possible pairs.
        """

        def _sample(n_neg: int) -> np.ndarray:
            negs: list[tuple[int, int]] = []
            buf  = n_neg * 4           # over-sample to reduce rejection loops
            while len(negs) < n_neg:
                us = self.rng.integers(0, self.n_tools, size=buf)
                vs = self.rng.integers(0, self.n_tools, size=buf)
                for u, v in zip(us.tolist(), vs.tolist()):
                    if u != v and (u, v) not in self._pos_set:
                        negs.append((u, v))
                        if len(negs) == n_neg:
                            break
            return np.array(negs, dtype=np.int64)

        n_tr = len(self._train_pos_idx) * self.neg_ratio
        n_va = len(self._val_pos_idx)   * self.neg_ratio
        n_te = len(self._test_pos_idx)  * self.neg_ratio

        print("[TransitionDataset] Sampling negatives …")
        self._train_neg = _sample(n_tr)    # (n_tr, 2)
        self._val_neg   = _sample(n_va)    # (n_va, 2)
        self._test_neg  = _sample(n_te)    # (n_te, 2)

        print(
            f"[TransitionDataset] Negative split — "
            f"train={n_tr:,}  val={n_va:,}  test={n_te:,}"
        )

    # ── PyG Data object ──────────────────────────────────────────────────

    @staticmethod
    def _to_edge_tensor(pairs: np.ndarray) -> torch.Tensor:
        """Convert (E, 2) numpy array to (2, E) torch.long tensor."""
        return torch.tensor(pairs.T, dtype=torch.long)

    def get_pyg_data(self) -> Data:
        """
        Return a single PyG Data object.  Layout:

          data.x                     — (N, 384)  node features
          data.edge_index            — (2, E)    full graph edges for message-passing
          data.{split}_{sign}_edge_index  — split edge index pairs
            where split ∈ {train, val, test}  and  sign ∈ {pos, neg}
          data.n_tools               — int
          data.tool_list             — list[str], index i ↔ row i of x
        """
        train_pos_ei = self._to_edge_tensor(self._pos_pairs[self._train_pos_idx])
        val_pos_ei   = self._to_edge_tensor(self._pos_pairs[self._val_pos_idx])
        test_pos_ei  = self._to_edge_tensor(self._pos_pairs[self._test_pos_idx])
        train_neg_ei = self._to_edge_tensor(self._train_neg)
        val_neg_ei   = self._to_edge_tensor(self._val_neg)
        test_neg_ei  = self._to_edge_tensor(self._test_neg)

        data = Data(x=self.x, edge_index=self.graph_edge_index)

        data.train_pos_edge_index = train_pos_ei
        data.train_neg_edge_index = train_neg_ei
        data.val_pos_edge_index   = val_pos_ei
        data.val_neg_edge_index   = val_neg_ei
        data.test_pos_edge_index  = test_pos_ei
        data.test_neg_edge_index  = test_neg_ei

        data.n_tools   = self.n_tools
        data.tool_list = self._active_tools
        return data

    @property
    def tool_to_idx(self) -> dict[str, int]:
        return self._tool_to_idx

    @property
    def active_tools(self) -> list[str]:
        return self._active_tools


# ============================================================================
# 2. GNNEncoder
# ============================================================================

class GNNEncoder(nn.Module):
    """
    Three 2-layer GNN encoder variants behind a unified interface.

    GCN   : GCNConv(384→256) → BN → ReLU → Dropout → GCNConv(256→256) → BN → ReLU
    GAT   : GATConv(384, 64, heads=4, concat=True)  → 256 dim output per layer
    SAGE  : SAGEConv(384→256) → BN → ReLU → Dropout → SAGEConv(256→256) → BN → ReLU

    Returns : (N, hidden_dim) node embeddings.
    """

    SUPPORTED = ("gcn", "gat", "sage")

    def __init__(
        self,
        encoder_type: str   = "gcn",
        in_dim:       int   = IN_DIM,
        hidden_dim:   int   = HIDDEN_DIM,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()
        enc = encoder_type.lower()
        if enc not in self.SUPPORTED:
            raise ValueError(
                f"encoder_type must be one of {self.SUPPORTED}, got '{enc}'"
            )
        self.encoder_type = enc
        self.dropout      = dropout

        if enc == "gcn":
            self.conv1 = GCNConv(in_dim,     hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        elif enc == "gat":
            n_heads  = 4
            if hidden_dim % n_heads != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
                )
            head_dim = hidden_dim // n_heads
            # concat=True: output = head_dim × n_heads = hidden_dim
            self.conv1 = GATConv(
                in_dim, head_dim, heads=n_heads, concat=True, dropout=dropout
            )
            self.conv2 = GATConv(
                hidden_dim, head_dim, heads=n_heads, concat=True, dropout=dropout
            )

        elif enc == "sage":
            self.conv1 = SAGEConv(in_dim,     hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:   # → (N, hidden_dim)
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return h


# ============================================================================
# 3. TransitionDecoder
# ============================================================================

class TransitionDecoder(nn.Module):
    """
    MLP decoder for directed link prediction.

    Input  : concatenate [h_u ‖ h_v ‖ h_u⊙h_v]  →  3 × hidden_dim = 768
    Layers : Linear(768 → 256) → ReLU → Dropout(0.2)
             Linear(256 → 128) → ReLU
             Linear(128 → 1)
    Output : raw logit (apply sigmoid externally for probability)
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        in_dim = 3 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim,    hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        h_u: torch.Tensor,   # (B, hidden_dim)
        h_v: torch.Tensor,   # (B, hidden_dim)
    ) -> torch.Tensor:       # (B,)  raw logits
        feat = torch.cat([h_u, h_v, h_u * h_v], dim=-1)   # (B, 768)
        return self.mlp(feat).squeeze(-1)


# ============================================================================
# 4. TransitionModel
# ============================================================================

class TransitionModel(nn.Module):
    """
    Full model: GNNEncoder  +  TransitionDecoder.

    forward() returns raw logits.
    Use BCEWithLogitsLoss for training; sigmoid(logits) for inference.
    """

    def __init__(
        self,
        encoder_type: str   = "gcn",
        in_dim:       int   = IN_DIM,
        hidden_dim:   int   = HIDDEN_DIM,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = GNNEncoder(encoder_type, in_dim, hidden_dim, dropout)
        self.decoder = TransitionDecoder(hidden_dim)

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:                              # (N, hidden_dim)
        return self.encoder(x, edge_index)

    def decode(
        self,
        h:      torch.Tensor,   # (N, hidden_dim)
        pairs:  torch.Tensor,   # (2, B)  [src_idx; dst_idx]
    ) -> torch.Tensor:          # (B,)  raw logits
        h_u = h[pairs[0]]
        h_v = h[pairs[1]]
        return self.decoder(h_u, h_v)

    def forward(
        self,
        x:          torch.Tensor,   # (N, in_dim)
        edge_index: torch.Tensor,   # (2, E)
        pairs:      torch.Tensor,   # (2, B)
    ) -> torch.Tensor:              # (B,)
        h = self.encode(x, edge_index)
        return self.decode(h, pairs)


# ============================================================================
# 5. Training helpers
# ============================================================================

def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _compute_metrics(
    logits: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    probs = _sigmoid_np(logits)
    auc   = roc_auc_score(labels, probs)
    ap    = average_precision_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
    f1    = f1_score(labels, preds, zero_division=0)
    return {"auc": auc, "ap": ap, "f1": f1}


@torch.no_grad()
def _evaluate_split(
    model:     TransitionModel,
    x:         torch.Tensor,       # (N, 384) on device
    graph_ei:  torch.Tensor,       # (2, E) on device
    pos_ei:    torch.Tensor,       # (2, E_pos) on device
    neg_ei:    torch.Tensor,       # (2, E_neg) on device
) -> dict[str, float]:
    model.eval()
    h = model.encode(x, graph_ei)

    pos_logits = model.decode(h, pos_ei).cpu().numpy()
    neg_logits = model.decode(h, neg_ei).cpu().numpy()

    logits = np.concatenate([pos_logits, neg_logits])
    labels = np.concatenate([
        np.ones(len(pos_logits),  dtype=np.float32),
        np.zeros(len(neg_logits), dtype=np.float32),
    ])
    return _compute_metrics(logits, labels)


# ============================================================================
# 6. train_transition_model
# ============================================================================

def train_transition_model(
    encoder_type: str   = "gcn",
    hidden_dim:   int   = HIDDEN_DIM,
    lr:           float = 1e-3,
    max_epochs:   int   = 200,
    patience:     int   = 20,
    neg_ratio:    int   = NEG_RATIO,
    dropout:      float = 0.3,
    device_str:   str   = "auto",
) -> TransitionModel:
    """
    Complete training pipeline.  Returns the best TransitionModel.

    Training uses full-batch gradient descent per epoch:
      - One GNN forward pass over all nodes  → node embeddings h
      - One decoder forward over all training edges  → logits
      - BCEWithLogitsLoss  →  backward  →  Adam step

    This is efficient because the graph (4 988 nodes) fits comfortably
    in memory.  For larger graphs, switch to PyG's NeighborLoader.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    print(f"[train] Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    print("[train] Preparing dataset …")
    ds   = TransitionDataset(neg_ratio=neg_ratio)
    data = ds.get_pyg_data()

    # Move static tensors to device once
    x        = data.x.to(device)
    graph_ei = data.edge_index.to(device)

    # Training pairs (positive + negative), concatenated and shuffled each epoch
    tr_pos = data.train_pos_edge_index.to(device)   # (2, E+)
    tr_neg = data.train_neg_edge_index.to(device)   # (2, E-)

    # Evaluation edge tensors
    val_pos  = data.val_pos_edge_index.to(device)
    val_neg  = data.val_neg_edge_index.to(device)
    test_pos = data.test_pos_edge_index.to(device)
    test_neg = data.test_neg_edge_index.to(device)

    n_pos = tr_pos.shape[1]
    n_neg = tr_neg.shape[1]

    # ── Model ─────────────────────────────────────────────────────────────
    model = TransitionModel(
        encoder_type=encoder_type,
        in_dim=IN_DIM,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[train] Encoder: {encoder_type.upper()}  "
        f"hidden_dim={hidden_dim}  params={n_params:,}"
    )
    print(
        f"[train] Train edges — pos={n_pos:,}  neg={n_neg:,}  "
        f"total={n_pos + n_neg:,}"
    )

    # ── Checkpoint path ────────────────────────────────────────────────────
    ckpt_path = MODEL_DIR / f"gnn_transition_{encoder_type}_best.pt"

    # ── Epoch loop ────────────────────────────────────────────────────────
    best_val_auc  = -1.0
    patience_ctr  = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()

        # Shuffle training pairs every epoch
        perm_pos = torch.randperm(n_pos, device=device)
        perm_neg = torch.randperm(n_neg, device=device)

        all_pairs = torch.cat(
            [tr_pos[:, perm_pos], tr_neg[:, perm_neg]], dim=1
        )  # (2, n_pos + n_neg)
        labels = torch.cat([
            torch.ones(n_pos,  device=device),
            torch.zeros(n_neg, device=device),
        ])

        # Single full-batch forward + backward
        optimizer.zero_grad()
        logits = model(x, graph_ei, all_pairs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0

        # ── Validation ────────────────────────────────────────────────────
        val_m   = _evaluate_split(model, x, graph_ei, val_pos, val_neg)
        val_auc = val_m["auc"]

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>4}/{max_epochs}  "
                f"loss={loss.item():.4f}  "
                f"val_auc={val_auc:.4f}  "
                f"val_ap={val_m['ap']:.4f}  "
                f"val_f1={val_m['f1']:.4f}  "
                f"({elapsed:.1f}s)"
            )

        # ── Checkpoint / early stopping ────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            torch.save(
                {
                    "epoch":        epoch,
                    "encoder_type": encoder_type,
                    "hidden_dim":   hidden_dim,
                    "model_state":  model.state_dict(),
                    "val_auc":      val_auc,
                    "tool_list":    ds.active_tools,
                },
                ckpt_path,
            )
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(
                    f"[train] Early stopping at epoch {epoch}  "
                    f"(best val_auc={best_val_auc:.4f})"
                )
                break

    # ── Load best checkpoint and evaluate on test set ─────────────────────
    print(f"\n[train] Loading best checkpoint (val_auc={best_val_auc:.4f}) …")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_m = _evaluate_split(model, x, graph_ei, test_pos, test_neg)
    print(
        f"\n[train] ── TEST RESULTS ({encoder_type.upper()}) ──\n"
        f"  AUC : {test_m['auc']:.4f}\n"
        f"  AP  : {test_m['ap']:.4f}\n"
        f"  F1  : {test_m['f1']:.4f}\n"
        f"  Checkpoint saved → {ckpt_path}\n"
    )

    # Save metadata JSON alongside checkpoint
    meta_path = MODEL_DIR / f"gnn_transition_{encoder_type}_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "encoder_type":  encoder_type,
                "hidden_dim":    hidden_dim,
                "best_val_auc":  best_val_auc,
                "test_auc":      test_m["auc"],
                "test_ap":       test_m["ap"],
                "test_f1":       test_m["f1"],
                "n_tools":       ds.n_tools,
                "n_pos_pairs":   int(len(ds._pos_pairs)),
                "neg_ratio":     neg_ratio,
            },
            fh,
            indent=2,
        )
    print(f"[train] Metadata → {meta_path}")

    return model


# ============================================================================
# 7. Inference utilities
# ============================================================================

def load_transition_model(
    encoder_type: str = "gcn",
    device_str:   str = "cpu",
) -> tuple[TransitionModel, dict[str, int], list[str]]:
    """
    Load a trained checkpoint from models/.

    Returns
    -------
    model        : TransitionModel in eval mode
    tool_to_idx  : {tool_name: row_index}
    active_tools : list of tool names, index i ↔ row i of embeddings
    """
    device    = torch.device(device_str)
    ckpt_path = MODEL_DIR / f"gnn_transition_{encoder_type}_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train it first:  python src/gnn_transition.py --encoder {encoder_type}"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = TransitionModel(
        encoder_type=ckpt["encoder_type"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tool_list   = ckpt["tool_list"]
    tool_to_idx = {name: i for i, name in enumerate(tool_list)}

    print(
        f"[load_transition_model] Loaded {encoder_type.upper()} "
        f"(val_auc={ckpt.get('val_auc', float('nan')):.4f}, "
        f"{len(tool_list):,} tools)"
    )
    return model, tool_to_idx, tool_list


@torch.no_grad()
def precompute_transition_scores(
    model:      TransitionModel,
    device_str: str  = "cpu",
    graph_edges_only: bool = True,
) -> dict[tuple[str, str], float]:
    """
    Pre-compute GNN transition scores and return as a lookup dict.

    Parameters
    ----------
    model            : trained TransitionModel (in eval mode)
    device_str       : device for inference
    graph_edges_only : if True (default), compute scores only for edges that
                       exist in final_graph.gpickle (fast, ~39 K pairs).
                       If False, compute for ALL n² tool pairs (24.9 M for
                       n=4988; stores only scores > 0.01 to save memory).

    Returns
    -------
    dict  { (tool_i_name, tool_j_name) : float probability in [0,1] }
    """
    device = torch.device(device_str)
    model.to(device).eval()

    # Re-load graph and embeddings (authoritative source of truth)
    with GRAPH_FILE.open("rb") as fh:
        G = pickle.load(fh)

    active_tools: list[str] = sorted(G.nodes())
    t2i = {name: i for i, name in enumerate(active_tools)}
    n   = len(active_tools)

    emb = np.load(str(EMB_FILE))
    x   = torch.tensor(emb, dtype=torch.float32).to(device)

    rows, cols = [], []
    for u, v in G.edges():
        if u in t2i and v in t2i:
            rows.append(t2i[u])
            cols.append(t2i[v])
    graph_ei = torch.tensor([rows, cols], dtype=torch.long).to(device)

    # Single GNN forward pass → all node embeddings
    h = model.encode(x, graph_ei)   # (N, 256)

    score_cache: dict[tuple[str, str], float] = {}
    BATCH = 8192

    if graph_edges_only:
        # Compute scores only for existing graph edges
        E          = graph_ei.shape[1]
        all_logits = []
        for start in range(0, E, BATCH):
            end    = min(start + BATCH, E)
            ei_b   = graph_ei[:, start:end]
            logits = model.decode(h, ei_b)
            all_logits.append(logits.cpu().numpy())

        logits_np = np.concatenate(all_logits)
        probs_np  = _sigmoid_np(logits_np)

        for idx, (ri, ci) in enumerate(zip(rows, cols)):
            score_cache[(active_tools[ri], active_tools[ci])] = float(probs_np[idx])

    else:
        # All n² pairs — compute row-by-row to avoid OOM
        for i in tqdm(range(n), desc="Precomputing all pairs"):
            u_name = active_tools[i]
            # All targets in one batch
            v_idx  = torch.arange(n, device=device)
            h_u    = h[i].unsqueeze(0).expand(n, -1)   # (N, 256)
            logits = model.decoder(h_u, h).cpu().numpy()
            probs  = _sigmoid_np(logits)
            for j, p in enumerate(probs):
                if p > 0.01:   # sparse: skip near-zero scores
                    score_cache[(u_name, active_tools[j])] = float(p)

    print(f"[precompute] Score cache built: {len(score_cache):,} entries.")
    return score_cache


# ============================================================================
# Full N×N score matrix  (Fix 1: covers all tool pairs, not just graph edges)
# ============================================================================

@torch.no_grad()
def precompute_full_score_matrix(
    model:      TransitionModel,
    device_str: str = "auto",
    batch_rows: int = 16,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute GNN transition score for ALL N×N tool pairs.

    Unlike precompute_transition_scores() which only scores existing edges,
    this function scores every (tool_i, tool_j) pair by running the learned
    GNN encoder once and batching the MLP decoder over all N² combinations.

    Parameters
    ----------
    model      : trained TransitionModel (eval mode)
    device_str : 'auto' | 'cpu' | 'cuda'
    batch_rows : number of source tools to process per GPU/CPU batch.
                 Each batch allocates ≈ batch_rows × N × 768 × 4 bytes.
                 batch_rows=16 → ~325 MB on GPU (safe for 8 GB VRAM).

    Returns
    -------
    score_matrix : ndarray (N, N) float32
        score_matrix[i, j] = sigmoid(decoder(h_i, h_j)) = P(j | i)
    active_tools : list[str]
        Row/col index i  ↔  sorted(G.nodes())[i]
    """
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model.to(device).eval()

    with GRAPH_FILE.open("rb") as fh:
        G = pickle.load(fh)

    active_tools: list[str] = sorted(G.nodes())
    N   = len(active_tools)
    t2i = {name: i for i, name in enumerate(active_tools)}

    emb = np.load(str(EMB_FILE))
    x   = torch.tensor(emb, dtype=torch.float32).to(device)

    edge_rows, edge_cols = [], []
    for u, v in G.edges():
        if u in t2i and v in t2i:
            edge_rows.append(t2i[u])
            edge_cols.append(t2i[v])
    graph_ei = torch.tensor([edge_rows, edge_cols], dtype=torch.long).to(device)

    # Single GNN forward pass → node embeddings (N, hidden_dim)
    h = model.encode(x, graph_ei)

    score_matrix = np.empty((N, N), dtype=np.float32)

    for i_start in tqdm(range(0, N, batch_rows), desc="N×N GNN scores"):
        i_end = min(i_start + batch_rows, N)
        batch = i_end - i_start

        h_i = h[i_start:i_end]                          # (batch, D)
        # Views — zero extra memory until cat materialises them
        h_i_exp = h_i.unsqueeze(1).expand(-1, N, -1)    # (batch, N, D)
        h_j_exp = h.unsqueeze(0).expand(batch, -1, -1)  # (batch, N, D)

        feat = torch.cat(
            [h_i_exp, h_j_exp, h_i_exp * h_j_exp], dim=-1
        ).reshape(batch * N, -1)                         # (batch*N, 3D)

        logits = model.decoder.mlp(feat).squeeze(-1)     # (batch*N,)
        probs  = torch.sigmoid(logits).reshape(batch, N) # (batch, N)
        score_matrix[i_start:i_end] = probs.cpu().numpy()

    mb = score_matrix.nbytes / 1e6
    print(f"[precompute_full] {N}×{N} = {N * N / 1e6:.1f}M pairs  ({mb:.1f} MB)")
    return score_matrix, active_tools


def save_score_matrix(
    score_matrix: np.ndarray,
    active_tools: list[str],
    encoder_type: str = "sage",
) -> None:
    """Persist (N, N) score matrix and tool list to models/."""
    mat_path   = MODEL_DIR / f"gnn_score_matrix_{encoder_type}.npy"
    tools_path = MODEL_DIR / f"gnn_score_tools_{encoder_type}.json"
    np.save(str(mat_path), score_matrix)
    with tools_path.open("w", encoding="utf-8") as fh:
        json.dump(active_tools, fh, ensure_ascii=False)
    print(f"[save_score_matrix] {mat_path.name}  "
          f"({mat_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[save_score_matrix] {tools_path.name}")


def load_score_matrix(
    encoder_type: str = "sage",
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """
    Load the pre-computed (N, N) score matrix from models/.

    Returns
    -------
    score_matrix : ndarray (N, N) float32
    active_tools : list[str]
    t2i          : dict {tool_name: row/col index}
    """
    mat_path   = MODEL_DIR / f"gnn_score_matrix_{encoder_type}.npy"
    tools_path = MODEL_DIR / f"gnn_score_tools_{encoder_type}.json"

    if not mat_path.exists():
        raise FileNotFoundError(
            f"{mat_path} not found.\n"
            f"Generate it with:\n"
            f"  python src/gnn_transition.py --encoder {encoder_type} "
            f"--precompute-full"
        )

    score_matrix = np.load(str(mat_path))
    with tools_path.open(encoding="utf-8") as fh:
        active_tools = json.load(fh)
    t2i = {name: i for i, name in enumerate(active_tools)}

    print(f"[load_score_matrix] {score_matrix.shape}  "
          f"({score_matrix.nbytes / 1e6:.1f} MB)")
    return score_matrix, active_tools, t2i


def get_gnn_transition_score(
    tool_i:      str,
    tool_j:      str,
    score_cache: dict[tuple[str, str], float],
    default:     float = 0.0,
) -> float:
    """
    Look up the GNN-predicted transition probability  P(tool_j | tool_i).

    Parameters
    ----------
    tool_i, tool_j : str
        Exact tool names as they appear in the graph nodes.
    score_cache    : dict
        Pre-computed cache from precompute_transition_scores().
    default        : float
        Returned for pairs not present in the cache (default 0.0).

    Returns
    -------
    float in [0, 1]
    """
    return score_cache.get((tool_i, tool_j), default)


def build_score_lookup(
    encoder_type:     str  = "gcn",
    device_str:       str  = "cpu",
    graph_edges_only: bool = True,
    cache_to_disk:    bool = True,
) -> tuple[dict[tuple[str, str], float], Callable[[str, str], float]]:
    """
    Convenience wrapper: load model → precompute scores → return
    (score_cache, callable).

    The returned callable has signature:
        score_fn(tool_i: str, tool_j: str) -> float

    Optionally persists the cache to  models/gnn_transition_{enc}_score_cache.pkl
    for instant reload on the next run.

    Example
    -------
    >>> cache, score_fn = build_score_lookup("gcn")
    >>> score_fn("search_for_google", "click_for_browser")
    0.73
    """
    cache_path = MODEL_DIR / f"gnn_transition_{encoder_type}_score_cache.pkl"

    # Fast path: load pre-computed cache if it exists
    if cache_to_disk and cache_path.exists():
        print(f"[build_score_lookup] Loading cached scores from {cache_path} …")
        with cache_path.open("rb") as fh:
            score_cache = pickle.load(fh)
        print(f"[build_score_lookup] {len(score_cache):,} entries loaded.")
    else:
        model, _, _ = load_transition_model(encoder_type, device_str)
        score_cache = precompute_transition_scores(
            model, device_str=device_str, graph_edges_only=graph_edges_only
        )
        if cache_to_disk:
            with cache_path.open("wb") as fh:
                pickle.dump(score_cache, fh, protocol=4)
            print(f"[build_score_lookup] Cache persisted → {cache_path}")

    def score_fn(tool_i: str, tool_j: str) -> float:
        return get_gnn_transition_score(tool_i, tool_j, score_cache)

    return score_cache, score_fn


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train / evaluate GNN tool-transition model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder", choices=["gcn", "gat", "sage"], default="gcn",
        help="GNN encoder variant",
    )
    parser.add_argument("--hidden-dim",  type=int,   default=HIDDEN_DIM,
                        help="GNN hidden dimension")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--patience",    type=int,   default=20,
                        help="Early-stopping patience on val AUC")
    parser.add_argument("--neg-ratio",   type=int,   default=NEG_RATIO,
                        help="Negative edges per positive edge")
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--device",      type=str,   default="auto",
                        help="'auto', 'cpu', or 'cuda'")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; load existing checkpoint and print test metrics",
    )
    parser.add_argument(
        "--precompute", action="store_true",
        help="After training / loading, precompute and cache all edge scores",
    )
    parser.add_argument(
        "--precompute-full", action="store_true",
        help="Compute GNN scores for ALL N×N tool pairs and save as score matrix",
    )
    parser.add_argument(
        "--all-pairs", action="store_true",
        help="With --precompute: compute scores for ALL n² pairs, not just graph edges",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Resolve "auto" → concrete device string early so all functions receive a valid value
    _resolved_device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    if args.eval_only:
        # Load existing checkpoint and evaluate on test set
        model, _, _ = load_transition_model(
            encoder_type=args.encoder, device_str=_resolved_device
        )
        ds   = TransitionDataset(neg_ratio=args.neg_ratio)
        data = ds.get_pyg_data()

        device = torch.device(_resolved_device)
        x        = data.x.to(device)
        graph_ei = data.edge_index.to(device)
        test_pos = data.test_pos_edge_index.to(device)
        test_neg = data.test_neg_edge_index.to(device)

        model.to(device)
        test_m = _evaluate_split(model, x, graph_ei, test_pos, test_neg)
        print(
            f"\nTEST ({args.encoder.upper()}):\n"
            f"  AUC={test_m['auc']:.4f}  "
            f"AP={test_m['ap']:.4f}  "
            f"F1={test_m['f1']:.4f}"
        )
    else:
        model = train_transition_model(
            encoder_type=args.encoder,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            max_epochs=args.epochs,
            patience=args.patience,
            neg_ratio=args.neg_ratio,
            dropout=args.dropout,
            device_str=_resolved_device,
        )

    if args.precompute:
        if args.eval_only:
            model, _, _ = load_transition_model(args.encoder, _resolved_device)

        cache_path = MODEL_DIR / f"gnn_transition_{args.encoder}_score_cache.pkl"
        score_cache = precompute_transition_scores(
            model,
            device_str=_resolved_device,
            graph_edges_only=not args.all_pairs,
        )
        with cache_path.open("wb") as fh:
            pickle.dump(score_cache, fh, protocol=4)
        print(f"[precompute] Score cache saved → {cache_path}")

    if args.precompute_full:
        if args.eval_only and not args.precompute:
            model, _, _ = load_transition_model(args.encoder, _resolved_device)

        score_matrix, active_tools = precompute_full_score_matrix(
            model, device_str=_resolved_device
        )
        save_score_matrix(score_matrix, active_tools, encoder_type=args.encoder)
