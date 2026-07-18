"""ContactPairNet inference helpers: logits → top-k distance anchors."""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import (
    AA_LIST,
    CKPT_DIR,
    CONTACT_CKPT_NAME,
    CONTACT_D_FF,
    CONTACT_D_MODEL,
    CONTACT_INFER_MAX_LEN,
    CONTACT_MIN_SEP,
    CONTACT_N_HEADS,
    CONTACT_N_LAYERS,
    CONTACT_SCORE_THRESH,
    CONTACT_TOP_K,
    PAD_IDX,
    UNK_IDX,
    VOCAB_SIZE,
)
from .model.contact_net import ContactPairNet


def _aa_to_idx(ch: str) -> int:
    i = AA_LIST.find(ch)
    return i if i >= 0 else UNK_IDX


class ContactPredictor:
    """Lazy-loaded ContactPairNet → sparse (i, j, dist) anchors."""

    def __init__(self, ckpt_path: Optional[Path] = None, device: Optional[torch.device] = None) -> None:
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = Path(ckpt_path or CKPT_DIR / CONTACT_CKPT_NAME)
        self.enabled = path.exists()
        self.ckpt_path = str(path) if self.enabled else ""
        self.model: Optional[ContactPairNet] = None
        self.max_len = CONTACT_INFER_MAX_LEN
        self.min_sep = CONTACT_MIN_SEP
        if not self.enabled:
            return
        ckpt = torch.load(path, map_location=self.dev)
        cfg = ckpt.get("config") or {}
        self.max_len = int(cfg.get("max_len", CONTACT_INFER_MAX_LEN))
        self.min_sep = int(cfg.get("min_sep", CONTACT_MIN_SEP))
        self.model = ContactPairNet(
            vocab_size=int(cfg.get("vocab_size", VOCAB_SIZE)),
            max_len=self.max_len,
            d_model=int(cfg.get("d_model", CONTACT_D_MODEL)),
            n_heads=int(cfg.get("n_heads", CONTACT_N_HEADS)),
            n_layers=int(cfg.get("n_layers", CONTACT_N_LAYERS)),
            d_ff=int(cfg.get("d_ff", CONTACT_D_FF)),
            dropout=0.0,
        ).to(self.dev)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _forward_window(self, seq: str) -> Tuple[np.ndarray, np.ndarray]:
        assert self.model is not None
        L = len(seq)
        assert L <= self.max_len
        tokens = [_aa_to_idx(c) for c in seq] + [PAD_IDX] * (self.max_len - L)
        mask = [1] * L + [0] * (self.max_len - L)
        t = torch.tensor([tokens], dtype=torch.long, device=self.dev)
        m = torch.tensor([mask], dtype=torch.bool, device=self.dev)
        with torch.no_grad():
            logits, dist = self.model(t, m)
        logits_np = logits[0, :L, :L].float().cpu().numpy()
        dist_np = dist[0, :L, :L].float().cpu().numpy()
        return logits_np, dist_np

    def predict_maps(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (contact_prob LxL, dist_pred LxL) for short chains only.

        For N > max_len this used to allocate three N×N float32 matrices
        (≈ 3 × N² × 4 bytes — ~7.5 GB at N=25k). Callers that need anchors
        on long chains must use top_anchors() which stays sparse.
        """
        seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
        n = len(seq)
        if not self.enabled or self.model is None or n < self.min_sep + 1:
            z = np.zeros((n, n), dtype=np.float32)
            return z, z.copy()

        if n > self.max_len:
            raise ValueError(
                f"predict_maps refuses N={n} > max_len={self.max_len}; "
                "use top_anchors() for sparse long-chain inference"
            )

        logits, dist = self._forward_window(seq)
        return 1.0 / (1.0 + np.exp(-logits)), dist

    def _crop_candidates(
        self,
        seq: str,
        offset: int,
        min_sep: int,
        score_thresh: float,
        keep: int,
    ) -> List[Tuple[float, int, int, float]]:
        """Top contacts inside one crop, mapped to global indices."""
        logits, dist = self._forward_window(seq)
        L = len(seq)
        probs = 1.0 / (1.0 + np.exp(-logits))
        local: List[Tuple[float, int, int, float]] = []
        for i in range(L):
            for j in range(i + min_sep, L):
                p = float(probs[i, j])
                if p < score_thresh:
                    continue
                d = float(min(max(float(dist[i, j]), 3.8), 12.0))
                local.append((p, offset + i, offset + j, d))
        local.sort(key=lambda x: -x[0])
        return local[:keep]

    def top_anchors(
        self,
        sequence: str,
        top_k: int = CONTACT_TOP_K,
        score_thresh: float = CONTACT_SCORE_THRESH,
        min_sep: Optional[int] = None,
    ) -> Dict:
        """
        Select top-k long-range contacts as (i, j, target_dist_Å) anchors.

        Long chains use overlapping crops and a bounded heap — never N×N maps.
        """
        seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
        n = len(seq)
        min_sep = int(min_sep if min_sep is not None else self.min_sep)
        empty = {
            "enabled": self.enabled,
            "anchors": [],
            "contacts": [],
            "n_candidates": 0,
            "mean_score": 0.0,
            "ckpt": self.ckpt_path,
        }
        if not self.enabled or self.model is None or n < min_sep + 1:
            return empty

        # Min-heap of (score, i, j, dist); keep more than top_k for the API list
        pool_k = max(int(top_k) * 4, 50)
        heap: List[Tuple[float, int, int, float]] = []

        def push_cand(p: float, i: int, j: int, d: float) -> None:
            item = (p, i, j, d)
            if len(heap) < pool_k:
                heapq.heappush(heap, item)
            elif p > heap[0][0]:
                heapq.heapreplace(heap, item)

        if n <= self.max_len:
            for p, i, j, d in self._crop_candidates(
                seq, 0, min_sep, score_thresh, keep=pool_k * 2
            ):
                push_cand(p, i, j, d)
        else:
            stride = max(self.max_len // 2, 32)
            starts = list(range(0, max(n - self.max_len, 0) + 1, stride))
            if starts[-1] != n - self.max_len:
                starts.append(n - self.max_len)
            per_crop = max(pool_k // 2, top_k)
            for s in starts:
                sub = seq[s : s + self.max_len]
                for p, i, j, d in self._crop_candidates(
                    sub, s, min_sep, score_thresh, keep=per_crop
                ):
                    push_cand(p, i, j, d)

        candidates = sorted(heap, key=lambda x: -x[0])
        k = min(int(top_k), max(1, n // 4), len(candidates))
        chosen = candidates[:k]
        anchors = [(i, j, d) for _, i, j, d in chosen]
        contact_list = [
            {"i": i, "j": j, "score": round(p, 4), "dist": round(d, 2)}
            for p, i, j, d in candidates[:50]
        ]
        return {
            "enabled": True,
            "anchors": anchors,
            "contacts": contact_list,
            "n_candidates": len(candidates),
            "mean_score": float(np.mean([c[0] for c in chosen])) if chosen else 0.0,
            "ckpt": self.ckpt_path,
            "top_k": k,
        }
