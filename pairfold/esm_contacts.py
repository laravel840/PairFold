"""ESM-2 contact / embedding helpers (consumer-GPU friendly).

Uses fair-esm pretrained models. Default: esm2_t6_8M_UR50D (~56 MB VRAM).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # Dict used by predictor cache

import numpy as np
import torch

from .config import (
    CONTACT_MIN_SEP,
    CONTACT_SCORE_THRESH,
    CONTACT_TOP_K,
    DATA_DIR,
)

ESM_CACHE_DIR = DATA_DIR / "esm_cache"
DEFAULT_ESM_NAME = "esm2_t12_35M_UR50D"
# Typical Cα–Cα distance for a true long-range contact
DEFAULT_CONTACT_DIST_A = 7.0


class ESMContactPredictor:
    """Pretrained ESM-2 → long-range contact anchors (+ optional embeddings)."""

    def __init__(
        self,
        model_name: str = DEFAULT_ESM_NAME,
        device: Optional[torch.device] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir or ESM_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = False
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self.repr_layer: Optional[int] = None
        self._load()

    def _load(self) -> None:
        try:
            import esm
        except ImportError as e:
            raise ImportError(
                "fair-esm is required for ESM contacts. Install: pip install fair-esm"
            ) from e

        loaders = {
            "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
            "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
            "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
        }
        if self.model_name not in loaders:
            raise ValueError(
                f"Unknown ESM model {self.model_name}; choose from {list(loaders)}"
            )

        model, alphabet = loaders[self.model_name]()
        model = model.eval().to(self.dev)
        if self.dev.type == "cuda":
            model = model.half()
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.repr_layer = model.num_layers
        self.enabled = True

    def _cache_key(self, sequence: str, kind: str) -> Path:
        h = hashlib.sha1(
            f"{self.model_name}|{kind}|{sequence}".encode("utf-8")
        ).hexdigest()[:20]
        return self.cache_dir / f"{kind}_{h}.npy"

    def contact_probs(self, sequence: str, use_cache: bool = True) -> np.ndarray:
        """Return L×L contact probability matrix (float32)."""
        seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
        n = len(seq)
        if n < 2:
            return np.zeros((n, n), dtype=np.float32)

        path = self._cache_key(seq, "contacts")
        if use_cache and path.exists():
            arr = np.load(path)
            if arr.shape == (n, n):
                return arr.astype(np.float32)

        assert self.model is not None and self.batch_converter is not None
        _, _, tokens = self.batch_converter([("query", seq)])
        tokens = tokens.to(self.dev)
        with torch.no_grad():
            if self.dev.type == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    contacts = self.model.predict_contacts(tokens)[0]
            else:
                contacts = self.model.predict_contacts(tokens)[0]
        probs = contacts.float().cpu().numpy().astype(np.float32)
        probs = 0.5 * (probs + probs.T)
        if use_cache:
            np.save(path, probs)
        return probs

    def embeddings(self, sequence: str, use_cache: bool = True) -> np.ndarray:
        """Per-residue representations (L, D) float32 from final layer."""
        seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
        n = len(seq)
        if n < 1:
            return np.zeros((0, 0), dtype=np.float32)

        path = self._cache_key(seq, "emb")
        if use_cache and path.exists():
            arr = np.load(path)
            if arr.shape[0] == n:
                return arr.astype(np.float32)

        assert self.model is not None and self.batch_converter is not None
        assert self.repr_layer is not None
        _, _, tokens = self.batch_converter([("query", seq)])
        tokens = tokens.to(self.dev)
        with torch.no_grad():
            if self.dev.type == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    out = self.model(
                        tokens, repr_layers=[self.repr_layer], return_contacts=False
                    )
            else:
                out = self.model(
                    tokens, repr_layers=[self.repr_layer], return_contacts=False
                )
            emb_t = out["representations"][self.repr_layer][0, 1 : n + 1].float().cpu()
        emb_np = emb_t.numpy().astype(np.float32)
        if use_cache:
            np.save(path, emb_np)
        return emb_np

    def top_anchors(
        self,
        sequence: str,
        top_k: int = CONTACT_TOP_K,
        score_thresh: float = CONTACT_SCORE_THRESH,
        min_sep: int = CONTACT_MIN_SEP,
        contact_dist: float = DEFAULT_CONTACT_DIST_A,
        use_cache: bool = True,
    ) -> Dict:
        """Sparse long-range anchors: (i, j, target_dist_Å)."""
        seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
        n = len(seq)
        empty = {
            "enabled": self.enabled,
            "anchors": [],
            "contacts": [],
            "n_candidates": 0,
            "mean_score": 0.0,
            "ckpt": f"esm:{self.model_name}",
            "source": "esm",
        }
        if not self.enabled or n < min_sep + 1:
            return empty

        probs = self.contact_probs(seq, use_cache=use_cache)

        def collect(thresh: float) -> List[Tuple[float, int, int, float]]:
            out: List[Tuple[float, int, int, float]] = []
            for i in range(n):
                for j in range(i + min_sep, n):
                    p = float(probs[i, j])
                    if p < thresh:
                        continue
                    d = float(contact_dist) - 1.2 * max(0.0, p - 0.5)
                    d = float(min(max(d, 4.5), 10.0))
                    out.append((p, i, j, d))
            out.sort(key=lambda x: -x[0])
            return out

        candidates = collect(score_thresh)
        # Mild relax only if almost empty
        if len(candidates) < 2:
            candidates = collect(max(0.30, score_thresh * 0.7))

        # Adaptive k: sharp maps keep slightly more; never flood with weak pairs
        if candidates:
            top5 = [c[0] for c in candidates[:5]]
            sharp = float(np.mean(top5)) >= 0.90
            k_cap = int(top_k) + (2 if sharp else 0)
        else:
            k_cap = int(top_k)
        k = min(k_cap, max(1, n // 4), len(candidates))
        chosen = candidates[:k]
        anchors = [(i, j, d) for _, i, j, d in chosen]
        contact_list = [
            {"i": i, "j": j, "score": round(p, 4), "dist": round(d, 2)}
            for p, i, j, d in candidates[:80]
        ]
        return {
            "enabled": True,
            "anchors": anchors,
            "contacts": contact_list,
            "n_candidates": len(candidates),
            "mean_score": float(np.mean([c[0] for c in chosen])) if chosen else 0.0,
            "ckpt": f"esm:{self.model_name}",
            "source": "esm",
            "top_k": k,
            "model_name": self.model_name,
        }


_ESM_CACHE: Dict[str, ESMContactPredictor] = {}


def get_esm_predictor(
    model_name: str = DEFAULT_ESM_NAME,
    device: Optional[torch.device] = None,
) -> ESMContactPredictor:
    """Cache one predictor per model name (allows t12 + t30 side-by-side)."""
    key = str(model_name)
    if key not in _ESM_CACHE:
        _ESM_CACHE[key] = ESMContactPredictor(model_name=model_name, device=device)
    return _ESM_CACHE[key]
