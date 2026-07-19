"""Inference: ESM fold-head distance restraints → CA optimize → torsion fit."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .clash_assembly import dihedrals_to_ca
from .config import (
    CKPT_DIR,
    ESM_MODEL_NAME,
    FOLD_HEAD_CA_STEPS,
    FOLD_HEAD_CKPT_NAME,
    FOLD_HEAD_D_MODEL,
    FOLD_HEAD_EMB_DIM,
    FOLD_HEAD_FIT_STEPS,
    FOLD_HEAD_MAX_LEN,
    FOLD_HEAD_PAIR_DIM,
)
from .contact_ca_fold import fit_torsions_to_ca
from .model.fold_head import ESMFoldHead
from .soft_contact_fold import rank_decoy


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_HEAD_CACHE: Dict[str, Optional[ESMFoldHead]] = {}


def load_fold_head(ckpt_path: Optional[Path] = None) -> Optional[ESMFoldHead]:
    path = Path(ckpt_path or (CKPT_DIR / FOLD_HEAD_CKPT_NAME))
    key = str(path)
    if key in _HEAD_CACHE:
        return _HEAD_CACHE[key]
    if not path.exists():
        _HEAD_CACHE[key] = None
        return None
    dev = _device()
    ckpt = torch.load(path, map_location=dev)
    emb_dim = int(ckpt.get("emb_dim", FOLD_HEAD_EMB_DIM))
    model = ESMFoldHead(
        emb_dim=emb_dim,
        d_model=int(ckpt.get("d_model", FOLD_HEAD_D_MODEL)),
        pair_dim=int(ckpt.get("pair_dim", FOLD_HEAD_PAIR_DIM)),
        max_len=int(ckpt.get("max_len", FOLD_HEAD_MAX_LEN)),
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _HEAD_CACHE[key] = model
    return model


@torch.no_grad()
def predict_distance_map(
    sequence: str,
    emb: Optional[np.ndarray] = None,
    model: Optional[ESMFoldHead] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (dist_Å, conf) or (None, None)."""
    seq = "".join(c for c in sequence.upper() if c.isalpha())
    n = len(seq)
    if n < 8 or n > FOLD_HEAD_MAX_LEN + 32:
        return None, None
    head = model or load_fold_head()
    if head is None:
        return None, None
    if emb is None:
        from .esm_contacts import get_esm_predictor

        esm = get_esm_predictor(ESM_MODEL_NAME)
        if not esm.enabled:
            return None, None
        emb = esm.embeddings(seq, use_cache=True)
    if emb.shape[0] != n:
        return None, None
    dev = next(head.parameters()).device
    # Crop center if slightly over max_len training size — still run full if small overflow
    emb_t = torch.tensor(emb, dtype=torch.float32, device=dev).unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool, device=dev)
    dist, conf_logit = head(emb_t, mask)
    conf = torch.sigmoid(conf_logit)
    return (
        dist[0].cpu().numpy().astype(np.float64),
        conf[0].cpu().numpy().astype(np.float64),
    )


def optimize_ca_to_distances(
    ca0: np.ndarray,
    dist_target: np.ndarray,
    conf: np.ndarray,
    min_sep: int = 3,
    n_steps: int = 500,
    lr: float = 0.06,
    seed: int = 0,
    conf_thresh: float = 0.25,
    local_weight: float = 0.04,
) -> Tuple[np.ndarray, float, float]:
    """Pull Cα beads toward predicted pairwise distances."""
    dev = _device()
    torch.manual_seed(seed)
    ca0_t = torch.tensor(ca0, dtype=torch.float32, device=dev)
    n = ca0_t.shape[0]
    idx_i, idx_j, tgt, wts = [], [], [], []
    for i in range(n):
        for j in range(i + min_sep, n):
            c = float(conf[i, j])
            if c < conf_thresh:
                continue
            idx_i.append(i)
            idx_j.append(j)
            tgt.append(float(dist_target[i, j]))
            # Prefer confident mid-range restraints
            wts.append(c * c)
    if not idx_i:
        return ca0, 0.0, 0.0
    ii = torch.tensor(idx_i, dtype=torch.long, device=dev)
    jj = torch.tensor(idx_j, dtype=torch.long, device=dev)
    tt = torch.tensor(tgt, dtype=torch.float32, device=dev)
    ww = torch.tensor(wts, dtype=torch.float32, device=dev)

    delta = torch.zeros_like(ca0_t, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    def energy(x: torch.Tensor) -> torch.Tensor:
        dvec = x[1:] - x[:-1]
        bl = torch.sqrt((dvec * dvec).sum(-1) + 1e-8)
        e_bond = 40.0 * ((bl - 3.8) ** 2).sum()
        dv = x[jj] - x[ii]
        dist = torch.sqrt((dv * dv).sum(-1) + 1e-8)
        e_dist = (ww * (dist - tt) ** 2).sum()
        e_clash = x.new_zeros(())
        for sep in range(3, min(10, n)):
            d2 = x[sep:] - x[: n - sep]
            d = torch.sqrt((d2 * d2).sum(-1) + 1e-8)
            e_clash = e_clash + 4.0 * (torch.relu(3.4 - d) ** 2).sum()
        e_local = float(local_weight) * ((x - ca0_t) ** 2).sum()
        return e_bond + 2.5 * e_dist + e_clash + e_local

    e0 = float(energy(ca0_t).detach().cpu())
    best_x = ca0_t.detach().clone()
    best_e = e0
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        e = energy(ca0_t + delta)
        e.backward()
        torch.nn.utils.clip_grad_norm_([delta], 5.0)
        opt.step()
        if step == n_steps // 2:
            for g in opt.param_groups:
                g["lr"] *= 0.4
        with torch.no_grad():
            val = float(e.detach().cpu())
            if val < best_e:
                best_e = val
                best_x = (ca0_t + delta).detach().clone()
    return best_x.cpu().numpy().astype(np.float64), e0, best_e


def decoy_distance_score(
    sequence: str,
    phis: Sequence[float],
    psis: Sequence[float],
) -> Optional[float]:
    """Higher is better: negative MAE vs fold-head distance map. None if unavailable."""
    dist, conf = predict_distance_map(sequence)
    if dist is None or conf is None:
        return None
    ca = dihedrals_to_ca(list(phis), list(psis))
    return -_dist_mae(ca, dist, conf)


def _dist_mae(ca: np.ndarray, dist_t: np.ndarray, conf: np.ndarray, thresh: float = 0.3) -> float:
    n = len(ca)
    err = 0.0
    wsum = 0.0
    for i in range(n):
        for j in range(i + 3, n):
            c = float(conf[i, j])
            if c < thresh:
                continue
            d = float(np.linalg.norm(ca[j] - ca[i]))
            err += c * abs(d - float(dist_t[i, j]))
            wsum += c
    return err / max(wsum, 1e-6)


def fold_head_scaffold(
    sequence: str,
    phis0: Sequence[float],
    psis0: Sequence[float],
    contact_probs: Optional[np.ndarray] = None,
    n_steps: int = FOLD_HEAD_CA_STEPS,
    fit_steps: int = FOLD_HEAD_FIT_STEPS,
    seed: int = 0,
) -> Dict:
    """Apply fold head; accept via rank_decoy OR clear distance-MAE gain + physics OK."""
    from .tertiary import score_tertiary

    seq = "".join(c for c in sequence.upper() if c.isalpha())
    ph0 = list(phis0)
    ps0 = list(psis0)
    base_rank = rank_decoy(seq, ph0, ps0, contact_probs)
    phys0 = float(score_tertiary(seq, ph0, ps0, anchors=None)["total"])
    dist, conf = predict_distance_map(seq)
    if dist is None or conf is None:
        return {
            "phis_deg": ph0,
            "psis_deg": ps0,
            "improved": False,
            "base_rank": base_rank,
            "best_rank": base_rank,
            "source": "none",
        }
    ca0 = dihedrals_to_ca(ph0, ps0)
    mae0 = _dist_mae(ca0, dist, conf)
    ca1, e0, e1 = optimize_ca_to_distances(
        ca0, dist, conf, n_steps=n_steps, seed=seed, conf_thresh=0.22, local_weight=0.03
    )
    if e1 >= 0.995 * e0 and e0 > 0:
        return {
            "phis_deg": ph0,
            "psis_deg": ps0,
            "improved": False,
            "base_rank": base_rank,
            "best_rank": base_rank,
            "source": "local",
        }
    ph, ps, fit_r = fit_torsions_to_ca(ca1, ph0, ps0, n_steps=fit_steps, seed=seed + 3)
    cand_rank = rank_decoy(seq, ph, ps, contact_probs)
    phys1 = float(score_tertiary(seq, ph, ps, anchors=None)["total"])
    mae1 = _dist_mae(dihedrals_to_ca(ph, ps), dist, conf)
    # Require BOTH rank and distance-MAE gains — either alone caused RMSD regressions
    rank_ok = cand_rank > base_rank + 0.35
    mae_ok = mae1 < mae0 - 0.55
    phys_ok = phys1 >= phys0 - 0.25
    improved = rank_ok and mae_ok and phys_ok
    if not improved:
        return {
            "phis_deg": ph0,
            "psis_deg": ps0,
            "improved": False,
            "base_rank": base_rank,
            "best_rank": cand_rank,
            "fit_rmsd": fit_r,
            "mae0": mae0,
            "mae1": mae1,
            "source": "local",
        }
    return {
        "phis_deg": ph.tolist(),
        "psis_deg": ps.tolist(),
        "improved": True,
        "base_rank": base_rank,
        "best_rank": max(cand_rank, base_rank + (mae0 - mae1)),
        "fit_rmsd": fit_r,
        "mae0": mae0,
        "mae1": mae1,
        "source": "fold_head",
        "e0": e0,
        "e1": e1,
    }
