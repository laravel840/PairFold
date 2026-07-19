"""Soft full-map contact folding via torch CA optimization.

Uses the entire ESM contact probability matrix (not just top-k anchors),
which keeps weak-but-useful signal (important for 1CRN-like maps).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .clash_assembly import dihedrals_to_ca
from .contact_ca_fold import fit_torsions_to_ca
from .tertiary import score_tertiary


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_ca_soft_contacts(
    ca0: np.ndarray,
    contact_probs: np.ndarray,
    min_sep: int = 6,
    n_steps: int = 400,
    lr: float = 0.05,
    contact_thresh: float = 0.15,
    seed: int = 0,
    contact_weight: float = 2.2,
    local_weight: float = 0.08,
) -> Tuple[np.ndarray, float, float]:
    """
    Optimize Cα beads to satisfy soft long-range contacts.

    Energy:
      bonds + soft clash + sum p_ij * relu(d_ij - 8)^2  for p_ij >= contact_thresh
    """
    dev = _device()
    torch.manual_seed(seed)
    ca0_t = torch.tensor(ca0, dtype=torch.float32, device=dev)
    probs = torch.tensor(contact_probs, dtype=torch.float32, device=dev)
    n = ca0_t.shape[0]
    if n < 6:
        return ca0, 0.0, 0.0

    # Pair index list (upper triangle, long-range, above thresh)
    idx_i, idx_j, weights = [], [], []
    for i in range(n):
        for j in range(i + min_sep, n):
            p = float(probs[i, j])
            if p < contact_thresh:
                continue
            idx_i.append(i)
            idx_j.append(j)
            # Emphasize confident contacts without ignoring mid ones
            weights.append(p ** 1.5)
    if not idx_i:
        return ca0, 0.0, 0.0

    ii = torch.tensor(idx_i, dtype=torch.long, device=dev)
    jj = torch.tensor(idx_j, dtype=torch.long, device=dev)
    w = torch.tensor(weights, dtype=torch.float32, device=dev)

    # Optimize displacement from local scaffold (keeps chain more stable)
    delta = torch.zeros_like(ca0_t, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    def energy(x: torch.Tensor) -> torch.Tensor:
        # Bonds
        dvec = x[1:] - x[:-1]
        bl = torch.sqrt((dvec * dvec).sum(-1) + 1e-8)
        e_bond = 35.0 * ((bl - 3.8) ** 2).sum()
        # Soft contacts: pull pairs under ~8 Å
        dv = x[jj] - x[ii]
        dist = torch.sqrt((dv * dv).sum(-1) + 1e-8)
        e_contact = (w * torch.relu(dist - 8.0) ** 2).sum()
        # Mild preferred distance ~6.5 for very confident pairs
        hi = w > 0.7
        if hi.any():
            e_contact = e_contact + 0.35 * (w[hi] * (dist[hi] - 6.5) ** 2).sum()
        # Local clash
        e_clash = x.new_zeros(())
        for sep in range(3, min(10, n)):
            d2 = x[sep:] - x[: n - sep]
            d = torch.sqrt((d2 * d2).sum(-1) + 1e-8)
            e_clash = e_clash + 5.0 * (torch.relu(3.5 - d) ** 2).sum()
        # Stay near local init (regularizer) — lower for gated maps that need motion
        e_anchor = float(local_weight) * ((x - ca0_t) ** 2).sum()
        return e_bond + float(contact_weight) * e_contact + e_clash + e_anchor

    e0 = float(energy(ca0_t).detach().cpu())
    best_x = ca0_t.detach().clone()
    best_e = e0

    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        x = ca0_t + delta
        e = energy(x)
        e.backward()
        torch.nn.utils.clip_grad_norm_([delta], 5.0)
        opt.step()
        # anneal lr
        if step == n_steps // 2:
            for g in opt.param_groups:
                g["lr"] *= 0.4
        with torch.no_grad():
            val = float(e.detach().cpu())
            if val < best_e:
                best_e = val
                best_x = (ca0_t + delta).detach().clone()

    out = best_x.cpu().numpy().astype(np.float64)
    return out, e0, best_e


def rank_decoy(
    sequence: str,
    phis: Sequence[float],
    psis: Sequence[float],
    contact_probs: Optional[np.ndarray] = None,
    min_sep: int = 6,
) -> float:
    """
    Higher is better. Physics + soft contact satisfaction (not sparse-anchor cheating).
    """
    phys = float(score_tertiary(sequence, phis, psis, anchors=None)["total"])
    if contact_probs is None:
        return phys
    ca = dihedrals_to_ca(phis, psis)
    n = len(ca)
    bonus = 0.0
    wsum = 0.0
    for i in range(n):
        for j in range(i + min_sep, n):
            p = float(contact_probs[i, j])
            if p < 0.2:
                continue
            d = float(np.linalg.norm(ca[j] - ca[i]))
            # Reward being under 8Å; mild reward near 6–7
            sat = max(0.0, 1.0 - max(0.0, d - 8.0) / 8.0)
            if d < 8.0:
                sat += 0.15 * max(0.0, 1.0 - abs(d - 6.5) / 3.0)
            bonus += (p ** 1.5) * sat
            wsum += p ** 1.5
    contact_term = bonus / max(wsum, 1e-6)
    return phys + 1.8 * contact_term


def soft_map_scaffold(
    sequence: str,
    phis0: Sequence[float],
    psis0: Sequence[float],
    contact_probs: np.ndarray,
    n_steps: int = 400,
    fit_steps: int = 1400,
    seed: int = 0,
    contact_thresh: float = 0.15,
) -> Dict:
    """Soft-map CA optimize → torsion fit → ranked vs local."""
    seq = "".join(c for c in sequence.upper() if c.isalpha())
    ph0 = list(phis0)
    ps0 = list(psis0)
    ca0 = dihedrals_to_ca(ph0, ps0)
    base_rank = rank_decoy(seq, ph0, ps0, contact_probs)

    ca1, e0, e1 = optimize_ca_soft_contacts(
        ca0,
        contact_probs,
        n_steps=n_steps,
        seed=seed,
        contact_thresh=contact_thresh,
        contact_weight=4.0,
        local_weight=0.02,
        lr=0.08,
    )
    if e1 >= 0.98 * e0:
        return {
            "phis_deg": ph0,
            "psis_deg": ps0,
            "improved": False,
            "base_rank": base_rank,
            "best_rank": base_rank,
            "source": "local",
        }

    ph, ps, fit_r = fit_torsions_to_ca(ca1, ph0, ps0, n_steps=fit_steps, seed=seed + 11)
    cand_rank = rank_decoy(seq, ph, ps, contact_probs)
    # Also require physics not to collapse
    phys_base = float(score_tertiary(seq, ph0, ps0, anchors=None)["total"])
    phys_cand = float(score_tertiary(seq, ph, ps, anchors=None)["total"])
    # Gated-only path: allow moderate gains (1CRN needs this)
    improved = cand_rank > base_rank + 0.15 and phys_cand >= phys_base - 1.0
    # Selection key: physics first (contact-heavy rank overfits wrong pairs)
    select_score = phys_cand + 0.35 * (cand_rank - phys_cand)
    if not improved:
        return {
            "phis_deg": ph0,
            "psis_deg": ps0,
            "improved": False,
            "base_rank": base_rank,
            "best_rank": cand_rank,
            "phys_base": phys_base,
            "phys_cand": phys_cand,
            "select_score": select_score,
            "fit_rmsd": fit_r,
            "source": "local",
        }
    return {
        "phis_deg": ph.tolist(),
        "psis_deg": ps.tolist(),
        "improved": True,
        "base_rank": base_rank,
        "best_rank": cand_rank,
        "phys_base": phys_base,
        "phys_cand": phys_cand,
        "select_score": select_score,
        "fit_rmsd": fit_r,
        "source": "soft_map",
        "e0": e0,
        "e1": e1,
    }
