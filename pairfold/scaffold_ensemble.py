"""Stronger Stage-1 global scaffold: distance geometry + ensemble selection.

Builds Cα scaffolds from contact restraints (MDS → gradient refine → torsion fit),
runs multiple candidates (seeds / contact sets), and keeps the best by a mixed
physics + high-confidence contact score.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .clash_assembly import dihedrals_to_ca
from .contact_ca_fold import fit_torsions_to_ca, fold_ca_gradient
from .tertiary import score_tertiary


def _scores_for_anchors(
    anchors: Sequence[Tuple[int, int, float]],
    contact_info: Optional[Dict],
) -> List[float]:
    if not contact_info:
        return [0.7] * len(anchors)
    top = contact_info.get("contacts") or []
    score_map = {(int(c["i"]), int(c["j"])): float(c["score"]) for c in top}
    out = []
    for a in anchors:
        i, j = int(a[0]), int(a[1])
        out.append(score_map.get((i, j), score_map.get((j, i), 0.55)))
    return out


def build_distance_targets(
    n: int,
    anchors: Sequence[Tuple[int, int, float]],
    scores: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (target_dist LxL, weight LxL).
    High weight on bonds + high-confidence contacts.
    """
    D = np.zeros((n, n), dtype=np.float64)
    W = np.zeros((n, n), dtype=np.float64)

    # Sequence-separation priors (virtual bonds)
    for i in range(n):
        for j in range(i + 1, n):
            sep = j - i
            if sep == 1:
                D[i, j] = D[j, i] = 3.80
                W[i, j] = W[j, i] = 50.0
            elif sep == 2:
                D[i, j] = D[j, i] = 5.40
                W[i, j] = W[j, i] = 8.0
            elif sep == 3:
                D[i, j] = D[j, i] = 5.80
                W[i, j] = W[j, i] = 2.0
            elif sep < 6:
                # Soft Flory-like growth
                D[i, j] = D[j, i] = 3.8 * (sep ** 0.5)
                W[i, j] = W[j, i] = 0.15
            else:
                # Weak upper preference toward compact domain size
                D[i, j] = D[j, i] = min(3.8 * (sep ** 0.38) * 2.2, 22.0)
                W[i, j] = W[j, i] = 0.02

    # Contact restraints override
    if scores is None:
        scores = [0.7] * len(anchors)
    for (i, j, td), sc in zip(anchors, scores):
        i, j = int(i), int(j)
        if not (0 <= i < n and 0 <= j < n and i != j):
            continue
        w = 12.0 * (max(0.25, float(sc)) ** 1.8)
        D[i, j] = D[j, i] = float(td)
        W[i, j] = W[j, i] = max(W[i, j], w)

    return D, W


def classical_mds(D: np.ndarray, n_dim: int = 3, seed: int = 0) -> np.ndarray:
    """Classical multidimensional scaling from a distance matrix."""
    n = D.shape[0]
    # Double-center squared distances
    D2 = D * D
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Symmetrize for numerical stability
    B = 0.5 * (B + B.T)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    # Keep positive eigenvalues
    pos = np.maximum(evals[:n_dim], 0.0)
    X = evecs[:, :n_dim] * np.sqrt(pos)
    # Random reflection / small noise for ensemble diversity
    rng = np.random.default_rng(seed)
    for d in range(n_dim):
        if rng.random() < 0.5:
            X[:, d] *= -1.0
    X = X + rng.normal(0.0, 0.15, size=X.shape)
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    return X.astype(np.float64)


def smacof_refine(
    X0: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    n_iter: int = 80,
    lr: float = 0.08,
) -> np.ndarray:
    """Weighted stress minimization (vectorized SMACOF-like gradient)."""
    X = X0.copy()
    n = X.shape[0]
    # Work on upper triangle mask where weight > 0
    iu = np.triu_indices(n, k=1)
    w_vec = W[iu]
    d_tgt = D[iu]
    active = w_vec > 0
    if not np.any(active):
        return X
    iu_i = iu[0][active]
    iu_j = iu[1][active]
    w_a = w_vec[active]
    d_a = d_tgt[active]

    for it in range(n_iter):
        diff = X[iu_i] - X[iu_j]
        dist = np.sqrt(np.sum(diff * diff, axis=1) + 1e-12)
        err = dist - d_a
        scale = (2.0 * w_a * err / dist)[:, None]
        force = scale * diff
        g = np.zeros_like(X)
        np.add.at(g, iu_i, force)
        np.add.at(g, iu_j, -force)
        gn = float(np.linalg.norm(g) + 1e-12)
        if gn > 80.0:
            g *= 80.0 / gn
        step = lr * (1.0 - 0.7 * it / max(n_iter - 1, 1))
        X = X - step * g
        X = X - X.mean(axis=0, keepdims=True)
    return X


def contact_violation(
    ca: np.ndarray,
    anchors: Sequence[Tuple[int, int, float]],
    scores: Optional[Sequence[float]] = None,
    min_score: float = 0.75,
) -> float:
    """Mean squared violation of high-confidence contacts only."""
    if not anchors:
        return 0.0
    if scores is None:
        scores = [1.0] * len(anchors)
    errs = []
    for (i, j, td), sc in zip(anchors, scores):
        if float(sc) < min_score:
            continue
        i, j = int(i), int(j)
        if not (0 <= i < len(ca) and 0 <= j < len(ca)):
            continue
        d = float(np.linalg.norm(ca[j] - ca[i]))
        err = d - float(td)
        errs.append(err * err if err > 0 else 0.25 * err * err)
    if not errs:
        return 0.0
    return float(np.mean(errs))


def rank_scaffold(
    sequence: str,
    phis: Sequence[float],
    psis: Sequence[float],
    anchors: Sequence[Tuple[int, int, float]],
    scores: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    """
    Returns (rank, phys, viol). Higher rank is better.
    """
    phys = float(score_tertiary(sequence, phis, psis, anchors=None)["total"])
    ca = dihedrals_to_ca(phis, psis)
    viol = contact_violation(ca, anchors, scores, min_score=0.80)
    # Prefer physics; contacts only break ties / mild preference
    rank = phys - 0.04 * viol
    return rank, phys, viol


def build_one_scaffold(
    sequence: str,
    phis0: Sequence[float],
    psis0: Sequence[float],
    anchors: Sequence[Tuple[int, int, float]],
    scores: Optional[Sequence[float]] = None,
    seed: int = 0,
    use_mds: bool = True,
    ca_steps: int = 700,
    fit_steps: int = 1400,
) -> Dict:
    """One candidate: optional MDS init → gradient CA → torsion fit."""
    n = len(phis0)
    anchors = list(anchors or [])
    ph_local = np.asarray(phis0, dtype=np.float64)
    ps_local = np.asarray(psis0, dtype=np.float64)
    ca_local = dihedrals_to_ca(ph_local, ps_local)

    if use_mds and anchors and n >= 8:
        D, W = build_distance_targets(n, anchors, scores)
        X = classical_mds(D, seed=seed)
        X = smacof_refine(X, D, W, n_iter=100, lr=0.10)
        # Blend with local trace to keep chain connectivity sane
        # Align MDS to local then mix
        Pc = X - X.mean(axis=0)
        Qc = ca_local - ca_local.mean(axis=0)
        H = Pc.T @ Qc
        U, _S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T
        X_al = Pc @ R + ca_local.mean(axis=0)
        blend = 0.65  # favor DG more when contacts exist
        ca0 = blend * X_al + (1.0 - blend) * ca_local
    else:
        # Perturb local as ensemble diversity
        rng = np.random.default_rng(seed)
        ca0 = ca_local + rng.normal(0.0, 0.4, size=ca_local.shape)

    ca1, e0, e1 = fold_ca_gradient(
        ca0, anchors, scores=scores, n_steps=ca_steps, lr=0.09
    )
    ph, ps, fit_r = fit_torsions_to_ca(
        ca1, ph_local, ps_local, n_steps=fit_steps, seed=seed + 17
    )
    rank, phys, viol = rank_scaffold(sequence, ph, ps, anchors, scores)
    return {
        "phis_deg": ph.tolist(),
        "psis_deg": ps.tolist(),
        "rank": rank,
        "phys": phys,
        "viol": viol,
        "ca_energy_before": e0,
        "ca_energy_after": e1,
        "fit_rmsd": fit_r,
        "seed": seed,
        "use_mds": use_mds,
    }


def run_scaffold_ensemble(
    sequence: str,
    phis0: Sequence[float],
    psis0: Sequence[float],
    anchors: Sequence[Tuple[int, int, float]],
    contact_info: Optional[Dict] = None,
    alt_anchors: Optional[Sequence[Tuple[int, int, float]]] = None,
    alt_contact_info: Optional[Dict] = None,
    n_members: int = 6,
    sharp_contacts: bool = False,
) -> Dict:
    """
    Build an ensemble of scaffolds and return the best by rank_scaffold,
    compared against the local baseline.
    """
    seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
    base_ph = list(phis0)
    base_ps = list(psis0)
    anchors = list(anchors or [])
    scores = _scores_for_anchors(anchors, contact_info)

    base_rank, base_phys, base_viol = rank_scaffold(seq, base_ph, base_ps, anchors, scores)
    candidates = [
        {
            "phis_deg": base_ph,
            "psis_deg": base_ps,
            "rank": base_rank,
            "phys": base_phys,
            "viol": base_viol,
            "source": "local",
            "seed": -1,
        }
    ]

    # Sparse / mid maps: MDS often destroys good local geometry — skip heavy ensemble
    if len(anchors) < 6 or (not sharp_contacts and len(anchors) < 8):
        n_mem = min(3, int(n_members))
        allow_mds = False
    else:
        n_mem = int(n_members)
        allow_mds = True
        if sharp_contacts:
            n_mem = max(n_mem, 5)

    # Primary contact set (usually t12)
    if anchors and len(anchors) >= 3:
        for k in range(n_mem):
            use_mds = allow_mds and (k % 2 == 0)
            cand = build_one_scaffold(
                seq,
                base_ph,
                base_ps,
                anchors,
                scores=scores,
                seed=11 + 17 * k,
                use_mds=use_mds,
                ca_steps=400 if not sharp_contacts else 550,
                fit_steps=800 if not sharp_contacts else 1100,
            )
            cand["source"] = "t12_mds" if use_mds else "t12_local"
            candidates.append(cand)

    # Optional alternate contact set (t30) — only when sharp + enough anchors
    if alt_anchors and sharp_contacts and len(alt_anchors) >= 6:
        alt_scores = _scores_for_anchors(alt_anchors, alt_contact_info)
        for k in range(2):
            cand = build_one_scaffold(
                seq,
                base_ph,
                base_ps,
                list(alt_anchors),
                scores=alt_scores,
                seed=101 + 23 * k,
                use_mds=True,
                ca_steps=500,
                fit_steps=1000,
            )
            cand["source"] = "t30_mds"
            r, p, v = rank_scaffold(
                seq, cand["phis_deg"], cand["psis_deg"], alt_anchors, alt_scores
            )
            cand["rank"], cand["phys"], cand["viol"] = r, p, v
            candidates.append(cand)

    def _acceptable(c: Dict) -> bool:
        # Never accept a candidate that wrecks physics vs local
        if float(c["phys"]) < base_phys - 1.5:
            return False
        # Prefer lower high-conf contact violation when contacts are sharp
        if sharp_contacts and float(c["viol"]) > base_viol + 8.0:
            return False
        return True

    viable = [c for c in candidates if c.get("source") == "local" or _acceptable(c)]
    best = max(viable, key=lambda c: float(c["rank"]))
    improved = (
        best.get("source") != "local"
        and float(best["rank"]) > base_rank + 0.15
        and float(best["phys"]) >= base_phys - 0.75
    )
    if not improved:
        best = candidates[0]
    return {
        "phis_deg": best["phis_deg"],
        "psis_deg": best["psis_deg"],
        "improved": improved,
        "kept_source": best.get("source", "local"),
        "best_rank": float(best["rank"]),
        "base_rank": float(base_rank),
        "n_candidates": len(candidates),
        "n_anchors": len(anchors),
    }
