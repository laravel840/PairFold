"""Cα-bead contact folding + torsion refit (gradient + Metropolis)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .clash_assembly import dihedrals_to_ca


def _weights_from_scores(
    anchors: Sequence[Tuple[int, int, float]],
    scores: Optional[Sequence[float]],
) -> np.ndarray:
    if scores is not None and len(scores) == len(anchors):
        return np.asarray([max(0.3, float(s)) ** 1.5 for s in scores], dtype=np.float64)
    return np.ones(len(anchors), dtype=np.float64)


def fold_ca_gradient(
    ca0: np.ndarray,
    anchors: Sequence[Tuple[int, int, float]],
    scores: Optional[Sequence[float]] = None,
    n_steps: int = 800,
    lr: float = 0.08,
    contact_boost: float = 2.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Differentiable-ish Cα optimization via finite-difference / analytical springs.
    Much better at large contact closures than tiny torsion Metropolis.
    """
    ca = np.asarray(ca0, dtype=np.float64).copy()
    n = ca.shape[0]
    anchors = list(anchors or [])
    if n < 6 or not anchors:
        return ca, 0.0, 0.0

    w = _weights_from_scores(anchors, scores)
    idx_i = np.asarray([int(a[0]) for a in anchors], dtype=np.int64)
    idx_j = np.asarray([int(a[1]) for a in anchors], dtype=np.int64)
    td = np.asarray([float(a[2]) for a in anchors], dtype=np.float64)

    def energy_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
        g = np.zeros_like(x)
        e = 0.0
        # Bonds
        dvec = x[1:] - x[:-1]
        bl = np.sqrt(np.sum(dvec * dvec, axis=1) + 1e-12)
        err = bl - 3.8
        e += 40.0 * float(np.sum(err * err))
        # grad bond
        for i in range(n - 1):
            if bl[i] < 1e-8:
                continue
            force = 80.0 * err[i] * dvec[i] / bl[i]
            g[i] -= force
            g[i + 1] += force
        # Contacts — boost tunable (sharp high-precision maps can pull harder)
        boost = float(contact_boost)
        for k in range(len(anchors)):
            i, j = int(idx_i[k]), int(idx_j[k])
            if i == j:
                continue
            dv = x[j] - x[i]
            d = float(np.linalg.norm(dv) + 1e-12)
            errc = d - td[k]
            wk = float(w[k]) * boost
            scale = wk * (2.0 * errc if errc >= 0 else 0.7 * errc)
            e += wk * (errc * errc if errc >= 0 else 0.35 * errc * errc)
            force = scale * dv / d
            g[i] -= force
            g[j] += force
        # Soft clash (repulsive)
        for sep in range(3, min(12, n)):
            dv = x[sep:] - x[: n - sep]
            dist = np.sqrt(np.sum(dv * dv, axis=1) + 1e-12)
            for t in range(dist.shape[0]):
                if dist[t] < 3.6:
                    err = 3.6 - dist[t]
                    e += 6.0 * err * err
                    force = -12.0 * err * dv[t] / dist[t]
                    g[t] -= force
                    g[t + sep] += force
        # Weak compact Rg prior for short domains
        com = x.mean(axis=0)
        rg2 = float(np.mean(np.sum((x - com) ** 2, axis=1)))
        rg0 = 2.2 * (n ** 0.38)
        e += 0.15 * (rg2 - rg0 * rg0) ** 2 / max(rg0 ** 2, 1.0)
        return e, g

    e0, _ = energy_and_grad(ca)
    best = ca.copy()
    best_e = e0
    v = np.zeros_like(ca)  # momentum
    for step in range(n_steps):
        e, g = energy_and_grad(ca)
        # gradient clipping
        gn = float(np.sqrt(np.sum(g * g)) + 1e-12)
        if gn > 50.0:
            g *= 50.0 / gn
        # Adam-ish momentum
        v = 0.85 * v + g
        step_lr = lr * (1.0 - 0.7 * step / max(n_steps - 1, 1))
        ca = ca - step_lr * v
        if e < best_e:
            best_e = e
            best = ca.copy()
    return best, float(e0), float(best_e)


def _kabsch_R(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    P = a - a.mean(axis=0)
    Q = b - b.mean(axis=0)
    H = P.T @ Q
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T


def fit_torsions_to_ca(
    target_ca: np.ndarray,
    phis_deg: Sequence[float],
    psis_deg: Sequence[float],
    n_steps: int = 2000,
    seed: int = 11,
) -> Tuple[np.ndarray, np.ndarray, float]:
    ph = np.asarray(phis_deg, dtype=np.float64).copy()
    ps = np.asarray(psis_deg, dtype=np.float64).copy()
    tgt = np.asarray(target_ca, dtype=np.float64)
    n = len(ph)
    rng = np.random.default_rng(seed)
    Qc = tgt - tgt.mean(axis=0)

    def loss_with_R(p_h, p_s, R: np.ndarray) -> float:
        ca = dihedrals_to_ca(p_h, p_s)
        Pc = ca - ca.mean(axis=0)
        return float(np.mean(np.sum((Pc @ R - Qc) ** 2, axis=1)))

    # Recompute Kabsch only periodically — SVD every Metropolis step was O(minutes)
    # for ~100 aa chains and froze the UI at the early-contact stage.
    align_every = max(20, n_steps // 40)
    R = _kabsch_R(dihedrals_to_ca(ph, ps), tgt)
    best_ph, best_ps = ph.copy(), ps.copy()
    best_e = loss_with_R(ph, ps, R)
    cur_ph, cur_ps, cur_e = best_ph.copy(), best_ps.copy(), best_e

    for step in range(n_steps):
        if step % align_every == 0 and step > 0:
            R = _kabsch_R(dihedrals_to_ca(cur_ph, cur_ps), tgt)
            cur_e = loss_with_R(cur_ph, cur_ps, R)
            if cur_e < best_e:
                best_e, best_ph, best_ps = cur_e, cur_ph.copy(), cur_ps.copy()
        frac = step / max(n_steps - 1, 1)
        amp = 25.0 * (1.0 - 0.88 * frac) + 1.0
        temp = 1.2 * (1.0 - frac) ** 2 + 0.02
        i = int(rng.integers(0, n))
        trial_ph, trial_ps = cur_ph.copy(), cur_ps.copy()
        delta = float(rng.uniform(-amp, amp))
        if rng.random() < 0.5:
            trial_ph[i] = ((trial_ph[i] + delta + 180.0) % 360.0) - 180.0
        else:
            trial_ps[i] = ((trial_ps[i] + delta + 180.0) % 360.0) - 180.0
        # Coupled move: also tweak neighbors for smoother backbone
        if rng.random() < 0.3:
            j = int(np.clip(i + rng.choice([-1, 1]), 0, n - 1))
            d2 = float(rng.uniform(-amp * 0.4, amp * 0.4))
            trial_ph[j] = ((trial_ph[j] + d2 + 180.0) % 360.0) - 180.0
        e = loss_with_R(trial_ph, trial_ps, R)
        if e <= cur_e or rng.random() < np.exp(-(e - cur_e) / max(temp, 1e-6)):
            cur_ph, cur_ps, cur_e = trial_ph, trial_ps, e
            if cur_e < best_e:
                best_ph, best_ps, best_e = cur_ph.copy(), cur_ps.copy(), cur_e

    # Final aligned score for the returned RMSD
    R = _kabsch_R(dihedrals_to_ca(best_ph, best_ps), tgt)
    best_e = loss_with_R(best_ph, best_ps, R)
    return best_ph, best_ps, float(np.sqrt(best_e))


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    P = a - a.mean(axis=0)
    Q = b - b.mean(axis=0)
    H = P.T @ Q
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T
    return float(np.sqrt(np.mean(np.sum((P @ R - Q) ** 2, axis=1))))


def ca_contact_scaffold(
    sequence: str,
    phis_deg: Sequence[float],
    psis_deg: Sequence[float],
    anchors: Sequence[Tuple[int, int, float]],
    scores: Optional[Sequence[float]] = None,
    ca_steps: int = 900,
    fit_steps: int = 2000,
    seed: int = 3,
    contact_boost: float = 2.0,
) -> Dict:
    ph0 = np.asarray(phis_deg, dtype=np.float64)
    ps0 = np.asarray(psis_deg, dtype=np.float64)
    anchors = list(anchors or [])
    if len(ph0) < 6 or not anchors:
        return {
            "phis_deg": ph0.tolist(),
            "psis_deg": ps0.tolist(),
            "improved": False,
            "ca_energy_before": 0.0,
            "ca_energy_after": 0.0,
            "fit_rmsd": 0.0,
            "n_anchors": 0,
        }

    ca0 = dihedrals_to_ca(ph0, ps0)
    ca1, e0, e1 = fold_ca_gradient(
        ca0,
        anchors,
        scores=scores,
        n_steps=ca_steps,
        contact_boost=contact_boost,
    )
    if e1 >= 0.95 * e0:
        return {
            "phis_deg": ph0.tolist(),
            "psis_deg": ps0.tolist(),
            "improved": False,
            "ca_energy_before": e0,
            "ca_energy_after": e1,
            "fit_rmsd": 0.0,
            "n_anchors": len(anchors),
        }

    ph, ps, fit_r = fit_torsions_to_ca(ca1, ph0, ps0, n_steps=fit_steps, seed=seed + 17)
    ca_fit = dihedrals_to_ca(ph, ps)
    # Accept only if fitted scaffold stayed close to folded Cα
    drift = _kabsch_rmsd(ca_fit, ca1)
    improved = bool(e1 < 0.95 * e0 and drift < 6.0)
    if not improved:
        return {
            "phis_deg": ph0.tolist(),
            "psis_deg": ps0.tolist(),
            "improved": False,
            "ca_energy_before": e0,
            "ca_energy_after": e1,
            "fit_rmsd": fit_r,
            "n_anchors": len(anchors),
        }
    return {
        "phis_deg": ph.tolist(),
        "psis_deg": ps.tolist(),
        "improved": True,
        "ca_energy_before": e0,
        "ca_energy_after": e1,
        "fit_rmsd": fit_r,
        "n_anchors": len(anchors),
    }
