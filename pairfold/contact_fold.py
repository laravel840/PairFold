"""Early contact-guided scaffold assembly (fast path).

Steers global fold with long-range contacts after local φ/ψ consensus.
Hot loop avoids full N×N clash matrices — anchors dominate, clash sampled rarely.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .clash_assembly import dihedrals_to_ca


def _anchor_energy(
    ca: np.ndarray,
    anchors: Sequence[Tuple[int, int, float]],
    weights: Optional[Sequence[float]] = None,
) -> float:
    n = ca.shape[0]
    e = 0.0
    for k, (i, j, td) in enumerate(anchors):
        i, j = int(i), int(j)
        if not (0 <= i < n and 0 <= j < n and i != j):
            continue
        w = float(weights[k]) if weights is not None else 1.0
        d = float(np.linalg.norm(ca[j] - ca[i]))
        err = d - float(td)
        e += w * (err * err if err >= 0 else 0.35 * err * err)
    return e


def _cheap_clash(ca: np.ndarray, min_sep: int = 3, thresh: float = 3.6) -> float:
    """O(N·w) local clash — no full distance matrix."""
    n = ca.shape[0]
    e = 0.0
    # Check pairs with sequence separation 3..12 (local sterics)
    for sep in range(min_sep, min(13, n)):
        d = ca[sep:] - ca[: n - sep]
        dist = np.sqrt(np.sum(d * d, axis=1) + 1e-12)
        bad = dist < thresh
        if bad.any():
            diff = thresh - dist[bad]
            e += float(np.sum(diff * diff))
    return e


def early_contact_fold(
    sequence: str,
    phis_deg: Sequence[float],
    psis_deg: Sequence[float],
    anchors: Sequence[Tuple[int, int, float]],
    scores: Optional[Sequence[float]] = None,
    n_restarts: int = 3,
    steps_per_restart: int = 280,
    seed: int = 7,
) -> Dict:
    """
    Multi-restart simulated annealing on hinge torsions with strong contact weight.
    """
    ph0 = np.asarray(phis_deg, dtype=np.float64)
    ps0 = np.asarray(psis_deg, dtype=np.float64)
    n = len(ph0)
    anchors = list(anchors or [])
    if n < 6 or not anchors:
        return {
            "phis_deg": ph0.tolist(),
            "psis_deg": ps0.tolist(),
            "improved": False,
            "anchor_energy_before": 0.0,
            "anchor_energy_after": 0.0,
            "n_anchors": 0,
            "n_steps": 0,
        }

    if scores is not None and len(scores) == len(anchors):
        weights = [max(0.25, float(s)) ** 1.5 for s in scores]
    else:
        weights = [1.0] * len(anchors)

    ca0 = dihedrals_to_ca(ph0, ps0)
    e0 = _anchor_energy(ca0, anchors, weights) + 0.1 * _cheap_clash(ca0)

    hinges = {0, n - 1, n // 2, n // 4, (3 * n) // 4}
    for i, j, _ in anchors:
        i, j = int(i), int(j)
        hinges.update({i, j, max(0, i - 1), min(n - 1, j + 1)})
        span = abs(j - i)
        if span > 4:
            for t in (0.25, 0.5, 0.75):
                hinges.add(int(min(i, j) + t * span))
    hinge_idx = np.asarray(sorted(h for h in hinges if 0 <= h < n), dtype=np.int64)

    best_ph, best_ps = ph0.copy(), ps0.copy()
    best_e = e0
    total_steps = 0
    rng_master = np.random.default_rng(seed)

    for r in range(max(1, n_restarts)):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        if r == 0:
            ph, ps = ph0.copy(), ps0.copy()
        else:
            ph, ps = best_ph.copy(), best_ps.copy()
            for _ in range(max(2, n // 12)):
                i = int(rng.choice(hinge_idx))
                amp = 22.0 if r < 2 else 10.0
                if rng.random() < 0.5:
                    ph[i] = ((ph[i] + rng.uniform(-amp, amp) + 180.0) % 360.0) - 180.0
                else:
                    ps[i] = ((ps[i] + rng.uniform(-amp, amp) + 180.0) % 360.0) - 180.0

        ca = dihedrals_to_ca(ph, ps)
        cur_e = _anchor_energy(ca, anchors, weights) + 0.1 * _cheap_clash(ca)
        cur_ph, cur_ps = ph, ps
        n_steps = int(steps_per_restart)

        for step in range(n_steps):
            frac = step / max(n_steps - 1, 1)
            amp = 16.0 * (1.0 - 0.85 * frac) + 1.5
            temp = 6.0 * (1.0 - frac) ** 2 + 0.12
            i = int(rng.choice(hinge_idx))
            trial_ph, trial_ps = cur_ph.copy(), cur_ps.copy()
            delta = float(rng.uniform(-amp, amp))
            if rng.random() < 0.5:
                trial_ph[i] = ((trial_ph[i] + delta + 180.0) % 360.0) - 180.0
            else:
                trial_ps[i] = ((trial_ps[i] + delta + 180.0) % 360.0) - 180.0
            if rng.random() < 0.2 and anchors:
                ai, aj, _ = anchors[int(rng.integers(0, len(anchors)))]
                for idx in (int(ai), int(aj)):
                    if 0 <= idx < n:
                        d2 = float(rng.uniform(-amp * 0.5, amp * 0.5))
                        if rng.random() < 0.5:
                            trial_ph[idx] = ((trial_ph[idx] + d2 + 180.0) % 360.0) - 180.0
                        else:
                            trial_ps[idx] = ((trial_ps[idx] + d2 + 180.0) % 360.0) - 180.0

            trial_ca = dihedrals_to_ca(trial_ph, trial_ps)
            # Clash only every 4th step — keeps loop ~3× faster
            if step % 4 == 0:
                e = _anchor_energy(trial_ca, anchors, weights) + 0.12 * _cheap_clash(trial_ca)
            else:
                e = _anchor_energy(trial_ca, anchors, weights)
            if e <= cur_e or rng.random() < math.exp(-(e - cur_e) / max(temp, 1e-6)):
                cur_e, cur_ph, cur_ps = e, trial_ph, trial_ps
                if cur_e < best_e:
                    best_e, best_ph, best_ps = cur_e, cur_ph.copy(), cur_ps.copy()
            total_steps += 1

    ca_f = dihedrals_to_ca(best_ph, best_ps)
    e_f = _anchor_energy(ca_f, anchors, weights)
    return {
        "phis_deg": best_ph.tolist(),
        "psis_deg": best_ps.tolist(),
        "improved": bool(best_e < e0 - 1e-6),
        "anchor_energy_before": float(e0),
        "anchor_energy_after": float(e_f),
        "total_energy_after": float(best_e),
        "n_anchors": len(anchors),
        "n_steps": int(total_steps),
        "n_restarts": int(n_restarts),
    }
