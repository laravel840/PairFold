"""
Lightweight tertiary structure scoring / refine for PairFold (prototype).

Fast path (default for longer chains):
  - One clash + Rg + hydrophobic score, then assemble 3D

Refine path (short chains ≤ TERTIARY_REFINE_MAX_LEN):
  - Short Metropolis on hinge torsions using a cheap clash+Rg objective
  - Full hydro score only at the end

Not AlphaFold.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .clash_assembly import clash_energy, dihedrals_to_ca
from .config import TERTIARY_MAX_LEN, TERTIARY_REFINE_MAX_LEN

_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
_HYDROPHOBIC = set("AILMFVWYC")


def expected_rg(n: int) -> float:
    n = max(int(n), 2)
    return float(2.2 * (n ** 0.38))


def radius_of_gyration(ca: np.ndarray) -> float:
    ca = np.asarray(ca, dtype=np.float64)
    c = ca.mean(axis=0)
    d2 = np.sum((ca - c) ** 2, axis=1)
    return float(math.sqrt(np.mean(d2) + 1e-12))


def hydrophobic_burial_score(sequence: str, ca: np.ndarray, cutoff: float = 8.0) -> float:
    ca = np.asarray(ca, dtype=np.float64)
    hydro_idx = [i for i, aa in enumerate(sequence) if aa in _HYDROPHOBIC]
    if len(hydro_idx) < 2:
        return 0.0
    # Vectorized: only hydrophobic subset distances
    idx = np.asarray(hydro_idx, dtype=np.int64)
    sub = ca[idx]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    scores = []
    for a, i in enumerate(idx):
        cnt = 0.0
        for b, j in enumerate(idx):
            if abs(int(i) - int(j)) < 2:
                continue
            if d[a, b] < cutoff:
                cnt += 1.0
                cnt += 0.05 * max(0.0, _KD.get(sequence[i], 0) + _KD.get(sequence[j], 0))
        scores.append(cnt)
    return float(np.mean(scores))


def _score_from_ca(
    sequence: str,
    ca: np.ndarray,
    with_hydro: bool = True,
    anchors: Optional[Sequence[Tuple[int, int, float]]] = None,
) -> Dict[str, float]:
    n = len(sequence)
    clash = clash_energy(ca, soft=True, min_seq_sep=3)
    rg = radius_of_gyration(ca)
    rg0 = expected_rg(n)
    rg_term = -((rg - rg0) / max(rg0, 1.0)) ** 2
    hydro = hydrophobic_burial_score(sequence, ca) if with_hydro else 0.0
    contact_e = 0.0
    if anchors:
        for i, j, td in anchors:
            if not (0 <= int(i) < n and 0 <= int(j) < n and int(i) != int(j)):
                continue
            d = float(np.linalg.norm(ca[int(j)] - ca[int(i)]))
            contact_e += (d - float(td)) ** 2
    # Stronger contact pull when ESM / DL anchors are present (Phase C)
    w_clash, w_rg, w_hydro, w_contact = 1.0, 2.5, 0.35, 0.35 if anchors else 0.08
    total = (
        -w_clash * clash
        + w_rg * rg_term
        + (w_hydro * hydro if with_hydro else 0.0)
        - w_contact * contact_e
    )
    return {
        "total": float(total),
        "clash_energy": float(clash),
        "rg": float(rg),
        "rg_expected": float(rg0),
        "rg_term": float(rg_term),
        "hydrophobic_burial": float(hydro),
        "contact_energy": float(contact_e),
        "n": float(n),
    }


def score_tertiary(
    sequence: str,
    phis_deg: Sequence[float],
    psis_deg: Sequence[float],
    with_hydro: bool = True,
    anchors: Optional[Sequence[Tuple[int, int, float]]] = None,
) -> Dict[str, float]:
    ca = dihedrals_to_ca(phis_deg, psis_deg)
    return _score_from_ca(sequence, ca, with_hydro=with_hydro, anchors=anchors)


def _fast_objective(
    sequence: str,
    phis: np.ndarray,
    psis: np.ndarray,
    anchors: Optional[Sequence[Tuple[int, int, float]]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Cheap MC objective: clash + Rg (+ contact if anchors)."""
    sc = score_tertiary(sequence, phis, psis, with_hydro=False, anchors=anchors)
    return sc["total"], sc


def _movable_indices(sequence: str, frozen: Optional[np.ndarray] = None) -> np.ndarray:
    n = len(sequence)
    if frozen is None:
        pref = [i for i, aa in enumerate(sequence) if aa in "GPNSD"]
        if len(pref) >= max(4, n // 8):
            return np.asarray(pref, dtype=np.int64)
        return np.arange(n, dtype=np.int64)
    mov = np.where(~np.asarray(frozen, dtype=bool))[0]
    if mov.size == 0:
        return np.arange(n, dtype=np.int64)
    return mov.astype(np.int64)


def _budget(n: int) -> Tuple[int, int, float]:
    """restarts, steps, proposal sigma — kept small on purpose."""
    if n <= 40:
        return 1, 40, 18.0
    if n <= 80:
        return 1, 32, 15.0
    return 1, 24, 12.0  # up to TERTIARY_REFINE_MAX_LEN


def refine_tertiary(
    sequence: str,
    phis_deg: Sequence[float],
    psis_deg: Sequence[float],
    frozen_mask: Optional[Sequence[bool]] = None,
    seed: int = 0,
    progress: Optional[Callable[[float, str], None]] = None,
    anchors: Optional[Sequence[Tuple[int, int, float]]] = None,
) -> Dict:
    seq = "".join(c for c in sequence.upper() if c.isalpha())
    if len(seq) > TERTIARY_MAX_LEN:
        raise ValueError(f"tertiary limited to ≤{TERTIARY_MAX_LEN}")

    ph0 = np.asarray(phis_deg, dtype=np.float64).copy()
    ps0 = np.asarray(psis_deg, dtype=np.float64).copy()
    if ph0.shape[0] != len(seq) or ps0.shape[0] != len(seq):
        raise ValueError("angle length mismatch")

    frozen = None
    if frozen_mask is not None:
        frozen = np.asarray(frozen_mask, dtype=bool)
        if frozen.shape[0] != len(seq):
            frozen = None

    if frozen is None:
        try:
            from .structure_blocks import detect_ss_blocks, freeze_ss_angles

            blocks = detect_ss_blocks(seq)
            ph0, ps0, blocks, frozen = freeze_ss_angles(seq, ph0, ps0, blocks)
        except Exception:
            frozen = np.zeros(len(seq), dtype=bool)

    # Prefer moving hinge residues near contact anchors
    if anchors:
        prefer = set()
        for i, j, _ in anchors:
            prefer.add(int(i))
            prefer.add(int(j))
            prefer.add(max(0, int(i) - 1))
            prefer.add(min(len(seq) - 1, int(j) + 1))
        if frozen is not None:
            for idx in prefer:
                if 0 <= idx < len(seq):
                    frozen[idx] = False

    # Fast path: score once, no MC (long chains)
    if len(seq) > TERTIARY_REFINE_MAX_LEN:
        if progress:
            progress(0.2, "Tertiary score (fast)")
        base = score_tertiary(seq, ph0, ps0, with_hydro=True, anchors=anchors)
        if progress:
            progress(1.0, "Tertiary score done")
        return {
            "phis_deg": ph0,
            "psis_deg": ps0,
            "score": base,
            "score_before": base,
            "improved": False,
            "trials": 1,
            "accepted": 0,
            "restarts": 0,
            "steps": 0,
            "n_movable": 0,
            "n_frozen": int(np.sum(frozen)) if frozen is not None else 0,
            "mode": "score_only",
        }

    movable = _movable_indices(seq, frozen)
    restarts, steps, sigma = _budget(len(seq))
    if anchors:
        n_anc = len(list(anchors))
        restarts = max(restarts, 3)
        steps = int(min(220, steps + 12 * n_anc))
        sigma = max(sigma, 22.0)
    rng = np.random.default_rng(seed)

    if progress:
        progress(0.05, "Tertiary refine (short)")

    base_full = score_tertiary(seq, ph0, ps0, with_hydro=True, anchors=anchors)
    best_ph, best_ps = ph0.copy(), ps0.copy()
    best_fast, _ = _fast_objective(seq, ph0, ps0, anchors=anchors)
    trials = 1
    accepted = 0
    total_work = max(restarts * steps, 1)
    done = 0

    for r in range(restarts):
        cur_ph, cur_ps = ph0.copy(), ps0.copy()
        if r > 0 and movable.size:
            kicks = min(4, movable.size)
            for idx in rng.choice(movable, size=kicks, replace=False):
                cur_ph[idx] = ((cur_ph[idx] + float(rng.normal(0, sigma)) + 180) % 360) - 180
                cur_ps[idx] = ((cur_ps[idx] + float(rng.normal(0, sigma)) + 180) % 360) - 180
            if frozen is not None:
                cur_ph[frozen] = ph0[frozen]
                cur_ps[frozen] = ps0[frozen]

        cur_fast, _ = _fast_objective(seq, cur_ph, cur_ps, anchors=anchors)
        trials += 1
        if cur_fast > best_fast:
            best_fast = cur_fast
            best_ph, best_ps = cur_ph.copy(), cur_ps.copy()

        for t in range(steps):
            temp = 0.9 * (1.0 - t / max(steps - 1, 1)) + 0.2
            idx = int(rng.choice(movable))
            prop_ph, prop_ps = cur_ph.copy(), cur_ps.copy()
            delta = float(rng.normal(0.0, sigma))
            if rng.random() < 0.5:
                prop_ph[idx] = ((prop_ph[idx] + delta + 180) % 360) - 180
            else:
                prop_ps[idx] = ((prop_ps[idx] + delta + 180) % 360) - 180
            if frozen is not None:
                prop_ph[frozen] = ph0[frozen]
                prop_ps[frozen] = ps0[frozen]

            prop_fast, _ = _fast_objective(seq, prop_ph, prop_ps, anchors=anchors)
            trials += 1
            dE = prop_fast - cur_fast
            if dE >= 0 or rng.random() < math.exp(dE / max(temp, 1e-6)):
                cur_ph, cur_ps, cur_fast = prop_ph, prop_ps, prop_fast
                accepted += 1
                if cur_fast > best_fast:
                    best_fast = cur_fast
                    best_ph, best_ps = cur_ph.copy(), cur_ps.copy()

            done += 1
            if progress and (done % 8 == 0 or done == total_work):
                progress(0.1 + 0.8 * done / total_work, f"Tertiary refine {done}/{total_work}")

    best = score_tertiary(seq, best_ph, best_ps, with_hydro=True, anchors=anchors)
    if progress:
        progress(1.0, "Tertiary refine done")

    improved = best["total"] > base_full["total"] + 1e-9
    if not improved:
        best_ph, best_ps, best = ph0, ps0, base_full

    return {
        "phis_deg": best_ph,
        "psis_deg": best_ps,
        "score": best,
        "score_before": base_full,
        "improved": bool(improved),
        "trials": int(trials),
        "accepted": int(accepted),
        "restarts": int(restarts),
        "steps": int(steps),
        "n_movable": int(movable.size),
        "n_frozen": int(np.sum(frozen)) if frozen is not None else 0,
        "mode": "refine",
    }


def run_tertiary_pipeline(
    sequence: str,
    phis_deg: Sequence[float],
    psis_deg: Sequence[float],
    progress: Optional[Callable[[float, str], None]] = None,
    anchors: Optional[Sequence[Tuple[int, int, float]]] = None,
) -> Dict:
    seq = "".join(c for c in sequence.upper() if c.isalpha())
    out = refine_tertiary(seq, phis_deg, psis_deg, progress=progress, anchors=anchors)
    sc = out["score"]
    mode = out.get("mode", "refine")
    n_anc = len(list(anchors)) if anchors else 0
    return {
        "phis": [float(x) for x in out["phis_deg"]],
        "psis": [float(x) for x in out["psis_deg"]],
        "tertiary": {
            "enabled": True,
            "mode": mode,
            "improved": out["improved"],
            "score": round(sc["total"], 4),
            "clash_energy": round(sc["clash_energy"], 4),
            "rg": round(sc["rg"], 3),
            "rg_expected": round(sc["rg_expected"], 3),
            "hydrophobic_burial": round(sc["hydrophobic_burial"], 3),
            "contact_energy": round(sc.get("contact_energy", 0.0), 3),
            "n_anchors": n_anc,
            "score_before": round(out["score_before"]["total"], 4),
            "trials": out["trials"],
            "accepted": out["accepted"],
            "restarts": out["restarts"],
            "n_frozen": out["n_frozen"],
            "note": (
                "Prototype tertiary "
                + ("score-only (fast)" if mode == "score_only" else "short refine")
                + ": clash + compactness + hydrophobic burial"
                + (" + contact anchors" if n_anc else "")
                + ". Not a full folding predictor."
            ),
        },
    }
