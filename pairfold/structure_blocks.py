"""
Secondary-structure block freezing + softmax temperature confidence calibration.

1. Detect helix / sheet blocks from sequence propensities, freeze internal
   φ/ψ to canonical values, and optimize only boundary torsions so the rigid
   block places without steric clashes (integrates with clash_assembly).

2. Softmax / logit temperature scaling (T) to sharpen over-conservative Platt
   scores, with an explicit disorder penalty for Gly-rich / loop-like regions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .clash_assembly import (
    AngleHypothesis,
    FragmentSlot,
    clash_energy,
    dihedrals_to_backbone,
    dihedrals_to_ca,
    has_steric_clash,
    new_residue_clash,
)
from .config import CALIB_DIR

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Canonical secondary-structure dihedrals (degrees)
HELIX_PHI, HELIX_PSI = -57.0, -47.0
SHEET_PHI, SHEET_PSI = -119.0, 113.0
COIL_PHI, COIL_PSI = -80.0, 70.0

# Chou–Fasman-like propensities (normalized around 1.0)
_P_HELIX = {
    "A": 1.42, "L": 1.21, "M": 1.45, "E": 1.51, "Q": 1.11, "K": 1.16,
    "R": 0.98, "H": 1.00, "V": 1.06, "I": 1.08, "W": 1.08, "F": 1.13,
    "Y": 0.69, "C": 0.70, "S": 0.77, "T": 0.83, "N": 0.67, "D": 1.01,
    "G": 0.57, "P": 0.57,
}
_P_SHEET = {
    "V": 1.70, "I": 1.60, "Y": 1.47, "F": 1.38, "W": 1.37, "L": 1.30,
    "T": 1.19, "C": 1.19, "Q": 1.10, "M": 1.05, "R": 0.93, "N": 0.89,
    "A": 0.83, "S": 0.75, "G": 0.75, "K": 0.74, "H": 0.87, "D": 0.54,
    "E": 0.37, "P": 0.55,
}


# ---------------------------------------------------------------------------
# 2. Softmax / logit temperature calibration
# ---------------------------------------------------------------------------


@dataclass
class TempCalibration:
    """
    Two-stage confidence transform:

      1) optional Platt:  p0 = σ(a · logit(c) + b)
      2) temperature:     p  = σ(logit(p0) / T)     # T<1 sharpens, T>1 smooths
      3) disorder gate:   p *= (1 − γ · disorder)    # Gly-rich loops down-weighted
    """

    T: float = 0.65
    platt_a: float = 1.0
    platt_b: float = 0.0
    use_platt: bool = False
    disorder_gamma: float = 0.45
    # floor so we never fully zero a residue
    conf_floor: float = 0.05

    @classmethod
    def from_json(cls, path: Optional[Path] = None) -> "TempCalibration":
        path = Path(path or CALIB_DIR / "confidence_calibration.json")
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        # Prefer explicit sharpening_T; else derive a sharpening default from
        # the stored temperature (often T>1 = too smooth → invert toward <1).
        stored_T = float(data.get("temperature_T", 1.0))
        sharpen = data.get("sharpening_T")
        if sharpen is None:
            # If Platt was conservative (b≪0) or T>1, default to sharpening
            b = float(data.get("platt_b", 0.0))
            sharpen = 0.55 if (b < -0.5 or stored_T > 1.05) else min(stored_T, 0.85)
        return cls(
            T=float(sharpen),
            platt_a=float(data.get("platt_a", 1.0)),
            platt_b=float(data.get("platt_b", 0.0)),
            use_platt=bool(data.get("use_platt_before_temp", False)),
            disorder_gamma=float(data.get("disorder_gamma", 0.45)),
            conf_floor=float(data.get("conf_floor", 0.05)),
        )

    def to_dict(self) -> dict:
        return {
            "sharpening_T": self.T,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "use_platt_before_temp": self.use_platt,
            "disorder_gamma": self.disorder_gamma,
            "conf_floor": self.conf_floor,
            "method": "temp_sharpen",
        }


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def temperature_scale(
    conf: np.ndarray,
    T: float,
    platt_a: float = 1.0,
    platt_b: float = 0.0,
    use_platt: bool = False,
) -> np.ndarray:
    """
    Softmax-style temperature on binary confidence logits.

    T < 1 → sharper (pushes mass toward 0/1)
    T > 1 → smoother (pushes toward 0.5)
    """
    c = np.asarray(conf, dtype=np.float64)
    z = logit(c)
    if use_platt:
        z = platt_a * z + platt_b
    return sigmoid(z / max(T, 1e-3))


def disorder_profile(sequence: str, window: int = 5) -> np.ndarray:
    """
    Vectorized local disorder score in [0, 1].

    High for Gly/Pro-rich and low-helix/sheet propensity stretches
    (typical flexible loops).
    """
    seq = sequence.upper()
    n = len(seq)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    gly = np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord("G")
    pro = np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord("P")
    # sliding fraction of G+P
    gp = (gly | pro).astype(np.float64)
    # box filter via cumsum
    w = max(1, window)
    cs = np.concatenate([[0.0], np.cumsum(gp)])
    # centered window
    half = w // 2
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx - half + w, 0, n)
    frac_gp = (cs[hi] - cs[lo]) / w

    # inverse mean helix/sheet propensity in window
    h = np.array([_P_HELIX.get(aa, 0.8) for aa in seq], dtype=np.float64)
    s = np.array([_P_SHEET.get(aa, 0.8) for aa in seq], dtype=np.float64)
    ss = np.maximum(h, s)
    cs_ss = np.concatenate([[0.0], np.cumsum(ss)])
    mean_ss = (cs_ss[hi] - cs_ss[lo]) / w
    low_ss = np.clip(1.2 - mean_ss, 0.0, 1.0)

    return np.clip(0.65 * frac_gp + 0.35 * low_ss, 0.0, 1.0)


def calibrate_confidence(
    conf: np.ndarray,
    sequence: str,
    calib: Optional[TempCalibration] = None,
) -> np.ndarray:
    """Apply temperature scaling + disorder penalty. Fully vectorized."""
    calib = calib or TempCalibration()
    p = temperature_scale(
        conf, calib.T, calib.platt_a, calib.platt_b, calib.use_platt
    )
    d = disorder_profile(sequence)
    if d.shape != p.shape:
        # broadcast if scalar / fragment-level conf repeated
        if p.ndim == 0:
            p = np.full_like(d, float(p))
        elif p.shape[0] != d.shape[0]:
            raise ValueError("conf length must match sequence length")
    p = p * (1.0 - calib.disorder_gamma * d)
    return np.clip(p, calib.conf_floor, 1.0)


def fit_sharpening_T(
    raw_conf: np.ndarray,
    y_correct: np.ndarray,
    T_grid: Optional[np.ndarray] = None,
) -> float:
    """
    Choose T that minimizes ECE after sharpening (no Platt).
    Prefer T<1 when raw/Platt masses near 0.5.
    """
    if T_grid is None:
        T_grid = np.concatenate(
            [np.linspace(0.25, 0.95, 15), np.linspace(1.0, 2.5, 10)]
        )
    best_t, best_ece = 1.0, float("inf")
    for t in T_grid:
        p = temperature_scale(raw_conf, float(t))
        e = _ece(p, y_correct)
        if e < best_ece:
            best_ece = e
            best_t = float(t)
    return best_t


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    n = len(p)
    for i in range(n_bins):
        m = (p >= edges[i]) & (
            p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1]
        )
        if not np.any(m):
            continue
        total += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(total)


# ---------------------------------------------------------------------------
# 1. Secondary-structure detection + block freezing
# ---------------------------------------------------------------------------


@dataclass
class SSBlock:
    start: int  # inclusive
    end: int  # exclusive
    kind: str  # "helix" | "sheet"
    phi_interior: float
    psi_interior: float

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def interior(self) -> slice:
        """Residues with fully frozen torsions (exclude entry/exit boundaries)."""
        if self.length <= 2:
            return slice(self.start, self.start)  # empty — all boundary
        return slice(self.start + 1, self.end - 1)


def detect_ss_blocks(
    sequence: str,
    helix_thresh: float = 1.05,
    sheet_thresh: float = 1.10,
    min_helix: int = 4,
    min_sheet: int = 3,
    smooth: int = 5,
) -> List[SSBlock]:
    """
    Propensity-based SSE detection (Chou–Fasman style), O(N).

    Returns non-overlapping blocks; helix wins ties.
    """
    seq = sequence.upper()
    n = len(seq)
    if n < min_sheet:
        return []

    h = np.array([_P_HELIX.get(aa, 0.8) for aa in seq], dtype=np.float64)
    s = np.array([_P_SHEET.get(aa, 0.8) for aa in seq], dtype=np.float64)

    # moving average
    w = max(1, smooth)
    kernel = np.ones(w, dtype=np.float64) / w
    h_s = np.convolve(h, kernel, mode="same")
    s_s = np.convolve(s, kernel, mode="same")

    helix_mask = h_s >= helix_thresh
    sheet_mask = (s_s >= sheet_thresh) & ~helix_mask

    def runs(mask: np.ndarray, kind: str, min_len: int) -> List[SSBlock]:
        out: List[SSBlock] = []
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i + 1
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                if kind == "helix":
                    out.append(SSBlock(i, j, "helix", HELIX_PHI, HELIX_PSI))
                else:
                    out.append(SSBlock(i, j, "sheet", SHEET_PHI, SHEET_PSI))
            i = j
        return out

    blocks = runs(helix_mask, "helix", min_helix) + runs(sheet_mask, "sheet", min_sheet)
    blocks.sort(key=lambda b: b.start)
    return blocks


def freeze_ss_angles(
    sequence: str,
    phis_deg: np.ndarray,
    psis_deg: np.ndarray,
    blocks: Optional[Sequence[SSBlock]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[SSBlock], np.ndarray]:
    """
    Freeze interior dihedrals of each SSE to canonical values.

    Returns (phis, psis, blocks, frozen_mask) where frozen_mask[i]=True
    means residue i should skip DB fine search.
    """
    phis = np.asarray(phis_deg, dtype=np.float64).copy()
    psis = np.asarray(psis_deg, dtype=np.float64).copy()
    n = len(sequence)
    if phis.shape[0] != n or psis.shape[0] != n:
        raise ValueError("angle arrays must match sequence length")

    blocks = list(blocks) if blocks is not None else detect_ss_blocks(sequence)
    frozen = np.zeros(n, dtype=bool)

    for b in blocks:
        # freeze all interiors; boundaries kept for optimization
        sl = b.interior
        if sl.start < sl.stop:
            phis[sl] = b.phi_interior
            psis[sl] = b.psi_interior
            frozen[sl] = True
        # also mark whole block as "structural" for pruning metadata
        # but only interiors are angle-locked
    return phis, psis, blocks, frozen


def block_transform_matrix(
    phis_deg: np.ndarray,
    psis_deg: np.ndarray,
    block: SSBlock,
) -> np.ndarray:
    """
    4×4 homogeneous transform taking the block's first Cα frame to its last Cα.

    Encodes the rigid-body geometry implied by the (frozen) internal torsions.
    """
    ca = dihedrals_to_ca(phis_deg, psis_deg)
    i0, i1 = block.start, block.end - 1
    # local frames from consecutive CA triples when possible
    def frame(i: int) -> np.ndarray:
        # origin at CA[i], x along CA[i]->CA[min(i+1,end)], ...
        o = ca[i]
        if i < len(ca) - 1:
            x = ca[i + 1] - o
        else:
            x = o - ca[i - 1]
        x = x / (np.linalg.norm(x) + 1e-12)
        # arbitrary helper
        tmp = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y = np.cross(tmp, x)
        y /= np.linalg.norm(y) + 1e-12
        z = np.cross(x, y)
        R = np.stack([x, y, z], axis=1)  # columns = basis
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = o
        return T

    T0 = frame(i0)
    T1 = frame(i1)
    # transform that maps frame0 coords → frame1: T1 @ inv(T0)
    return T1 @ np.linalg.inv(T0)


def _boundary_indices(block: SSBlock, n: int) -> List[int]:
    """Entry and exit residues whose φ/ψ we may optimize."""
    idx = {block.start, block.end - 1}
    # also allow one residue just outside if present (hinge)
    if block.start > 0:
        idx.add(block.start - 1)
    if block.end < n:
        idx.add(block.end)
    return sorted(i for i in idx if 0 <= i < n)


def optimize_block_boundaries(
    sequence: str,
    phis_deg: np.ndarray,
    psis_deg: np.ndarray,
    blocks: Sequence[SSBlock],
    clash_thresh: float = 3.8,
    n_grid: int = 7,
    refine_steps: int = 40,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Grid + local Metropolis search over boundary torsions only.

    Interior frozen angles are held fixed. Objective: minimize soft clash
    energy of the full Cα trace (compatible with clash_assembly backtracking).
    """
    rng = np.random.default_rng(seed)
    phis = np.asarray(phis_deg, dtype=np.float64).copy()
    psis = np.asarray(psis_deg, dtype=np.float64).copy()
    n = len(sequence)

    # ensure interiors frozen
    phis, psis, blocks, _frozen = freeze_ss_angles(sequence, phis, psis, blocks)

    # Collect unique boundary DOFs: (res_idx, 'phi'|'psi')
    boundary_res = []
    for b in blocks:
        boundary_res.extend(_boundary_indices(b, n))
    boundary_res = sorted(set(boundary_res))
    if not boundary_res:
        ca = dihedrals_to_ca(phis, psis)
        return phis, psis, {"clash_energy": clash_energy(ca), "n_boundary": 0}

    # Candidate angle grid (degrees) around helix/sheet/coil basins
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(-150, -40, n_grid),
                np.linspace(40, 150, n_grid),
                np.array([HELIX_PHI, HELIX_PSI, SHEET_PHI, SHEET_PSI, COIL_PHI, COIL_PSI]),
            ]
        )
    )

    def energy(ph: np.ndarray, ps: np.ndarray) -> float:
        ca = dihedrals_to_ca(ph, ps)
        return clash_energy(ca, clash_thresh=clash_thresh, soft=True)

    best_e = energy(phis, psis)
    best_ph, best_ps = phis.copy(), psis.copy()

    # Coordinate descent over boundary residues with vectorized grid eval
    for res in boundary_res:
        # freeze others; scan phi then psi
        for which in ("phi", "psi"):
            base_ph, base_ps = best_ph.copy(), best_ps.copy()
            # evaluate all grid values
            energies = np.empty(grid.shape[0], dtype=np.float64)
            for gi, ang in enumerate(grid):
                ph, ps = base_ph.copy(), base_ps.copy()
                if which == "phi":
                    ph[res] = ang
                else:
                    ps[res] = ang
                # keep interiors locked
                for b in blocks:
                    sl = b.interior
                    if sl.start < sl.stop:
                        ph[sl] = b.phi_interior
                        ps[sl] = b.psi_interior
                energies[gi] = energy(ph, ps)
            gbest = int(np.argmin(energies))
            if energies[gbest] < best_e - 1e-9:
                best_e = float(energies[gbest])
                if which == "phi":
                    best_ph[res] = grid[gbest]
                else:
                    best_ps[res] = grid[gbest]

    # Short MH refine on boundary angles
    cur_ph, cur_ps = best_ph.copy(), best_ps.copy()
    cur_e = best_e
    for _ in range(refine_steps):
        res = int(rng.choice(boundary_res))
        which = "phi" if rng.random() < 0.5 else "psi"
        prop_ph, prop_ps = cur_ph.copy(), cur_ps.copy()
        delta = float(rng.normal(0.0, 18.0))
        if which == "phi":
            prop_ph[res] = ((prop_ph[res] + delta + 180) % 360) - 180
        else:
            prop_ps[res] = ((prop_ps[res] + delta + 180) % 360) - 180
        for b in blocks:
            sl = b.interior
            if sl.start < sl.stop:
                prop_ph[sl] = b.phi_interior
                prop_ps[sl] = b.psi_interior
        e = energy(prop_ph, prop_ps)
        if e <= cur_e or rng.random() < math.exp(-(e - cur_e) / 0.35):
            cur_ph, cur_ps, cur_e = prop_ph, prop_ps, e
            if e < best_e:
                best_e, best_ph, best_ps = e, prop_ph.copy(), prop_ps.copy()

    transforms = {f"{b.kind}_{b.start}_{b.end}": block_transform_matrix(best_ph, best_ps, b).tolist()
                  for b in blocks}

    info = {
        "clash_energy": float(best_e),
        "n_boundary": len(boundary_res),
        "n_blocks": len(blocks),
        "blocks": [
            {"start": b.start, "end": b.end, "kind": b.kind, "length": b.length}
            for b in blocks
        ],
        "has_clash": has_steric_clash(dihedrals_to_ca(best_ph, best_ps), clash_thresh),
        "transforms": transforms,
    }
    return best_ph, best_ps, info


def prune_slots_with_ss_blocks(
    sequence: str,
    slots: Sequence[FragmentSlot],
    blocks: Optional[Sequence[SSBlock]] = None,
    block_confidence: float = 0.97,
) -> List[FragmentSlot]:
    """
    Replace fragment slots fully inside a frozen SSE with a single rigid
    hypothesis — pruning the DB candidate tree for backtracking/MCTS.
    """
    blocks = list(blocks) if blocks is not None else detect_ss_blocks(sequence)
    if not blocks:
        return list(slots)

    def covering_block(start: int, end: int) -> Optional[SSBlock]:
        for b in blocks:
            # slot strictly inside interior+boundaries of block
            if start >= b.start and end <= b.end and (end - start) >= 2:
                # prefer slots that don't need DB diversity
                if start >= b.start + 1 and end <= b.end - 1:
                    return b
                if b.length >= 4 and start >= b.start and end <= b.end:
                    return b
        return None

    out: List[FragmentSlot] = []
    for slot in slots:
        b = covering_block(slot.start, slot.end)
        if b is None:
            out.append(slot)
            continue
        L = slot.length
        hyp = AngleHypothesis(
            confidence=block_confidence,
            phis_deg=np.full(L, b.phi_interior),
            psis_deg=np.full(L, b.psi_interior),
            source=f"frozen_{b.kind}",
        )
        out.append(FragmentSlot(slot.start, slot.end, [hyp]))
    return out


def apply_ss_pipeline(
    sequence: str,
    phis_deg: np.ndarray,
    psis_deg: np.ndarray,
    conf: Optional[np.ndarray] = None,
    calib: Optional[TempCalibration] = None,
    optimize_boundaries: bool = True,
    max_optimize_len: int = 64,
) -> Dict:
    """
    End-to-end upgrade hook for the existing backbone pipeline.

    - Detect + freeze SSE interiors
    - Optionally optimize boundary torsions against clashes
    - Recalibrate confidence with sharpening T + disorder penalty

    Boundary optimize is disabled for sequences longer than max_optimize_len
    (default 64) because full-chain clash scans are too expensive.
    """
    seq = "".join(c for c in sequence.upper() if c.isalpha())
    blocks = detect_ss_blocks(seq)
    phis, psis, blocks, frozen = freeze_ss_angles(seq, phis_deg, psis_deg, blocks)

    boundary_info: Dict = {"n_blocks": len(blocks)}
    run_opt = bool(optimize_boundaries and blocks and len(seq) <= max_optimize_len)
    if run_opt:
        # Scale down search for mid-length chains
        n_grid = 7 if len(seq) <= 40 else 5
        refine_steps = 40 if len(seq) <= 40 else 16
        phis, psis, boundary_info = optimize_block_boundaries(
            seq,
            phis,
            psis,
            blocks,
            n_grid=n_grid,
            refine_steps=refine_steps,
        )
    elif blocks:
        # clash_energy → dense N×N Cα matrix; only safe for short chains
        from .config import SS_PIPELINE_MAX_LEN

        if len(seq) <= SS_PIPELINE_MAX_LEN:
            ca = dihedrals_to_ca(phis, psis)
            ce = float(clash_energy(ca))
        else:
            ce = 0.0
        boundary_info = {
            "clash_energy": ce,
            "n_boundary": 0,
            "n_blocks": len(blocks),
            "blocks": [
                {"start": b.start, "end": b.end, "kind": b.kind, "length": b.length}
                for b in blocks
            ],
            "skipped_optimize": True,
        }

    calib = calib or TempCalibration.from_json()
    if conf is None:
        conf = np.full(len(seq), 0.7, dtype=np.float64)
        conf[frozen] = 0.95
    conf_cal = calibrate_confidence(np.asarray(conf, dtype=np.float64), seq, calib)

    from .config import SS_PIPELINE_MAX_LEN

    # Full backbone .tolist() is huge and unused by the API for long chains
    if len(seq) <= SS_PIPELINE_MAX_LEN:
        bb = dihedrals_to_backbone(phis, psis, sequence=seq)
        ca = bb["CA"]
        ce = float(clash_energy(ca))
        backbone = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in bb.items()
        }
    else:
        ce = float(boundary_info.get("clash_energy", 0.0))
        backbone = {}
    return {
        "sequence": seq,
        "phis_deg": phis,
        "psis_deg": psis,
        "confidence": conf_cal,
        "frozen_mask": frozen,
        "blocks": boundary_info.get("blocks", [
            {"start": b.start, "end": b.end, "kind": b.kind, "length": b.length}
            for b in blocks
        ]),
        "boundary_opt": boundary_info,
        "calibration": calib.to_dict(),
        "clash_energy": ce,
        "backbone": backbone,
    }


# ---------------------------------------------------------------------------
# Persist sharpening defaults into calibration JSON (non-destructive merge)
# ---------------------------------------------------------------------------


def update_calibration_file(
    sharpening_T: float = 0.55,
    disorder_gamma: float = 0.45,
    use_platt_before_temp: bool = False,
) -> Path:
    path = CALIB_DIR / "confidence_calibration.json"
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data["sharpening_T"] = float(sharpening_T)
    data["disorder_gamma"] = float(disorder_gamma)
    data["use_platt_before_temp"] = bool(use_platt_before_temp)
    data["conf_floor"] = float(data.get("conf_floor", 0.05))
    # Keep legacy fields; inference prefers sharpening path when present
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def main() -> None:
    seq = "AAAAEAAAKAAAAAGGGGPAAAAVVIAAAA"
    # Start from noisy "model" angles
    rng = np.random.default_rng(0)
    phis = rng.normal(-70, 35, size=len(seq))
    psis = rng.normal(0, 50, size=len(seq))
    raw_conf = rng.uniform(0.45, 0.62, size=len(seq))  # conservative ~50%

    update_calibration_file(sharpening_T=0.55)
    out = apply_ss_pipeline(seq, phis, psis, conf=raw_conf, optimize_boundaries=True)

    print(f"sequence:  {out['sequence']}")
    print(f"blocks:    {out['blocks']}")
    print(f"frozen:    {int(out['frozen_mask'].sum())}/{len(seq)} residues")
    print(f"clash_E:   {out['clash_energy']:.4f}")
    print(
        f"conf raw mean={raw_conf.mean():.3f}  "
        f"calibrated mean={out['confidence'].mean():.3f}  "
        f"calibrated max={out['confidence'].max():.3f}"
    )
    print(f"calib:     {out['calibration']}")


if __name__ == "__main__":
    main()
