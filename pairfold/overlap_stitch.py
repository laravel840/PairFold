"""
O(N) multi-scale overlapping fragment stitcher.

Assembles a full-length backbone from a k-mer (default 5-mer) fragment
database by sliding overlapping windows and confidence-weighted circular
averaging of φ/ψ. Low-confidence sites fall back to secondary-structure
defaults and are flagged for local relaxation.

Complexity: with fixed window sizes W and strides S, number of windows is
Θ(N). All per-residue reductions use NumPy scatter-add (np.add.at) — no
Python nested loops over overlaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .assemble import build_backbone

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Canonical secondary-structure (φ, ψ) in degrees
SS_DEFAULTS = {
    "helix": (-57.0, -47.0),  # α-helix
    "sheet": (-120.0, 113.0),  # β-strand
    "coil": (-80.0, 70.0),  # polyproline-II-ish coil
}

# Light residue propensities → default SS when DB is weak
_HELIX_LIKE = set("AELMKRQH")
_SHEET_LIKE = set("VIFTYW")
_COIL_LIKE = set("GPSNDC")


@dataclass(frozen=True)
class FragmentRecord:
    """One database hit for a peptide oligomer."""

    sequence: str
    phis_deg: np.ndarray  # (L,)
    psis_deg: np.ndarray  # (L,)
    confidence: Union[float, np.ndarray]  # scalar or (L,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phis_deg", np.asarray(self.phis_deg, dtype=np.float64))
        object.__setattr__(self, "psis_deg", np.asarray(self.psis_deg, dtype=np.float64))
        conf = np.asarray(self.confidence, dtype=np.float64)
        object.__setattr__(self, "confidence", conf)
        L = len(self.sequence)
        if self.phis_deg.shape != (L,) or self.psis_deg.shape != (L,):
            raise ValueError("angle arrays must match sequence length")
        if conf.ndim == 0:
            object.__setattr__(self, "confidence", np.full(L, float(conf), dtype=np.float64))
        elif conf.shape != (L,):
            raise ValueError("confidence must be scalar or length-L")


@dataclass
class ScaleSpec:
    """One sliding-window scale."""

    length: int
    stride: int
    weight: float = 1.0  # relative importance of this scale


# Default multi-scale schedule: 5-mers overlap by 3 (stride 2), plus finer scales
DEFAULT_SCALES: Tuple[ScaleSpec, ...] = (
    ScaleSpec(5, 2, 1.00),  # primary pentamers, overlap 3
    ScaleSpec(4, 1, 0.65),
    ScaleSpec(3, 1, 0.45),
)


@dataclass
class StitchResult:
    sequence: str
    phis_deg: np.ndarray
    psis_deg: np.ndarray
    confidence: np.ndarray  # effective per-residue weight / coverage
    need_relax: np.ndarray  # bool mask — flag for local relaxation
    used_fallback: np.ndarray  # bool — default SS injected
    n_windows: int
    structure: Optional[dict] = None

    @property
    def mean_confidence(self) -> float:
        return float(self.confidence.mean()) if self.confidence.size else 0.0


class FragmentDatabase:
    """
    O(1) average lookup of oligomer → FragmentRecord.

    Accepts either FragmentRecord objects or plain dicts with keys
    phis/psis/confidence (degrees).
    """

    __slots__ = ("_table", "default_length")

    def __init__(
        self,
        records: Optional[Iterable[Union[FragmentRecord, Mapping]]] = None,
        default_length: int = 5,
    ) -> None:
        self._table: Dict[str, FragmentRecord] = {}
        self.default_length = int(default_length)
        if records:
            for r in records:
                self.add(r)

    def add(self, record: Union[FragmentRecord, Mapping]) -> None:
        if not isinstance(record, FragmentRecord):
            seq = str(record["sequence"])
            conf = record.get("confidence", record.get("conf", 1.0))
            record = FragmentRecord(
                sequence=seq,
                phis_deg=np.asarray(record["phis_deg"] if "phis_deg" in record else record["phis"]),
                psis_deg=np.asarray(record["psis_deg"] if "psis_deg" in record else record["psis"]),
                confidence=conf,
            )
        self._table[record.sequence] = record

    def __len__(self) -> int:
        return len(self._table)

    def get(self, seq: str) -> Optional[FragmentRecord]:
        return self._table.get(seq)

    def batch_get(self, seqs: Sequence[str]) -> List[Optional[FragmentRecord]]:
        t = self._table
        return [t.get(s) for s in seqs]


def default_ss_angles(sequence: str) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized per-residue default (φ, ψ) from simple AA propensity."""
    # Ordinal map A..Z → row in table (unused letters → coil)
    helix_phi, helix_psi = SS_DEFAULTS["helix"]
    sheet_phi, sheet_psi = SS_DEFAULTS["sheet"]
    coil_phi, coil_psi = SS_DEFAULTS["coil"]

    phi_table = np.full(26, coil_phi, dtype=np.float64)
    psi_table = np.full(26, coil_psi, dtype=np.float64)
    for aa in _HELIX_LIKE:
        phi_table[ord(aa) - 65] = helix_phi
        psi_table[ord(aa) - 65] = helix_psi
    for aa in _SHEET_LIKE:
        phi_table[ord(aa) - 65] = sheet_phi
        psi_table[ord(aa) - 65] = sheet_psi

    idx = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).astype(np.int64) - 65
    idx = np.clip(idx, 0, 25)
    return phi_table[idx].copy(), psi_table[idx].copy()


def _window_starts(n: int, length: int, stride: int) -> np.ndarray:
    """Start indices covering [0, n) with given window length / stride."""
    if n < length:
        return np.array([0], dtype=np.int64) if n > 0 else np.array([], dtype=np.int64)
    starts = np.arange(0, n - length + 1, stride, dtype=np.int64)
    # ensure C-terminus is covered
    last = n - length
    if starts.size == 0 or starts[-1] != last:
        starts = np.concatenate([starts, np.array([last], dtype=np.int64)])
    return starts


def _scatter_circular(
    n: int,
    residue_index: np.ndarray,
    angles_deg: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Confidence-weighted circular accumulator via np.add.at.

    residue_index, angles_deg, weights: flat arrays of equal length.
    Returns (sum_w_sin, sum_w_cos, sum_w) each shape (n,).
    """
    sin_acc = np.zeros(n, dtype=np.float64)
    cos_acc = np.zeros(n, dtype=np.float64)
    w_acc = np.zeros(n, dtype=np.float64)

    rad = angles_deg * DEG2RAD
    w_sin = weights * np.sin(rad)
    w_cos = weights * np.cos(rad)

    np.add.at(sin_acc, residue_index, w_sin)
    np.add.at(cos_acc, residue_index, w_cos)
    np.add.at(w_acc, residue_index, weights)
    return sin_acc, cos_acc, w_acc


def _finalize_circular(
    sin_acc: np.ndarray, cos_acc: np.ndarray, w_acc: np.ndarray
) -> np.ndarray:
    """atan2 of accumulated weighted sin/cos → degrees. NaN where w==0."""
    out = np.full_like(w_acc, np.nan, dtype=np.float64)
    mask = w_acc > 1e-12
    out[mask] = np.arctan2(sin_acc[mask], cos_acc[mask]) * RAD2DEG
    return out


class OverlapStitcher:
    """
    Multi-scale overlapping k-mer assembler.

    Parameters
    ----------
    database : FragmentDatabase
        Oligomer → angles + confidence.
    scales : sequence of ScaleSpec
        Sliding-window schedule. Primary scale should be length=5, stride=2
        (overlap of 3 residues).
    low_conf_threshold : float
        Residues whose total accumulated weight falls below this (or that
        never receive a hit) use default SS angles and are flagged.
    build_structure : bool
        If True, also emit Cartesian backbone via assemble.build_backbone.
    """

    def __init__(
        self,
        database: FragmentDatabase,
        scales: Sequence[ScaleSpec] = DEFAULT_SCALES,
        low_conf_threshold: float = 0.25,
        missing_penalty: float = 0.0,
        build_structure: bool = True,
    ) -> None:
        self.db = database
        self.scales = tuple(scales)
        self.low_conf_threshold = float(low_conf_threshold)
        self.missing_penalty = float(missing_penalty)
        self.build_structure = bool(build_structure)

    def stitch(self, sequence: str) -> StitchResult:
        seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
        n = len(seq)
        if n < 2:
            raise ValueError("sequence must be length >= 2")

        sin_phi = np.zeros(n, dtype=np.float64)
        cos_phi = np.zeros(n, dtype=np.float64)
        sin_psi = np.zeros(n, dtype=np.float64)
        cos_psi = np.zeros(n, dtype=np.float64)
        w_tot = np.zeros(n, dtype=np.float64)
        n_windows = 0

        for scale in self.scales:
            W = scale.length
            if n < 2:
                break
            # allow short terminal windows by clamping length
            win_len = min(W, n)
            starts = _window_starts(n, win_len, scale.stride)
            if starts.size == 0:
                continue

            # --- gather all window sequences (O(N) Python, W fixed) ---
            # Building strings is the only per-window Python work; angle
            # fusion below is fully vectorized.
            frags: List[Optional[FragmentRecord]] = []
            for s in starts:
                sub = seq[int(s) : int(s) + win_len]
                hit = self.db.get(sub)
                # try exact length-W key if we clamped (short protein)
                if hit is None and win_len != W:
                    hit = self.db.get(sub)
                frags.append(hit)

            # pack hits into flat arrays for scatter-add
            res_idx_list: List[np.ndarray] = []
            phi_list: List[np.ndarray] = []
            psi_list: List[np.ndarray] = []
            w_list: List[np.ndarray] = []

            for s, hit in zip(starts, frags):
                s = int(s)
                L = win_len
                idx = np.arange(s, s + L, dtype=np.int64)
                if hit is None:
                    if self.missing_penalty <= 0:
                        continue
                    # optional: still count coverage gap (no angles)
                    continue

                # per-residue confidence × scale weight
                conf = np.asarray(hit.confidence, dtype=np.float64)
                if conf.shape[0] != L:
                    # DB entry longer/shorter than window — align prefix
                    conf = conf[:L]
                    ph = hit.phis_deg[:L]
                    ps = hit.psis_deg[:L]
                else:
                    ph = hit.phis_deg
                    ps = hit.psis_deg

                w = conf * scale.weight
                res_idx_list.append(idx)
                phi_list.append(ph)
                psi_list.append(ps)
                w_list.append(w)
                n_windows += 1

            if not res_idx_list:
                continue

            residue_index = np.concatenate(res_idx_list)
            phis = np.concatenate(phi_list)
            psis = np.concatenate(psi_list)
            weights = np.concatenate(w_list)

            s_phi, c_phi, w_phi = _scatter_circular(n, residue_index, phis, weights)
            s_psi, c_psi, w_psi = _scatter_circular(n, residue_index, psis, weights)

            # φ and ψ share the same weights by construction
            sin_phi += s_phi
            cos_phi += c_phi
            sin_psi += s_psi
            cos_psi += c_psi
            w_tot += w_phi  # == w_psi

        phis = _finalize_circular(sin_phi, cos_phi, w_tot)
        psis = _finalize_circular(sin_psi, cos_psi, w_tot)

        # effective confidence ∈ [0, 1]: normalize by max observed weight
        conf = w_tot.copy()
        peak = conf.max() if conf.size else 0.0
        if peak > 1e-12:
            conf = np.clip(conf / peak, 0.0, 1.0)
        else:
            conf = np.zeros(n, dtype=np.float64)

        # low-confidence / uncovered → default SS + relax flag
        need_relax = (w_tot < self.low_conf_threshold) | np.isnan(phis)
        used_fallback = need_relax.copy()
        if need_relax.any():
            dphi, dpsi = default_ss_angles(seq)
            phis = np.where(need_relax, dphi, phis)
            psis = np.where(need_relax, dpsi, psis)

        structure = None
        if self.build_structure:
            structure = build_backbone(seq, phis.tolist(), psis.tolist())

        return StitchResult(
            sequence=seq,
            phis_deg=phis,
            psis_deg=psis,
            confidence=conf,
            need_relax=need_relax,
            used_fallback=used_fallback,
            n_windows=n_windows,
            structure=structure,
        )


def stitch_from_records(
    sequence: str,
    records: Iterable[Union[FragmentRecord, Mapping]],
    **kwargs,
) -> StitchResult:
    """Convenience: build DB from records and stitch in one call."""
    db = FragmentDatabase(records)
    return OverlapStitcher(db, **kwargs).stitch(sequence)


# ---------------------------------------------------------------------------
# Fast path when ALL windows of a scale are pre-materialized as arrays
# ---------------------------------------------------------------------------


def stitch_array_bank(
    sequence: str,
    *,
    window_length: int,
    stride: int,
    # banks indexed by window order from _window_starts
    phis_bank: np.ndarray,  # (n_windows, W)
    psis_bank: np.ndarray,  # (n_windows, W)
    conf_bank: np.ndarray,  # (n_windows, W) or (n_windows,)
    scale_weight: float = 1.0,
    low_conf_threshold: float = 0.25,
    build_structure: bool = True,
) -> StitchResult:
    """
    Fully vectorized stitch when the caller already has dense angle banks
    aligned to sliding windows (no dict lookups). Ideal for GPU/batch models.

    Complexity: O(N) scatter with fixed W.
    """
    seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
    n = len(seq)
    W = window_length
    starts = _window_starts(n, min(W, n), stride)
    n_win = starts.shape[0]
    if phis_bank.shape[0] != n_win or psis_bank.shape[0] != n_win:
        raise ValueError("bank rows must equal number of windows")

    W_eff = phis_bank.shape[1]
    # residue index matrix: starts[:, None] + arange(W)
    offsets = np.arange(W_eff, dtype=np.int64)
    residue_index = (starts[:, None] + offsets[None, :]).ravel()

    if conf_bank.ndim == 1:
        weights = np.broadcast_to(conf_bank[:, None], phis_bank.shape).ravel() * scale_weight
    else:
        weights = conf_bank.ravel() * scale_weight

    phis_flat = np.asarray(phis_bank, dtype=np.float64).ravel()
    psis_flat = np.asarray(psis_bank, dtype=np.float64).ravel()

    # clamp indices (terminal short windows may have been padded — mask them)
    valid = residue_index < n
    residue_index = residue_index[valid]
    phis_flat = phis_flat[valid]
    psis_flat = psis_flat[valid]
    weights = weights[valid]

    s_phi, c_phi, w = _scatter_circular(n, residue_index, phis_flat, weights)
    s_psi, c_psi, w2 = _scatter_circular(n, residue_index, psis_flat, weights)

    phis = _finalize_circular(s_phi, c_phi, w)
    psis = _finalize_circular(s_psi, c_psi, w2)

    conf = w.copy()
    peak = conf.max() if conf.size else 0.0
    conf = np.clip(conf / peak, 0.0, 1.0) if peak > 1e-12 else np.zeros(n)

    need_relax = (w < low_conf_threshold) | np.isnan(phis)
    used_fallback = need_relax.copy()
    if need_relax.any():
        dphi, dpsi = default_ss_angles(seq)
        phis = np.where(need_relax, dphi, phis)
        psis = np.where(need_relax, dpsi, psis)

    structure = build_backbone(seq, phis.tolist(), psis.tolist()) if build_structure else None
    return StitchResult(
        sequence=seq,
        phis_deg=phis,
        psis_deg=psis,
        confidence=conf,
        need_relax=need_relax,
        used_fallback=used_fallback,
        n_windows=int(n_win),
        structure=structure,
    )


def main() -> None:
    # Tiny synthetic 5-mer DB
    records = []
    motifs = {
        "AAAAA": ([-57] * 5, [-47] * 5, 0.95),
        "AAAAG": ([-58, -57, -56, -55, -70], [-47, -46, -45, -44, 60], 0.80),
        "AAAGA": ([-57, -56, -70, -60, -55], [-47, -45, 60, -40, -45], 0.55),
        "AAGAA": ([-56, -70, -65, -57, -57], [-45, 55, -40, -47, -47], 0.40),
        "AGAAA": ([-70, -60, -57, -57, -57], [50, -40, -47, -47, -47], 0.35),
        "GAAAA": ([-75, -57, -57, -57, -57], [70, -47, -47, -47, -47], 0.30),
        # low-conf junk to trigger fallback on unseen regions
        "VVVVV": ([-120] * 5, [120] * 5, 0.90),
    }
    for seq, (ph, ps, c) in motifs.items():
        records.append(FragmentRecord(seq, ph, ps, c))

    # Also add 3-mers / 4-mers for multi-scale
    for seq, (ph, ps, c) in list(motifs.items()):
        records.append(FragmentRecord(seq[:4], ph[:4], ps[:4], c * 0.9))
        records.append(FragmentRecord(seq[:3], ph[:3], ps[:3], c * 0.8))

    stitcher = OverlapStitcher(
        FragmentDatabase(records),
        scales=DEFAULT_SCALES,
        low_conf_threshold=0.20,
    )
    query = "AAAAAAAAAAAA"  # 12-mer helix-like
    result = stitcher.stitch(query)

    print(f"sequence:     {result.sequence}")
    print(f"n_windows:    {result.n_windows}")
    print(f"mean_conf:    {result.mean_confidence:.3f}")
    print(f"need_relax:   {result.need_relax.astype(int).tolist()}")
    print(f"fallback:     {result.used_fallback.astype(int).tolist()}")
    print(f"phi[:4]:      {np.round(result.phis_deg[:4], 1)}")
    print(f"psi[:4]:      {np.round(result.psis_deg[:4], 1)}")
    if result.structure:
        print(f"atoms:        {len(result.structure['atoms'])}")


if __name__ == "__main__":
    main()
