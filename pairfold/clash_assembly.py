"""
Clash-aware fragment assembly for φ/ψ-based backbone models.

Solves angle-chain error accumulation by:
  1. Converting predicted dihedrals → Cartesian Cα (and optional N/C) coords
  2. Scoring steric clashes with a fast NumPy Cα–Cα contact check
  3. Assembling pentamer (or 2–5-mer) fragments via greedy backtracking or MCTS,
     falling back to the next-best confidence hypothesis when a placement clashes

This module is intentionally model-agnostic: pass ranked angle hypotheses per
window (from your network / PDB fragment DB). No torch dependency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .assemble import ANG, LEN, place_atom

# Ideal consecutive Cα–Cα virtual bond (~trans peptide)
CA_CA_IDEAL_A = 3.8
# Default hard clash: non-local Cα pairs closer than this (Å)
DEFAULT_CLASH_A = 3.8
# Ignore sequential neighbors |i−j| < MIN_SEQ_SEP (1 = bonded, often skip < 2)
DEFAULT_MIN_SEQ_SEP = 2


# ---------------------------------------------------------------------------
# 1. Dihedral → Cartesian
# ---------------------------------------------------------------------------


def dihedrals_to_backbone(
    phis_deg: Sequence[float],
    psis_deg: Sequence[float],
    omega_deg: float = 180.0,
    sequence: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Build N / Cα / C coordinates from backbone dihedrals (degrees).

    Returns dict with keys:
      N, CA, C : (L, 3) float64
      sequence : str (optional labels)
    """
    phis = np.asarray(phis_deg, dtype=np.float64)
    psis = np.asarray(psis_deg, dtype=np.float64)
    L = int(phis.shape[0])
    if psis.shape[0] != L:
        raise ValueError("phis and psis must have the same length")
    if sequence is not None and len(sequence) != L:
        raise ValueError("sequence length must match dihedral arrays")

    N = np.zeros((L, 3), dtype=np.float64)
    CA = np.zeros((L, 3), dtype=np.float64)
    C = np.zeros((L, 3), dtype=np.float64)

    N[0] = (0.0, 0.0, 0.0)
    CA[0] = (LEN["N_CA"], 0.0, 0.0)
    C[0] = place_atom(
        np.array([0.0, 1.0, 0.0]),
        N[0],
        CA[0],
        LEN["CA_C"],
        ANG["N_CA_C"],
        0.0,
    )

    for i in range(L - 1):
        N[i + 1] = place_atom(N[i], CA[i], C[i], LEN["C_N"], ANG["CA_C_N"], float(psis[i]))
        CA[i + 1] = place_atom(
            CA[i], C[i], N[i + 1], LEN["N_CA"], ANG["C_N_CA"], omega_deg
        )
        C[i + 1] = place_atom(
            C[i], N[i + 1], CA[i + 1], LEN["CA_C"], ANG["N_CA_C"], float(phis[i + 1])
        )

    out: Dict[str, np.ndarray] = {"N": N, "CA": CA, "C": C}
    if sequence is not None:
        out["sequence"] = sequence  # type: ignore[assignment]
    return out


def dihedrals_to_ca(
    phis_deg: Sequence[float],
    psis_deg: Sequence[float],
    omega_deg: float = 180.0,
) -> np.ndarray:
    """Convenience: return only Cα coordinates, shape (L, 3)."""
    return dihedrals_to_backbone(phis_deg, psis_deg, omega_deg)["CA"]


def center_coords(xyz: np.ndarray) -> np.ndarray:
    """Translate so centroid is at origin (copy)."""
    return xyz - xyz.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# 2. Fast steric / clash scoring (Cα)
# ---------------------------------------------------------------------------


def ca_distance_matrix(ca: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances, shape (L, L)."""
    ca = np.asarray(ca, dtype=np.float64)
    from .mem_guard import guard_matrix

    guard_matrix(int(ca.shape[0]), itemsize=8, label="CA distance matrix")
    # (L,1,3) - (1,L,3) → (L,L,3)
    d = ca[:, None, :] - ca[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d))


def clash_pairs(
    ca: np.ndarray,
    clash_thresh: float = DEFAULT_CLASH_A,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (i_idx, j_idx, distances) for upper-triangle pairs with
    |i−j| >= min_seq_sep and d < clash_thresh.
    """
    ca = np.asarray(ca, dtype=np.float64)
    L = ca.shape[0]
    if L < min_seq_sep + 1:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float64)

    dist = ca_distance_matrix(ca)
    ii, jj = np.triu_indices(L, k=min_seq_sep)
    d = dist[ii, jj]
    mask = d < clash_thresh
    return ii[mask], jj[mask], d[mask]


def clash_energy(
    ca: np.ndarray,
    clash_thresh: float = DEFAULT_CLASH_A,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
    soft: bool = True,
) -> float:
    """
    Physical-ish clash score (lower is better).

    Hard: count of violating pairs.
    Soft: sum_i max(0, thresh − d_i)^2  (van der Waals-like repulsion).
    """
    _, _, d = clash_pairs(ca, clash_thresh, min_seq_sep)
    if d.size == 0:
        return 0.0
    if not soft:
        return float(d.size)
    pen = clash_thresh - d
    return float(np.dot(pen, pen))


def has_steric_clash(
    ca: np.ndarray,
    clash_thresh: float = DEFAULT_CLASH_A,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
) -> bool:
    i, _, _ = clash_pairs(ca, clash_thresh, min_seq_sep)
    return bool(i.size > 0)


def new_residue_clash(
    ca: np.ndarray,
    start_new: int,
    clash_thresh: float = DEFAULT_CLASH_A,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
) -> bool:
    """
    Check only clashes involving newly appended residues ca[start_new:].
    Faster incremental test during assembly.
    """
    ca = np.asarray(ca, dtype=np.float64)
    L = ca.shape[0]
    if start_new >= L:
        return False
    for j in range(start_new, L):
        # distances to all earlier residues with sufficient sequence separation
        i_max = j - min_seq_sep
        if i_max < 0:
            continue
        d = np.linalg.norm(ca[: i_max + 1] - ca[j], axis=1)
        if np.any(d < clash_thresh):
            return True
    return False


def expected_end_to_end(n_steps: int, mode: str = "coil") -> float:
    """
    Expected Cα–Cα end-to-end distance over `n_steps` virtual bonds.
    coil ≈ random-walk; helix ≈ short pitch along axis.
    """
    n_steps = max(int(n_steps), 1)
    if mode == "helix":
        return float(1.5 * n_steps)  # ~1.5 Å rise per residue
    # Flory-like polypeptide: ~3.8 * sqrt(N)
    return float(CA_CA_IDEAL_A * math.sqrt(n_steps))


def bend_deviation_score(
    ca: np.ndarray,
    start: int,
    window: int = 4,
) -> float:
    """
    Soft penalty if a local window is unnaturally collapsed or over-extended
    vs a relaxed coil (lever / kink detector). Lower is better; 0 is ideal.
    """
    ca = np.asarray(ca, dtype=np.float64)
    L = ca.shape[0]
    if L < 3:
        return 0.0
    w = max(2, min(int(window), L - 1))
    i0 = max(0, min(int(start), L - 1))
    i1 = min(L - 1, i0 + w)
    if i1 <= i0:
        return 0.0
    steps = i1 - i0
    d = float(np.linalg.norm(ca[i1] - ca[i0]))
    d_coil = expected_end_to_end(steps, "coil")
    d_helix = expected_end_to_end(steps, "helix")
    # Allow anything between helix and ~1.35× coil without penalty
    lo = 0.55 * d_helix
    hi = 1.35 * d_coil
    if lo <= d <= hi:
        return 0.0
    if d < lo:
        return float((lo - d) ** 2)
    return float((d - hi) ** 2)


def lookahead_ok(
    ca: np.ndarray,
    placed_start: int,
    clash_thresh: float = DEFAULT_CLASH_A,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
    bend_window: int = 4,
    bend_thresh: float = 8.0,
) -> bool:
    """
    Dynamic look-ahead: reject placement if new residues clash OR induce a
    severe local bend (global lever risk) over a short window.
    """
    if new_residue_clash(ca, placed_start, clash_thresh, min_seq_sep):
        return False
    # Check bend at a few anchors near the join
    L = ca.shape[0]
    for s in range(max(0, placed_start - 2), min(L - 1, placed_start + 2)):
        if bend_deviation_score(ca, s, window=bend_window) > bend_thresh:
            return False
    return True


def local_torsion_relax(
    phis_deg: np.ndarray,
    psis_deg: np.ndarray,
    focus: Sequence[int],
    n_steps: int = 16,
    delta_deg: float = 4.0,
    clash_thresh: float = DEFAULT_CLASH_A,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Lightweight local φ/ψ perturbation (±delta) on `focus` residues to reduce
    clash energy — localized energy descent without DB re-search.

    Returns (phis, psis, best_energy).
    """
    ph = np.asarray(phis_deg, dtype=np.float64).copy()
    ps = np.asarray(psis_deg, dtype=np.float64).copy()
    focus_idx = [int(i) for i in focus if 0 <= int(i) < len(ph)]
    if not focus_idx:
        ca = dihedrals_to_ca(ph, ps)
        return ph, ps, clash_energy(ca, clash_thresh, min_seq_sep, soft=True)

    rng = np.random.default_rng(seed)
    ca = dihedrals_to_ca(ph, ps)
    best_e = clash_energy(ca, clash_thresh, min_seq_sep, soft=True)
    # Add mild bend terms near focus
    for i in focus_idx:
        best_e += 0.15 * bend_deviation_score(ca, max(0, i - 2), window=4)
    best_ph, best_ps = ph.copy(), ps.copy()

    for step in range(max(1, n_steps)):
        i = int(rng.choice(focus_idx))
        which = "phi" if rng.random() < 0.5 else "psi"
        delta = float(rng.uniform(-delta_deg, delta_deg))
        trial_ph, trial_ps = best_ph.copy(), best_ps.copy()
        if which == "phi":
            trial_ph[i] = ((trial_ph[i] + delta + 180.0) % 360.0) - 180.0
        else:
            trial_ps[i] = ((trial_ps[i] + delta + 180.0) % 360.0) - 180.0
        trial_ca = dihedrals_to_ca(trial_ph, trial_ps)
        e = clash_energy(trial_ca, clash_thresh, min_seq_sep, soft=True)
        for j in focus_idx:
            e += 0.15 * bend_deviation_score(trial_ca, max(0, j - 2), window=4)
        # Greedy accept improvements; occasional mild uphill early on
        temp = 0.35 * (1.0 - step / max(n_steps - 1, 1)) + 0.05
        if e <= best_e or rng.random() < math.exp(-(e - best_e) / max(temp, 1e-6)):
            best_e, best_ph, best_ps = e, trial_ph, trial_ps

    return best_ph, best_ps, float(best_e)


def end_to_end_anchor_relax(
    phis_deg: np.ndarray,
    psis_deg: np.ndarray,
    anchors: Optional[Sequence[Tuple[int, int, float]]] = None,
    n_steps: int = 24,
    delta_deg: float = 3.5,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Optimize φ/ψ so key residue pairs approach target distances (Å).

    anchors: iterable of (i, j, target_dist). If None, only corrects severe
    lever over-/under-extension toward a mild Flory-like target (no aggressive
    compaction when the chain is already in a plausible range).

    When explicit DL / experimental anchors are provided, step count and
    proposal amplitude are increased so long-range contacts can close.
    """
    ph = np.asarray(phis_deg, dtype=np.float64).copy()
    ps = np.asarray(psis_deg, dtype=np.float64).copy()
    n = len(ph)
    if n < 3:
        ca = dihedrals_to_ca(ph, ps)
        return ph, ps, {"anchor_energy": 0.0, "rg": float(np.linalg.norm(ca[-1] - ca[0]))}

    ca0 = dihedrals_to_ca(ph, ps)
    ee0 = float(np.linalg.norm(ca0[-1] - ca0[0]))
    coil_t = expected_end_to_end(n - 1, "coil")
    helix_t = expected_end_to_end(n - 1, "helix")

    explicit = bool(anchors)
    if not anchors:
        # Skip unless clearly over-extended (lever) or collapsed
        lo, hi = 0.45 * helix_t, 1.55 * coil_t
        if lo <= ee0 <= hi:
            return ph, ps, {
                "anchor_energy": 0.0,
                "end_to_end": ee0,
                "clash_energy": float(clash_energy(ca0, soft=True)),
                "skipped": 1.0,
                "n_anchors": 0.0,
            }
        target = coil_t * 0.95 if ee0 > hi else max(helix_t, coil_t * 0.55)
        anchors = [(0, n - 1, target)]
        n_steps = min(n_steps, max(8, n // 6))
    else:
        n_anc = len(list(anchors))
        # Stronger search when DL contacts are available
        n_steps = max(n_steps, min(400, 24 + n_anc * max(8, n // 6)))
        delta_deg = max(delta_deg, 6.0)

    anchors = list(anchors)

    def anchor_energy(ca: np.ndarray) -> float:
        e = 0.0
        for i, j, td in anchors:
            if not (0 <= i < n and 0 <= j < n and i != j):
                continue
            d = float(np.linalg.norm(ca[j] - ca[i]))
            e += (d - float(td)) ** 2
        return e

    rng = np.random.default_rng(seed)
    # Hinges: ends + mid + anchor residues (+ path samples between pairs)
    hinges = {0, n - 1, n // 2, n // 4, (3 * n) // 4}
    for i, j, _ in anchors:
        i, j = int(i), int(j)
        hinges.add(i)
        hinges.add(j)
        hinges.add(max(0, i - 1))
        hinges.add(min(n - 1, j + 1))
        if explicit and abs(j - i) > 4:
            for t in (0.25, 0.5, 0.75):
                hinges.add(int(i + t * (j - i)))
    hinge_idx = sorted(h for h in hinges if 0 <= h < n)

    ca = ca0
    clash_w = 0.15 if explicit else 0.25
    best_e = anchor_energy(ca) + clash_w * clash_energy(ca, soft=True)
    best_ph, best_ps = ph.copy(), ps.copy()

    for step in range(max(1, n_steps)):
        i = int(rng.choice(hinge_idx))
        trial_ph, trial_ps = best_ph.copy(), best_ps.copy()
        # Anneal proposal size: large early moves, fine late
        frac = step / max(n_steps - 1, 1)
        amp = delta_deg * (1.0 - 0.65 * frac)
        delta = float(rng.uniform(-amp, amp))
        if rng.random() < 0.5:
            trial_ph[i] = ((trial_ph[i] + delta + 180.0) % 360.0) - 180.0
        else:
            trial_ps[i] = ((trial_ps[i] + delta + 180.0) % 360.0) - 180.0
        trial_ca = dihedrals_to_ca(trial_ph, trial_ps)
        e = anchor_energy(trial_ca) + clash_w * clash_energy(trial_ca, soft=True)
        temp = 1.2 * (1.0 - frac) + 0.05 if explicit else 0.5 * (1.0 - frac) + 0.08
        if e <= best_e or rng.random() < math.exp(-(e - best_e) / max(temp, 1e-6)):
            best_e, best_ph, best_ps = e, trial_ph, trial_ps

    ca = dihedrals_to_ca(best_ph, best_ps)
    return best_ph, best_ps, {
        "anchor_energy": float(anchor_energy(ca)),
        "end_to_end": float(np.linalg.norm(ca[-1] - ca[0])),
        "clash_energy": float(clash_energy(ca, soft=True)),
        "skipped": 0.0,
        "n_anchors": float(len(anchors)),
        "n_steps": float(n_steps),
    }


def correct_lever_effect(
    sequence: str,
    phis_deg: Sequence[float],
    psis_deg: Sequence[float],
    lookahead: int = 4,
    relax_steps: int = 14,
    anchors: Optional[Sequence[Tuple[int, int, float]]] = None,
    seed: int = 0,
) -> Dict:
    """
    O(N)-ish post-assembly polish against cumulative lever error.

    1) Sweep joins every `lookahead` residues; if clash/bend fires, locally relax.
    2) End-to-end / anchor distance fine-tune.
    """
    seq = "".join(c for c in sequence.upper() if c.isalpha())
    ph = np.asarray(phis_deg, dtype=np.float64).copy()
    ps = np.asarray(psis_deg, dtype=np.float64).copy()
    n = len(seq)
    if n != len(ph) or n != len(ps):
        raise ValueError("sequence/angle length mismatch")

    repairs = 0
    # Sliding look-ahead windows — linear number of checks
    step = max(2, lookahead)
    for start in range(0, max(n - 2, 1), step):
        ca = dihedrals_to_ca(ph, ps)
        if lookahead_ok(ca, start, bend_window=lookahead):
            continue
        lo = max(0, start - 1)
        hi = min(n - 1, start + lookahead + 1)
        focus = list(range(lo, hi + 1))
        ph, ps, _ = local_torsion_relax(
            ph, ps, focus, n_steps=relax_steps, delta_deg=4.0, seed=seed + start
        )
        repairs += 1

    base_steps = max(12, n // 4)
    if anchors:
        base_steps = max(base_steps, 48 + 10 * len(list(anchors)))
    ph, ps, anchor_info = end_to_end_anchor_relax(
        ph, ps, anchors=anchors, n_steps=base_steps, seed=seed + 17
    )
    ca = dihedrals_to_ca(ph, ps)
    return {
        "phis_deg": ph,
        "psis_deg": ps,
        "ca": ca,
        "repairs": repairs,
        "anchor_info": anchor_info,
        "clash_energy": float(clash_energy(ca, soft=True)),
    }


# ---------------------------------------------------------------------------
# 3. Hypotheses + assembly search
# ---------------------------------------------------------------------------


@dataclass
class AngleHypothesis:
    """One ranked (φ, ψ) prediction for a sequence window."""

    confidence: float
    phis_deg: np.ndarray
    psis_deg: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        self.phis_deg = np.asarray(self.phis_deg, dtype=np.float64)
        self.psis_deg = np.asarray(self.psis_deg, dtype=np.float64)
        if self.phis_deg.shape != self.psis_deg.shape:
            raise ValueError("hypothesis phis/psis shape mismatch")


@dataclass
class FragmentSlot:
    """A contiguous window with ranked alternative angle sets."""

    start: int
    end: int  # exclusive
    candidates: List[AngleHypothesis]

    @property
    def length(self) -> int:
        return self.end - self.start

    def ranked(self) -> List[AngleHypothesis]:
        return sorted(self.candidates, key=lambda h: -h.confidence)


@dataclass
class AssemblyResult:
    sequence: str
    phis_deg: np.ndarray
    psis_deg: np.ndarray
    ca: np.ndarray
    backbone: Dict[str, np.ndarray]
    chosen_ranks: List[int]
    clash_energy: float
    n_clashes: int
    method: str
    note: str = ""


def _merge_angles(
    n: int,
    slots: Sequence[FragmentSlot],
    ranks: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Stitch chosen hypotheses into full-length φ/ψ (degrees).

    `ranks` may be a prefix (partial assembly); uncovered residues stay NaN.
    """
    phis = np.full(n, np.nan, dtype=np.float64)
    psis = np.full(n, np.nan, dtype=np.float64)
    if len(ranks) > len(slots):
        raise ValueError("more ranks than slots")
    for slot, rank in zip(slots, ranks):
        cands = slot.ranked()
        if rank >= len(cands):
            raise IndexError(f"rank {rank} out of range for slot {slot.start}:{slot.end}")
        hyp = cands[rank]
        L = slot.length
        if hyp.phis_deg.shape[0] != L:
            raise ValueError(
                f"hypothesis length {hyp.phis_deg.shape[0]} != slot length {L}"
            )
        phis[slot.start : slot.end] = hyp.phis_deg
        psis[slot.start : slot.end] = hyp.psis_deg
    return phis, psis


def _prefix_length(slots: Sequence[FragmentSlot], n_chosen: int) -> int:
    if n_chosen <= 0:
        return 0
    return slots[n_chosen - 1].end


def _build_prefix_ca(
    slots: Sequence[FragmentSlot],
    ranks: Sequence[int],
) -> np.ndarray:
    """Build Cα for the covered prefix only (used during search)."""
    if not ranks:
        return np.zeros((0, 3), dtype=np.float64)
    end = _prefix_length(slots, len(ranks))
    phis, psis = _merge_angles(end, slots[: len(ranks)], ranks)
    if np.isnan(phis).any():
        raise ValueError("prefix angles incomplete")
    return dihedrals_to_ca(phis, psis)


def _score_assembly(
    ca: np.ndarray,
    ranks: Sequence[int],
    slots: Sequence[FragmentSlot],
    clash_thresh: float,
    min_seq_sep: int,
    clash_weight: float = 1.0,
) -> float:
    """Higher is better: mean confidence − clash_weight * soft clash energy."""
    confs = []
    for slot, rank in zip(slots, ranks):
        confs.append(slot.ranked()[rank].confidence)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    e = clash_energy(ca, clash_thresh, min_seq_sep, soft=True)
    return mean_conf - clash_weight * e


def make_pentamer_slots(
    sequence: str,
    hypothesis_fn: Callable[[str, int, int], List[AngleHypothesis]],
    frag_len: int = 5,
) -> List[FragmentSlot]:
    """
    Tile the chain into non-overlapping windows of `frag_len` (last may be shorter,
    minimum length 2). `hypothesis_fn(subseq, start, end)` returns ranked candidates.
    """
    n = len(sequence)
    slots: List[FragmentSlot] = []
    i = 0
    while i < n:
        rem = n - i
        if rem > frag_len and rem < frag_len + 2:
            # avoid orphan length-1 tail: take rem-2 then 2, or shrink last
            L = rem - 2 if rem - 2 >= 2 else rem
        else:
            L = min(frag_len, rem)
        if L < 2:
            # merge into previous slot if possible
            if slots:
                prev = slots[-1]
                slots[-1] = FragmentSlot(
                    prev.start,
                    n,
                    hypothesis_fn(sequence[prev.start : n], prev.start, n),
                )
            break
        j = i + L
        slots.append(FragmentSlot(i, j, hypothesis_fn(sequence[i:j], i, j)))
        i = j
    return slots


def assemble_greedy_backtrack(
    sequence: str,
    slots: Sequence[FragmentSlot],
    clash_thresh: float = DEFAULT_CLASH_A,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
    max_nodes: int = 50_000,
    lookahead: int = 4,
    relax_on_clash: bool = True,
    anchors: Optional[Sequence[Tuple[int, int, float]]] = None,
) -> AssemblyResult:
    """
    Depth-first greedy backtracking with dynamic look-ahead and local torsion
    relaxation (lever-effect mitigation).

    At each fragment slot try candidates in descending confidence.
    Placement is rejected if look-ahead clash/bend fires; optional local φ/ψ
    relaxation (±1–5°) is tried before skipping to the next hypothesis.
    """
    n = len(sequence)
    slots = list(slots)
    if not slots:
        raise ValueError("no fragment slots")

    ranked = [s.ranked() for s in slots]
    max_rank = [len(r) for r in ranked]
    if any(m == 0 for m in max_rank):
        raise ValueError("every slot needs ≥1 hypothesis")

    lookahead = int(max(3, min(5, lookahead)))
    choice: List[int] = []
    nodes = 0
    relax_hits = 0
    best: Optional[Tuple[List[int], float, np.ndarray, np.ndarray]] = None
    # Persistent local φ/ψ overrides from successful torsion relax (NaN = none)
    override_ph = np.full(n, np.nan, dtype=np.float64)
    override_ps = np.full(n, np.nan, dtype=np.float64)

    def _apply_overrides(phis: np.ndarray, psis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = ~np.isnan(override_ph)
        if not mask.any():
            return phis, psis
        ph = np.array(phis, copy=True)
        ps = np.array(psis, copy=True)
        ph[mask] = override_ph[mask]
        ps[mask] = override_ps[mask]
        return ph, ps

    def rebuild(ranks: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        phis, psis = _merge_angles(n, slots, ranks)
        phis, psis = _apply_overrides(phis, psis)
        if len(ranks) < len(slots):
            end = _prefix_length(slots, len(ranks))
            ph_full = np.array(phis, copy=True)
            ps_full = np.array(psis, copy=True)
            # Provisional look-ahead: fill upcoming top-1 into a short window
            filled_to = end
            for sidx in range(len(ranks), len(slots)):
                if filled_to >= end + lookahead:
                    break
                hyp = ranked[sidx][0]
                sl = slots[sidx]
                for k in range(sl.length):
                    idx = sl.start + k
                    if idx >= end + lookahead:
                        break
                    if np.isnan(ph_full[idx]):
                        ph_full[idx] = hyp.phis_deg[k]
                        ps_full[idx] = hyp.psis_deg[k]
                        filled_to = idx + 1
            use_n = 0
            while use_n < n and not np.isnan(ph_full[use_n]):
                use_n += 1
            ca = dihedrals_to_ca(ph_full[:use_n], ps_full[:use_n])
            # Return angles that match CA (prefix + provisional LA), not raw slots
            return ph_full, ps_full, ca
        if np.isnan(phis).any():
            missing = np.where(np.isnan(phis))[0]
            raise ValueError(f"incomplete coverage; missing {missing.tolist()}")
        ca = dihedrals_to_ca(phis, psis)
        return phis, psis, ca

    def try_relax_prefix(ranks: Sequence[int], placed: int) -> bool:
        """Perturb hinge torsions near `placed`; on success write into overrides."""
        nonlocal relax_hits
        if not relax_on_clash:
            return False
        phis, psis, ca = rebuild(ranks)
        use_n = ca.shape[0]
        if use_n < 3:
            return False
        ph_pref = np.asarray(phis[:use_n], dtype=np.float64)
        ps_pref = np.asarray(psis[:use_n], dtype=np.float64)
        if np.isnan(ph_pref).any() or np.isnan(ps_pref).any():
            return False
        focus = list(range(max(0, placed - 1), min(use_n, placed + lookahead + 1)))
        delta = float(np.random.default_rng(nodes + placed).uniform(1.0, 5.0))
        ph_r, ps_r, _ = local_torsion_relax(
            ph_pref,
            ps_pref,
            focus,
            n_steps=12,
            delta_deg=delta,
            clash_thresh=clash_thresh,
            min_seq_sep=min_seq_sep,
            seed=nodes + placed,
        )
        ca_r = dihedrals_to_ca(ph_r, ps_r)
        if not lookahead_ok(ca_r, placed, clash_thresh, min_seq_sep, bend_window=lookahead):
            return False
        relax_hits += 1
        # Only persist overrides on the *committed* prefix (not provisional LA)
        end = _prefix_length(slots, len(ranks))
        override_ph[:end] = ph_r[:end]
        override_ps[:end] = ps_r[:end]
        return True

    def dfs() -> bool:
        nonlocal nodes, best
        nodes += 1
        if nodes > max_nodes:
            return False

        depth = len(choice)
        if depth == len(slots):
            phis, psis, ca = rebuild(choice)
            phis, psis, _info = end_to_end_anchor_relax(
                phis, psis, anchors=anchors, n_steps=max(8, n // 5), seed=nodes
            )
            ca = dihedrals_to_ca(phis, psis)
            e = clash_energy(ca, clash_thresh, min_seq_sep)
            score = _score_assembly(ca, choice, slots, clash_thresh, min_seq_sep)
            ee = float(np.linalg.norm(ca[-1] - ca[0]))
            score -= 0.001 * abs(ee - expected_end_to_end(n - 1, "coil") * 0.85)
            if best is None or score > best[1]:
                best = (list(choice), score, phis.copy(), psis.copy())
            return e == 0.0 or not has_steric_clash(ca, clash_thresh, min_seq_sep)

        placed = slots[depth].start
        for rank in range(max_rank[depth]):
            choice.append(rank)
            # Snapshot overrides so failed branches / backtracks restore cleanly
            snap_ph = override_ph.copy()
            snap_ps = override_ps.copy()
            _, _, ca = rebuild(choice)
            ok_place = lookahead_ok(
                ca, placed, clash_thresh, min_seq_sep, bend_window=lookahead
            )
            if not ok_place:
                ok_place = try_relax_prefix(choice, placed)
            progressed = False
            if ok_place:
                progressed = dfs()
            override_ph[:] = snap_ph
            override_ps[:] = snap_ps
            choice.pop()
            if progressed:
                return True
        return False

    ok = dfs()
    if best is None:
        ranks = [0] * len(slots)
        # Clear overrides for fallback rebuild from raw hypotheses
        override_ph[:] = np.nan
        override_ps[:] = np.nan
        phis, psis, ca = rebuild(ranks)
        phis, psis, _ = end_to_end_anchor_relax(phis, psis, anchors=anchors, n_steps=12)
        ca = dihedrals_to_ca(phis, psis)
    else:
        ranks, _, phis, psis = best
        ca = dihedrals_to_ca(phis, psis)

    # Optional global lever polish
    polished = correct_lever_effect(
        sequence, phis, psis, lookahead=lookahead, relax_steps=10, anchors=anchors
    )
    phis, psis, ca = polished["phis_deg"], polished["psis_deg"], polished["ca"]

    bb = dihedrals_to_backbone(phis, psis, sequence=sequence)
    i_cl, _, _ = clash_pairs(ca, clash_thresh, min_seq_sep)
    return AssemblyResult(
        sequence=sequence,
        phis_deg=phis,
        psis_deg=psis,
        ca=ca,
        backbone=bb,
        chosen_ranks=ranks,
        clash_energy=clash_energy(ca, clash_thresh, min_seq_sep),
        n_clashes=int(i_cl.size),
        method="greedy_backtrack_lookahead",
        note=(
            f"{'clash-free' if ok and i_cl.size == 0 else 'best-effort'} "
            f"after {nodes} nodes; relax_hits={relax_hits}; "
            f"lever_repairs={polished['repairs']}"
        ),
    )


# ---- MCTS -----------------------------------------------------------------


@dataclass
class _MCTSNode:
    ranks: Tuple[int, ...]  # partial assignment
    visits: int = 0
    value: float = 0.0
    children: Dict[int, "_MCTSNode"] = field(default_factory=dict)
    untried: Optional[List[int]] = None


def assemble_mcts(
    sequence: str,
    slots: Sequence[FragmentSlot],
    clash_thresh: float = DEFAULT_CLASH_A,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
    n_simulations: int = 400,
    exploration: float = 1.4,
    clash_weight: float = 1.0,
    seed: int = 0,
) -> AssemblyResult:
    """
    Monte Carlo Tree Search over fragment-hypothesis ranks.

    Action at depth d = which confidence-ranked candidate to use for slot d.
    Rollouts fill remaining slots greedily (highest confidence that does not
    clash; else top-1). Reward = mean confidence − clash_weight * soft energy.
    """
    rng = np.random.default_rng(seed)
    slots = list(slots)
    n = len(sequence)
    ranked = [s.ranked() for s in slots]
    n_cand = [len(r) for r in ranked]

    def rebuild(ranks: Sequence[int]) -> np.ndarray:
        if len(ranks) < len(slots):
            return _build_prefix_ca(slots, ranks)
        phis, psis = _merge_angles(n, slots, ranks)
        if np.isnan(phis).any():
            raise ValueError("incomplete coverage in MCTS rebuild")
        return dihedrals_to_ca(phis, psis)

    def legal_actions(partial: Sequence[int]) -> List[int]:
        d = len(partial)
        return list(range(n_cand[d]))

    def rollout(partial: List[int]) -> Tuple[float, List[int]]:
        ranks = list(partial)
        while len(ranks) < len(slots):
            d = len(ranks)
            placed = slots[d].start
            picked = None
            for rank in range(n_cand[d]):
                trial = ranks + [rank]
                ca = rebuild(trial)
                if not new_residue_clash(ca, placed, clash_thresh, min_seq_sep):
                    picked = rank
                    break
            if picked is None:
                picked = 0
            ranks.append(picked)
        ca = rebuild(ranks)
        return _score_assembly(
            ca, ranks, slots, clash_thresh, min_seq_sep, clash_weight
        ), ranks

    def ucb(parent: _MCTSNode, child: _MCTSNode) -> float:
        if child.visits == 0:
            return float("inf")
        return child.value / child.visits + exploration * math.sqrt(
            math.log(parent.visits + 1) / child.visits
        )

    root = _MCTSNode(ranks=tuple())
    root.untried = legal_actions([])
    best_ranks: List[int] = [0] * len(slots)
    best_score = -1e18

    for _ in range(n_simulations):
        node = root
        path = [node]

        # selection
        while node.untried == [] and node.children and len(node.ranks) < len(slots):
            node = max(node.children.values(), key=lambda c: ucb(node, c))
            path.append(node)

        # expansion
        if node.untried is None:
            node.untried = legal_actions(list(node.ranks))
        if node.untried and len(node.ranks) < len(slots):
            # prefer high-confidence (low rank index) when expanding
            node.untried.sort()
            a = node.untried.pop(0)
            child = _MCTSNode(ranks=node.ranks + (a,))
            if not child.untried and len(child.ranks) < len(slots):
                child.untried = legal_actions(list(child.ranks))
            elif len(child.ranks) == len(slots):
                child.untried = []
            node.children[a] = child
            node = child
            path.append(node)

        # simulation
        reward, full_ranks = rollout(list(node.ranks))
        if reward > best_score:
            best_score = reward
            best_ranks = full_ranks

        # backprop
        for p in path:
            p.visits += 1
            p.value += reward

    phis, psis = _merge_angles(n, slots, best_ranks)
    ca = dihedrals_to_ca(phis, psis)
    bb = dihedrals_to_backbone(phis, psis, sequence=sequence)
    i_cl, _, _ = clash_pairs(ca, clash_thresh, min_seq_sep)
    return AssemblyResult(
        sequence=sequence,
        phis_deg=phis,
        psis_deg=psis,
        ca=ca,
        backbone=bb,
        chosen_ranks=best_ranks,
        clash_energy=clash_energy(ca, clash_thresh, min_seq_sep),
        n_clashes=int(i_cl.size),
        method="mcts",
        note=f"best_score={best_score:.4f} after {n_simulations} sims",
    )


def assemble_clash_aware(
    sequence: str,
    slots: Sequence[FragmentSlot],
    method: str = "backtrack",
    **kwargs,
) -> AssemblyResult:
    """Dispatch helper: method in {'backtrack', 'mcts'}."""
    method = method.lower()
    if method in ("backtrack", "greedy", "greedy_backtrack"):
        return assemble_greedy_backtrack(sequence, slots, **kwargs)
    if method == "mcts":
        return assemble_mcts(sequence, slots, **kwargs)
    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------


def _demo_hypotheses(sub: str, _start: int, _end: int) -> List[AngleHypothesis]:
    """Synthetic ranked alternatives: helix-like + sheet-like + noisy."""
    L = len(sub)
    helix = AngleHypothesis(
        confidence=0.85,
        phis_deg=np.full(L, -57.0),
        psis_deg=np.full(L, -47.0),
        source="helix",
    )
    sheet = AngleHypothesis(
        confidence=0.70,
        phis_deg=np.full(L, -120.0),
        psis_deg=np.full(L, 120.0),
        source="sheet",
    )
    # Deliberately clash-prone compact coil (for demo diversity)
    coil = AngleHypothesis(
        confidence=0.55,
        phis_deg=np.full(L, 60.0),
        psis_deg=np.full(L, 40.0),
        source="coil",
    )
    return [helix, sheet, coil]


def main() -> None:
    seq = "AAEAAKAAEAAKAA"  # 14 residues → pentamers 5+5+4
    slots = make_pentamer_slots(seq, _demo_hypotheses, frag_len=5)

    t0 = time.perf_counter()
    bt = assemble_clash_aware(seq, slots, method="backtrack")
    t1 = time.perf_counter()
    mc = assemble_clash_aware(seq, slots, method="mcts", n_simulations=200)
    t2 = time.perf_counter()

    for name, res, dt in (("backtrack", bt, t1 - t0), ("mcts", mc, t2 - t1)):
        print(f"\n=== {name} ({dt * 1000:.1f} ms) ===")
        print(f"ranks={res.chosen_ranks} clashes={res.n_clashes} E={res.clash_energy:.4f}")
        print(f"CA shape={res.ca.shape} note={res.note}")
        # consecutive CA–CA should be ~3.8 Å
        d01 = np.linalg.norm(res.ca[1] - res.ca[0])
        print(f"CA0-CA1 distance={d01:.3f} A (ideal ~{CA_CA_IDEAL_A})")


if __name__ == "__main__":
    main()
