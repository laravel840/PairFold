"""Stage-2: lightweight sidechain packing (backbone locked)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .assemble import build_backbone
from .sidechain_geom import (
    build_sidechain_atoms,
    clash_energy,
    rotamers_for,
)


def pack_sidechains(
    sequence: str,
    phis: Sequence[float],
    psis: Sequence[float],
    backbone: Optional[Dict] = None,
    max_len: int = 256,
) -> Dict:
    """
    Pack sidechain heavy atoms onto a fixed backbone.

    Tries a small rotamer library per residue; greedily minimizes clashes
    against backbone + already packed sidechains. Does not move N/CA/C.
    """
    seq = "".join(c for c in sequence.upper() if c.isalpha())
    n = len(seq)
    if n == 0:
        return {"enabled": False, "note": "empty sequence", "chi_angles": [], "atoms": []}
    if n > max_len:
        return {
            "enabled": False,
            "note": f"Stage-2 skipped (length {n} > {max_len}).",
            "chi_angles": [[] for _ in seq],
            "atoms": [],
            "n_residues": n,
            "backbone_locked": True,
        }

    if backbone is None:
        backbone = build_backbone(seq, phis, psis)
    residues = backbone["residues"]

    bb_pts = []
    for r in residues:
        for key in ("N", "CA", "C"):
            bb_pts.append(np.asarray(r[key], dtype=np.float64))
    bb_pts = np.stack(bb_pts, axis=0) if bb_pts else np.zeros((0, 3))

    packed_sc: List[np.ndarray] = []
    chi_angles: List[List[float]] = []
    sc_atom_records: List[Dict] = []
    total_clash = 0.0

    for i, aa in enumerate(seq):
        N = np.asarray(residues[i]["N"], dtype=np.float64)
        CA = np.asarray(residues[i]["CA"], dtype=np.float64)
        C = np.asarray(residues[i]["C"], dtype=np.float64)
        # Exclude this residue's backbone from clash refs (bonded)
        mask = np.ones(len(bb_pts), dtype=bool)
        mask[i * 3 : (i + 1) * 3] = False
        others = bb_pts[mask]
        if packed_sc:
            others = np.concatenate([others, np.stack(packed_sc, axis=0)], axis=0)

        best_chi: Tuple[float, ...] = ()
        best_atoms: Dict[str, np.ndarray] = {}
        best_e = 1e18
        for rot in rotamers_for(aa):
            atoms = build_sidechain_atoms(aa, N, CA, C, rot)
            if not atoms:
                best_chi, best_atoms, best_e = (), {}, 0.0
                break
            pts = np.stack(list(atoms.values()), axis=0)
            e = clash_energy(pts, others, cutoff=2.55)
            if e < best_e:
                best_e = e
                best_chi = tuple(float(x) for x in rot)
                best_atoms = atoms
        total_clash += best_e
        chi_angles.append(list(best_chi))
        for name, xyz in best_atoms.items():
            packed_sc.append(xyz)
            sc_atom_records.append(
                {
                    "element": name,
                    "name": name,
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                    "residue": i,
                    "code": aa,
                    "sidechain": True,
                }
            )

    return {
        "enabled": True,
        "sequence": seq,
        "chi_angles": chi_angles,
        "atoms": sc_atom_records,
        "n_residues": n,
        "n_sidechain_atoms": len(sc_atom_records),
        "clash_energy": float(total_clash),
        "backbone_locked": True,
        "note": (
            f"Stage-2 rotamer pack: {len(sc_atom_records)} sidechain atoms, "
            f"clash_E={total_clash:.2f}."
        ),
    }


def sidechain_clash_score(
    coords: np.ndarray,
    min_sep: int = 1,
    cutoff: float = 2.5,
) -> float:
    if coords is None or len(coords) < 2:
        return 0.0
    e = 0.0
    n = len(coords)
    for i in range(n):
        for j in range(i + min_sep, n):
            d = float(np.linalg.norm(coords[j] - coords[i]))
            if d < cutoff:
                e += (cutoff - d) ** 2
    return float(e)
