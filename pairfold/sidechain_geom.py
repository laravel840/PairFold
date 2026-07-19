"""Lightweight sidechain geometry + rotamer packing (consumer-GPU friendly)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .assemble import place_atom

_CA_CB = 1.522
_N_CA_CB = 110.1
_C_N_CA_CB = -122.5

# Most-common / backup χ rotamers (degrees)
_ROTAMERS: Dict[str, List[Tuple[float, ...]]] = {
    "A": [()],
    "G": [()],
    "C": [(-60,), (180,), (60,)],
    "S": [(-60,), (60,), (180,)],
    "T": [(-60,), (60,), (180,)],
    "V": [(-60,), (180,), (60,)],
    "I": [(-60, 170), (-60, 60), (60, 170), (180, 170)],
    "L": [(-60, 180), (-60, 60), (180, 180), (60, 180)],
    "D": [(-70, 0), (-70, 40), (180, 0), (60, 0)],
    "N": [(-70, 0), (-70, 40), (180, 0), (60, 0)],
    "E": [(-70, 180, 0), (-70, 60, 0), (180, 180, 0), (60, 180, 0)],
    "Q": [(-70, 180, 0), (-70, 60, 0), (180, 180, 0), (60, 180, 0)],
    "M": [(-70, 180, 60), (-70, 180, 180), (180, 180, 60), (60, 180, 60)],
    "K": [(-70, 180, 180, 180), (-70, 180, 180, 60), (180, 180, 180, 180)],
    "R": [(-70, 180, 180, 180), (-70, 180, 180, 0), (180, 180, 180, 180)],
    "P": [(-20, 30), (20, -30)],
    "H": [(-70, -70), (-70, 180), (180, -70), (60, -70)],
    "F": [(-70, 90), (-70, -90), (180, 90), (60, 90)],
    "Y": [(-70, 90), (-70, -90), (180, 90), (60, 90)],
    "W": [(-70, 90), (-70, -90), (180, 90), (60, -90)],
}


def place_cb(N: np.ndarray, CA: np.ndarray, C: np.ndarray) -> np.ndarray:
    return place_atom(C, N, CA, _CA_CB, _N_CA_CB, _C_N_CA_CB)


def _chi(chis: Sequence[float], i: int, default: float = 180.0) -> float:
    return float(chis[i]) if i < len(chis) else default


def build_sidechain_atoms(
    aa: str,
    N: np.ndarray,
    CA: np.ndarray,
    C: np.ndarray,
    chis: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Return {atom_name: xyz} for sidechain heavy atoms (includes CB except Gly)."""
    aa = aa.upper()
    if aa == "G":
        return {}
    cb = place_cb(N, CA, C)
    atoms: Dict[str, np.ndarray] = {"CB": cb}
    c1 = _chi(chis, 0, -60.0)

    if aa == "A":
        return atoms
    if aa == "C":
        atoms["SG"] = place_atom(N, CA, cb, 1.808, 110.8, c1)
        return atoms
    if aa == "S":
        atoms["OG"] = place_atom(N, CA, cb, 1.417, 111.0, c1)
        return atoms
    if aa == "T":
        atoms["OG1"] = place_atom(N, CA, cb, 1.433, 109.6, c1)
        atoms["CG2"] = place_atom(N, CA, cb, 1.529, 111.6, c1 + 120.0)
        return atoms
    if aa == "V":
        atoms["CG1"] = place_atom(N, CA, cb, 1.527, 110.9, c1)
        atoms["CG2"] = place_atom(N, CA, cb, 1.527, 110.9, c1 + 120.0)
        return atoms
    if aa == "I":
        c2 = _chi(chis, 1, 170.0)
        cg1 = place_atom(N, CA, cb, 1.530, 111.0, c1)
        atoms["CG1"] = cg1
        atoms["CG2"] = place_atom(N, CA, cb, 1.530, 111.0, c1 - 120.0)
        atoms["CD1"] = place_atom(CA, cb, cg1, 1.516, 113.8, c2)
        return atoms
    if aa == "L":
        c2 = _chi(chis, 1, 180.0)
        cg = place_atom(N, CA, cb, 1.530, 116.3, c1)
        atoms["CG"] = cg
        atoms["CD1"] = place_atom(CA, cb, cg, 1.524, 111.2, c2)
        atoms["CD2"] = place_atom(CA, cb, cg, 1.524, 111.2, c2 + 120.0)
        return atoms
    if aa == "D":
        c2 = _chi(chis, 1, 0.0)
        cg = place_atom(N, CA, cb, 1.516, 112.8, c1)
        atoms["CG"] = cg
        atoms["OD1"] = place_atom(CA, cb, cg, 1.249, 118.6, c2)
        atoms["OD2"] = place_atom(CA, cb, cg, 1.249, 118.6, c2 + 180.0)
        return atoms
    if aa == "N":
        c2 = _chi(chis, 1, 0.0)
        cg = place_atom(N, CA, cb, 1.516, 112.8, c1)
        atoms["CG"] = cg
        atoms["OD1"] = place_atom(CA, cb, cg, 1.231, 120.8, c2)
        atoms["ND2"] = place_atom(CA, cb, cg, 1.328, 116.6, c2 + 180.0)
        return atoms
    if aa == "E":
        c2, c3 = _chi(chis, 1, 180.0), _chi(chis, 2, 0.0)
        cg = place_atom(N, CA, cb, 1.522, 114.2, c1)
        cd = place_atom(CA, cb, cg, 1.524, 114.2, c2)
        atoms["CG"], atoms["CD"] = cg, cd
        atoms["OE1"] = place_atom(cb, cg, cd, 1.249, 118.6, c3)
        atoms["OE2"] = place_atom(cb, cg, cd, 1.249, 118.6, c3 + 180.0)
        return atoms
    if aa == "Q":
        c2, c3 = _chi(chis, 1, 180.0), _chi(chis, 2, 0.0)
        cg = place_atom(N, CA, cb, 1.522, 114.2, c1)
        cd = place_atom(CA, cb, cg, 1.524, 114.2, c2)
        atoms["CG"], atoms["CD"] = cg, cd
        atoms["OE1"] = place_atom(cb, cg, cd, 1.231, 120.8, c3)
        atoms["NE2"] = place_atom(cb, cg, cd, 1.328, 116.6, c3 + 180.0)
        return atoms
    if aa == "M":
        c2, c3 = _chi(chis, 1, 180.0), _chi(chis, 2, 60.0)
        cg = place_atom(N, CA, cb, 1.522, 114.0, c1)
        sd = place_atom(CA, cb, cg, 1.803, 112.8, c2)
        atoms["CG"], atoms["SD"] = cg, sd
        atoms["CE"] = place_atom(cb, cg, sd, 1.791, 100.9, c3)
        return atoms
    if aa == "K":
        c2, c3, c4 = _chi(chis, 1, 180.0), _chi(chis, 2, 180.0), _chi(chis, 3, 180.0)
        cg = place_atom(N, CA, cb, 1.530, 114.2, c1)
        cd = place_atom(CA, cb, cg, 1.521, 111.9, c2)
        ce = place_atom(cb, cg, cd, 1.521, 111.9, c3)
        atoms["CG"], atoms["CD"], atoms["CE"] = cg, cd, ce
        atoms["NZ"] = place_atom(cg, cd, ce, 1.489, 111.9, c4)
        return atoms
    if aa == "R":
        c2, c3, c4 = _chi(chis, 1, 180.0), _chi(chis, 2, 180.0), _chi(chis, 3, 180.0)
        cg = place_atom(N, CA, cb, 1.520, 114.2, c1)
        cd = place_atom(CA, cb, cg, 1.520, 111.9, c2)
        ne = place_atom(cb, cg, cd, 1.461, 112.0, c3)
        cz = place_atom(cg, cd, ne, 1.330, 124.2, c4)
        atoms.update({"CG": cg, "CD": cd, "NE": ne, "CZ": cz})
        atoms["NH1"] = place_atom(cd, ne, cz, 1.326, 120.3, 0.0)
        atoms["NH2"] = place_atom(cd, ne, cz, 1.326, 120.3, 180.0)
        return atoms
    if aa == "P":
        c2 = _chi(chis, 1, 30.0)
        cg = place_atom(N, CA, cb, 1.492, 104.0, c1)
        atoms["CG"] = cg
        atoms["CD"] = place_atom(CA, cb, cg, 1.503, 105.0, c2)
        return atoms
    if aa == "H":
        c2 = _chi(chis, 1, -70.0)
        cg = place_atom(N, CA, cb, 1.497, 113.8, c1)
        atoms["CG"] = cg
        atoms["ND1"] = place_atom(CA, cb, cg, 1.378, 122.8, c2)
        atoms["CD2"] = place_atom(CA, cb, cg, 1.356, 130.8, c2 + 180.0)
        atoms["CE1"] = place_atom(cb, cg, atoms["ND1"], 1.347, 108.9, 180.0)
        atoms["NE2"] = place_atom(cb, cg, atoms["CD2"], 1.374, 109.8, 180.0)
        return atoms
    if aa in ("F", "Y"):
        c2 = _chi(chis, 1, 90.0)
        cg = place_atom(N, CA, cb, 1.502, 113.8, c1)
        atoms["CG"] = cg
        atoms["CD1"] = place_atom(CA, cb, cg, 1.389, 120.7, c2)
        atoms["CD2"] = place_atom(CA, cb, cg, 1.389, 120.7, c2 + 180.0)
        atoms["CE1"] = place_atom(cb, cg, atoms["CD1"], 1.382, 120.7, 180.0)
        atoms["CE2"] = place_atom(cb, cg, atoms["CD2"], 1.382, 120.7, 180.0)
        atoms["CZ"] = place_atom(cg, atoms["CD1"], atoms["CE1"], 1.382, 120.0, 0.0)
        if aa == "Y":
            atoms["OH"] = place_atom(atoms["CD1"], atoms["CE1"], atoms["CZ"], 1.376, 119.9, 180.0)
        return atoms
    if aa == "W":
        c2 = _chi(chis, 1, 90.0)
        cg = place_atom(N, CA, cb, 1.498, 113.6, c1)
        atoms["CG"] = cg
        atoms["CD1"] = place_atom(CA, cb, cg, 1.365, 126.9, c2)
        atoms["CD2"] = place_atom(CA, cb, cg, 1.433, 126.9, c2 + 180.0)
        atoms["NE1"] = place_atom(cb, cg, atoms["CD1"], 1.374, 110.3, 180.0)
        atoms["CE2"] = place_atom(cg, atoms["CD1"], atoms["NE1"], 1.413, 106.5, 0.0)
        atoms["CE3"] = place_atom(cb, cg, atoms["CD2"], 1.400, 133.8, 180.0)
        atoms["CZ2"] = place_atom(atoms["CD1"], atoms["NE1"], atoms["CE2"], 1.394, 122.0, 180.0)
        atoms["CZ3"] = place_atom(cg, atoms["CD2"], atoms["CE3"], 1.394, 118.0, 180.0)
        atoms["CH2"] = place_atom(atoms["NE1"], atoms["CE2"], atoms["CZ2"], 1.368, 120.0, 0.0)
        return atoms
    return atoms


def rotamers_for(aa: str) -> List[Tuple[float, ...]]:
    return list(_ROTAMERS.get(aa.upper(), [()]))


def clash_energy(points: np.ndarray, others: np.ndarray, cutoff: float = 2.6) -> float:
    if len(points) == 0 or len(others) == 0:
        return 0.0
    # Vectorized pairwise
    d = np.linalg.norm(points[:, None, :] - others[None, :, :], axis=-1)
    pen = np.maximum(0.0, cutoff - d)
    return float(np.sum(pen * pen))
