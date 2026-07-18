"""Build 3D backbone coordinates from φ/ψ (degrees or radians)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEG = math.pi / 180.0
LEN = {"N_CA": 1.458, "CA_C": 1.525, "C_N": 1.329}
ANG = {"N_CA_C": 111.2, "CA_C_N": 116.2, "C_N_CA": 121.7}


def _v(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def _norm(a):
    n = np.linalg.norm(a)
    return a / n if n > 1e-8 else a


def place_atom(A, B, C, bond_length, bond_angle_deg, dihedral_deg):
    bc = _norm(C - B)
    n = np.cross(B - A, bc)
    if np.linalg.norm(n) < 1e-8:
        n = np.array([1.0, 0.0, 0.0]) if abs(bc[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n = _norm(np.cross(bc, n))
    else:
        n = _norm(n)
    m = np.cross(n, bc)
    theta = bond_angle_deg * DEG
    phi = dihedral_deg * DEG
    st = math.sin(theta)
    direction = -bc * math.cos(theta) + m * st * math.cos(phi) + n * st * math.sin(phi)
    return C + _norm(direction) * bond_length


def build_backbone(
    sequence: str,
    phis_deg: Sequence[float],
    psis_deg: Sequence[float],
    omega_deg: float = 180.0,
) -> Dict:
    n = len(sequence)
    assert n == len(phis_deg) == len(psis_deg)

    residues = []
    N0 = _v(0, 0, 0)
    CA0 = _v(LEN["N_CA"], 0, 0)
    C0 = place_atom(_v(0, 1, 0), N0, CA0, LEN["CA_C"], ANG["N_CA_C"], 0)
    residues.append({"code": sequence[0], "N": N0, "CA": CA0, "C": C0, "phi": phis_deg[0], "psi": psis_deg[0]})

    for i in range(n - 1):
        res = residues[i]
        Nnext = place_atom(res["N"], res["CA"], res["C"], LEN["C_N"], ANG["CA_C_N"], psis_deg[i])
        CAnext = place_atom(res["CA"], res["C"], Nnext, LEN["N_CA"], ANG["C_N_CA"], omega_deg)
        Cnext = place_atom(res["C"], Nnext, CAnext, LEN["CA_C"], ANG["N_CA_C"], phis_deg[i + 1])
        residues.append(
            {
                "code": sequence[i + 1],
                "N": Nnext,
                "CA": CAnext,
                "C": Cnext,
                "phi": phis_deg[i + 1],
                "psi": psis_deg[i + 1],
            }
        )

    ca = np.stack([r["CA"] for r in residues], axis=0)
    center = ca.mean(axis=0)
    atoms = []
    bonds = []
    for i, r in enumerate(residues):
        for key in ("N", "CA", "C"):
            r[key] = (r[key] - center).tolist()
        base = len(atoms)
        atoms.append({"element": "N", "x": r["N"][0], "y": r["N"][1], "z": r["N"][2], "residue": i})
        atoms.append(
            {
                "element": "CA",
                "x": r["CA"][0],
                "y": r["CA"][1],
                "z": r["CA"][2],
                "residue": i,
                "code": r["code"],
            }
        )
        atoms.append({"element": "C", "x": r["C"][0], "y": r["C"][1], "z": r["C"][2], "residue": i})
        bonds.extend([[base, base + 1], [base + 1, base + 2]])
    for i in range(n - 1):
        bonds.append([i * 3 + 2, (i + 1) * 3])

    return {
        "sequence": sequence,
        "phis": list(map(float, phis_deg)),
        "psis": list(map(float, psis_deg)),
        "residues": [
            {"code": r["code"], "phi": r["phi"], "psi": r["psi"], "CA": r["CA"], "N": r["N"], "C": r["C"]}
            for r in residues
        ],
        "atoms": atoms,
        "bonds": bonds,
    }
