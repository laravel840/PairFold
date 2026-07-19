"""Stage-3: all-atom completion (backbone O + Stage-2 sidechains) and PDB export."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .assemble import build_backbone, place_atom
from .stage2_sidechains import pack_sidechains

# PDB residue names
_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _element_from_name(name: str) -> str:
    if name.startswith("CA") or name == "C":
        return "C"
    if name.startswith("N"):
        return "N"
    if name.startswith("O"):
        return "O"
    if name.startswith("S"):
        return "S"
    if name.startswith("H"):
        return "H"
    # CB, CG, CD, CE, CZ, CH2…
    return "C"


def complete_all_atom(
    sequence: str,
    phis: Sequence[float],
    psis: Sequence[float],
    chi_angles: Optional[Sequence[Sequence[float]]] = None,
    sidechain: Optional[Dict] = None,
    add_hydrogens: bool = False,
    max_len: int = 256,
) -> Dict:
    """
    Merge backbone (N/CA/C/O) + packed sidechains into an all-atom structure.

    Hydrogens are optional (off by default for speed/payload size).
    """
    seq = "".join(c for c in sequence.upper() if c.isalpha())
    n = len(seq)
    if n == 0:
        return {"enabled": False, "skipped": True, "atoms": [], "bonds": [], "note": "empty"}
    if n > max_len:
        return {
            "enabled": False,
            "skipped": True,
            "atoms": [],
            "bonds": [],
            "note": f"Stage-3 skipped (length {n} > {max_len}).",
        }

    backbone = build_backbone(seq, phis, psis)
    residues = backbone["residues"]

    if sidechain is None:
        sidechain = pack_sidechains(seq, phis, psis, backbone=backbone, max_len=max_len)

    atoms: List[Dict] = []
    bonds: List[List[int]] = []

    # Per-residue atom index map for bonding
    res_atom_idx: List[Dict[str, int]] = [{} for _ in range(n)]

    for i, r in enumerate(residues):
        aa = seq[i]
        for key in ("N", "CA", "C"):
            xyz = r[key]
            idx = len(atoms)
            atoms.append(
                {
                    "element": key if key != "CA" else "C",
                    "name": key,
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                    "residue": i,
                    "code": aa,
                    "resname": _THREE.get(aa, "UNK"),
                }
            )
            res_atom_idx[i][key] = idx
        # Carbonyl O
        N = np.asarray(r["N"], dtype=np.float64)
        CA = np.asarray(r["CA"], dtype=np.float64)
        C = np.asarray(r["C"], dtype=np.float64)
        if i + 1 < n:
            Nnext = np.asarray(residues[i + 1]["N"], dtype=np.float64)
            O = place_atom(Nnext, CA, C, 1.231, 120.8, 0.0)
        else:
            O = place_atom(N, CA, C, 1.231, 120.8, 180.0)
        idx = len(atoms)
        atoms.append(
            {
                "element": "O",
                "name": "O",
                "x": float(O[0]),
                "y": float(O[1]),
                "z": float(O[2]),
                "residue": i,
                "code": aa,
                "resname": _THREE.get(aa, "UNK"),
            }
        )
        res_atom_idx[i]["O"] = idx

        # Backbone bonds
        bonds.append([res_atom_idx[i]["N"], res_atom_idx[i]["CA"]])
        bonds.append([res_atom_idx[i]["CA"], res_atom_idx[i]["C"]])
        bonds.append([res_atom_idx[i]["C"], res_atom_idx[i]["O"]])
        if i > 0:
            bonds.append([res_atom_idx[i - 1]["C"], res_atom_idx[i]["N"]])

        # Optional amide H
        if add_hydrogens and aa != "P" and i > 0:
            H = place_atom(
                np.asarray(residues[i - 1]["C"], dtype=np.float64),
                CA,
                N,
                1.01,
                119.0,
                180.0,
            )
            idx = len(atoms)
            atoms.append(
                {
                    "element": "H",
                    "name": "H",
                    "x": float(H[0]),
                    "y": float(H[1]),
                    "z": float(H[2]),
                    "residue": i,
                    "code": aa,
                    "resname": _THREE.get(aa, "UNK"),
                }
            )
            bonds.append([res_atom_idx[i]["N"], idx])

    # Sidechain parent map for sensible viz bonds (not chemistry-complete)
    _SC_PARENT = {
        "CB": "CA",
        "SG": "CB", "OG": "CB", "OG1": "CB", "CG": "CB", "CG1": "CB", "CG2": "CB",
        "CD": "CG", "CD1": "CG", "CD2": "CG", "SD": "CG",
        "CE": "CD", "CE1": "CD1", "CE2": "CD2", "CE3": "CD2", "NZ": "CE", "NE": "CD",
        "CZ": "CE1", "CZ2": "CE2", "CZ3": "CE3", "OH": "CZ",
        "OD1": "CG", "OD2": "CG", "ND2": "CG", "OE1": "CD", "OE2": "CD", "NE2": "CD",
        "NH1": "CZ", "NH2": "CZ", "NE1": "CD1", "CH2": "CZ2",
        "ND1": "CG",
    }

    for a in sidechain.get("atoms") or []:
        i = int(a["residue"])
        name = str(a.get("name") or a.get("element") or "X")
        idx = len(atoms)
        atoms.append(
            {
                "element": _element_from_name(name),
                "name": name,
                "x": float(a["x"]),
                "y": float(a["y"]),
                "z": float(a["z"]),
                "residue": i,
                "code": seq[i],
                "resname": _THREE.get(seq[i], "UNK"),
                "sidechain": True,
            }
        )
        res_atom_idx[i][name] = idx
        parent = _SC_PARENT.get(name)
        if parent and parent in res_atom_idx[i]:
            bonds.append([res_atom_idx[i][parent], idx])
        elif name == "CB" and "CA" in res_atom_idx[i]:
            bonds.append([res_atom_idx[i]["CA"], idx])
        elif "CB" in res_atom_idx[i]:
            bonds.append([res_atom_idx[i]["CB"], idx])

    # Ring closures for aromatics / His / Trp
    for i in range(n):
        idxmap = res_atom_idx[i]
        for a, b in (
            ("CE1", "CZ"),
            ("CE2", "CZ"),
            ("CD2", "CE2"),
            ("NE2", "CE1"),
            ("NE1", "CE2"),
            ("CZ2", "CH2"),
            ("CZ3", "CH2"),
        ):
            if a in idxmap and b in idxmap:
                bonds.append([idxmap[a], idxmap[b]])

    note = (
        f"Stage-3 all-atom: {len(atoms)} atoms"
        f" (sidechain {sidechain.get('n_sidechain_atoms', 0)})."
    )

    return {
        "enabled": True,
        "skipped": False,
        "sequence": seq,
        "phis": list(map(float, phis)),
        "psis": list(map(float, psis)),
        "residues": residues,
        "atoms": atoms,
        "bonds": bonds,
        "chi_angles": sidechain.get("chi_angles") or chi_angles or [],
        "sidechain": {
            "clash_energy": sidechain.get("clash_energy", 0.0),
            "n_sidechain_atoms": sidechain.get("n_sidechain_atoms", 0),
        },
        "n_atoms": len(atoms),
        "note": note,
    }


def write_pdb(path: Union[str, Path], structure: Dict) -> Path:
    """Write ATOM records for an all-atom structure dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "REMARK  PairFold Stage-3 all-atom export",
        f"REMARK  {structure.get('note', '')}",
    ]
    serial = 1
    for a in structure.get("atoms") or []:
        name = str(a.get("name") or a.get("element") or "X")
        # PDB atom name column: right-ish align 4 chars
        aname = f"{name:>4}"[:4]
        resname = str(a.get("resname") or _THREE.get(a.get("code", "G"), "UNK"))
        resi = int(a.get("residue", 0)) + 1
        x, y, z = float(a["x"]), float(a["y"]), float(a["z"])
        elem = str(a.get("element") or _element_from_name(name))[:2]
        lines.append(
            f"ATOM  {serial:5d} {aname} {resname:>3} A{resi:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}"
        )
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# Back-compat alias
def write_pdb_stub(path: str, structure: Dict) -> None:
    write_pdb(path, structure)
