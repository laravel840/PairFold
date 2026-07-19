#!/usr/bin/env python3
"""Test soft-map AFTER full pipeline angles on 1CRN."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

from pairfold.clash_assembly import dihedrals_to_ca
from pairfold.contact_predict import ContactPredictor
from pairfold.predict import FragmentPredictor
from pairfold.soft_contact_fold import soft_map_scaffold


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T
    return float(np.sqrt(np.mean(np.sum((Pc @ R - Qc) ** 2, 1))))


def main():
    pred = FragmentPredictor()
    cp = ContactPredictor(device=pred.dev)
    path = ROOT / "benchmarks" / "pdbs" / "1crn.pdb"
    model = next(PDBParser(QUIET=True).get_structure("x", str(path)).get_models())
    pp = max(PPBuilder().build_peptides(model), key=lambda x: len(x))
    seq = str(pp.get_sequence())
    native = np.array([r["CA"].get_coord() for r in pp], dtype=np.float64)

    r = pred.predict_sequence(seq)
    ph, ps = r["phis"], r["psis"]
    base = kabsch(dihedrals_to_ca(ph, ps), native)
    print("pipeline", base)

    probs = cp.contact_probs(seq, model="primary")
    best = None
    for seed in (5, 19, 37, 51, 73, 91):
        soft = soft_map_scaffold(
            seq, ph, ps, probs, n_steps=550, fit_steps=1800, seed=seed, contact_thresh=0.10
        )
        rms = kabsch(dihedrals_to_ca(soft["phis_deg"], soft["psis_deg"]), native)
        print(
            f"seed{seed} improved={soft['improved']} "
            f"rank {soft['base_rank']:.3f}->{soft['best_rank']:.3f} "
            f"sel {soft.get('select_score', float('nan')):.3f} rmsd={rms:.3f}"
        )
        if soft["improved"] and (best is None or soft["select_score"] > best[0]):
            best = (soft["select_score"], rms, seed)
    print("selected", best)


if __name__ == "__main__":
    main()
