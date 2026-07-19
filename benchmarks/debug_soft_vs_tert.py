#!/usr/bin/env python3
"""Compare soft-map RMSD before/after tertiary on 1CRN (soft disabled in pipeline)."""
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

import pairfold.config as cfg
from pairfold.clash_assembly import dihedrals_to_ca
from pairfold.contact_predict import ContactPredictor
from pairfold.predict import FragmentPredictor
from pairfold.soft_contact_fold import soft_map_scaffold
from pairfold.tertiary import run_tertiary_pipeline

cfg.USE_SOFT_CONTACT_FOLD = False


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
    print("no_soft", kabsch(dihedrals_to_ca(ph, ps), native))

    maps = [("t12", cp.contact_probs(seq, model="primary")), ("t30", cp.contact_probs(seq, model="alt"))]
    best = None
    for name, pmap in maps:
        for seed in (5, 19, 37, 51):
            soft = soft_map_scaffold(
                seq, ph, ps, pmap, n_steps=550, fit_steps=1800, seed=seed, contact_thresh=0.10
            )
            if not soft["improved"]:
                continue
            rms = kabsch(dihedrals_to_ca(soft["phis_deg"], soft["psis_deg"]), native)
            up = run_tertiary_pipeline(seq, soft["phis_deg"], soft["psis_deg"], anchors=None)
            rms2 = kabsch(dihedrals_to_ca(up["phis"], up["psis"]), native)
            print(
                f"{name}@{seed} soft={rms:.3f} after_tert={rms2:.3f} "
                f"tert_imp={up['tertiary']['improved']} sel={soft['select_score']:.3f}"
            )
            if best is None or soft["select_score"] > best[0]:
                best = (soft["select_score"], rms, rms2, f"{name}@{seed}")
    print("pick", best)


if __name__ == "__main__":
    main()
