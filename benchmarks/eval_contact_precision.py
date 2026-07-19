#!/usr/bin/env python3
"""Compare ESM contact precision @ top-k vs native Cα contacts."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pairfold.contact_predict import ContactPredictor  # noqa: E402

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

PDB_IDS = ["1A8O", "1CRN", "2GB1", "1UBQ", "3GB1"]
THRESH = 8.0
MIN_SEP = 6
PDB_DIR = BENCH_DIR / "pdbs"


def load_native(pdb_id: str):
    path = PDB_DIR / f"{pdb_id.lower()}.pdb"
    parser = PDBParser(QUIET=True)
    model = next(parser.get_structure(pdb_id, str(path)).get_models())
    pp = max(PPBuilder().build_peptides(model), key=lambda x: len(x))
    seq = str(pp.get_sequence())
    ca = np.asarray([r["CA"].get_coord() for r in pp], dtype=np.float64)
    return seq, ca


def native_contacts(ca: np.ndarray):
    n = len(ca)
    true = set()
    for i in range(n):
        for j in range(i + MIN_SEP, n):
            if np.linalg.norm(ca[j] - ca[i]) < THRESH:
                true.add((i, j))
    return true


def main():
    cp = ContactPredictor()
    print(f"source={cp.source} ckpt={cp.ckpt_path}")
    for pdb_id in PDB_IDS:
        seq, ca = load_native(pdb_id)
        true = native_contacts(ca)
        info = cp.top_anchors(seq)
        pred = [(c["i"], c["j"], c["score"]) for c in info["contacts"][:40]]
        for k in (5, 10, 20, 40):
            top = pred[:k]
            if not top:
                print(f"{pdb_id} k={k}: no preds")
                continue
            hit = sum(1 for i, j, _ in top if (i, j) in true or (j, i) in true)
            print(
                f"{pdb_id} L={len(seq)} k={k}: prec={hit/len(top):.2f} "
                f"hits={hit}/{len(top)} native_contacts={len(true)} "
                f"mean_score={np.mean([s for _,_,s in top]):.3f}"
            )
        print()


if __name__ == "__main__":
    main()
