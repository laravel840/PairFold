#!/usr/bin/env python3
"""Smoke-test Stage-2/3 all-atom packing on benchmark PDBs."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

from pairfold.config import EXPORT_DIR, STAGE23_MAX_LEN
from pairfold.predict import FragmentPredictor
from pairfold.stage3_atoms import write_pdb

PDB_IDS = ["1CRN", "1A8O", "2GB1"]


def main() -> None:
    pred = FragmentPredictor()
    out_dir = EXPORT_DIR / "all_atom"
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdb_id in PDB_IDS:
        path = ROOT / "benchmarks" / "pdbs" / f"{pdb_id.lower()}.pdb"
        model = next(PDBParser(QUIET=True).get_structure(pdb_id, str(path)).get_models())
        pp = max(PPBuilder().build_peptides(model), key=lambda x: len(x))
        seq = str(pp.get_sequence())
        print(f"=== {pdb_id} len={len(seq)} ===", flush=True)
        if len(seq) > STAGE23_MAX_LEN:
            print("  skip: over STAGE23_MAX_LEN", flush=True)
            continue
        r = pred.predict_sequence(seq)
        st = r.get("structure") or {}
        n_atoms = len(st.get("atoms") or [])
        s2 = st.get("stage2") or {}
        print(
            f"  atoms={n_atoms} sidechain={s2.get('n_sidechain_atoms', '?')} "
            f"clash={s2.get('clash_energy', '?')} enabled={st.get('enabled', st.get('stage2') is not None)}",
            flush=True,
        )
        print(f"  note: {(st.get('note') or r.get('note') or '')[:180]}", flush=True)
        pdb_out = write_pdb(out_dir / f"{pdb_id.lower()}_pairfold.pdb", st)
        print(f"  wrote {pdb_out}", flush=True)


if __name__ == "__main__":
    main()
