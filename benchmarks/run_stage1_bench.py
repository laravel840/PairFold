#!/usr/bin/env python3
"""Quick Stage-1 upgrade benchmark (ESM contacts + early fold)."""

from __future__ import annotations

import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pairfold.clash_assembly import dihedrals_to_ca  # noqa: E402
from pairfold.predict import FragmentPredictor  # noqa: E402

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

PDB_IDS = ["1A8O", "1CRN", "2GB1", "1UBQ", "3GB1"]
CACHE = BENCH_DIR / "pdbs"
OUT = BENCH_DIR / "results" / "benchmark_results_stage1.csv"
SCOREBOARD = BENCH_DIR / "results" / "stage1_scoreboard.csv"


def kabsch_rmsd(pred: np.ndarray, native: np.ndarray) -> float:
    P = np.asarray(pred, dtype=np.float64)
    Q = np.asarray(native, dtype=np.float64)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T
    return float(np.sqrt(np.mean(np.sum(((Pc @ R) - Qc) ** 2, axis=1))))


def load_native(pdb_id: str):
    path = CACHE / f"{pdb_id.lower()}.pdb"
    parser = PDBParser(QUIET=True)
    model = next(parser.get_structure(pdb_id, str(path)).get_models())
    peptides = list(PPBuilder().build_peptides(model))
    pp = max(peptides, key=lambda x: len(x))
    seq = str(pp.get_sequence())
    ca = np.asarray([r["CA"].get_coord() for r in pp], dtype=np.float64)
    return seq, ca


def main() -> None:
    print("Loading FragmentPredictor…", flush=True)
    pred = FragmentPredictor()
    print(f"Device: {pred.dev}", flush=True)
    rows = []
    for pdb_id in PDB_IDS:
        print(f"=== {pdb_id} ===", flush=True)
        seq, native = load_native(pdb_id)
        t0 = time.perf_counter()
        result = pred.predict_sequence(seq)
        elapsed = time.perf_counter() - t0
        ca = dihedrals_to_ca(result["phis"], result["psis"])
        m = min(len(ca), len(native))
        rmsd = kabsch_rmsd(ca[:m], native[:m])
        contacts = result.get("contacts") or {}
        row = {
            "pdb_id": pdb_id,
            "length": m,
            "time_s": elapsed,
            "rmsd": rmsd,
            "mode": result.get("mode", ""),
            "n_anchors": int(contacts.get("n_anchors") or 0),
            "contact_mean_score": float(contacts.get("mean_score") or 0.0),
            "source": contacts.get("source", ""),
        }
        rows.append(row)
        print(
            f"  t={elapsed:.1f}s RMSD={rmsd:.3f} A anchors={row['n_anchors']} "
            f"src={row['source']} mode={row['mode']}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mean_r = float(np.mean([r["rmsd"] for r in rows]))
    mean_t = float(np.mean([r["time_s"] for r in rows]))
    print(f"\nMEAN RMSD={mean_r:.3f} A | MEAN time={mean_t:.1f}s", flush=True)
    print(f"Wrote {OUT}", flush=True)

    from pairfold.config import ESM_MODEL_NAME

    with SCOREBOARD.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "E3_soft_map",
                f"{ESM_MODEL_NAME}+soft_contact_fold",
                f"{mean_r:.3f}",
                f"{mean_t:.2f}",
                *[f"{r['rmsd']:.3f}" for r in rows],
                time.strftime("%Y-%m-%d"),
            ]
        )


if __name__ == "__main__":
    main()
