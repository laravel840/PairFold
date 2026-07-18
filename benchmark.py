#!/usr/bin/env python3
"""
PairFold automated structure benchmark.

Uses the project's FragmentPredictor.predict_sequence() to predict backbone
torsions, builds Cα coordinates via clash_assembly.dihedrals_to_ca(), then
reports Kabsch-aligned Cα RMSD vs experimental structures from the RCSB PDB.

Run from anywhere:
  python benchmark.py
"""

from __future__ import annotations

import csv
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Bio.Data.IUPACData import protein_letters_3to1  # noqa: E402
from Bio.PDB import PDBList, PDBParser, PPBuilder  # noqa: E402
from Bio.PDB.PDBExceptions import PDBConstructionWarning  # noqa: E402

from pairfold.clash_assembly import dihedrals_to_ca  # noqa: E402
from pairfold.predict import FragmentPredictor  # noqa: E402

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# Standard diverse small domains (all within PairFold tertiary / export limits)
PDB_IDS = ["1A8O", "1CRN", "2GB1", "1UBQ", "3GB1"]

OUT_CSV = ROOT / "benchmark_results.csv"
CACHE_DIR = ROOT / "benchmark_pdbs"


def three_to_one(resname: str) -> Optional[str]:
    key = resname.strip().capitalize()
    # BioPython map is title-case 3-letter → 1-letter
    if key in protein_letters_3to1:
        return protein_letters_3to1[key]
    upper = resname.strip().upper()
    # common MSE → M
    if upper == "MSE":
        return "M"
    return None


def download_pdb(pdb_id: str, cache_dir: Path) -> Path:
    """Download PDB coordinates from RCSB (requests first, Bio.PDB fallback)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.upper().strip()
    out = cache_dir / f"{pdb_id.lower()}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out

    # Direct RCSB download (most reliable)
    import requests

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        resp = requests.get(url, timeout=120)
        if resp.ok and len(resp.content) > 100:
            out.write_bytes(resp.content)
            return out
    except Exception:
        pass

    # Bio.PDB fallback
    pdbl = PDBList(verbose=False)
    path = Path(
        pdbl.retrieve_pdb_file(
            pdb_id,
            pdir=str(cache_dir),
            file_format="pdb",
            overwrite=True,
        )
    )
    if path.exists() and path.stat().st_size > 0:
        if path.resolve() != out.resolve():
            out.write_bytes(path.read_bytes())
        return out

    raise FileNotFoundError(f"Failed to download PDB {pdb_id} from RCSB")


def extract_native_ca_and_sequence(pdb_path: Path) -> Tuple[str, np.ndarray]:
    """
    Take the longest continuous peptide from the first model.
    Returns (sequence, CA coords shape (L, 3)).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())

    best_seq = ""
    best_ca: Optional[np.ndarray] = None

    # Prefer Bio.PDB peptide builder (skips het/water, handles gaps as separate peptides)
    ppb = PPBuilder()
    for pp in ppb.build_peptides(model):
        seq = str(pp.get_sequence())
        ca_list = []
        ok = True
        for res in pp:
            if "CA" not in res:
                ok = False
                break
            ca_list.append(res["CA"].get_coord())
        if not ok or not ca_list:
            continue
        if len(seq) != len(ca_list):
            continue
        if len(seq) > len(best_seq):
            best_seq = seq
            best_ca = np.asarray(ca_list, dtype=np.float64)

    # Fallback: walk first chain residues with CA
    if best_ca is None or len(best_seq) < 2:
        for chain in model:
            seq_chars: List[str] = []
            ca_list = []
            for res in chain:
                if res.id[0] != " ":
                    continue
                aa = three_to_one(res.get_resname())
                if aa is None or "CA" not in res:
                    # break continuous stretch on nonstandard
                    if len(seq_chars) > len(best_seq):
                        best_seq = "".join(seq_chars)
                        best_ca = np.asarray(ca_list, dtype=np.float64)
                    seq_chars, ca_list = [], []
                    continue
                seq_chars.append(aa)
                ca_list.append(res["CA"].get_coord())
            if len(seq_chars) > len(best_seq):
                best_seq = "".join(seq_chars)
                best_ca = np.asarray(ca_list, dtype=np.float64)
            break

    if best_ca is None or len(best_seq) < 2:
        raise ValueError(f"No usable peptide found in {pdb_path}")

    # Keep only standard 20 AA (PairFold vocabulary)
    from pairfold.config import AA_LIST

    aa_set = set(AA_LIST)
    if any(c not in aa_set for c in best_seq):
        # drop nonstandard positions if any slipped through
        keep = [i for i, c in enumerate(best_seq) if c in aa_set]
        if len(keep) < 2:
            raise ValueError(f"Sequence has too few standard residues in {pdb_path}")
        # only keep if contiguous block
        if keep[-1] - keep[0] + 1 == len(keep):
            best_seq = "".join(best_seq[i] for i in keep)
            best_ca = best_ca[keep]
        else:
            # take longest contiguous standard stretch
            raise ValueError(f"Non-contiguous nonstandard residues in {pdb_path.name}")

    return best_seq, best_ca


def kabsch_rmsd(pred: np.ndarray, native: np.ndarray) -> float:
    """Kabsch-aligned Cα RMSD (Å)."""
    P = np.asarray(pred, dtype=np.float64)
    Q = np.asarray(native, dtype=np.float64)
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch pred {P.shape} vs native {Q.shape}")
    if P.shape[0] < 2:
        return float("nan")

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T
    P_aligned = Pc @ R
    diff = P_aligned - Qc
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def predicted_ca_from_result(result: Dict) -> np.ndarray:
    """
    Prefer Cα rebuilt from predicted φ/ψ (always present).
    Fall back to exported structure residues if needed.
    """
    phis = result.get("phis")
    psis = result.get("psis")
    if phis is not None and psis is not None and len(phis) == len(psis) and len(phis) >= 2:
        return dihedrals_to_ca(phis, psis)

    structure = result.get("structure") or {}
    residues = structure.get("residues") or []
    if residues:
        coords = []
        for r in residues:
            ca = r.get("CA")
            if ca is None:
                raise ValueError("Structure residue missing CA")
            if isinstance(ca, dict):
                coords.append([ca["x"], ca["y"], ca["z"]])
            else:
                coords.append(list(ca))
        return np.asarray(coords, dtype=np.float64)

    raise ValueError("Prediction result has neither phis/psis nor structure CA")


def markdown_table(rows: List[Dict]) -> str:
    headers = ["PDB ID", "Length", "Time (s)", "CA RMSD (A)", "Anchors"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        rmsd = r["rmsd"]
        rmsd_s = f"{rmsd:.3f}" if rmsd == rmsd else "n/a"
        lines.append(
            f"| {r['pdb_id']} | {r['length']} | {r['time_s']:.3f} | {rmsd_s} | {r.get('n_anchors', 0)} |"
        )
    if rows:
        mean_t = float(np.mean([r["time_s"] for r in rows]))
        rmsds = [r["rmsd"] for r in rows if r["rmsd"] == r["rmsd"]]
        mean_r = float(np.mean(rmsds)) if rmsds else float("nan")
        mean_rs = f"{mean_r:.3f}" if mean_r == mean_r else "n/a"
        lines.append(f"| **mean** |  | **{mean_t:.3f}** | **{mean_rs}** |  |")
    return "\n".join(lines)


def run_benchmark() -> List[Dict]:
    print("Loading PairFold FragmentPredictor (GPU if available)…")
    predictor = FragmentPredictor()
    print(f"Device: {predictor.dev}")
    print(f"Checkpoint: {predictor.ckpt_path}")
    print()

    rows: List[Dict] = []
    for pdb_id in PDB_IDS:
        print(f"=== {pdb_id} ===")
        try:
            pdb_path = download_pdb(pdb_id, CACHE_DIR)
            print(f"  PDB file: {pdb_path.name}")
            sequence, native_ca = extract_native_ca_and_sequence(pdb_path)
            n = len(sequence)
            print(f"  Sequence length: {n}")
            print(f"  Sequence: {sequence[:40]}{'…' if n > 40 else ''}")

            t0 = time.perf_counter()
            result = predictor.predict_sequence(sequence)
            elapsed = time.perf_counter() - t0

            pred_ca = predicted_ca_from_result(result)
            if pred_ca.shape[0] != native_ca.shape[0]:
                # Align to min length (should not happen if sequence matches)
                m = min(pred_ca.shape[0], native_ca.shape[0])
                print(f"  WARNING: length mismatch pred={pred_ca.shape[0]} native={native_ca.shape[0]}; using first {m}")
                pred_ca = pred_ca[:m]
                native_ca = native_ca[:m]

            rmsd = kabsch_rmsd(pred_ca, native_ca)
            contacts = result.get("contacts") or {}
            row = {
                "pdb_id": pdb_id,
                "length": int(native_ca.shape[0]),
                "time_s": float(elapsed),
                "rmsd": float(rmsd),
                "mode": result.get("mode", ""),
                "n_anchors": int(contacts.get("n_anchors") or 0),
                "contact_mean_score": float(contacts.get("mean_score") or 0.0),
            }
            rows.append(row)
            print(
                f"  Time: {elapsed:.3f}s | CA RMSD: {rmsd:.3f} A | mode={row['mode']}"
                f" | anchors={row['n_anchors']}"
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "length": 0,
                    "time_s": float("nan"),
                    "rmsd": float("nan"),
                    "mode": "error",
                    "n_anchors": 0,
                    "contact_mean_score": 0.0,
                }
            )
        print()

    return rows


def save_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdb_id",
                "length",
                "time_s",
                "rmsd",
                "mode",
                "n_anchors",
                "contact_mean_score",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "pdb_id": r["pdb_id"],
                    "length": r["length"],
                    "time_s": "" if r["time_s"] != r["time_s"] else f"{r['time_s']:.6f}",
                    "rmsd": "" if r["rmsd"] != r["rmsd"] else f"{r['rmsd']:.6f}",
                    "mode": r.get("mode", ""),
                    "n_anchors": r.get("n_anchors", 0),
                    "contact_mean_score": f"{float(r.get('contact_mean_score', 0.0)):.4f}",
                }
            )


def main() -> None:
    print("PairFold structure benchmark")
    print("Predictor: ml.predict.FragmentPredictor.predict_sequence")
    print("CA builder: ml.clash_assembly.dihedrals_to_ca")
    print(f"Targets: {', '.join(PDB_IDS)}")
    print()

    rows = run_benchmark()
    save_csv(rows, OUT_CSV)

    print("Results")
    print(markdown_table(rows))
    print()
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
