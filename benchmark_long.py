#!/usr/bin/env python3
"""
Long-chain PairFold benchmark (sequences ≥ 10,000 aa).

Experimental PDB almost never contains continuous modeled chains >10k residues,
so each case tiles a real short domain to length ≥ MIN_LEN and scores accuracy as
mean Kabsch Cα RMSD of each tile against that domain's experimental structure.

Run:
  python benchmark_long.py
"""

from __future__ import annotations

import csv
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Bio.PDB.PDBExceptions import PDBConstructionWarning  # noqa: E402

from pairfold.clash_assembly import dihedrals_to_ca  # noqa: E402
from pairfold.predict import FragmentPredictor  # noqa: E402

# Reuse download / extract / kabsch from the short benchmark
from benchmark import (  # noqa: E402
    CACHE_DIR,
    download_pdb,
    extract_native_ca_and_sequence,
    kabsch_rmsd,
)

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

MIN_LEN = 10_000
# Ten distinct seed domains (diverse folds / lengths)
SEED_PDBS = [
    "1A8O",  # HIV capsid helix bundle
    "1CRN",  # crambin
    "2GB1",  # protein G
    "1UBQ",  # ubiquitin
    "3GB1",  # protein G variant
    "1PGB",  # protein G B1
    "1VII",  # villin headpiece
    "1L2Y",  # trp-cage
    "1BZ4",  # SH3
    "2JVD",  # WW domain
]

OUT_CSV = ROOT / "benchmark_long_results.csv"


def tile_to_length(seq: str, native_ca: np.ndarray, min_len: int) -> Tuple[str, np.ndarray, int]:
    """Repeat domain until length ≥ min_len. Returns (seq, tiled_native_ca, n_tiles)."""
    L = len(seq)
    if L < 2:
        raise ValueError("seed too short")
    n_tiles = int(np.ceil(min_len / L))
    tiled_seq = seq * n_tiles
    # Native reference: stack the same experimental CA for each tile
    tiled_ca = np.vstack([native_ca for _ in range(n_tiles)])
    return tiled_seq, tiled_ca, n_tiles


def tile_rmsds(pred_ca: np.ndarray, native_tile: np.ndarray, n_tiles: int) -> List[float]:
    L = native_tile.shape[0]
    out: List[float] = []
    for t in range(n_tiles):
        a = t * L
        b = a + L
        out.append(kabsch_rmsd(pred_ca[a:b], native_tile))
    return out


def markdown_table(rows: List[Dict]) -> str:
    headers = [
        "Case",
        "Seed",
        "Length",
        "Tiles",
        "Time (s)",
        "Mean tile RMSD (A)",
        "Median",
        "Best",
        "Worst",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append(
            "| {case} | {seed} | {length} | {n_tiles} | {time_s:.2f} | "
            "{mean_rmsd:.3f} | {median_rmsd:.3f} | {best_rmsd:.3f} | {worst_rmsd:.3f} |".format(
                **r
            )
        )
    if rows:
        mean_t = float(np.mean([r["time_s"] for r in rows]))
        mean_r = float(np.mean([r["mean_rmsd"] for r in rows]))
        lines.append(
            f"| **mean** |  |  |  | **{mean_t:.2f}** | **{mean_r:.3f}** |  |  |  |"
        )
    return "\n".join(lines)


def run() -> List[Dict]:
    print("Loading PairFold FragmentPredictor…")
    predictor = FragmentPredictor()
    print(f"Device: {predictor.dev}")
    print(f"Min sequence length: {MIN_LEN}")
    print(f"Cases: {len(SEED_PDBS)}")
    print()

    rows: List[Dict] = []
    for i, pdb_id in enumerate(SEED_PDBS, start=1):
        case = f"L{i:02d}"
        print(f"=== {case} · seed {pdb_id} ===")
        try:
            path = download_pdb(pdb_id, CACHE_DIR)
            seq0, ca0 = extract_native_ca_and_sequence(path)
            tiled_seq, tiled_native, n_tiles = tile_to_length(seq0, ca0, MIN_LEN)
            print(
                f"  Seed length {len(seq0)} → tiled {len(tiled_seq)} aa "
                f"({n_tiles}×{len(seq0)})"
            )

            t0 = time.perf_counter()
            result = predictor.predict_sequence(tiled_seq)
            elapsed = time.perf_counter() - t0

            phis = result["phis"]
            psis = result["psis"]
            pred_ca = dihedrals_to_ca(phis, psis)
            if pred_ca.shape[0] != len(tiled_seq):
                m = min(pred_ca.shape[0], len(tiled_seq))
                pred_ca = pred_ca[:m]
                n_tiles = m // len(seq0)
                tiled_native = ca0
                print(f"  WARNING: truncated to {m} residues, tiles={n_tiles}")

            rms = tile_rmsds(pred_ca, ca0, n_tiles)
            row = {
                "case": case,
                "seed": pdb_id,
                "length": int(len(tiled_seq)),
                "n_tiles": int(n_tiles),
                "seed_len": int(len(seq0)),
                "time_s": float(elapsed),
                "mean_rmsd": float(np.mean(rms)),
                "median_rmsd": float(np.median(rms)),
                "best_rmsd": float(np.min(rms)),
                "worst_rmsd": float(np.max(rms)),
                "mode": result.get("mode", ""),
            }
            rows.append(row)
            print(
                f"  Time {elapsed:.2f}s | mean tile RMSD {row['mean_rmsd']:.3f} A "
                f"(best {row['best_rmsd']:.3f}, worst {row['worst_rmsd']:.3f}) | "
                f"mode={row['mode']}"
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            rows.append(
                {
                    "case": case,
                    "seed": pdb_id,
                    "length": 0,
                    "n_tiles": 0,
                    "seed_len": 0,
                    "time_s": float("nan"),
                    "mean_rmsd": float("nan"),
                    "median_rmsd": float("nan"),
                    "best_rmsd": float("nan"),
                    "worst_rmsd": float("nan"),
                    "mode": "error",
                }
            )
        print()

    return rows


def save_csv(rows: List[Dict], path: Path) -> None:
    fields = [
        "case",
        "seed",
        "length",
        "n_tiles",
        "seed_len",
        "time_s",
        "mean_rmsd",
        "median_rmsd",
        "best_rmsd",
        "worst_rmsd",
        "mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = dict(r)
            for k in ("time_s", "mean_rmsd", "median_rmsd", "best_rmsd", "worst_rmsd"):
                v = out[k]
                out[k] = "" if v != v else f"{v:.6f}"
            w.writerow(out)


def main() -> None:
    print("PairFold LONG-CHAIN structure benchmark")
    print(
        "Note: no continuous experimental chains >10k aa exist in the PDB archive;"
    )
    print(
        "accuracy = mean Kabsch CA RMSD of each tiled experimental domain vs prediction."
    )
    print()
    rows = run()
    save_csv(rows, OUT_CSV)
    print("Results")
    print(markdown_table(rows))
    print()
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
