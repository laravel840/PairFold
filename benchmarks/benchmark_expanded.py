#!/usr/bin/env python3
"""
Expanded PairFold paper benchmark:
  - 100 diverse short domains (all-α / all-β / α/β)
  - length–RMSD correlation CSV
  - optional long-chain panel (10k–50k tiled)

Run:
  python -u benchmarks/benchmark_expanded.py
  python -u benchmarks/benchmark_expanded.py --long-only
  python -u benchmarks/benchmark_expanded.py --domains-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from Bio.PDB.PDBExceptions import PDBConstructionWarning  # noqa: E402

from pairfold.clash_assembly import dihedrals_to_ca  # noqa: E402
from pairfold.predict import FragmentPredictor  # noqa: E402

from benchmark import (  # noqa: E402
    CACHE_DIR,
    download_pdb,
    extract_native_ca_and_sequence,
    kabsch_rmsd,
)

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

SET_PATH = BENCH_DIR / "sets" / "domains_100.json"
OUT_DOMAINS = BENCH_DIR / "results" / "benchmark_domains100.csv"
OUT_LONG = BENCH_DIR / "results" / "benchmark_long100.csv"
OUT_SUMMARY = BENCH_DIR / "results" / "benchmark_expanded_summary.json"
MIN_DOM_LEN = 20
MAX_DOM_LEN = 180


def load_entries() -> List[Dict]:
    data = json.loads(SET_PATH.read_text(encoding="utf-8"))
    return list(data["entries"])


def _flush_domains_csv(rows: List[Dict]) -> None:
    OUT_DOMAINS.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with OUT_DOMAINS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_domains(
    pred: FragmentPredictor, limit: Optional[int] = None, resume: bool = True
) -> List[Dict]:
    entries = load_entries()
    if limit:
        entries = entries[:limit]
    rows: List[Dict] = []
    done = set()
    if resume and OUT_DOMAINS.exists():
        with OUT_DOMAINS.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "pdb_id": r["pdb_id"],
                        "class": r["class"],
                        "length": int(r["length"]),
                        "time_s": float(r["time_s"]),
                        "rmsd": float(r["rmsd"]),
                        "mode": r.get("mode", ""),
                        "n_anchors": int(float(r.get("n_anchors") or 0)),
                        "contact_mean_score": float(r.get("contact_mean_score") or 0.0),
                        "source": r.get("source", ""),
                    }
                )
                done.add(r["pdb_id"].upper())
        if done:
            print(f"Resuming Domains100 with {len(done)} cached rows", flush=True)

    for i, ent in enumerate(entries, 1):
        pdb_id = ent["pdb"].upper()
        cls = ent["class"]
        if pdb_id in done:
            print(f"[{i}/{len(entries)}] {pdb_id} (cached)", flush=True)
            continue
        print(f"[{i}/{len(entries)}] {pdb_id} ({cls})", flush=True)
        try:
            path = download_pdb(pdb_id, CACHE_DIR)
            seq, native = extract_native_ca_and_sequence(path)
        except Exception as e:
            print(f"  SKIP download/parse: {type(e).__name__}: {e}", flush=True)
            continue
        n = len(seq)
        if n < MIN_DOM_LEN or n > MAX_DOM_LEN:
            print(f"  SKIP length={n}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            result = pred.predict_sequence(seq)
        except Exception as e:
            print(f"  SKIP predict: {type(e).__name__}: {e}", flush=True)
            continue
        elapsed = time.perf_counter() - t0
        ca = dihedrals_to_ca(result["phis"], result["psis"])
        m = min(len(ca), len(native))
        rmsd = kabsch_rmsd(ca[:m], native[:m])
        contacts = result.get("contacts") or {}
        row = {
            "pdb_id": pdb_id,
            "class": cls,
            "length": m,
            "time_s": round(elapsed, 3),
            "rmsd": round(rmsd, 4),
            "mode": result.get("mode", ""),
            "n_anchors": int(contacts.get("n_anchors") or 0),
            "contact_mean_score": float(contacts.get("mean_score") or 0.0),
            "source": contacts.get("source", ""),
        }
        rows.append(row)
        done.add(pdb_id)
        _flush_domains_csv(rows)
        print(f"  t={elapsed:.1f}s RMSD={rmsd:.3f}Å", flush=True)

    if rows:
        print(f"Wrote {OUT_DOMAINS} ({len(rows)} rows)", flush=True)
    return rows


def tile_to_length(seq: str, native_ca: np.ndarray, target: int) -> Tuple[str, np.ndarray, int]:
    L = len(seq)
    n_tiles = int(np.ceil(target / L))
    return seq * n_tiles, np.vstack([native_ca for _ in range(n_tiles)]), n_tiles


def run_long100(pred: FragmentPredictor, n_cases: int = 100) -> List[Dict]:
    """100 long sequences with lengths spaced from 10k to 50k."""
    seeds = load_entries()
    # Prefer seeds that already downloaded successfully
    usable = []
    for ent in seeds:
        try:
            path = download_pdb(ent["pdb"], CACHE_DIR)
            seq, native = extract_native_ca_and_sequence(path)
            if MIN_DOM_LEN <= len(seq) <= MAX_DOM_LEN:
                usable.append((ent["pdb"].upper(), ent["class"], seq, native))
        except Exception:
            continue
    if not usable:
        raise SystemExit("No usable seed domains for long bench")

    lengths = np.linspace(10_000, 50_000, n_cases, dtype=int)
    rows: List[Dict] = []
    for i, target in enumerate(lengths):
        pdb_id, cls, seq, native = usable[i % len(usable)]
        tiled_seq, tiled_native, n_tiles = tile_to_length(seq, native, int(target))
        # Trim to exact target
        tiled_seq = tiled_seq[: int(target)]
        tiled_native = tiled_native[: int(target)]
        # For tile RMSD use full repeats only
        L = len(seq)
        n_full = len(tiled_seq) // L
        print(
            f"[long {i+1}/{n_cases}] {pdb_id} → len={len(tiled_seq)} tiles≈{n_full}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            result = pred.predict_sequence(tiled_seq)
        except Exception as e:
            print(f"  SKIP: {type(e).__name__}: {e}", flush=True)
            continue
        elapsed = time.perf_counter() - t0
        ca = dihedrals_to_ca(result["phis"], result["psis"])
        tile_rms = []
        for t in range(n_full):
            a, b = t * L, (t + 1) * L
            if b > len(ca):
                break
            tile_rms.append(kabsch_rmsd(ca[a:b], native))
        if not tile_rms:
            continue
        row = {
            "case": f"L{len(tiled_seq)}_{pdb_id}",
            "seed": pdb_id,
            "class": cls,
            "length": len(tiled_seq),
            "n_tiles": n_full,
            "time_s": round(elapsed, 3),
            "mean_tile_rmsd": round(float(np.mean(tile_rms)), 4),
            "median_tile_rmsd": round(float(np.median(tile_rms)), 4),
            "best_tile_rmsd": round(float(np.min(tile_rms)), 4),
            "worst_tile_rmsd": round(float(np.max(tile_rms)), 4),
            "mode": result.get("mode", ""),
        }
        rows.append(row)
        _flush_long_csv(rows)
        print(
            f"  t={elapsed:.1f}s mean_tile_RMSD={row['mean_tile_rmsd']:.3f}Å",
            flush=True,
        )

    if rows:
        print(f"Wrote {OUT_LONG} ({len(rows)} rows)", flush=True)
    return rows


def _flush_long_csv(rows: List[Dict]) -> None:
    if not rows:
        return
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LONG.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(domain_rows: List[Dict], long_rows: List[Dict]) -> Dict:
    summary: Dict = {"n_domains": len(domain_rows), "n_long": len(long_rows)}
    if domain_rows:
        rmsds = [r["rmsd"] for r in domain_rows]
        times = [r["time_s"] for r in domain_rows]
        summary["domains"] = {
            "mean_rmsd": float(np.mean(rmsds)),
            "median_rmsd": float(np.median(rmsds)),
            "std_rmsd": float(np.std(rmsds)),
            "mean_time_s": float(np.mean(times)),
            "by_class": {},
        }
        for cls in ("all_alpha", "all_beta", "alpha_beta"):
            sub = [r for r in domain_rows if r["class"] == cls]
            if sub:
                summary["domains"]["by_class"][cls] = {
                    "n": len(sub),
                    "mean_rmsd": float(np.mean([r["rmsd"] for r in sub])),
                    "mean_time_s": float(np.mean([r["time_s"] for r in sub])),
                }
        # Pearson length vs RMSD
        lens = np.array([r["length"] for r in domain_rows], dtype=float)
        rarr = np.array(rmsds, dtype=float)
        if len(lens) > 2 and np.std(lens) > 0:
            corr = float(np.corrcoef(lens, rarr)[0, 1])
            summary["domains"]["length_rmsd_pearson"] = corr
    if long_rows:
        summary["long"] = {
            "mean_time_s": float(np.mean([r["time_s"] for r in long_rows])),
            "mean_tile_rmsd": float(np.mean([r["mean_tile_rmsd"] for r in long_rows])),
            "min_length": int(min(r["length"] for r in long_rows)),
            "max_length": int(max(r["length"] for r in long_rows)),
        }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains-only", action="store_true")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="limit domain count (debug)")
    ap.add_argument("--long-cases", type=int, default=100)
    args = ap.parse_args()

    # Keep Stage-2/3 off for paper Cα timing/RMSD (heavy and irrelevant to Kabsch Cα)
    import pairfold.config as cfg
    import pairfold.predict as pred_mod

    # Paper panel (Domains100 / Long100): interactive local assembly path.
    # Skip all contact towers / hinge search (CONTACT_USE_MAX_LEN=0) so n≈100
    # finishes quickly. Ablation "full_pipeline" restores contacts + early fold.
    for name, val in (
        ("USE_STAGE2_SIDECHAINS", False),
        ("USE_STAGE3_ATOMS", False),
        ("USE_FOLD_HEAD", False),
        ("USE_FOLD_HEAD_SELECT", False),
        ("USE_ESM_ALT_CONTACTS", False),
        ("USE_SOFT_CONTACT_FOLD", False),
        ("EARLY_CONTACT_FOLD", False),
        ("USE_ESM_CONTACTS", False),
        ("CONTACT_USE_MAX_LEN", 0),
        # Domains100 / Long100: skip combinatorial look-ahead (can stall for minutes
        # on a few folds). Consensus + SS freeze stays on; ablation covers look-ahead.
        ("LEVER_ASSEMBLY_MAX_LEN", 0),
        ("USE_LEVER_POLISH", False),
        ("SS_PIPELINE_MAX_LEN", 256),
    ):
        setattr(cfg, name, val)
        if hasattr(pred_mod, name):
            setattr(pred_mod, name, val)

    print("Loading FragmentPredictor…", flush=True)
    pred = FragmentPredictor()
    print(f"Device: {pred.dev}", flush=True)

    domain_rows: List[Dict] = []
    long_rows: List[Dict] = []
    if not args.long_only:
        domain_rows = run_domains(pred, limit=args.limit or None)
    if not args.domains_only:
        long_rows = run_long100(pred, n_cases=args.long_cases)
    summarize(domain_rows, long_rows)


if __name__ == "__main__":
    main()
