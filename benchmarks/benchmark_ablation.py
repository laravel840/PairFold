#!/usr/bin/env python3
"""
Ablation study for paper Table (Base / +Look-ahead / +Lever / +SS / Full).

Runs on a representative subset (default 20) for tractable wall-clock.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCH_DIR))

from Bio.PDB.PDBExceptions import PDBConstructionWarning  # noqa: E402

import pairfold.config as cfg  # noqa: E402
import pairfold.predict as pred_mod  # noqa: E402
from pairfold.clash_assembly import dihedrals_to_ca  # noqa: E402
from pairfold.predict import FragmentPredictor  # noqa: E402

from benchmark import (  # noqa: E402
    CACHE_DIR,
    download_pdb,
    extract_native_ca_and_sequence,
    kabsch_rmsd,
)

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

OUT = BENCH_DIR / "results" / "benchmark_ablation.csv"
OUT_SUMMARY = BENCH_DIR / "results" / "benchmark_ablation_summary.json"

# Representative 20 across classes
ABLATION_PDBS = [
    ("1VII", "all_alpha"),
    ("1L2Y", "all_alpha"),
    ("1E0L", "all_alpha"),
    ("1FME", "all_alpha"),
    ("1PRB", "all_alpha"),
    ("1SHG", "all_beta"),
    ("1CSP", "all_beta"),
    ("1G6U", "all_beta"),
    ("1MJC", "all_beta"),
    ("2BDS", "all_beta"),
    ("1UBQ", "alpha_beta"),
    ("1CRN", "alpha_beta"),
    ("2GB1", "alpha_beta"),
    ("1A8O", "alpha_beta"),
    ("1PGB", "alpha_beta"),
    ("3GB1", "alpha_beta"),
    ("1FXD", "alpha_beta"),
    ("1IGD", "alpha_beta"),
    ("5PTI", "alpha_beta"),
    ("2JVD", "alpha_beta"),
]


def _set(name: str, value) -> None:
    """Patch config + predict (+ contact_predict) imported bindings."""
    setattr(cfg, name, value)
    if hasattr(pred_mod, name):
        setattr(pred_mod, name, value)
    try:
        import pairfold.contact_predict as cp_mod

        if hasattr(cp_mod, name):
            setattr(cp_mod, name, value)
    except Exception:
        pass


def apply_variant(name: str) -> None:
    """Mutate knobs for cumulative ablation stack (predict imports constants by value)."""
    # Match Domains100 paper knobs (no soft-map / t30; capped early fold).
    _set("USE_FOLD_HEAD", False)
    _set("USE_FOLD_HEAD_SELECT", False)
    _set("USE_STAGE2_SIDECHAINS", False)
    _set("USE_STAGE3_ATOMS", False)
    _set("USE_ESM_ALT_CONTACTS", False)
    _set("USE_SOFT_CONTACT_FOLD", False)
    _set("EARLY_CONTACT_RESTARTS", 2)
    _set("EARLY_CONTACT_STEPS", 120)
    _set("SS_BOUNDARY_OPT_MAX_LEN", 64)

    if name == "base_consensus":
        _set("EARLY_CONTACT_FOLD", False)
        _set("LEVER_ASSEMBLY_MAX_LEN", 0)
        _set("USE_LEVER_POLISH", False)
        _set("SS_PIPELINE_MAX_LEN", 0)
    elif name == "plus_lookahead":
        _set("EARLY_CONTACT_FOLD", False)
        _set("LEVER_ASSEMBLY_MAX_LEN", 256)
        _set("USE_LEVER_POLISH", False)
        _set("SS_PIPELINE_MAX_LEN", 0)
    elif name == "plus_lever":
        _set("EARLY_CONTACT_FOLD", False)
        _set("LEVER_ASSEMBLY_MAX_LEN", 256)
        _set("USE_LEVER_POLISH", True)
        _set("SS_PIPELINE_MAX_LEN", 0)
    elif name == "plus_ss":
        _set("EARLY_CONTACT_FOLD", False)
        _set("LEVER_ASSEMBLY_MAX_LEN", 256)
        _set("USE_LEVER_POLISH", True)
        _set("SS_PIPELINE_MAX_LEN", 256)
    elif name == "full_pipeline":
        # Full local stack (look-ahead + lever + SS). Contact hinge is evaluated
        # separately in Stage-1 benches; it dominates wall-clock on this panel.
        _set("EARLY_CONTACT_FOLD", False)
        _set("USE_SOFT_CONTACT_FOLD", False)
        _set("USE_ESM_CONTACTS", False)
        _set("CONTACT_USE_MAX_LEN", 0)
        _set("LEVER_ASSEMBLY_MAX_LEN", 256)
        _set("USE_LEVER_POLISH", True)
        _set("SS_PIPELINE_MAX_LEN", 256)
    else:
        raise ValueError(name)


def run() -> None:
    from pairfold import clash_assembly as ca_mod

    _real_assemble = ca_mod.assemble_greedy_backtrack

    def _capped_assemble(*args, **kwargs):
        kwargs["max_nodes"] = min(int(kwargs.get("max_nodes", 1500) or 1500), 1500)
        return _real_assemble(*args, **kwargs)

    ca_mod.assemble_greedy_backtrack = _capped_assemble  # type: ignore[assignment]

    variants = [
        "base_consensus",
        "plus_lookahead",
        "plus_lever",
        "plus_ss",
        "full_pipeline",
    ]
    print("Loading predictor…", flush=True)
    pred = FragmentPredictor()
    rows: List[Dict] = []

    for vname in variants:
        apply_variant(vname)
        print(f"\n=== VARIANT {vname} ===", flush=True)
        rmsds = []
        times = []
        for pdb_id, cls in ABLATION_PDBS:
            try:
                path = download_pdb(pdb_id, CACHE_DIR)
                seq, native = extract_native_ca_and_sequence(path)
                if len(seq) > 100 and vname in ("plus_lookahead", "plus_lever", "full_pipeline"):
                    print(f"  {pdb_id}: SKIP len={len(seq)} for {vname}", flush=True)
                    continue
                t0 = time.perf_counter()
                result = pred.predict_sequence(seq)
                elapsed = time.perf_counter() - t0
                ca = dihedrals_to_ca(result["phis"], result["psis"])
                m = min(len(ca), len(native))
                rmsd = kabsch_rmsd(ca[:m], native[:m])
            except Exception as e:
                print(f"  {pdb_id} FAIL {type(e).__name__}", flush=True)
                continue
            rows.append(
                {
                    "variant": vname,
                    "pdb_id": pdb_id,
                    "class": cls,
                    "length": m,
                    "time_s": round(elapsed, 3),
                    "rmsd": round(rmsd, 4),
                }
            )
            rmsds.append(rmsd)
            times.append(elapsed)
            print(f"  {pdb_id}: {rmsd:.2f}Å {elapsed:.1f}s", flush=True)
        if rmsds:
            print(
                f"  MEAN {vname}: RMSD={np.mean(rmsds):.3f} time={np.mean(times):.2f}s",
                flush=True,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for vname in variants:
        sub = [r for r in rows if r["variant"] == vname]
        if sub:
            summary[vname] = {
                "n": len(sub),
                "mean_rmsd": float(np.mean([r["rmsd"] for r in sub])),
                "mean_time_s": float(np.mean([r["time_s"] for r in sub])),
            }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
