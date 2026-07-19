"""Extract short-chain Cα crops + precompute ESM embeddings for fold-head training."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pairfold.config import (  # noqa: E402
    ESM_MODEL_NAME,
    FOLD_HEAD_CROPS_PER_CHAIN,
    FOLD_HEAD_DIR,
    FOLD_HEAD_MAX_CHAINS,
    FOLD_HEAD_MAX_LEN,
    FOLD_HEAD_MIN_LEN,
    RAW_DIR,
    SEED,
)

THREE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _chain_ca(path: Path):
    from Bio.PDB import PDBParser, PPBuilder

    parser = PDBParser(QUIET=True)
    model = next(parser.get_structure(path.stem, str(path)).get_models())
    out = []
    for ci, pp in enumerate(PPBuilder().build_peptides(model)):
        residues = list(pp)
        n = len(residues)
        if n < FOLD_HEAD_MIN_LEN:
            continue
        seq_chars, ca_list = [], []
        ok = True
        for res in residues:
            name = res.get_resname().upper()
            if name not in THREE or "CA" not in res:
                ok = False
                break
            seq_chars.append(THREE[name])
            ca_list.append(np.asarray(res["CA"].get_coord(), dtype=np.float32))
        if not ok:
            continue
        out.append(("".join(seq_chars), np.stack(ca_list, axis=0), f"{path.stem}_{ci}"))
    return out


def main() -> None:
    FOLD_HEAD_DIR.mkdir(parents=True, exist_ok=True)
    emb_dir = FOLD_HEAD_DIR / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    pdbs = sorted(RAW_DIR.glob("*.pdb"))
    rng.shuffle(pdbs)
    pdbs = pdbs[: FOLD_HEAD_MAX_CHAINS]

    from pairfold.esm_contacts import get_esm_predictor

    esm = get_esm_predictor(ESM_MODEL_NAME)
    if not esm.enabled:
        raise SystemExit("ESM not available — install fair-esm")

    index = []
    n_ok = 0
    for pi, path in enumerate(pdbs):
        try:
            chains = _chain_ca(path)
        except Exception:
            continue
        for seq, ca, source in chains:
            n = len(seq)
            if n <= FOLD_HEAD_MAX_LEN:
                crops = [(0, n)]
            else:
                max_start = n - FOLD_HEAD_MAX_LEN
                starts = {0, max_start}
                while len(starts) < FOLD_HEAD_CROPS_PER_CHAIN:
                    starts.add(rng.randint(0, max_start))
                crops = [(s, FOLD_HEAD_MAX_LEN) for s in sorted(starts)[: FOLD_HEAD_CROPS_PER_CHAIN]]
            for start, length in crops:
                sub_seq = seq[start : start + length]
                sub_ca = ca[start : start + length]
                # Skip benchmark IDs from train leakage (lowercase stems)
                if path.stem.lower() in {"1a8o", "1crn", "2gb1", "1ubq", "3gb1", "1l2y", "1vii", "1pgb", "1bz4", "2jvd"}:
                    continue
                try:
                    emb = esm.embeddings(sub_seq, use_cache=True)
                except Exception:
                    continue
                if emb.shape[0] != length:
                    continue
                key = f"{source}_{start}_{length}"
                np.savez_compressed(
                    emb_dir / f"{key}.npz",
                    emb=emb.astype(np.float32),
                    ca=sub_ca.astype(np.float32),
                    seq=np.asarray(list(sub_seq)),
                )
                index.append({"key": key, "n": length, "source": source, "start": start})
                n_ok += 1
        if (pi + 1) % 25 == 0:
            print(f"  scanned {pi+1}/{len(pdbs)} files → {n_ok} crops", flush=True)

    (FOLD_HEAD_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {n_ok} crops to {FOLD_HEAD_DIR}", flush=True)


if __name__ == "__main__":
    main()
