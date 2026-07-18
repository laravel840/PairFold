"""Extract short backbone fragments (len 2–5) with φ/ψ from downloaded PDBs."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from Bio.PDB import PDBParser, PPBuilder
from tqdm import tqdm

from ..config import (  # noqa: E402
    AA_LIST,
    FRAG_DIR,
    FRAGMENT_STRIDE,
    MAX_FRAGMENTS_PER_LENGTH,
    MAX_LEN,
    MAX_PER_SEQUENCE,
    MIN_LEN,
    RAW_DIR,
    SEED,
)

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _valid_angle(a: float) -> bool:
    return a is not None and not math.isnan(a) and -math.pi <= a <= math.pi


def extract_from_file(path: Path) -> List[dict]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    ppb = PPBuilder()
    frags: List[dict] = []

    for model in structure:
        for pp in ppb.build_peptides(model):
            residues = list(pp)
            n = len(residues)
            if n < MIN_LEN:
                continue

            seq_chars = []
            ok = True
            for res in residues:
                name = res.get_resname().upper()
                if name not in THREE_TO_ONE:
                    ok = False
                    break
                seq_chars.append(THREE_TO_ONE[name])
            if not ok:
                continue

            torsions = pp.get_phi_psi_list()
            if len(torsions) != n:
                continue

            phis = []
            psis = []
            for phi, psi in torsions:
                phis.append(float(phi) if _valid_angle(phi) else float("nan"))
                psis.append(float(psi) if _valid_angle(psi) else float("nan"))

            seq = "".join(seq_chars)
            for L in range(MIN_LEN, MAX_LEN + 1):
                for i in range(0, n - L + 1, FRAGMENT_STRIDE):
                    window_phi = phis[i : i + L]
                    window_psi = psis[i : i + L]
                    if any(math.isnan(window_phi[j]) and 0 < j < L - 1 for j in range(L)):
                        continue
                    if any(math.isnan(window_psi[j]) and 0 < j < L - 1 for j in range(L)):
                        continue
                    if all(math.isnan(x) for x in window_phi + window_psi):
                        continue
                    # Prefer windows with at least half of torsions defined
                    defined = sum(
                        0 if math.isnan(x) else 1 for x in window_phi + window_psi
                    )
                    if defined < L:
                        continue
                    frags.append(
                        {
                            "seq": seq[i : i + L],
                            "phi": window_phi,
                            "psi": window_psi,
                            "length": L,
                            "source": path.stem,
                        }
                    )
    return frags


def soft_dedup(rows: List[dict], max_per_seq: int, rng: random.Random) -> List[dict]:
    by_seq: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_seq[r["seq"]].append(r)
    out = []
    for seq, items in by_seq.items():
        rng.shuffle(items)
        out.extend(items[:max_per_seq])
    rng.shuffle(out)
    return out


def main() -> None:
    FRAG_DIR.mkdir(parents=True, exist_ok=True)
    pdb_files = sorted(RAW_DIR.glob("*.pdb"))
    if not pdb_files:
        raise SystemExit(f"No PDB files in {RAW_DIR}. Run fetch_pdb.py first.")

    buckets: Dict[int, List[dict]] = defaultdict(list)
    for path in tqdm(pdb_files, desc="Extracting fragments"):
        try:
            for frag in extract_from_file(path):
                buckets[frag["length"]].append(frag)
        except Exception as e:
            tqdm.write(f"skip {path.name}: {e}")

    rng = random.Random(SEED)
    all_rows = []
    stats = {}
    seq_freq = Counter()
    for L in range(MIN_LEN, MAX_LEN + 1):
        rows = soft_dedup(buckets[L], MAX_PER_SEQUENCE, rng)
        rng.shuffle(rows)
        if len(rows) > MAX_FRAGMENTS_PER_LENGTH:
            rows = rows[:MAX_FRAGMENTS_PER_LENGTH]
        stats[L] = len(rows)
        for r in rows:
            seq_freq[r["seq"]] += 1
        all_rows.extend(rows)

    rng.shuffle(all_rows)
    # Attach empirical frequency (how often this exact oligomer appears in the set)
    for r in all_rows:
        r["seq_count"] = int(seq_freq[r["seq"]])

    out = FRAG_DIR / "fragments.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    # Sequence prior file for confidence prior at inference
    prior = {s: c for s, c in seq_freq.most_common()}
    (FRAG_DIR / "seq_prior.json").write_text(json.dumps(prior), encoding="utf-8")

    meta = {
        "n_total": len(all_rows),
        "by_length": stats,
        "n_structures": len(pdb_files),
        "n_unique_sequences": len(seq_freq),
        "max_per_sequence": MAX_PER_SEQUENCE,
        "aa_list": AA_LIST,
    }
    (FRAG_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
