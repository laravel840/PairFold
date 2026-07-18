"""Extract long-range Cα contact labels from downloaded PDBs for ContactPairNet."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from ..config import (  # noqa: E402
    AA_LIST,
    CONTACT_DIR,
    CONTACT_MAX_LEN,
    CONTACT_MIN_SEP,
    CONTACT_THRESH_A,
    RAW_DIR,
    SEED,
)

MAX_FILE_BYTES = 600_000
FILE_TIMEOUT_S = 12
TARGET_RECORDS = 2500

_WORKER = r'''
import json, random, sys
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, PPBuilder

THREE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}
path = Path(sys.argv[1])
out = Path(sys.argv[2])
max_len = int(sys.argv[3])
min_sep = int(sys.argv[4])
thresh = float(sys.argv[5])
max_chain = 400
crops_n = 4

parser = PDBParser(QUIET=True)
structure = parser.get_structure(path.stem, str(path))
model = next(structure.get_models())
ppb = PPBuilder()
rng = random.Random(hash(path.stem) & 0xFFFFFFFF)
rows = []
for ci, pp in enumerate(ppb.build_peptides(model)):
    residues = list(pp)
    n = len(residues)
    if n < min_sep + 1 or n > max_chain:
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
    seq = "".join(seq_chars)
    ca = np.stack(ca_list, axis=0)
    source = f"{path.stem}_{ci}"
    if n <= max_len:
        starts_lens = [(0, n)]
    else:
        max_start = n - max_len
        starts = {0, max_start}
        while len(starts) < crops_n:
            starts.add(rng.randint(0, max_start))
        starts_lens = [(s, max_len) for s in sorted(starts)[:crops_n]]
    for start, length in starts_lens:
        sub = ca[start:start+length]
        L = sub.shape[0]
        contacts = []
        for i in range(L - min_sep):
            d = np.linalg.norm(sub[i+min_sep:] - sub[i], axis=1)
            for off in np.flatnonzero(d < thresh):
                contacts.append([float(i), float(i+min_sep+int(off)), float(d[off])])
        rows.append({
            "seq": seq[start:start+length],
            "length": length,
            "start": start,
            "source": source,
            "contacts": contacts,
            "n_contacts": len(contacts),
        })
out.write_text(json.dumps(rows), encoding="utf-8")
'''


def main() -> None:
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    pdb_files = [
        p for p in sorted(RAW_DIR.glob("*.pdb")) if p.stat().st_size <= MAX_FILE_BYTES
    ]
    print(f"Candidate PDBs: {len(pdb_files)}", flush=True)

    rows: List[dict] = []
    n_ok = n_timeout = n_fail = 0

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        worker_py = td_path / "worker.py"
        worker_py.write_text(_WORKER, encoding="utf-8")

        for i, path in enumerate(pdb_files):
            if len(rows) >= TARGET_RECORDS:
                break
            out_json = td_path / f"{path.stem}.json"
            cmd = [
                sys.executable,
                str(worker_py),
                str(path),
                str(out_json),
                str(CONTACT_MAX_LEN),
                str(CONTACT_MIN_SEP),
                str(CONTACT_THRESH_A),
            ]
            try:
                subprocess.run(
                    cmd,
                    timeout=FILE_TIMEOUT_S,
                    check=True,
                    capture_output=True,
                )
                part = json.loads(out_json.read_text(encoding="utf-8"))
                rows.extend(part)
                n_ok += 1
            except subprocess.TimeoutExpired:
                n_timeout += 1
                print(f"  timeout {path.name}", flush=True)
            except Exception as e:
                n_fail += 1
                if n_fail <= 8:
                    print(f"  fail {path.name}: {type(e).__name__}", flush=True)

            if (i + 1) % 40 == 0:
                print(
                    f"  scanned {i + 1} ok={n_ok} timeout={n_timeout} "
                    f"fail={n_fail} records={len(rows)}",
                    flush=True,
                )

    rng = random.Random(SEED)
    rng.shuffle(rows)
    out = CONTACT_DIR / "contacts.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n_contacts = sum(int(r.get("n_contacts", 0)) for r in rows)
    length_hist: Dict[int, int] = {}
    for r in rows:
        length_hist[int(r["length"])] = length_hist.get(int(r["length"]), 0) + 1

    meta = {
        "n_records": len(rows),
        "n_ok_files": n_ok,
        "n_timeout": n_timeout,
        "n_fail": n_fail,
        "n_contacts_total": n_contacts,
        "contact_thresh_A": CONTACT_THRESH_A,
        "min_sep": CONTACT_MIN_SEP,
        "max_crop_len": CONTACT_MAX_LEN,
        "length_hist": {str(k): v for k, v in sorted(length_hist.items())},
        "aa_list": AA_LIST,
    }
    (CONTACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
