"""Download an expanded high-res PDB subset via RCSB Search + files API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

from tqdm import tqdm

from ..config import MAX_RESOLUTION, MAX_STRUCTURES, RAW_DIR

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "PairFold/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_pdb_ids(limit: int = MAX_STRUCTURES) -> List[str]:
    """X-ray proteins only, resolution cutoff, sorted by resolution."""
    ids: List[str] = []
    page = 100  # RCSB max rows per request often 1000; keep 100 for safety
    start = 0
    while len(ids) < limit:
        rows = min(page, limit - len(ids))
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "negation": False,
                            "value": MAX_RESOLUTION,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "negation": False,
                            "value": "X-RAY DIFFRACTION",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                            "operator": "exact_match",
                            "value": "Protein (only)",
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": start, "rows": rows},
                "results_content_type": ["experimental"],
                "sort": [
                    {
                        "sort_by": "rcsb_entry_info.resolution_combined",
                        "direction": "asc",
                    }
                ],
                "scoring_strategy": "combined",
            },
        }
        result = _post_json(SEARCH_URL, query)
        batch = [hit["identifier"] for hit in result.get("result_set", [])]
        if not batch:
            break
        ids.extend(batch)
        start += len(batch)
        if len(batch) < rows:
            break
    # unique preserve order
    seen = set()
    uniq = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq[:limit]


def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{pdb_id.upper()}.pdb"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    req = urllib.request.Request(url, headers={"User-Agent": "PairFold/0.2"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    id_file = RAW_DIR / "pdb_ids.json"

    print("Searching RCSB for high-resolution protein structures…")
    ids = search_pdb_ids(MAX_STRUCTURES)
    id_file.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    print(f"Target entries: {len(ids)} (resolution <= {MAX_RESOLUTION} A)")

    existing = {p.stem.upper() for p in RAW_DIR.glob("*.pdb")}
    todo = [i for i in ids if i.upper() not in existing]
    print(f"Already on disk: {len(existing)} | to download: {len(todo)}")

    ok = len(existing)
    for pdb_id in tqdm(todo, desc="Downloading PDB"):
        try:
            download_pdb(pdb_id, RAW_DIR)
            ok += 1
            time.sleep(0.03)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            tqdm.write(f"skip {pdb_id}: {e}")
    print(f"Downloaded/available: {ok} pdb files in {RAW_DIR}")


if __name__ == "__main__":
    main()
