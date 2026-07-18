"""One-shot: fetch PDB → extract fragments → train on GPU."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(module: str) -> None:
    print(f"\n=== {module} ===\n", flush=True)
    subprocess.check_call([sys.executable, "-m", module], cwd=str(REPO))


def main() -> None:
    run("pairfold.data.fetch_pdb")
    run("pairfold.data.extract_fragments")
    run("pairfold.train")
    print("\nPipeline complete. Start API with: python -m pairfold.server")


if __name__ == "__main__":
    main()
