# PairFold — PDB short-fragment + contact predictor

**Scope:** Trains a small Transformer on **real high-resolution PDB backbone fragments** (length 2–5) for φ/ψ, plus a lightweight **ContactPairNet** for long-range Cα contacts. Longer sequences are **segmented**, assembled with clash-aware look-ahead, and optionally **pulled toward predicted contact anchors**. This is **not** AlphaFold.

## Results on this machine

- GPU: NVIDIA GTX 1650 (CUDA)
- Fragment checkpoint: `checkpoints/best.pt`
- Contact checkpoint: `checkpoints/contact_best.pt`
- Contact data: ~2.4k crops from high-res PDBs (8 Å, |i−j|≥6)
- Five-domain mean Cα RMSD with contacts: **~17.2 Å** (was ~18.5 Å local-only)

## Pipeline

From the repository root:

```bash
python -m pip install -r pairfold/requirements.txt
python -m pairfold.data.fetch_pdb
python -m pairfold.data.extract_fragments
python -m pairfold.data.extract_contacts
python -m pairfold.train
python -m pairfold.train_contact
python -m pairfold.server          # API http://127.0.0.1:8000
```

Or: `npm run ml:pipeline` then `npm run ml:server`.

CLI test:

```bash
python -m pairfold.predict AGPVK
python -m pairfold.predict AGPVKLLTFGAA
```

## UI

With Vite running (`npm run dev`) and `python -m pairfold.server`, use **Predict (PDB)** in the web app. Vite proxies `/api` → `:8000`.

## Limitations

- Local torsion geometry (φ/ψ windows ≤5) + sparse sequence-only contacts
- Contact precision is moderate; wrong contacts can hurt some targets
- Not competitive with AlphaFold / ESMFold for full proteins
