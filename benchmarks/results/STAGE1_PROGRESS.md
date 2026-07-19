# Stage-1 Accuracy Upgrade — Progress Notes

## Goal
Move mean Kabsch Cα RMSD on (`1A8O`, `1CRN`, `2GB1`, `1UBQ`, `3GB1`) from ~17 Å → toward **&lt;10 Å** (soft: 7–9), then Stage-2/3.

## Locked baseline (this campaign)
| Phase | Mean Å | Notes |
|---|---:|---|
| Original PairFold | **17.17** | ContactPairNet + late polish |
| Prior Stage-1 lock | **11.25** | ESM-t12 + CA/torsion early fold |
| Soft-map post-assembly | **11.13** | Gated soft rescue after lever (1CRN 14.33→13.71) |
| ESM fold-head campaign | **11.13** | Trained head; direct fold regressed; select-only flat |

**Hard-stop:** mean ≤ 8.0 Å **or** 3 iterations with &lt;0.3 Å mean gain → **plateau declared**.  
**Phase-A gate for Stage-2/3:** mean **&lt; 10.0** — **NOT MET**.

## What was implemented (fold-head campaign)

| Module | Role |
|---|---|
| `pairfold/model/fold_head.py` | `ESMFoldHead` — pair distance + conf on frozen ESM emb |
| `pairfold/data/extract_fold_crops.py` | PDB crops + precomputed ESM embeddings (~1556) |
| `pairfold/train_fold_head.py` | AMP training → `checkpoints/fold_head_best.pt` (val≈2.87) |
| `pairfold/fold_head_infer.py` | Distance map, CA optimize, decoy MAE score |
| `pairfold/stage2_sidechains.py` | **Inert** Stage-2 stub (`USE_STAGE2_SIDECHAINS=False`) |
| `pairfold/stage3_atoms.py` | **Inert** Stage-3 stub (`USE_STAGE3_ATOMS=False`) |

### Config (best stable)
- `ESM_MODEL_NAME=esm2_t12_35M_UR50D`
- `USE_SCAFFOLD_ENSEMBLE=False`
- `USE_SOFT_CONTACT_FOLD=True` (gated-only, after lever/SS)
- `USE_FOLD_HEAD=False` — direct CA refine hurt RMSD
- `USE_FOLD_HEAD_SELECT=True` — rank decoys by predicted-distance MAE (no mean gain yet)
- `USE_STAGE2_SIDECHAINS=False` / `USE_STAGE3_ATOMS=False` until mean &lt; 10

## Iteration scoreboard (fold-head push)

| Iter | Mean Å | Notes |
|---|---:|---|
| Soft-map lock | **11.13** | 7.35 / 13.71 / 10.99 / 12.95 / 10.63 |
| Fold-head CA refine (all) | 11.93 | **1CRN 17.75** — rejected |
| Fold-head sharp-only post | 11.23 | 1A8O 7.85 regression |
| Fold-head early+sharp | ~15+ on 2GB1 mid-run | **killed** — ranking ≠ RMSD |
| Fold-head **select-only** | **11.13** | Safe; no mean gain |

### Best per-target (stable)
| PDB | Å |
|---|---:|
| 1A8O | **7.35** |
| 1CRN | **13.71** (gated soft rescue) |
| 2GB1 | **10.99** |
| 1UBQ | 12.95 |
| 3GB1 | **10.63** |

## What worked
1. ESM-t12 + early CA/torsion + post-assembly soft-map (gated).
2. Precomputing ESM embeddings → train tiny head on 4GB VRAM (works).
3. Dual-path / gating that **rejects** fold-head CA moves when they regress.

## What did not
1. Learned distance → CA optimize: rank/MAE accept still ≠ true RMSD (1CRN, 2GB1).
2. Did **not** reach mean &lt; 10 (need ~5.6 Å total cut; dominated by 1CRN+1UBQ).
3. Stage-2/3 **not started** (gate failed). Stubs only.

## How to run
```bash
pip install fair-esm
# (re)build fold-head data / train if needed:
python -u -m pairfold.data.extract_fold_crops
python -u -m pairfold.train_fold_head
python -u benchmarks/run_stage1_bench.py
```

## Stage 2/3
**Enabled anyway (user override).** Mean still 11.13 Å (scaffold not near-native).

| Module | Status |
|---|---|
| `pairfold/sidechain_geom.py` | Rotamer library + CB/SC geometry |
| `pairfold/stage2_sidechains.py` | Greedy clash-min rotamer pack (backbone locked) |
| `pairfold/stage3_atoms.py` | N/CA/C/O + sidechains → PDB export |
| Config | `USE_STAGE2_SIDECHAINS=True`, `USE_STAGE3_ATOMS=True`, `STAGE23_MAX_LEN=256` |

Smoke (predict → all-atom PDB): `python -u benchmarks/smoke_stage23.py`  
Exports: `pairfold/export/all_atom/*_pairfold.pdb`

## Honest status — PLATEAU
Best mean remains **11.13 Å**. Three fold-head iterations failed to gain ≥0.3 Å (two regressed, one flat).  
Checkpoint kept: `pairfold/checkpoints/fold_head_best.pt`.  
Next real lever: **train a decoy quality / RMSD-proxy ranker** (or MSA/larger structure module), not more contact-energy folding.
