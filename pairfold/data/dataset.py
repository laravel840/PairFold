"""PyTorch dataset over extracted PDB fragments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from ..config import AA_LIST, MAX_LEN, PAD_IDX, UNK_IDX
from ..model.fragment_net import angles_to_sincos


def aa_to_idx(ch: str) -> int:
    i = AA_LIST.find(ch)
    return i if i >= 0 else UNK_IDX


class FragmentDataset(Dataset):
    def __init__(self, rows: List[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        seq = row["seq"]
        L = len(seq)
        tokens = [aa_to_idx(c) for c in seq] + [PAD_IDX] * (MAX_LEN - L)
        mask = [1] * L + [0] * (MAX_LEN - L)

        phi = []
        psi = []
        ang_mask = []
        for i in range(MAX_LEN):
            if i < L:
                p = row["phi"][i]
                s = row["psi"][i]
                p_ok = not (p is None or (isinstance(p, float) and math.isnan(p)))
                s_ok = not (s is None or (isinstance(s, float) and math.isnan(s)))
                phi.append(float(p) if p_ok else 0.0)
                psi.append(float(s) if s_ok else 0.0)
                # 4 channels: sinφ cosφ sinψ cosψ
                ang_mask.append(
                    [1.0 if p_ok else 0.0, 1.0 if p_ok else 0.0, 1.0 if s_ok else 0.0, 1.0 if s_ok else 0.0]
                )
            else:
                phi.append(0.0)
                psi.append(0.0)
                ang_mask.append([0.0, 0.0, 0.0, 0.0])

        phi_t = torch.tensor(phi, dtype=torch.float32)
        psi_t = torch.tensor(psi, dtype=torch.float32)
        target = angles_to_sincos(phi_t, psi_t)
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target": target,
            "ang_mask": torch.tensor(ang_mask, dtype=torch.bool),
            "length": L,
        }


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_rows(rows: List[dict], val_frac: float, seed: int) -> Tuple[List[dict], List[dict]]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(rows), generator=g).tolist()
    n_val = max(1, int(len(rows) * val_frac))
    val_idx = set(idx[:n_val])
    train, val = [], []
    for i, r in enumerate(rows):
        (val if i in val_idx else train).append(r)
    return train, val
