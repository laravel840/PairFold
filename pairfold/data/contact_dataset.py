"""Dataset for ContactPairNet from contacts.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from ..config import (
    AA_LIST,
    CONTACT_MAX_LEN,
    CONTACT_MIN_SEP,
    PAD_IDX,
    UNK_IDX,
)


def aa_to_idx(ch: str) -> int:
    i = AA_LIST.find(ch)
    return i if i >= 0 else UNK_IDX


class ContactDataset(Dataset):
    def __init__(self, rows: List[dict], max_len: int = CONTACT_MAX_LEN) -> None:
        self.rows = rows
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        seq = row["seq"]
        L = min(len(seq), self.max_len)
        seq = seq[:L]

        tokens = [aa_to_idx(c) for c in seq] + [PAD_IDX] * (self.max_len - L)
        mask = [1] * L + [0] * (self.max_len - L)

        contact = torch.zeros(self.max_len, self.max_len, dtype=torch.float32)
        dist = torch.zeros(self.max_len, self.max_len, dtype=torch.float32)
        for item in row.get("contacts") or []:
            i, j, d = int(item[0]), int(item[1]), float(item[2])
            if i < 0 or j < 0 or i >= L or j >= L:
                continue
            if abs(i - j) < CONTACT_MIN_SEP:
                continue
            contact[i, j] = 1.0
            contact[j, i] = 1.0
            dist[i, j] = d
            dist[j, i] = d

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "contact": contact,
            "dist": dist,
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


def estimate_pos_weight(rows: List[dict], max_len: int = CONTACT_MAX_LEN) -> float:
    """neg/pos ratio for BCE pos_weight (clamped)."""
    n_pos = 0
    n_neg = 0
    for r in rows:
        L = min(len(r["seq"]), max_len)
        # long-range upper-triangle slots
        n_pairs = 0
        for i in range(L):
            for j in range(i + CONTACT_MIN_SEP, L):
                n_pairs += 1
        n_c = sum(
            1
            for item in (r.get("contacts") or [])
            if int(item[0]) < L and int(item[1]) < L and abs(int(item[0]) - int(item[1])) >= CONTACT_MIN_SEP
        )
        # contacts list is upper-only from extractor
        n_pos += n_c
        n_neg += max(0, n_pairs - n_c)
    if n_pos == 0:
        return 10.0
    return float(min(max(n_neg / n_pos, 1.0), 50.0))
