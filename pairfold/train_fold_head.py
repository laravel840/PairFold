"""Train ESMFoldHead on precomputed embeddings + native Cα distances."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .config import (
    CKPT_DIR,
    FOLD_HEAD_BATCH_SIZE,
    FOLD_HEAD_CKPT_NAME,
    FOLD_HEAD_D_MODEL,
    FOLD_HEAD_DIR,
    FOLD_HEAD_EMB_DIM,
    FOLD_HEAD_EPOCHS,
    FOLD_HEAD_LR,
    FOLD_HEAD_MAX_LEN,
    FOLD_HEAD_PAIR_DIM,
    NUM_WORKERS,
    SEED,
    VAL_FRAC,
)
from .model.fold_head import ESMFoldHead, fold_distance_loss


class FoldCropDataset(Dataset):
    def __init__(self, rows: List[Dict], emb_dir: Path, max_len: int = FOLD_HEAD_MAX_LEN):
        self.rows = rows
        self.emb_dir = emb_dir
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        key = self.rows[idx]["key"]
        z = np.load(self.emb_dir / f"{key}.npz")
        emb = z["emb"].astype(np.float32)
        ca = z["ca"].astype(np.float32)
        n = emb.shape[0]
        # Pad to max_len
        emb_p = np.zeros((self.max_len, emb.shape[1]), dtype=np.float32)
        ca_p = np.zeros((self.max_len, 3), dtype=np.float32)
        mask = np.zeros(self.max_len, dtype=np.bool_)
        emb_p[:n] = emb
        ca_p[:n] = ca
        mask[:n] = True
        # Distance matrix
        d = np.linalg.norm(ca_p[:, None, :] - ca_p[None, :, :], axis=-1)
        d = d.astype(np.float32)
        return {
            "emb": torch.from_numpy(emb_p),
            "dist": torch.from_numpy(d),
            "mask": torch.from_numpy(mask),
        }


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    index_path = FOLD_HEAD_DIR / "index.json"
    emb_dir = FOLD_HEAD_DIR / "emb"
    if not index_path.exists():
        raise SystemExit("Missing fold crops. Run: python -m pairfold.data.extract_fold_crops")

    rows = json.loads(index_path.read_text(encoding="utf-8"))
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_FRAC))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    print(f"Train {len(train_rows)} | Val {len(val_rows)}", flush=True)

    # Infer emb dim from first crop
    sample = np.load(emb_dir / f"{train_rows[0]['key']}.npz")
    emb_dim = int(sample["emb"].shape[1])
    print(f"emb_dim={emb_dim}", flush=True)

    train_loader = DataLoader(
        FoldCropDataset(train_rows, emb_dir),
        batch_size=FOLD_HEAD_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        FoldCropDataset(val_rows, emb_dir),
        batch_size=FOLD_HEAD_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    dev = device()
    print(f"Device: {dev}", flush=True)
    model = ESMFoldHead(
        emb_dim=emb_dim,
        d_model=FOLD_HEAD_D_MODEL,
        pair_dim=FOLD_HEAD_PAIR_DIM,
        max_len=FOLD_HEAD_MAX_LEN,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=FOLD_HEAD_LR, weight_decay=1e-4)
    scaler = GradScaler(enabled=dev.type == "cuda")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_val = 1e9
    history = []
    for epoch in range(1, FOLD_HEAD_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        tr_loss, tr_n = 0.0, 0
        for batch in train_loader:
            emb = batch["emb"].to(dev)
            dist_t = batch["dist"].to(dev)
            mask = batch["mask"].to(dev)
            pair = mask.unsqueeze(1) & mask.unsqueeze(2)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=dev.type == "cuda"):
                dist_p, conf_logit = model(emb, mask)
                loss = fold_distance_loss(dist_p, dist_t, conf_logit, pair)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            tr_loss += float(loss.detach()) * emb.size(0)
            tr_n += emb.size(0)

        model.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["emb"].to(dev)
                dist_t = batch["dist"].to(dev)
                mask = batch["mask"].to(dev)
                pair = mask.unsqueeze(1) & mask.unsqueeze(2)
                with autocast(enabled=dev.type == "cuda"):
                    dist_p, conf_logit = model(emb, mask)
                    loss = fold_distance_loss(dist_p, dist_t, conf_logit, pair)
                va_loss += float(loss.detach()) * emb.size(0)
                va_n += emb.size(0)

        tr = tr_loss / max(tr_n, 1)
        va = va_loss / max(va_n, 1)
        dt = time.perf_counter() - t0
        print(f"epoch {epoch:02d} train={tr:.4f} val={va:.4f} ({dt:.1f}s)", flush=True)
        history.append({"epoch": epoch, "train": tr, "val": va, "s": dt})
        ckpt = {
            "model": model.state_dict(),
            "emb_dim": emb_dim,
            "d_model": FOLD_HEAD_D_MODEL,
            "pair_dim": FOLD_HEAD_PAIR_DIM,
            "max_len": FOLD_HEAD_MAX_LEN,
            "epoch": epoch,
            "val": va,
        }
        torch.save(ckpt, CKPT_DIR / "fold_head_last.pt")
        if va < best_val:
            best_val = va
            torch.save(ckpt, CKPT_DIR / FOLD_HEAD_CKPT_NAME)
            print(f"  saved {FOLD_HEAD_CKPT_NAME} val={va:.4f}", flush=True)

    (CKPT_DIR / "fold_head_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Done. best_val={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
