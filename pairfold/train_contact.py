"""Train ContactPairNet on PDB long-range Cα contacts (GPU)."""

from __future__ import annotations

import json
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    CKPT_DIR,
    CONTACT_BATCH_SIZE,
    CONTACT_CKPT_NAME,
    CONTACT_D_FF,
    CONTACT_D_MODEL,
    CONTACT_DIR,
    CONTACT_DIST_LOSS_W,
    CONTACT_DROPOUT,
    CONTACT_EPOCHS,
    CONTACT_INFER_MAX_LEN,
    CONTACT_LR,
    CONTACT_MIN_SEP,
    CONTACT_N_HEADS,
    CONTACT_N_LAYERS,
    CONTACT_WEIGHT_DECAY,
    NUM_WORKERS,
    SEED,
    VAL_FRAC,
    VOCAB_SIZE,
)
from .data.contact_dataset import (
    ContactDataset,
    estimate_pos_weight,
    load_jsonl,
    split_rows,
)
from .model.contact_net import ContactPairNet, contact_loss, contact_pair_mask


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    torch.manual_seed(SEED)
    path = CONTACT_DIR / "contacts.jsonl"
    if not path.exists():
        raise SystemExit("Missing contacts. Run data/extract_contacts.py first.")

    rows = load_jsonl(path)
    train_rows, val_rows = split_rows(rows, VAL_FRAC, SEED)
    pos_w = estimate_pos_weight(train_rows)
    print(f"Train {len(train_rows)} | Val {len(val_rows)} | pos_weight={pos_w:.2f}", flush=True)

    train_loader = DataLoader(
        ContactDataset(train_rows),
        batch_size=CONTACT_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        ContactDataset(val_rows),
        batch_size=CONTACT_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (CONTACT_DIR / "val_split.jsonl").write_text(
        "\n".join(json.dumps(r) for r in val_rows) + "\n", encoding="utf-8"
    )

    dev = device()
    print(f"Device: {dev}", flush=True)
    if dev.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)

    model = ContactPairNet(
        vocab_size=VOCAB_SIZE,
        max_len=CONTACT_INFER_MAX_LEN,
        d_model=CONTACT_D_MODEL,
        n_heads=CONTACT_N_HEADS,
        n_layers=CONTACT_N_LAYERS,
        d_ff=CONTACT_D_FF,
        dropout=CONTACT_DROPOUT,
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=CONTACT_LR, weight_decay=CONTACT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONTACT_EPOCHS)
    scaler = GradScaler(enabled=dev.type == "cuda")
    pos_weight = torch.tensor([pos_w], device=dev)

    best_val = float("inf")
    history = []

    for epoch in range(1, CONTACT_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        tr_prec = 0.0
        tr_rec = 0.0
        n = 0
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{CONTACT_EPOCHS}", leave=False):
            tokens = batch["tokens"].to(dev)
            mask = batch["mask"].to(dev)
            contact = batch["contact"].to(dev)
            dist = batch["dist"].to(dev)
            pair_mask = contact_pair_mask(mask, CONTACT_MIN_SEP)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=dev.type == "cuda"):
                logits, dist_pred = model(tokens, mask)
                stats = contact_loss(
                    logits,
                    dist_pred,
                    contact,
                    dist,
                    pair_mask,
                    pos_weight=pos_weight,
                    dist_weight=CONTACT_DIST_LOSS_W,
                )
                loss = stats["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            bs = tokens.size(0)
            tr_loss += float(loss.item()) * bs
            tr_prec += float(stats["prec"]) * bs
            tr_rec += float(stats["rec"]) * bs
            n += bs

        sched.step()
        tr_loss /= max(n, 1)
        tr_prec /= max(n, 1)
        tr_rec /= max(n, 1)

        model.eval()
        va_loss = 0.0
        va_prec = 0.0
        va_rec = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(dev)
                mask = batch["mask"].to(dev)
                contact = batch["contact"].to(dev)
                dist = batch["dist"].to(dev)
                pair_mask = contact_pair_mask(mask, CONTACT_MIN_SEP)
                with autocast(enabled=dev.type == "cuda"):
                    logits, dist_pred = model(tokens, mask)
                    stats = contact_loss(
                        logits,
                        dist_pred,
                        contact,
                        dist,
                        pair_mask,
                        pos_weight=pos_weight,
                        dist_weight=CONTACT_DIST_LOSS_W,
                    )
                bs = tokens.size(0)
                va_loss += float(stats["loss"].item()) * bs
                va_prec += float(stats["prec"]) * bs
                va_rec += float(stats["rec"]) * bs
                vn += bs
        va_loss /= max(vn, 1)
        va_prec /= max(vn, 1)
        va_rec /= max(vn, 1)

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_prec": tr_prec,
                "train_rec": tr_rec,
                "val_loss": va_loss,
                "val_prec": va_prec,
                "val_rec": va_rec,
                "sec": time.time() - t0,
            }
        )
        print(
            f"epoch {epoch:03d}  train {tr_loss:.4f} P/R {tr_prec:.3f}/{tr_rec:.3f}  "
            f"val {va_loss:.4f} P/R {va_prec:.3f}/{va_rec:.3f}  "
            f"lr {sched.get_last_lr()[0]:.2e}  ({history[-1]['sec']:.1f}s)",
            flush=True,
        )

        ckpt = {
            "model": model.state_dict(),
            "config": {
                "vocab_size": VOCAB_SIZE,
                "max_len": CONTACT_INFER_MAX_LEN,
                "d_model": CONTACT_D_MODEL,
                "n_heads": CONTACT_N_HEADS,
                "n_layers": CONTACT_N_LAYERS,
                "d_ff": CONTACT_D_FF,
                "dropout": CONTACT_DROPOUT,
                "min_sep": CONTACT_MIN_SEP,
                "pos_weight": pos_w,
            },
            "epoch": epoch,
            "val_loss": va_loss,
            "val_prec": va_prec,
            "val_rec": va_rec,
        }
        torch.save(ckpt, CKPT_DIR / "contact_last.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, CKPT_DIR / CONTACT_CKPT_NAME)
            print(f"  saved {CONTACT_CKPT_NAME} (val_loss={best_val:.5f})", flush=True)

    (CKPT_DIR / "contact_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Done. Best val_loss={best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()
