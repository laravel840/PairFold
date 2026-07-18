"""Train fragment torsion model with Gaussian NLL uncertainty (GPU)."""

from __future__ import annotations

import json
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    BATCH_SIZE,
    CKPT_DIR,
    D_FF,
    D_MODEL,
    DROPOUT,
    EPOCHS,
    FRAG_DIR,
    LR,
    MAX_LEN,
    N_HEADS,
    N_LAYERS,
    NUM_WORKERS,
    SEED,
    VAL_FRAC,
    VOCAB_SIZE,
    WEIGHT_DECAY,
)
from .data.dataset import FragmentDataset, load_jsonl, split_rows
from .model.fragment_net import FragmentTorsionNet, gaussian_nll_sincos, torsion_mse


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    torch.manual_seed(SEED)
    frag_path = FRAG_DIR / "fragments.jsonl"
    if not frag_path.exists():
        raise SystemExit("Missing fragments. Run fetch + extract first.")

    rows = load_jsonl(frag_path)
    train_rows, val_rows = split_rows(rows, VAL_FRAC, SEED)
    print(f"Train {len(train_rows)} | Val {len(val_rows)}", flush=True)

    train_loader = DataLoader(
        FragmentDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        FragmentDataset(val_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Persist val split for calibration (same seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    val_path = FRAG_DIR / "val_split.jsonl"
    with val_path.open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote validation split -> {val_path}", flush=True)

    dev = device()
    print(f"Device: {dev}", flush=True)
    if dev.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)

    model = FragmentTorsionNet(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = GradScaler(enabled=dev.type == "cuda")

    best_val = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        n = 0
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}", leave=False):
            tokens = batch["tokens"].to(dev)
            mask = batch["mask"].to(dev)
            target = batch["target"].to(dev)
            ang_mask = batch["ang_mask"].to(dev)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=dev.type == "cuda"):
                pred, log_sigma, _ = model(tokens, mask)
                loss = gaussian_nll_sincos(pred, target, log_sigma, mask, ang_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * tokens.size(0)
            n += tokens.size(0)

        sched.step()
        tr_loss /= max(n, 1)

        model.eval()
        va_nll = 0.0
        va_mse = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(dev)
                mask = batch["mask"].to(dev)
                target = batch["target"].to(dev)
                ang_mask = batch["ang_mask"].to(dev)
                with autocast(enabled=dev.type == "cuda"):
                    pred, log_sigma, _ = model(tokens, mask)
                    nll = gaussian_nll_sincos(pred, target, log_sigma, mask, ang_mask)
                    mse = torsion_mse(pred, target, mask, ang_mask)
                va_nll += nll.item() * tokens.size(0)
                va_mse += mse.item() * tokens.size(0)
                vn += tokens.size(0)
        va_nll /= max(vn, 1)
        va_mse /= max(vn, 1)

        history.append(
            {
                "epoch": epoch,
                "train_nll": tr_loss,
                "val_nll": va_nll,
                "val_mse": va_mse,
                "sec": time.time() - t0,
            }
        )
        print(
            f"epoch {epoch:03d}  train_nll {tr_loss:.5f}  val_nll {va_nll:.5f}  "
            f"val_mse {va_mse:.5f}  lr {sched.get_last_lr()[0]:.2e}  ({history[-1]['sec']:.1f}s)",
            flush=True,
        )

        ckpt = {
            "model": model.state_dict(),
            "config": {
                "vocab_size": VOCAB_SIZE,
                "max_len": MAX_LEN,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "dropout": DROPOUT,
                "uncertainty": "log_sigma_phi_psi",
            },
            "epoch": epoch,
            "val_nll": va_nll,
            "val_mse": va_mse,
        }
        torch.save(ckpt, CKPT_DIR / "last.pt")
        if va_mse < best_val:
            best_val = va_mse
            torch.save(ckpt, CKPT_DIR / "best.pt")
            print(f"  saved best.pt (val_mse={best_val:.5f})", flush=True)

    (CKPT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Done. Best val_mse={best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()
