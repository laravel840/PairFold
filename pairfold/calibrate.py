"""Calibrate model confidence against real angular error on held-out PDB fragments."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    BATCH_SIZE,
    CALIB_DIR,
    CKPT_DIR,
    CONF_ANGLE_THRESH_DEG,
    FRAG_DIR,
    NUM_WORKERS,
)
from .data.dataset import FragmentDataset, load_jsonl
from .model.fragment_net import FragmentTorsionNet, sincos_to_angles


def circular_abs_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Smallest absolute difference between angles in radians → degrees."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d.abs() * (180.0 / math.pi)


def load_model(path: Path, device: torch.device) -> FragmentTorsionNet:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = FragmentTorsionNet(
        vocab_size=cfg["vocab_size"],
        max_len=cfg["max_len"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def collect_pairs(model, loader, device):
    raw_conf = []
    correct = []
    mean_err = []
    for batch in tqdm(loader, desc="Calibration eval"):
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        ang_mask = batch["ang_mask"].to(device)
        pred, log_sigma, conf = model(tokens, mask)
        pred_phi, pred_psi = sincos_to_angles(pred)
        tgt_phi, tgt_psi = sincos_to_angles(target)

        err_phi = circular_abs_deg(pred_phi, tgt_phi)
        err_psi = circular_abs_deg(pred_psi, tgt_psi)
        # residue correct if both available angles within threshold
        phi_ok = ang_mask[..., 0]
        psi_ok = ang_mask[..., 2]
        phi_good = (~phi_ok) | (err_phi <= CONF_ANGLE_THRESH_DEG)
        psi_good = (~psi_ok) | (err_psi <= CONF_ANGLE_THRESH_DEG)
        is_correct = (phi_good & psi_good & mask).cpu().numpy()
        conf_np = conf.cpu().numpy()
        mask_np = mask.cpu().numpy()
        err = ((err_phi + err_psi) / 2.0).cpu().numpy()

        for b in range(conf_np.shape[0]):
            for i in range(conf_np.shape[1]):
                if not mask_np[b, i]:
                    continue
                raw_conf.append(float(conf_np[b, i]))
                correct.append(1.0 if is_correct[b, i] else 0.0)
                mean_err.append(float(err[b, i]))
    return (
        np.asarray(raw_conf, dtype=np.float64),
        np.asarray(correct, dtype=np.float64),
        np.asarray(mean_err, dtype=np.float64),
    )


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """
    Fit scalar T for sigmoid(logit/T).
    conf = sigmoid(z) ⇒ logit z = log(c/(1-c)).
    """
    eps = 1e-6
    c = np.clip(logits, eps, 1 - eps)
    z = np.log(c / (1 - c))

    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.25, 5.0, 96):
        p = 1.0 / (1.0 + np.exp(-z / t))
        p = np.clip(p, eps, 1 - eps)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def fit_platt(raw: np.ndarray, y: np.ndarray):
    """Simple 1D logistic: p = sigmoid(a * logit(raw) + b)."""
    eps = 1e-6
    c = np.clip(raw, eps, 1 - eps)
    z = np.log(c / (1 - c))
    # grid search a,b
    best = (1.0, 0.0, float("inf"))
    for a in np.linspace(0.2, 3.0, 40):
        for b in np.linspace(-2.0, 2.0, 41):
            p = 1.0 / (1.0 + np.exp(-(a * z + b)))
            p = np.clip(p, eps, 1 - eps)
            nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if nll < best[2]:
                best = (float(a), float(b), float(nll))
    return {"a": best[0], "b": best[1], "nll": best[2]}


def apply_platt(raw: np.ndarray, a: float, b: float) -> np.ndarray:
    eps = 1e-6
    c = np.clip(raw, eps, 1 - eps)
    z = np.log(c / (1 - c))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        bins.append(
            {
                "bin": i,
                "count": int(m.sum()),
                "conf_mean": float(p[m].mean()),
                "accuracy": float(y[m].mean()),
            }
        )
    return bins


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    n = len(p)
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(total)


def main() -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    val_path = FRAG_DIR / "val_split.jsonl"
    if not val_path.exists():
        # fallback: rebuild split from full set is not available — use last 10% of fragments
        rows = load_jsonl(FRAG_DIR / "fragments.jsonl")
        n_val = max(1, int(len(rows) * 0.1))
        rows = rows[:n_val]
        print(f"No val_split.jsonl — using first {n_val} rows as proxy", flush=True)
    else:
        rows = load_jsonl(val_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = CKPT_DIR / "best.pt"
    if not ckpt.exists():
        raise SystemExit("Missing checkpoints/best.pt — train first")
    model = load_model(ckpt, device)

    loader = DataLoader(
        FragmentDataset(rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    raw, y, errs = collect_pairs(model, loader, device)
    print(f"Calibration samples: {len(raw)}", flush=True)
    print(
        f"Raw ECE={ece(raw, y):.4f}  accuracy={y.mean():.4f}  "
        f"mean_err_deg={errs.mean():.2f}",
        flush=True,
    )

    T = fit_temperature(raw, y)
    # temperature on logits of raw conf
    eps = 1e-6
    z = np.log(np.clip(raw, eps, 1 - eps) / (1 - np.clip(raw, eps, 1 - eps)))
    temp_p = 1.0 / (1.0 + np.exp(-z / T))

    platt = fit_platt(raw, y)
    platt_p = apply_platt(raw, platt["a"], platt["b"])

    report = {
        "threshold_deg": CONF_ANGLE_THRESH_DEG,
        "n_samples": int(len(raw)),
        "raw": {
            "ece": ece(raw, y),
            "accuracy": float(y.mean()),
            "mean_angular_error_deg": float(errs.mean()),
            "reliability": reliability_bins(raw, y),
        },
        "temperature": {
            "T": T,
            "ece": ece(temp_p, y),
            "reliability": reliability_bins(temp_p, y),
        },
        "platt": {
            **platt,
            "ece": ece(platt_p, y),
            "reliability": reliability_bins(platt_p, y),
        },
        "chosen": "platt" if ece(platt_p, y) <= ece(temp_p, y) else "temperature",
    }
    # Persist params for inference
    params = {
        "method": report["chosen"],
        "threshold_deg": CONF_ANGLE_THRESH_DEG,
        "temperature_T": T,
        "platt_a": platt["a"],
        "platt_b": platt["b"],
        "report": report,
    }
    out = CALIB_DIR / "confidence_calibration.json"
    out.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("raw", "temperature", "platt", "chosen")}, indent=2))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
