"""Lightweight sequence → long-range Cα contact / distance model."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactPairNet(nn.Module):
    """
    AA embed + 1D Pre-LN Transformer → outer-product pair features →
    small 2D tower → contact logit + distance (Å) per residue pair.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 96,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.12,
        pair_dim: int = 64,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.pair_dim = pair_dim

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=vocab_size - 1)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Project to pair factors for outer product
        self.to_left = nn.Linear(d_model, pair_dim)
        self.to_right = nn.Linear(d_model, pair_dim)
        # Relative sequence separation embedding (clamped)
        self.sep_embed = nn.Embedding(max_len, pair_dim)

        self.pair_proj = nn.Sequential(
            nn.Linear(pair_dim * 3, pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Light 2D conv refinement
        self.pair_cnn = nn.Sequential(
            nn.Conv2d(pair_dim, pair_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(pair_dim, pair_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.contact_head = nn.Linear(pair_dim, 1)
        self.dist_head = nn.Linear(pair_dim, 1)

    def _pair_features(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, L, D) → pair: (B, L, L, P)
        """
        b, l, _ = h.shape
        left = self.to_left(h)  # (B, L, P)
        right = self.to_right(h)
        # Outer product via broadcast: (B, L, 1, P) * (B, 1, L, P)
        outer = left.unsqueeze(2) * right.unsqueeze(1)

        idx = torch.arange(l, device=h.device)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().clamp(max=self.max_len - 1)
        sep_e = self.sep_embed(sep).unsqueeze(0).expand(b, -1, -1, -1)

        # Also include additive pair (i+j)/2 style
        add = 0.5 * (left.unsqueeze(2) + right.unsqueeze(1))
        feat = torch.cat([outer, add, sep_e], dim=-1)
        feat = self.pair_proj(feat)  # (B, L, L, P)
        # CNN expects (B, C, H, W)
        x = feat.permute(0, 3, 1, 2)
        x = x + self.pair_cnn(x)
        return x.permute(0, 2, 3, 1)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: (B, L) long
        mask: (B, L) bool, True = valid residue

        returns:
          contact_logits: (B, L, L)
          dist_pred: (B, L, L) positive Å predictions
        """
        b, l = tokens.shape
        pos_ids = torch.arange(l, device=tokens.device).unsqueeze(0).expand(b, -1)
        pos_ids = pos_ids.clamp(max=self.max_len - 1)
        x = self.embed(tokens) + self.pos(pos_ids)
        pad_mask = ~mask
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        pair = self._pair_features(h)
        logits = self.contact_head(pair).squeeze(-1)
        # Symmetrize
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        dist = F.softplus(self.dist_head(pair).squeeze(-1)) + 1.0
        dist = 0.5 * (dist + dist.transpose(-1, -2))
        return logits, dist


def contact_pair_mask(
    length_mask: torch.Tensor, min_sep: int
) -> torch.Tensor:
    """
    length_mask: (B, L) bool
    returns pair_mask (B, L, L) True where both residues valid and |i-j| >= min_sep
    """
    b, l = length_mask.shape
    valid = length_mask.unsqueeze(2) & length_mask.unsqueeze(1)
    idx = torch.arange(l, device=length_mask.device)
    sep_ok = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= min_sep
    # Upper triangle only for loss (avoid double-counting)
    upper = idx.unsqueeze(0) < idx.unsqueeze(1)
    return valid & sep_ok.unsqueeze(0) & upper.unsqueeze(0)


def contact_loss(
    logits: torch.Tensor,
    dist_pred: torch.Tensor,
    contact_target: torch.Tensor,
    dist_target: torch.Tensor,
    pair_mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    dist_weight: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """
    contact_target: (B, L, L) float {0,1}
    dist_target: (B, L, L) Å (meaningful on positives)
    pair_mask: (B, L, L) bool
    """
    if not pair_mask.any():
        zero = logits.sum() * 0.0
        return {"loss": zero, "bce": zero, "dist": zero, "prec": zero, "rec": zero}

    logits_m = logits[pair_mask]
    target_m = contact_target[pair_mask]
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits_m, target_m, pos_weight=pos_weight.to(logits_m.dtype)
        )
    else:
        bce = F.binary_cross_entropy_with_logits(logits_m, target_m)

    pos = pair_mask & (contact_target > 0.5)
    if pos.any():
        dist_l = F.smooth_l1_loss(dist_pred[pos], dist_target[pos])
    else:
        dist_l = logits.sum() * 0.0

    loss = bce + dist_weight * dist_l

    with torch.no_grad():
        pred = (torch.sigmoid(logits_m) >= 0.5).float()
        tp = (pred * target_m).sum()
        prec = tp / pred.sum().clamp_min(1.0)
        rec = tp / target_m.sum().clamp_min(1.0)

    return {"loss": loss, "bce": bce.detach(), "dist": dist_l.detach(), "prec": prec, "rec": rec}
