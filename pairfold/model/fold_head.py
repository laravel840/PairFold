"""Learned Cα distance head on frozen ESM residue embeddings."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESMFoldHead(nn.Module):
    """
    Frozen ESM emb (B,L,E) → pair features → distance (Å) + pair confidence.

    Small enough for GTX 1650; ESM itself is not inside this module.
    """

    def __init__(
        self,
        emb_dim: int = 480,
        d_model: int = 96,
        pair_dim: int = 48,
        max_len: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.to_left = nn.Linear(d_model, pair_dim)
        self.to_right = nn.Linear(d_model, pair_dim)
        self.sep_embed = nn.Embedding(max_len, pair_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim * 3, pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
        )
        self.dist_head = nn.Linear(pair_dim, 1)
        self.conf_head = nn.Linear(pair_dim, 1)

    def forward(
        self, emb: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        emb: (B, L, E)
        mask: (B, L) bool, True = valid
        returns dist (B,L,L) Å, conf (B,L,L) in (0,1)
        """
        h = self.proj(emb)
        b, l, _ = h.shape
        left = self.to_left(h)
        right = self.to_right(h)
        outer = left.unsqueeze(2) * right.unsqueeze(1)
        add = 0.5 * (left.unsqueeze(2) + right.unsqueeze(1))
        idx = torch.arange(l, device=h.device)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().clamp(max=self.max_len - 1)
        sep_e = self.sep_embed(sep).unsqueeze(0).expand(b, -1, -1, -1)
        feat = self.pair_mlp(torch.cat([outer, add, sep_e], dim=-1))
        dist = F.softplus(self.dist_head(feat).squeeze(-1)) + 2.0
        conf_logit = self.conf_head(feat).squeeze(-1)
        # Symmetrize
        dist = 0.5 * (dist + dist.transpose(-1, -2))
        conf_logit = 0.5 * (conf_logit + conf_logit.transpose(-1, -2))
        if mask is not None:
            pair = mask.unsqueeze(1) & mask.unsqueeze(2)
            dist = dist * pair.float()
            conf_logit = conf_logit.masked_fill(~pair, 0.0)
        return dist, conf_logit


def fold_distance_loss(
    dist_pred: torch.Tensor,
    dist_true: torch.Tensor,
    conf_logit: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Huber on distances + mild confidence calibration vs |error|."""
    err = F.smooth_l1_loss(dist_pred, dist_true, reduction="none")
    w = pair_mask.float()
    # Emphasize mid/long range and near-contact pairs
    n = dist_true.shape[-1]
    idx = torch.arange(n, device=dist_true.device)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    sep_w = (sep >= 3).float()
    near = (dist_true < 12.0).float()
    long = (sep >= 6).float()
    w = w * sep_w * (1.0 + 1.5 * near + 0.75 * long)
    dist_loss = (err * w).sum() / w.sum().clamp_min(1.0)
    # Confidence should be high when error is small (AMP-safe logits loss)
    target_conf = torch.exp(-err.detach() / 4.0)
    conf_loss = F.binary_cross_entropy_with_logits(conf_logit, target_conf, reduction="none")
    conf_loss = (conf_loss * w).sum() / w.sum().clamp_min(1.0)
    return dist_loss + 0.15 * conf_loss
