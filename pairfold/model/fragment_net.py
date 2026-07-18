"""Sequence → backbone torsion model with learned uncertainty."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FragmentTorsionNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 160,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 320,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.max_len = max_len
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
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),  # sinφ, cosφ, sinψ, cosψ
        )
        # log σ for φ and ψ (homoscedastic per-residue, two channels)
        self.log_sigma = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
          tors_sc: (B, L, 4)
          log_sigma: (B, L, 2) for (φ, ψ)
          conf: (B, L) derived soft confidence in (0,1)
        """
        b, l = tokens.shape
        pos_ids = torch.arange(l, device=tokens.device).unsqueeze(0).expand(b, -1)
        x = self.embed(tokens) + self.pos(pos_ids)
        pad_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        tors_sc = self.head(x)
        phi = tors_sc[..., 0:2]
        psi = tors_sc[..., 2:4]
        phi = phi / (phi.norm(dim=-1, keepdim=True) + 1e-6)
        psi = psi / (psi.norm(dim=-1, keepdim=True) + 1e-6)
        tors_sc = torch.cat([phi, psi], dim=-1)
        log_sigma = self.log_sigma(x).clamp(-4.0, 2.0)
        # conf ↓ when σ ↑
        conf = torch.sigmoid(-log_sigma.mean(dim=-1))
        return tors_sc, log_sigma, conf


def angles_to_sincos(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi)], dim=-1
    )


def sincos_to_angles(sc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    phi = torch.atan2(sc[..., 0], sc[..., 1])
    psi = torch.atan2(sc[..., 2], sc[..., 3])
    return phi, psi


def gaussian_nll_sincos(
    pred_sc: torch.Tensor,
    target_sc: torch.Tensor,
    log_sigma: torch.Tensor,
    mask: torch.Tensor,
    angle_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian NLL on sin/cos pairs.
    log_sigma: (B,L,2) → expand to 4 channels (φ,φ,ψ,ψ).
    """
    if angle_mask.dim() == 2:
        angle_mask = angle_mask.unsqueeze(-1).expand_as(pred_sc)
    valid = mask.unsqueeze(-1) & angle_mask
    ls = torch.stack(
        [log_sigma[..., 0], log_sigma[..., 0], log_sigma[..., 1], log_sigma[..., 1]],
        dim=-1,
    )
    inv_var = torch.exp(-2.0 * ls)
    nll = 0.5 * inv_var * (pred_sc - target_sc) ** 2 + ls
    denom = valid.float().sum().clamp_min(1.0)
    return (nll * valid.float()).sum() / denom


def torsion_mse(
    pred_sc: torch.Tensor,
    target_sc: torch.Tensor,
    mask: torch.Tensor,
    angle_mask: torch.Tensor,
) -> torch.Tensor:
    if angle_mask.dim() == 2:
        angle_mask = angle_mask.unsqueeze(-1).expand_as(pred_sc)
    valid = mask.unsqueeze(-1) & angle_mask
    diff = (pred_sc - target_sc) ** 2
    denom = valid.float().sum().clamp_min(1.0)
    return (diff * valid.float()).sum() / denom


# Back-compat alias
def torsion_loss(pred_sc, target_sc, mask, angle_mask):
    return torsion_mse(pred_sc, target_sc, mask, angle_mask)
