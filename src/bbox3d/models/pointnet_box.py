"""
Step 6 — PointNet box regression model (models/pointnet_box.py).

Architecture (from SPEC §2 diagram):
  Input: pts (B, P, 6)  xyz_canon + rgb   extra (B, 7)  log_n + extents + view
  Encoder: shared MLP  6 → 64 → 128 → 256 → 512  (Conv1d + BN + ReLU)
  Pooling: max-pool || avg-pool → (B, 1024)
  Concat extra → (B, 1031)
  Head: MLP  1028 → 512 → 256 → 12  (Linear + BN + ReLU + Dropout)
  Output: Δcenter(3), log_size(3), rot6d(6)  — all in canonical frame

Output decoding:
  pred_center = Δcenter
  pred_size   = exp(log_size)
  pred_R      = rot6d_to_matrix(rot6d)   (det +1 by construction)
  pred_corners = params_to_corners_torch(pred_center, pred_size, pred_R)

Initialization:
  Last linear weights ← zeros
  Bias: Δcenter=0, log_size=log_size_init (dataset median), rot6d=[1,0,0, 0,1,0] → R≈I
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from bbox3d.geometry.box import rot6d_to_matrix, params_to_corners_torch


# ---------------------------------------------------------------------------
# Encoder block helper
# ---------------------------------------------------------------------------

def _mlp_encoder(in_ch: int, channels: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    ch = in_ch
    for out_ch in channels:
        layers += [nn.Conv1d(ch, out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)]
        ch = out_ch
    return nn.Sequential(*layers)


def _mlp_head(in_ch: int, channels: list[int], out_ch: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    ch = in_ch
    for mid_ch in channels:
        layers += [nn.Linear(ch, mid_ch), nn.BatchNorm1d(mid_ch), nn.ReLU(inplace=True),
                   nn.Dropout(dropout)]
        ch = mid_ch
    layers.append(nn.Linear(ch, out_ch))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PointNetBoxHead(nn.Module):
    """
    PointNet-style encoder + MLP head for oriented bounding box regression.

    Predicts (Δcenter, log_size, rot6d) in the PCA-canonical frame.
    Call decode() to get (pred_center, pred_size, pred_R, pred_corners).
    """

    def __init__(
        self,
        in_ch: int = 6,
        mlp_channels: list[int] | None = None,
        head_channels: list[int] | None = None,
        dropout: float = 0.3,
        log_size_init: np.ndarray | list[float] | None = None,
        extra_dim: int = 7,
    ) -> None:
        super().__init__()
        if mlp_channels  is None: mlp_channels  = [64, 128, 256, 512]
        if head_channels is None: head_channels = [512, 256]

        self.encoder = _mlp_encoder(in_ch, mlp_channels)

        feat_dim  = mlp_channels[-1] * 2 + extra_dim  # max + avg + extra
        self.head = _mlp_head(feat_dim, head_channels, out_ch=12, dropout=dropout)

        self._init_output(log_size_init)

    def _init_output(self, log_size_init: np.ndarray | list[float] | None) -> None:
        """Zero last-layer weights; set bias so R≈I and size≈dataset median."""
        last = self.head[-1]          # final nn.Linear(?, 12)
        nn.init.zeros_(last.weight)
        bias = torch.zeros(12)
        if log_size_init is not None:
            ls = torch.as_tensor(log_size_init, dtype=torch.float32)
            bias[3:6] = ls[:3]
        # rot6d bias → identity rotation: first two columns of I
        bias[6:9]  = torch.tensor([1., 0., 0.])
        bias[9:12] = torch.tensor([0., 1., 0.])
        last.bias.data.copy_(bias)

    # ------------------------------------------------------------------
    def forward(self, pts: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """
        pts   : (B, P, 6)
        extra : (B, 7)
        returns: (B, 12)  — raw head output
        """
        x = pts.transpose(1, 2)             # (B, 6, P)
        x = self.encoder(x)                 # (B, 512, P)
        x_max = x.max(dim=2).values         # (B, 512)
        x_avg = x.mean(dim=2)               # (B, 512)
        feat  = torch.cat([x_max, x_avg, extra], dim=1)  # (B, 1031)
        return self.head(feat)              # (B, 12)

    def decode(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode raw head output into box parameters + corners.

        raw : (B, 12)
        returns:
          pred_center  (B, 3)    Δcenter in canonical frame
          pred_size    (B, 3)    positive edge lengths
          pred_R       (B, 3, 3) proper rotation (det +1)
          pred_corners (B, 8, 3) predicted box corners in canonical frame
        """
        pred_center = raw[:, :3]
        pred_size   = torch.exp(raw[:, 3:6])
        pred_R      = rot6d_to_matrix(raw[:, 6:12])
        pred_corners = params_to_corners_torch(pred_center, pred_size, pred_R)
        return pred_center, pred_size, pred_R, pred_corners

    def forward_decode(
        self, pts: torch.Tensor, extra: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience: forward + decode in one call."""
        return self.decode(self.forward(pts, extra))


# ---------------------------------------------------------------------------
# Factory from config
# ---------------------------------------------------------------------------

def build_model(cfg: dict, log_size_init: np.ndarray | None = None):
    """Build model from config. Dispatches on cfg['model']['type'] (default: 'pointnet')."""
    m = cfg.get("model", {})
    model_type = m.get("type", "pointnet")

    if model_type == "pointnetpp":
        from bbox3d.models.pointnetpp_box import build_model_pp
        return build_model_pp(cfg, log_size_init)

    extra_dim = int(cfg.get("dataset", {}).get("extra_dim", 7))
    return PointNetBoxHead(
        in_ch        = 6,
        mlp_channels = m.get("mlp_channels",  [64, 128, 256, 512]),
        head_channels= m.get("head_channels", [512, 256]),
        dropout      = m.get("dropout",       0.3),
        log_size_init= log_size_init,
        extra_dim    = extra_dim,
    )
