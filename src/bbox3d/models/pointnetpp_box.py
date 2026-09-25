"""
PointNet++ box regression model — lightweight 2 SA-layer variant (CPU-friendly).

Architecture:
  SA1: FPS(P→128), kNN(k=32), shared MLP(in_ch+3 → 64 → 64 → 128), max-pool
  SA2: FPS(128→32), kNN(k=32), shared MLP(128+3  → 128 → 128 → 256), max-pool
  Global: per-point MLP(256+3 → 256 → 512), max-pool ‖ avg-pool → (1024)
  Concat extra(7) → (1031)
  Head MLP: 1031 → 512 → 256 → 12
  Output: Δcenter(3), log_size(3), rot6d(6) in canonical frame

Same forward/decode interface as PointNetBoxHead so train.py is unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from bbox3d.geometry.box import rot6d_to_matrix, params_to_corners_torch


# ---------------------------------------------------------------------------
# FPS
# ---------------------------------------------------------------------------

def _fps(xyz: torch.Tensor, npoints: int) -> torch.Tensor:
    """Farthest point sampling. xyz: (B, N, 3) → indices (B, npoints)."""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoints, dtype=torch.long, device=device)
    dist      = torch.full((B, N), 1e10, device=device)
    farthest  = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(npoints):
        centroids[:, i] = farthest
        c = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)  # (B,1,3)
        d = ((xyz - c) ** 2).sum(-1)                                      # (B,N)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(1)

    return centroids  # (B, npoints)


# ---------------------------------------------------------------------------
# kNN grouping
# ---------------------------------------------------------------------------

def _group_knn(xyz: torch.Tensor, queries: torch.Tensor,
               feats: torch.Tensor, k: int) -> torch.Tensor:
    """
    xyz:     (B, N, 3)
    queries: (B, M, 3)  centroid positions (subset of xyz)
    feats:   (B, N, C)  per-point features

    Returns grouped (B, M, k, C+3):  [rel_xyz ‖ feats_k]
    """
    B, M, _ = queries.shape
    device = xyz.device

    # pairwise distances, take k nearest
    dist = torch.cdist(queries, xyz)                         # (B, M, N)
    _, idx = dist.topk(k, dim=2, largest=False)              # (B, M, k)

    # gather neighbors via advanced indexing
    b_idx    = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, k)
    nbr_xyz  = xyz[b_idx, idx]                               # (B, M, k, 3)
    rel_xyz  = nbr_xyz - queries.unsqueeze(2)                # (B, M, k, 3)
    nbr_feat = feats[b_idx, idx]                             # (B, M, k, C)

    return torch.cat([rel_xyz, nbr_feat], dim=-1)            # (B, M, k, C+3)


# ---------------------------------------------------------------------------
# SA layer
# ---------------------------------------------------------------------------

class _ConvBNReLU2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class SetAbstraction(nn.Module):
    """FPS → kNN grouping → shared MLP per group → max-pool."""

    def __init__(self, npoints: int, k: int,
                 in_feat_ch: int, mlp_ch: list[int]) -> None:
        super().__init__()
        self.npoints = npoints
        self.k       = k

        layers: list[nn.Module] = []
        ch = in_feat_ch + 3          # +3 for relative xyz
        for out_ch in mlp_ch:
            layers.append(_ConvBNReLU2d(ch, out_ch))
            ch = out_ch
        self.mlp    = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, xyz: torch.Tensor,
                feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        xyz:   (B, N, 3)
        feats: (B, N, C)
        returns: new_xyz (B, M, 3),  new_feats (B, M, out_ch)
        """
        B      = xyz.shape[0]
        device = xyz.device
        M      = self.npoints

        idx   = _fps(xyz, M)                                         # (B, M)
        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, M)
        new_xyz = xyz[b_idx, idx]                                    # (B, M, 3)

        grouped = _group_knn(xyz, new_xyz, feats, self.k)            # (B,M,k,C+3)

        # Conv2d layout: (B, C+3, k, M)
        x = grouped.permute(0, 3, 2, 1)    # (B, C+3, k, M)
        x = self.mlp(x)                    # (B, out_ch, k, M)
        new_feats = x.max(dim=2).values    # (B, out_ch, M)
        new_feats = new_feats.permute(0, 2, 1)  # (B, M, out_ch)

        return new_xyz, new_feats


# ---------------------------------------------------------------------------
# Helpers reused from pointnet_box
# ---------------------------------------------------------------------------

def _mlp1d(in_ch: int, channels: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    ch = in_ch
    for out_ch in channels:
        layers += [nn.Conv1d(ch, out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)]
        ch = out_ch
    return nn.Sequential(*layers)


def _mlp_head(in_ch: int, channels: list[int], out_ch: int,
              dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    ch = in_ch
    for mid_ch in channels:
        layers += [nn.Linear(ch, mid_ch), nn.BatchNorm1d(mid_ch),
                   nn.ReLU(inplace=True), nn.Dropout(dropout)]
        ch = mid_ch
    layers.append(nn.Linear(ch, out_ch))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PointNetPPBoxHead(nn.Module):
    """
    PointNet++ (2 SA layers) encoder + MLP head for oriented 3D box regression.

    Predicts (Δcenter, log_size, rot6d) in PCA-canonical frame.
    Identical call interface to PointNetBoxHead.
    """

    def __init__(
        self,
        in_ch: int = 6,
        sa1_npoints: int = 128,
        sa1_k: int = 32,
        sa1_mlp: list[int] | None = None,
        sa2_npoints: int = 32,
        sa2_k: int = 32,
        sa2_mlp: list[int] | None = None,
        global_mlp: list[int] | None = None,
        head_channels: list[int] | None = None,
        dropout: float = 0.3,
        log_size_init: np.ndarray | list[float] | None = None,
    ) -> None:
        super().__init__()
        if sa1_mlp       is None: sa1_mlp       = [64, 64, 128]
        if sa2_mlp       is None: sa2_mlp       = [128, 128, 256]
        if global_mlp    is None: global_mlp    = [256, 512]
        if head_channels is None: head_channels = [512, 256]

        self.sa1 = SetAbstraction(sa1_npoints, sa1_k, in_ch,        sa1_mlp)
        self.sa2 = SetAbstraction(sa2_npoints, sa2_k, sa1_mlp[-1],  sa2_mlp)

        # Global MLP applied pointwise to SA2 output (xyz concatenated)
        self.global_enc = _mlp1d(sa2_mlp[-1] + 3, global_mlp)

        feat_dim = global_mlp[-1] * 2 + 7   # max ‖ avg + extra(7)
        self.head = _mlp_head(feat_dim, head_channels, out_ch=12, dropout=dropout)

        self._init_output(log_size_init)

    def _init_output(self, log_size_init) -> None:
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        bias = torch.zeros(12)
        if log_size_init is not None:
            ls = torch.as_tensor(log_size_init, dtype=torch.float32)
            bias[3:6] = ls[:3]
        bias[6:9]  = torch.tensor([1., 0., 0.])
        bias[9:12] = torch.tensor([0., 1., 0.])
        last.bias.data.copy_(bias)

    def forward(self, pts: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """
        pts:   (B, P, 6)   xyz_canon + rgb in canonical frame
        extra: (B, 7)      [log_npts, extent_x, extent_y, extent_z, view_x, view_y, view_z]
        returns: (B, 12)   raw head output
        """
        xyz   = pts[:, :, :3]   # (B, P, 3)
        feats = pts             # (B, P, 6)  all 6 channels as SA input features

        xyz1, f1 = self.sa1(xyz,  feats)    # (B, 128, 3), (B, 128, 128)
        xyz2, f2 = self.sa2(xyz1, f1)       # (B,  32, 3), (B,  32, 256)

        # Append xyz to features before global MLP
        x = torch.cat([xyz2, f2], dim=-1)   # (B, 32, 259)
        x = x.transpose(1, 2)              # (B, 259, 32)
        x = self.global_enc(x)             # (B, 512, 32)
        x_max = x.max(dim=2).values        # (B, 512)
        x_avg = x.mean(dim=2)              # (B, 512)
        feat  = torch.cat([x_max, x_avg, extra], dim=1)  # (B, 1031)

        return self.head(feat)             # (B, 12)

    def decode(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_center  = raw[:, :3]
        pred_size    = torch.exp(raw[:, 3:6])
        pred_R       = rot6d_to_matrix(raw[:, 6:12])
        pred_corners = params_to_corners_torch(pred_center, pred_size, pred_R)
        return pred_center, pred_size, pred_R, pred_corners

    def forward_decode(
        self, pts: torch.Tensor, extra: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decode(self.forward(pts, extra))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model_pp(cfg: dict,
                   log_size_init: np.ndarray | None = None) -> PointNetPPBoxHead:
    m = cfg.get("model", {})
    return PointNetPPBoxHead(
        in_ch        = 6,
        sa1_npoints  = m.get("sa1_npoints",  128),
        sa1_k        = m.get("sa1_k",         32),
        sa1_mlp      = m.get("sa1_mlp",       [64, 64, 128]),
        sa2_npoints  = m.get("sa2_npoints",   32),
        sa2_k        = m.get("sa2_k",          32),
        sa2_mlp      = m.get("sa2_mlp",       [128, 128, 256]),
        global_mlp   = m.get("global_mlp",    [256, 512]),
        head_channels= m.get("head_channels", [512, 256]),
        dropout      = m.get("dropout",       0.3),
        log_size_init= log_size_init,
    )
