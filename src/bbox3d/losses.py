"""
Step 6 — Bounding-box losses (losses.py).

L_total = w_corner * L_corner + w_center * L_center + w_size * L_size

L_corner : min_{σ∈SYM_PERMS} mean_k SmoothL1(pred_corner_k, gt_corner_σ(k))
           symmetry-aware — handles the 24 equivalent corner labelings of a box.
L_center : SmoothL1(pred_center, gt_center)
L_size   : SmoothL1(pred_size, gt_size_under_argmin_σ)
           argmin σ is DETACHED — no gradient through permutation choice.

SmoothL1 beta = 0.002 m (user spec; thin objects can be 2.5 mm).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bbox3d.geometry.box import SYM_PERMS, get_sym_perms_torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batch_edge_lengths(corners: torch.Tensor) -> torch.Tensor:
    """
    Extract edge lengths [|u|, |v|, |w|] from a batch of (B, 8, 3) corner arrays.

    Uses the standard convention: u=c1-c0, v=c3-c0, w=c4-c0.
    Returns (B, 3).
    """
    u = corners[:, 1] - corners[:, 0]
    v = corners[:, 3] - corners[:, 0]
    w = corners[:, 4] - corners[:, 0]
    return torch.stack([u.norm(dim=1), v.norm(dim=1), w.norm(dim=1)], dim=1)


def _symmetric_corner_loss(
    pred_corners: torch.Tensor,   # (B, 8, 3)
    gt_corners:   torch.Tensor,   # (B, 8, 3)
    sym_perms:    torch.Tensor,   # (24, 8) long  — on same device
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetry-aware SmoothL1 corner loss.

    For each item in the batch, finds the best-matching permutation σ* in
    SYM_PERMS (the 24 proper rotations of the cube) and computes the mean
    SmoothL1 distance between pred and gt under σ*.

    Returns (mean_loss_over_batch (scalar), argmin_perm_indices (B,) long).
    """
    # gt under all 24 permutations: (B, 24, 8, 3)
    gt_perm = gt_corners[:, sym_perms, :]                        # (B, 24, 8, 3)
    pred_ex = pred_corners.unsqueeze(1).expand_as(gt_perm)       # (B, 24, 8, 3)

    # Per-element SmoothL1, then mean over corners+xyz → (B, 24)
    costs = F.smooth_l1_loss(pred_ex, gt_perm, beta=beta, reduction="none")
    costs = costs.mean(dim=(-2, -1))                             # (B, 24)

    min_costs, argmin = costs.min(dim=1)                         # (B,), (B,)
    return min_costs.mean(), argmin


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------

class BBoxLoss(nn.Module):
    """
    Total box regression loss.

    forward(pred_corners, pred_center, pred_size, gt_corners) → (total, components)

    components is a dict: {"corner", "center", "size", "total"} with detached scalars.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        loss_cfg = cfg.get("loss", {})
        self.corner_w = float(loss_cfg.get("corner_weight",  1.0))
        self.center_w = float(loss_cfg.get("center_weight",  1.0))
        self.size_w   = float(loss_cfg.get("size_weight",    0.5))
        self.beta     = float(loss_cfg.get("smooth_l1_beta", 0.002))

        # Symmetry permutations (24, 8) — moved with .to(device) via register_buffer
        self.register_buffer("sym_perms", torch.from_numpy(SYM_PERMS).long())

    def forward(
        self,
        pred_corners: torch.Tensor,   # (B, 8, 3) — canonical frame
        pred_center:  torch.Tensor,   # (B, 3)
        pred_size:    torch.Tensor,   # (B, 3) — positive
        gt_corners:   torch.Tensor,   # (B, 8, 3) — canonical frame
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        B = pred_corners.shape[0]

        # Corner loss + best permutation
        L_corner, argmin = _symmetric_corner_loss(
            pred_corners, gt_corners, self.sym_perms, self.beta
        )

        # Center loss
        gt_center = gt_corners.mean(dim=1)                       # (B, 3)
        L_center  = F.smooth_l1_loss(pred_center, gt_center, beta=self.beta)

        # Size loss under argmin permutation (detached)
        argmin_det = argmin.detach()
        best_perm  = self.sym_perms[argmin_det]                  # (B, 8)
        b_idx      = torch.arange(B, device=gt_corners.device).unsqueeze(1).expand(B, 8)
        gt_best    = gt_corners[b_idx, best_perm, :]             # (B, 8, 3)
        gt_size    = _batch_edge_lengths(gt_best).detach()       # (B, 3) — detached
        L_size     = F.smooth_l1_loss(pred_size, gt_size, beta=self.beta)

        total = self.corner_w * L_corner + self.center_w * L_center + self.size_w * L_size

        components = {
            "corner": L_corner.detach(),
            "center": L_center.detach(),
            "size":   L_size.detach(),
            "total":  total.detach(),
        }
        return total, components


# ---------------------------------------------------------------------------
# Dataset prior utility
# ---------------------------------------------------------------------------

def compute_log_size_prior(
    train_items: list[dict],
    cache_dir,
    n_samples: int = 500,
) -> np.ndarray:
    """
    Compute log of median GT edge lengths over a sample of training instances.

    Returns (3,) float64 array suitable for PointNetBoxHead log_size_init.
    Sizes are in camera frame (same magnitude as canonical frame after PCA).
    """
    from pathlib import Path
    from bbox3d.geometry.box import corners_to_params

    cache_dir = Path(cache_dir)
    rng   = np.random.default_rng(0)
    items = list(train_items)
    rng.shuffle(items)
    items = items[:n_samples]

    sizes = []
    for item in items:
        p = cache_dir / item["npz"]
        if not p.exists():
            continue
        d  = np.load(p, allow_pickle=True)
        _, s, _ = corners_to_params(d["gt_corners"].astype(np.float64))
        sizes.append(np.sort(s)[::-1])      # sort descending (matches PCA order)

    if not sizes:
        return np.array([-2.3, -2.3, -2.3])
    median_s = np.median(np.array(sizes), axis=0)
    return np.log(np.clip(median_s, 1e-4, None))
