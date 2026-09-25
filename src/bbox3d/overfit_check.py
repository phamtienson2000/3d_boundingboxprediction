"""
Overfit sanity check: train on 10 fixed samples, no augmentation.
Goal: all losses and rot_err should go near 0 if model + loss are bug-free.

Usage:
    python -m bbox3d.overfit_check --config configs/default.yaml [--n 10] [--epochs 300]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

import yaml

from bbox3d.data.dataset import BBox3DDataset, make_datasets
from bbox3d.geometry.box import SYM_PERMS, corners_to_params
from bbox3d.losses import BBoxLoss, compute_log_size_prior
from bbox3d.models.pointnet_box import build_model
from bbox3d.train import seed_everything


def rot_err_deg(pred_R: torch.Tensor, gt_R: torch.Tensor) -> torch.Tensor:
    """Mean geodesic rotation error in degrees. (B,3,3) each."""
    R_diff = pred_R.transpose(-1, -2) @ gt_R
    trace  = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos    = ((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean() * (180.0 / math.pi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--n",       type=int, default=10)
    parser.add_argument("--epochs",  type=int, default=300)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root      = Path(__file__).resolve().parents[2]
    cache_dir = root / cfg["data"]["cache_dir"]
    split_json = root / cfg["data"]["split_json"]

    with open(split_json) as f:
        split_data = json.load(f)

    seed_everything(42)

    # Build dataset with augmentation=False for overfit test
    cfg_no_aug = cfg.copy()
    cfg_no_aug["augmentation"] = {
        "rot_z_deg": 0.0, "tilt_deg": 0.0,
        "scale_min": 1.0, "scale_max": 1.0,
        "jitter_sigma_m": 0.0, "jitter_clip_m": 0.0,
        "dropout_max": 0.0, "rgb_jitter": 0.0,
    }

    items_10 = split_data["train"][:args.n]
    dataset  = BBox3DDataset(items_10, cache_dir, cfg_no_aug,
                             augment_data=False, seed=42)
    loader   = DataLoader(dataset, batch_size=args.n,
                          shuffle=False, num_workers=0)

    log_size_init = compute_log_size_prior(split_data["train"], cache_dir)
    model    = build_model(cfg, log_size_init)
    loss_fn  = BBoxLoss(cfg)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)

    sym_perms = torch.from_numpy(SYM_PERMS).long()

    print(f"Overfitting {args.n} samples for {args.epochs} epochs (no augmentation)\n")
    print(f"{'ep':>5}  {'loss':>8}  {'corner_mm':>10}  {'rot_err_deg':>12}")

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            pts      = batch["pts"]
            extra    = batch["extra"]
            gt_c     = batch["gt_corners_canon"]

            optimizer.zero_grad()
            pred_center, pred_size, pred_R, pred_corners = model.forward_decode(pts, extra)
            total, comps = loss_fn(pred_corners, pred_center, pred_size, gt_c)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                batch = next(iter(loader))
                pts, extra = batch["pts"], batch["extra"]
                gt_c   = batch["gt_corners_canon"]
                gt_R   = batch["R0"]          # camera→canonical rotation stored in dataset

                pred_center, pred_size, pred_R, pred_corners = model.forward_decode(pts, extra)
                total, comps = loss_fn(pred_corners, pred_center, pred_size, gt_c)

                # corner error in mm (canonical frame ~ metres)
                # find best-perm corners
                gt_perm = gt_c[:, sym_perms, :]          # (B,24,8,3)
                pred_ex = pred_corners.unsqueeze(1).expand_as(gt_perm)
                dists   = (pred_ex - gt_perm).norm(dim=-1).mean(dim=-1)  # (B,24)
                best_k  = dists.argmin(dim=1)             # (B,)
                B = pts.shape[0]
                b_idx = torch.arange(B).unsqueeze(1).expand(B, 8)
                best_perm_idx = sym_perms[best_k]          # (B,8)
                gt_best = gt_c[b_idx, best_perm_idx]       # (B,8,3)
                corner_mm = (pred_corners - gt_best).norm(dim=-1).mean().item() * 1000

                # Rotation error: use best-perm gt corners (same convention as L_corner)
                # Naive gt_R is ~180° from pred_R due to sign convention — use sym-aware min
                gt_R_list = []
                for b in range(B):
                    best_err = math.pi
                    best_R   = None
                    for perm in sym_perms.numpy():
                        gt_c_perm = gt_c[b].numpy()[perm].astype(np.float64)
                        _, _, R_cand = corners_to_params(gt_c_perm)
                        R_diff = pred_R[b].detach().numpy().T @ R_cand
                        tr  = float(R_diff[0,0] + R_diff[1,1] + R_diff[2,2])
                        err = math.acos(max(-1+1e-7, min(1-1e-7, (tr-1)/2)))
                        if err < best_err:
                            best_err = err
                            best_R   = R_cand
                    gt_R_list.append(torch.tensor(best_R, dtype=torch.float32))
                R_gt = torch.stack(gt_R_list)
                rot_deg = rot_err_deg(pred_R, R_gt).item()

                print(f"{epoch:5d}  {total.item():8.5f}  {corner_mm:10.2f}  {rot_deg:12.2f}")

    print(f"\nFinal rot error: {rot_deg:.2f} deg  corner: {corner_mm:.2f} mm")
    if rot_deg > 5.0:
        print("WARNING: rot error > 5 deg on 10 samples — possible bug in loss or model.")
    else:
        print("OK: model can overfit rotation on 10 samples.")


if __name__ == "__main__":
    main()
