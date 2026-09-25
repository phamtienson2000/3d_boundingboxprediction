"""
Step 5 — Dataset + augmentations (data/dataset.py).

torch.utils.data.Dataset that loads pre-processed .npz files,
samples P=512 points, optionally augments (train only),
then canonicalizes via PCA and returns all tensors the model needs.

Item keys:
  pts              (P, 6) float32 — [xyz_canon, rgb] (or zeros for rgb if use_rgb=False)
  extra            (7,)   float32 — [log1p(n_raw), extent_x, extent_y, extent_z, view_x, view_y, view_z]
  gt_corners_canon (8, 3) float32 — GT box corners in canonical (PCA) frame
  gt_corners_cam   (8, 3) float32 — GT box corners in camera frame (post-aug, pre-PCA)
  R0               (3, 3) float32 — PCA rotation; x_cam = x_canon @ R0.T + t0
  t0               (3,)   float32 — canonical origin in camera frame
  scene_id         str
  inst_id          int

Supported input modes (dataset.use_rgb / use_pca):
  xyz + PCA  (best, IoU 0.411 on test) — use_rgb=False, use_pca=True
  xyz+rgb + PCA (main, IoU 0.409)      — use_rgb=True,  use_pca=True
  xyz, no PCA  (IoU 0.400)             — use_rgb=False, use_pca=False
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bbox3d.geometry.box import pca_frame


# ---------------------------------------------------------------------------
# Small rotation-matrix helpers (numpy)
# ---------------------------------------------------------------------------

def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=np.float64)


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]], dtype=np.float64)


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment(
    xyz: np.ndarray,         # (N, 3) float64
    rgb: np.ndarray,         # (N, 3) float64
    gt_corners: np.ndarray,  # (8, 3) float64
    aug_cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train-time augmentations applied consistently to points + GT corners.

    Rotations and scale are applied to both xyz and gt_corners so the
    canonical frame (PCA applied later) remains consistent.
    Jitter and dropout apply to xyz/rgb only (not GT corners).
    """
    # 1. Random rotation about camera Z (uniform 0–360° by default)
    rot_z_max = np.radians(aug_cfg.get("rot_z_deg", 360.0))
    Rz         = _rot_z(rng.uniform(0., rot_z_max))
    xyz        = xyz        @ Rz.T
    gt_corners = gt_corners @ Rz.T

    # 2. Small tilt about X and Y (±tilt_deg)
    tilt       = np.radians(aug_cfg.get("tilt_deg", 10.0))
    R_tilt     = _rot_x(rng.uniform(-tilt, tilt)) @ _rot_y(rng.uniform(-tilt, tilt))
    xyz        = xyz        @ R_tilt.T
    gt_corners = gt_corners @ R_tilt.T

    # 3. Uniform scale about point-cloud centroid
    scale      = rng.uniform(aug_cfg.get("scale_min", 0.9), aug_cfg.get("scale_max", 1.1))
    centroid   = xyz.mean(axis=0)
    xyz        = centroid + scale * (xyz        - centroid)
    gt_corners = centroid + scale * (gt_corners - centroid)

    # 4. Per-point Gaussian jitter on xyz (not gt_corners)
    sigma      = aug_cfg.get("jitter_sigma_m", 0.002)
    clip_val   = aug_cfg.get("jitter_clip_m",  0.005)
    xyz       += np.clip(rng.normal(0., sigma, xyz.shape), -clip_val, clip_val)

    # 5. Random point dropout (up to dropout_max fraction)
    keep_frac  = rng.uniform(1.0 - aug_cfg.get("dropout_max", 0.20), 1.0)
    n_keep     = max(1, int(len(xyz) * keep_frac))
    keep_idx   = rng.choice(len(xyz), n_keep, replace=False)
    xyz        = xyz[keep_idx]
    rgb        = rgb[keep_idx]

    # 6. RGB brightness + contrast jitter
    alpha      = aug_cfg.get("rgb_jitter", 0.2)
    brightness = rng.uniform(-alpha, alpha)
    contrast   = rng.uniform(1.0 - alpha, 1.0 + alpha)
    rgb_mean   = rgb.mean(axis=0, keepdims=True)
    rgb        = np.clip(rgb_mean + contrast * (rgb - rgb_mean) + brightness, 0., 1.)

    return xyz, rgb, gt_corners


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BBox3DDataset(Dataset):
    """Per-instance oriented bounding box dataset."""

    def __init__(
        self,
        items: list[dict],
        cache_dir: Path,
        cfg: dict,
        augment_data: bool = False,
        seed: int = 42,
    ) -> None:
        self.items        = items
        self.cache_dir    = Path(cache_dir)
        self.augment_data = augment_data
        self.seed         = seed

        ds_cfg                = cfg.get("dataset", {})
        self.num_points       = int(ds_cfg.get("num_points",      512))
        self.use_rgb          = bool(ds_cfg.get("use_rgb",         True))
        self.use_pca          = bool(ds_cfg.get("use_pca",         True))
        self.use_new_features = bool(ds_cfg.get("use_new_features", False))
        self.aug_cfg          = cfg.get("augmentation", {})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item    = self.items[idx]
        d       = np.load(self.cache_dir / item["npz"], allow_pickle=True)

        pts_raw = d["pts"]                               # (N, 6) float32
        gt_cam  = d["gt_corners"].astype(np.float64)    # (8, 3)
        n_raw   = len(pts_raw)

        xyz     = pts_raw[:, :3].astype(np.float64)
        rgb     = pts_raw[:, 3:].astype(np.float64)

        # --- Train augmentations (before PCA canonicalization) ---
        if self.augment_data:
            xyz, rgb, gt_cam = augment(xyz, rgb, gt_cam, self.aug_cfg,
                                       np.random.default_rng())

        # --- Point sampling ---
        P = self.num_points
        N = len(xyz)
        rng_s = (np.random.default_rng() if self.augment_data
                 else np.random.default_rng(self.seed + idx))
        sel   = rng_s.choice(N, P, replace=(N < P))
        xyz_s = xyz[sel]    # (P, 3)
        rgb_s = rgb[sel]    # (P, 3)

        # --- PCA canonicalization ---
        if self.use_pca:
            R0, t0 = pca_frame(xyz_s)
        else:
            R0 = np.eye(3, dtype=np.float64)
            t0 = xyz_s.mean(axis=0)

        xyz_c = (xyz_s  - t0) @ R0    # (P, 3)
        gt_c  = (gt_cam - t0) @ R0    # (8, 3)

        # --- Extra features: [log1p(n_raw), canonical extents, view direction in canonical frame] ---
        extents = xyz_c.max(axis=0) - xyz_c.min(axis=0)    # (3,)
        view    = (t0 / (np.linalg.norm(t0) + 1e-12)) @ R0  # (3,) in canonical frame
        extra   = np.array([np.log1p(float(n_raw)), *extents, *view], dtype=np.float32)  # (7,)

        if self.use_new_features:
            # 4 additional depth-context features (no GT used)
            t0z  = float(t0[2])
            raw_gap50 = float(d["gap_p50"]) if "gap_p50" in d else float("nan")
            raw_gap90 = float(d["gap_p90"]) if "gap_p90" in d else float("nan")
            rof   = float(d["ring_other_frac"]) if "ring_other_frac" in d else float("nan")
            # Clip gap features to [0, 0.25]; replace NaN with 0
            gap50 = float(np.clip(raw_gap50, 0.0, 0.25)) if np.isfinite(raw_gap50) else 0.0
            gap90 = float(np.clip(raw_gap90, 0.0, 0.25)) if np.isfinite(raw_gap90) else 0.0
            rof   = float(rof)  if np.isfinite(rof)  else 0.0
            extra = np.concatenate([extra, [t0z, gap50, gap90, rof]], dtype=np.float32)  # (11,)

        # --- Point tensor (P, 6): xyz_canon + (rgb | zeros) ---
        if self.use_rgb:
            pts_out = np.concatenate([xyz_c, rgb_s], axis=1).astype(np.float32)
        else:
            pts_out = np.concatenate(
                [xyz_c, np.zeros_like(rgb_s)], axis=1
            ).astype(np.float32)

        return {
            "pts":              torch.from_numpy(pts_out),
            "extra":            torch.from_numpy(extra),
            "gt_corners_canon": torch.from_numpy(gt_c.astype(np.float32)),
            "gt_corners_cam":   torch.from_numpy(gt_cam.astype(np.float32)),
            "R0":               torch.from_numpy(R0.astype(np.float32)),
            "t0":               torch.from_numpy(t0.astype(np.float32)),
            "scene_id":         str(d["scene_id"]),
            "inst_id":          int(d["inst_id"]),
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_datasets(cfg: dict, cache_dir: Path, split_json: Path) -> dict[str, BBox3DDataset]:
    """Build train/val/test BBox3DDatasets from config and split.json."""
    with open(split_json) as f:
        split = json.load(f)

    train_seed = cfg.get("train", {}).get("seed", 42)
    return {
        "train": BBox3DDataset(split["train"], cache_dir, cfg,
                               augment_data=True,  seed=train_seed),
        "val":   BBox3DDataset(split["val"],   cache_dir, cfg,
                               augment_data=False, seed=train_seed),
        "test":  BBox3DDataset(split["test"],  cache_dir, cfg,
                               augment_data=False, seed=train_seed),
    }


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Step 5 — dataset sanity check")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root       = Path(__file__).resolve().parents[3]
    cache_dir  = root / cfg["data"]["cache_dir"]
    split_json = root / cfg["data"]["split_json"]

    datasets = make_datasets(cfg, cache_dir, split_json)
    for split_name, ds in datasets.items():
        item = ds[0]
        det  = float(np.linalg.det(item["R0"].numpy().astype(np.float64)))
        print(
            f"{split_name:5s}: n={len(ds):4d}  "
            f"pts={tuple(item['pts'].shape)}  "
            f"extra={item['extra'].tolist()}  "
            f"R0_det={det:.5f}  "
            f"t0={item['t0'].tolist()}"
        )


if __name__ == "__main__":
    main()
