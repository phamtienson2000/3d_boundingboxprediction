"""
Tests for Step 5 — dataset + augmentations.

Checks shapes, dtypes, geometric invariants (PCA centering, R0 det +1,
gt_corners roundtrip), determinism for val/test, randomness for train,
and config flags use_rgb / use_pca.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR    = PROJECT_ROOT / "outputs" / "cache"
SPLIT_JSON   = CACHE_DIR / "split.json"
CONFIG_PATH  = PROJECT_ROOT / "configs" / "default.yaml"


def _cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _split():
    with open(SPLIT_JSON) as f:
        return json.load(f)


from bbox3d.data.dataset import BBox3DDataset, make_datasets


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_dataset_lengths():
    cfg   = _cfg()
    split = _split()
    for key in ("train", "val", "test"):
        ds = BBox3DDataset(split[key], CACHE_DIR, cfg, augment_data=False)
        assert len(ds) == len(split[key]), f"{key}: len mismatch"


def test_item_keys():
    cfg  = _cfg()
    ds   = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False)
    item = ds[0]
    assert set(item.keys()) >= {
        "pts", "extra", "gt_corners_canon", "gt_corners_cam",
        "R0", "t0", "scene_id", "inst_id",
    }


def test_item_shapes():
    cfg = _cfg()
    P   = cfg["dataset"]["num_points"]
    ds  = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False)
    item = ds[0]
    assert item["pts"].shape              == (P, 6),    f"pts: {item['pts'].shape}"
    extra_dim = int(cfg.get("dataset", {}).get("extra_dim", 7))
    assert item["extra"].shape == (extra_dim,), f"extra: {item['extra'].shape} vs expected {extra_dim}"
    assert item["gt_corners_canon"].shape == (8, 3),    f"gt_canon: {item['gt_corners_canon'].shape}"
    assert item["gt_corners_cam"].shape   == (8, 3),    f"gt_cam: {item['gt_corners_cam'].shape}"
    assert item["R0"].shape               == (3, 3),    f"R0: {item['R0'].shape}"
    assert item["t0"].shape               == (3,),      f"t0: {item['t0'].shape}"


def test_item_dtypes():
    cfg  = _cfg()
    ds   = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False)
    item = ds[0]
    for key in ("pts", "extra", "gt_corners_canon", "gt_corners_cam", "R0", "t0"):
        assert item[key].dtype == torch.float32, f"{key} dtype={item[key].dtype}"


# ---------------------------------------------------------------------------
# Geometric invariants
# ---------------------------------------------------------------------------

def test_canonical_pts_centered():
    """Sampled pts in canonical frame must have mean xyz ≈ 0."""
    cfg = _cfg()
    ds  = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False)
    for i in range(5):
        mean_xyz = ds[i]["pts"][:, :3].mean(dim=0)
        assert mean_xyz.abs().max().item() < 1e-4, \
            f"idx={i} canonical mean={mean_xyz.tolist()}"


def test_pca_rotation_proper():
    """R0 must be a proper rotation: det ≈ +1, R R^T ≈ I."""
    cfg = _cfg()
    ds  = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False)
    for i in range(10):
        R0   = ds[i]["R0"].double()
        det  = torch.linalg.det(R0).item()
        orth = (R0 @ R0.T - torch.eye(3, dtype=torch.float64)).abs().max().item()
        assert abs(det - 1.0) < 1e-4,  f"idx={i} det={det:.6f}"
        assert orth           < 1e-4,  f"idx={i} orthogonality err={orth:.2e}"


def test_gt_corners_roundtrip():
    """gt_corners_cam == gt_corners_canon @ R0.T + t0 (within float32 precision)."""
    cfg = _cfg()
    ds  = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False)
    for i in range(5):
        item    = ds[i]
        R0      = item["R0"].double()
        t0      = item["t0"].double()
        canon   = item["gt_corners_canon"].double()
        cam_gt  = item["gt_corners_cam"].double()
        cam_rec = canon @ R0.T + t0
        err     = (cam_rec - cam_gt).abs().max().item()
        assert err < 5e-4, f"idx={i} roundtrip err={err:.2e}"


def test_extra_log_n_raw_positive():
    """extra[0] = log1p(n_raw) must be > 0."""
    cfg = _cfg()
    ds  = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False)
    for i in range(5):
        assert ds[i]["extra"][0].item() > 0., f"idx={i} log_n_raw <= 0"


def test_extra_extents_positive():
    """extra[1:4] = canonical extents must be > 0 (extra[4:] = view direction, may be negative)."""
    cfg = _cfg()
    ds  = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False)
    for i in range(5):
        extents = ds[i]["extra"][1:4].numpy()
        assert (extents > 0).all(), f"idx={i} non-positive extents: {extents}"


# ---------------------------------------------------------------------------
# Determinism (val) vs. randomness (train)
# ---------------------------------------------------------------------------

def test_val_deterministic():
    """Same val index must return identical pts on repeated access."""
    cfg = _cfg()
    ds  = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False, seed=42)
    a   = ds[0]["pts"]
    b   = ds[0]["pts"]
    assert torch.equal(a, b), "Val dataset is not deterministic"


def test_train_augmentation_varies():
    """Train dataset must produce different pts on repeated access (augmentation is random)."""
    cfg = _cfg()
    ds  = BBox3DDataset(_split()["train"], CACHE_DIR, cfg, augment_data=True)
    pts = [ds[0]["pts"] for _ in range(5)]
    n_different = sum(not torch.equal(pts[0], pts[i]) for i in range(1, 5))
    assert n_different >= 1, "Train augmentation appears non-random over 5 calls"


# ---------------------------------------------------------------------------
# Config flags
# ---------------------------------------------------------------------------

def test_use_rgb_false_zeroes_rgb():
    """With use_rgb=False the RGB columns (3:6) of pts must be all zeros."""
    cfg = copy.deepcopy(_cfg())
    cfg["dataset"]["use_rgb"] = False
    ds   = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False)
    item = ds[0]
    assert item["pts"][:, 3:].abs().max().item() == 0.0, "RGB not zeroed with use_rgb=False"


def test_use_pca_false_identity_r0():
    """With use_pca=False, R0 must be the identity matrix."""
    cfg = copy.deepcopy(_cfg())
    cfg["dataset"]["use_pca"] = False
    ds   = BBox3DDataset(_split()["val"], CACHE_DIR, cfg, augment_data=False)
    item = ds[0]
    diff = (item["R0"] - torch.eye(3)).abs().max().item()
    assert diff < 1e-6, f"R0 not identity with use_pca=False: max_diff={diff}"


# ---------------------------------------------------------------------------
# make_datasets factory
# ---------------------------------------------------------------------------

def test_make_datasets_returns_all_splits():
    dsets = make_datasets(_cfg(), CACHE_DIR, SPLIT_JSON)
    assert set(dsets.keys()) == {"train", "val", "test"}


def test_make_datasets_aug_flags():
    dsets = make_datasets(_cfg(), CACHE_DIR, SPLIT_JSON)
    assert dsets["train"].augment_data is True
    assert dsets["val"].augment_data   is False
    assert dsets["test"].augment_data  is False


def test_make_datasets_total_items():
    """Sum of all split lengths should equal total npz count."""
    dsets = make_datasets(_cfg(), CACHE_DIR, SPLIT_JSON)
    total = sum(len(ds) for ds in dsets.values())
    # 1901 instances in split (1917 - 16 skipped in preprocess)
    assert 1880 <= total <= 1917, f"Unexpected total items: {total}"
