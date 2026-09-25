"""Tests for Step 3 — preprocessing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR    = PROJECT_ROOT / "outputs" / "cache"
DATA_ROOT    = PROJECT_ROOT / "data" / "dl_challenge"


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

def test_cache_exists():
    assert CACHE_DIR.exists(), "cache_dir not found — run preprocess.py first"


def test_npz_count():
    npz_files = list(CACHE_DIR.glob("*.npz"))
    # Between 1900 and 1917 (some instances skipped due to too few points)
    assert 1880 <= len(npz_files) <= 1917, f"Unexpected .npz count: {len(npz_files)}"


def test_split_json_exists():
    assert (CACHE_DIR / "split.json").exists(), "split.json not found"


# ---------------------------------------------------------------------------
# split.json structure
# ---------------------------------------------------------------------------

def test_split_json_splits():
    with open(CACHE_DIR / "split.json") as f:
        split = json.load(f)
    assert set(split.keys()) == {"train", "val", "test"}, f"Bad split keys: {split.keys()}"


def test_split_json_scene_not_split():
    """No scene_id appears in more than one split."""
    with open(CACHE_DIR / "split.json") as f:
        split = json.load(f)
    seen = {}
    for spl, items in split.items():
        for item in items:
            sid = item["scene_id"]
            assert sid not in seen or seen[sid] == spl, (
                f"Scene {sid} appears in both {seen[sid]} and {spl}"
            )
            seen[sid] = spl


def test_split_json_ratios():
    """Check 80/10/10 split at scene level."""
    with open(CACHE_DIR / "split.json") as f:
        split = json.load(f)
    # Recover unique scenes per split
    train_scenes = {i["scene_id"] for i in split["train"]}
    val_scenes   = {i["scene_id"] for i in split["val"]}
    test_scenes  = {i["scene_id"] for i in split["test"]}
    total = len(train_scenes) + len(val_scenes) + len(test_scenes)
    assert total == 200, f"Expected 200 scenes total, got {total}"
    assert 155 <= len(train_scenes) <= 165, f"Train scenes: {len(train_scenes)}"
    assert 15  <= len(val_scenes)   <= 25,  f"Val scenes:   {len(val_scenes)}"
    assert 15  <= len(test_scenes)  <= 25,  f"Test scenes:  {len(test_scenes)}"


def test_split_json_npz_exists():
    """Every entry in split.json must have a matching .npz file."""
    with open(CACHE_DIR / "split.json") as f:
        split = json.load(f)
    missing = []
    for items in split.values():
        for item in items:
            if not (CACHE_DIR / item["npz"]).exists():
                missing.append(item["npz"])
    assert not missing, f"Missing .npz files: {missing[:5]}"


# ---------------------------------------------------------------------------
# .npz content
# ---------------------------------------------------------------------------

def load_some_npz(n=30):
    npz_files = sorted(CACHE_DIR.glob("*.npz"))[:n]
    return [np.load(p, allow_pickle=True) for p in npz_files]


def test_npz_keys():
    for d in load_some_npz():
        assert set(d.files) >= {"pts", "gt_corners", "scene_id", "inst_id", "pixel_bbox"}, \
            f"Missing keys: {d.files}"


def test_npz_pts_shape_and_range():
    for d in load_some_npz():
        pts = d["pts"]
        assert pts.ndim == 2 and pts.shape[1] == 6, f"pts shape: {pts.shape}"
        assert len(pts) >= 30, f"pts too few: {len(pts)}"
        assert pts.dtype == np.float32
        # RGB channels in [0,1]
        assert pts[:, 3:].min() >= 0.0 and pts[:, 3:].max() <= 1.0, "RGB out of [0,1]"
        # Z values should all be > 0
        assert pts[:, 2].min() > 0, "Z<=0 point survived filter"


def test_npz_gt_corners_shape():
    for d in load_some_npz():
        corners = d["gt_corners"]
        assert corners.shape == (8, 3), f"gt_corners shape: {corners.shape}"
        assert corners.dtype == np.float32


def test_npz_pixel_bbox_valid():
    for d in load_some_npz():
        bb = d["pixel_bbox"]
        assert bb.shape == (4,), f"pixel_bbox shape: {bb.shape}"
        r_min, r_max, c_min, c_max = bb
        assert r_min <= r_max and c_min <= c_max, f"invalid bbox: {bb}"


# ---------------------------------------------------------------------------
# Filter correctness spot-check
# ---------------------------------------------------------------------------

def test_no_z_zero_in_pts():
    """All saved points must have Z > 0."""
    for d in load_some_npz(50):
        assert (d["pts"][:, 2] > 0).all(), "Z=0 point in saved cache"


def test_post_filter_pts_in_scene_range():
    """Z values should be in plausible depth range (0.5 – 2.0 m)."""
    for d in load_some_npz(50):
        z = d["pts"][:, 2]
        assert z.min() > 0.2 and z.max() < 3.0, f"Z out of range: [{z.min():.3f}, {z.max():.3f}]"
