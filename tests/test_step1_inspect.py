"""Tests for Step 1 — data inspection."""
import numpy as np
import pytest
from pathlib import Path

# Locate project root (tests/ is one level below root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "dl_challenge"


def scene_dirs():
    return sorted(DATA_ROOT.iterdir()) if DATA_ROOT.exists() else []


def test_scene_count():
    dirs = scene_dirs()
    assert len(dirs) == 200, f"Expected 200 scenes, got {len(dirs)}"


def test_all_files_present():
    missing = []
    for sd in scene_dirs():
        for fname in ["rgb.jpg", "pc.npy", "mask.npy", "bbox3d.npy"]:
            if not (sd / fname).exists():
                missing.append(f"{sd.name}/{fname}")
    assert not missing, f"Missing files: {missing[:10]}"


def test_shape_consistency():
    from PIL import Image
    errors = []
    for sd in scene_dirs():
        rgb = np.array(Image.open(sd / "rgb.jpg"))
        pc  = np.load(sd / "pc.npy")
        mask = np.load(sd / "mask.npy")
        bbox = np.load(sd / "bbox3d.npy")
        H, W = rgb.shape[:2]
        if pc.shape != (3, H, W):
            errors.append(f"{sd.name}: pc {pc.shape}")
        if mask.shape[1:] != (H, W):
            errors.append(f"{sd.name}: mask H/W {mask.shape}")
        if bbox.shape[1:] != (8, 3):
            errors.append(f"{sd.name}: bbox {bbox.shape}")
        if len(mask) != len(bbox):
            errors.append(f"{sd.name}: N mismatch mask={len(mask)} bbox={len(bbox)}")
    assert not errors, f"Shape errors (first 10): {errors[:10]}"


def test_all_boxes_orthogonal():
    thresh = 0.01
    bad = []
    for sd in scene_dirs():
        bbox = np.load(sd / "bbox3d.npy")
        for i, corners in enumerate(bbox):
            u = corners[1] - corners[0]
            v = corners[3] - corners[0]
            w = corners[4] - corners[0]
            for a, b, name in [(u, v, "u·v"), (u, w, "u·w"), (v, w, "v·w")]:
                na = a / (np.linalg.norm(a) + 1e-12)
                nb = b / (np.linalg.norm(b) + 1e-12)
                dot = abs(float(np.dot(na, nb)))
                if dot > thresh:
                    bad.append(f"{sd.name} inst {i} {name}={dot:.4f}")
    assert not bad, f"Non-orthogonal boxes (first 5): {bad[:5]}"


def test_all_boxes_left_handed():
    bad = []
    for sd in scene_dirs():
        bbox = np.load(sd / "bbox3d.npy")
        for i, corners in enumerate(bbox):
            u = corners[1] - corners[0]; nu = u / (np.linalg.norm(u) + 1e-12)
            v = corners[3] - corners[0]; nv = v / (np.linalg.norm(v) + 1e-12)
            w = corners[4] - corners[0]; nw = w / (np.linalg.norm(w) + 1e-12)
            det = float(np.linalg.det(np.stack([nu, nv, nw], axis=1)))
            if det > -0.9:
                bad.append(f"{sd.name} inst {i} det={det:.4f}")
    assert not bad, f"Non-left-handed boxes (first 5): {bad[:5]}"


def test_report_generated():
    report = PROJECT_ROOT / "reports" / "data_report.md"
    assert report.exists(), "data_report.md not found — run inspect.py first"
    assert report.stat().st_size > 500, "report seems too small"
