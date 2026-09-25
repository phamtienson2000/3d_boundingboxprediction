"""
Tests for Step 4 (baseline) + Step 7 (metrics).

IoU tests per SPEC §7:
  - IoU(b,b) = 1
  - disjoint boxes → 0
  - shifted by half along one axis → 1/3
  - agrees with Monte-Carlo IoU within 0.01 on random pairs
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from bbox3d.geometry.box import params_to_corners_np, pca_frame
from bbox3d.metrics import (
    box_iou_3d, symmetric_corner_distance, evaluate_instance, aggregate_metrics
)
from bbox3d.baseline import predict_pca_box


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_box(center, size, R=None):
    """Build (8,3) corners for an axis-aligned or oriented box."""
    if R is None:
        R = np.eye(3)
    return params_to_corners_np(np.array(center), np.array(size, dtype=float), R)


def random_box(rng, scale=0.1):
    """Random proper-R box in reasonable size range."""
    center = rng.uniform(-0.3, 0.3, 3)
    size   = rng.uniform(0.02, scale, 3)
    M = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return params_to_corners_np(center, size, Q)


# ---------------------------------------------------------------------------
# IoU unit tests
# ---------------------------------------------------------------------------

def test_iou_identical():
    corners = make_box([0, 0, 0], [0.1, 0.1, 0.1])
    iou = box_iou_3d(corners, corners)
    assert abs(iou - 1.0) < 1e-3, f"IoU(b,b)={iou:.5f}, expected 1.0"


def test_iou_disjoint():
    a = make_box([0, 0, 0], [0.1, 0.1, 0.1])
    b = make_box([1, 0, 0], [0.1, 0.1, 0.1])
    assert box_iou_3d(a, b) == 0.0


def test_iou_half_shift():
    """Shift one box by half its width along x → IoU = 1/3."""
    a = make_box([0, 0, 0], [1.0, 1.0, 1.0])
    b = make_box([0.5, 0, 0], [1.0, 1.0, 1.0])
    iou = box_iou_3d(a, b)
    assert abs(iou - 1.0 / 3.0) < 0.01, f"IoU={iou:.4f}, expected 1/3"


def test_iou_monte_carlo():
    """IoU agrees with Monte-Carlo estimation within 0.01 on random pairs."""
    rng = np.random.default_rng(0)

    def mc_iou(a_corners, b_corners, n=1_000_000):
        from bbox3d.geometry.box import corners_to_params
        c_a, s_a, R_a = corners_to_params(a_corners.astype(np.float64))
        c_b, s_b, R_b = corners_to_params(b_corners.astype(np.float64))
        # Sample points inside a bounding cube
        lo_a = (c_a - s_a.max()).min()
        hi_a = (c_a + s_a.max()).max()
        lo_b = (c_b - s_b.max()).min()
        hi_b = (c_b + s_b.max()).max()
        lo = min(lo_a, lo_b); hi = max(hi_a, hi_b)
        pts = rng.uniform(lo, hi, (n, 3))
        def in_box(pts, c, s, R):
            local = (pts - c) @ R
            return np.all(np.abs(local) <= s / 2, axis=1)
        in_a = in_box(pts, c_a, s_a, R_a)
        in_b = in_box(pts, c_b, s_b, R_b)
        inter = np.sum(in_a & in_b)
        union = np.sum(in_a | in_b)
        return inter / union if union > 0 else 0.0

    max_err = 0.0
    for _ in range(10):
        a = random_box(rng, scale=0.12)
        b = random_box(rng, scale=0.12)
        iou_exact = box_iou_3d(a, b)
        iou_mc    = mc_iou(a, b)
        max_err   = max(max_err, abs(iou_exact - iou_mc))

    print(f"Monte-Carlo vs exact IoU max err: {max_err:.4f}")
    assert max_err < 0.01, f"Max MC vs exact IoU error: {max_err:.4f}"


# ---------------------------------------------------------------------------
# Corner distance
# ---------------------------------------------------------------------------

def test_corner_dist_identical():
    corners = random_box(np.random.default_rng(1))
    dist_mm, _ = symmetric_corner_distance(corners, corners)
    assert dist_mm < 1e-6, f"dist for identical boxes: {dist_mm:.2e} mm"


def test_corner_dist_permuted():
    """Corner dist is 0 for any SYM_PERM relabeling of the same box."""
    from bbox3d.geometry.box import SYM_PERMS
    corners = random_box(np.random.default_rng(2))
    for k, perm in enumerate(SYM_PERMS):
        dist_mm, _ = symmetric_corner_distance(corners, corners[perm])
        assert dist_mm < 1e-6, f"Perm {k}: dist={dist_mm:.2e} mm"


# ---------------------------------------------------------------------------
# evaluate_instance
# ---------------------------------------------------------------------------

def test_evaluate_instance_perfect():
    """Perfect prediction → IoU≈1, corner_dist≈0."""
    corners = random_box(np.random.default_rng(3))
    r = evaluate_instance(corners, corners)
    assert r["iou_3d"] > 0.99
    assert r["corner_dist_mm"] < 1e-3
    assert r["center_err_mm"] < 1e-3


def test_evaluate_instance_keys():
    a = random_box(np.random.default_rng(4))
    b = random_box(np.random.default_rng(5))
    r = evaluate_instance(a, b)
    assert set(r.keys()) >= {
        "corner_dist_mm", "iou_3d", "center_err_mm",
        "size_err_mm", "rot_err_deg", "is_thin"
    }


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------

def test_aggregate_metrics_structure():
    rng = np.random.default_rng(6)
    results = []
    for _ in range(20):
        a = random_box(rng)
        b = random_box(rng)
        results.append(evaluate_instance(a, b))
    agg = aggregate_metrics(results)
    assert "all" in agg and "thin" in agg and "normal" in agg
    assert agg["all"]["n"] == 20


# ---------------------------------------------------------------------------
# PCA baseline on preprocessed data
# ---------------------------------------------------------------------------

def test_pca_box_shape():
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((200, 3)) * 0.05 + np.array([0.5, 0.5, 1.0])
    corners = predict_pca_box(pts)
    assert corners.shape == (8, 3)


def test_baseline_report_exists():
    report = PROJECT_ROOT / "reports" / "baseline.md"
    assert report.exists(), "baseline.md not found — run baseline.py first"
    assert report.stat().st_size > 200
