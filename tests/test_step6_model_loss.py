"""
Tests for Step 6 — PointNet model + losses.

Model tests:
  - forward output shape
  - decode produces proper rotation (det +1)
  - at init, R ≈ I (rot6d bias initialised to [1,0,0, 0,1,0])
  - gradients flow without NaN/Inf

Loss tests (SPEC §6 acceptance criteria):
  - loss(pred = GT)                              ≈ 0
  - loss(pred = GT relabeled by any SYM_PERM σ) ≈ 0
  - loss is differentiable; gradients finite
  - all components non-negative
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH  = PROJECT_ROOT / "configs" / "default.yaml"


def _cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


from bbox3d.geometry.box import (
    SYM_PERMS, params_to_corners_np, params_to_corners_torch,
    corners_to_params, rot6d_to_matrix,
)
from bbox3d.models.pointnet_box import PointNetBoxHead, build_model
from bbox3d.losses import BBoxLoss, _batch_edge_lengths, _symmetric_corner_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_gt_corners(B: int, rng: np.random.Generator) -> torch.Tensor:
    """Create (B, 8, 3) GT corners from random proper boxes."""
    corners = []
    for _ in range(B):
        center = rng.uniform(-0.1, 0.1, 3)
        size   = rng.uniform(0.02, 0.15, 3)
        M      = rng.standard_normal((3, 3))
        Q, _   = np.linalg.qr(M)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        c = params_to_corners_np(center, size, Q)
        corners.append(c)
    return torch.tensor(np.array(corners), dtype=torch.float32)


def _gt_to_pred_params(gt_corners: torch.Tensor):
    """
    Derive (pred_center, pred_size, pred_R, pred_corners) that exactly
    reproduce gt_corners (within float32 precision) via corners_to_params.
    """
    B = gt_corners.shape[0]
    centers, sizes, Rs = [], [], []
    for b in range(B):
        c, s, R = corners_to_params(gt_corners[b].double().numpy())
        centers.append(c); sizes.append(s); Rs.append(R)
    center = torch.tensor(np.array(centers), dtype=torch.float32)
    size   = torch.tensor(np.array(sizes),   dtype=torch.float32)
    R      = torch.tensor(np.array(Rs),      dtype=torch.float32)
    pred_c = params_to_corners_torch(center, size, R)
    return center, size, R, pred_c


# ---------------------------------------------------------------------------
# Model — shape and output properties
# ---------------------------------------------------------------------------

def test_model_forward_shape():
    cfg   = _cfg()
    ED    = int(cfg.get("dataset", {}).get("extra_dim", 7))
    model = build_model(cfg)
    model.eval()
    B, P  = 4, 512
    pts   = torch.randn(B, P, 6)
    extra = torch.randn(B, ED)
    raw   = model(pts, extra)
    assert raw.shape == (B, 12), f"raw output shape: {raw.shape}"


def test_model_decode_shapes():
    cfg   = _cfg()
    ED    = int(cfg.get("dataset", {}).get("extra_dim", 7))
    model = build_model(cfg)
    model.eval()
    B, P  = 4, 512
    raw   = model(torch.randn(B, P, 6), torch.randn(B, ED))
    c, s, R, corners = model.decode(raw)
    assert c.shape       == (B, 3)
    assert s.shape       == (B, 3)
    assert R.shape       == (B, 3, 3)
    assert corners.shape == (B, 8, 3)


def test_model_pred_R_proper():
    """Predicted R must have det ≈ +1 for any input."""
    cfg   = _cfg()
    ED    = int(cfg.get("dataset", {}).get("extra_dim", 7))
    model = build_model(cfg)
    model.eval()
    B = 8
    _, _, R, _ = model.forward_decode(torch.randn(B, 512, 6), torch.randn(B, ED))
    dets = torch.linalg.det(R.double())
    assert (dets - 1.0).abs().max().item() < 1e-4, f"det(R) not +1: {dets.tolist()}"


def test_model_init_identity_rotation():
    """At initialization, rot6d bias → R ≈ I (before any training)."""
    model = PointNetBoxHead()
    model.eval()
    B = 4
    with torch.no_grad():
        raw = model(torch.zeros(B, 512, 6), torch.zeros(B, 7))
    pred_R = rot6d_to_matrix(raw[:, 6:12])
    err    = (pred_R - torch.eye(3)).abs().max().item()
    assert err < 0.01, f"Init R not close to I: max err={err:.4f}"


def test_model_pred_size_positive():
    """Predicted sizes must be positive (exp of log_size)."""
    cfg   = _cfg()
    ED    = int(cfg.get("dataset", {}).get("extra_dim", 7))
    model = build_model(cfg)
    model.eval()
    _, s, _, _ = model.forward_decode(torch.randn(4, 512, 6), torch.randn(4, ED))
    assert (s > 0).all(), "Predicted sizes contain non-positive values"


def test_model_gradients_finite():
    """Backward pass must produce finite gradients everywhere."""
    cfg   = _cfg()
    ED    = int(cfg.get("dataset", {}).get("extra_dim", 7))
    model = build_model(cfg).train()
    B     = 4
    pts   = torch.randn(B, 512, 6, requires_grad=False)
    extra = torch.randn(B, ED,     requires_grad=False)
    raw   = model(pts, extra)
    raw.sum().backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


# ---------------------------------------------------------------------------
# Loss — zero for exact GT
# ---------------------------------------------------------------------------

def test_loss_zero_for_exact_prediction():
    """Loss ≈ 0 when prediction exactly equals GT."""
    rng      = np.random.default_rng(42)
    gt       = _random_gt_corners(8, rng)
    center, size, _, pred_c = _gt_to_pred_params(gt)

    loss_fn  = BBoxLoss(_cfg())
    total, parts = loss_fn(pred_c, center, size, gt)

    assert total.item() < 1e-5, f"total loss for exact pred={total.item():.2e}"
    assert parts["corner"].item() < 1e-5
    assert parts["center"].item() < 1e-5
    assert parts["size"].item()   < 1e-5


def test_corner_loss_zero_for_all_sym_perms():
    """
    Corner loss must be 0 when pred = GT relabeled by ANY of the 24 SYM_PERMS.
    This is the key symmetry-awareness test from SPEC §6.
    """
    rng      = np.random.default_rng(7)
    gt       = _random_gt_corners(1, rng)                    # (1, 8, 3)
    sym      = torch.from_numpy(SYM_PERMS).long()            # (24, 8)

    for k in range(24):
        perm      = SYM_PERMS[k]
        gt_perm   = gt[:, perm, :]                           # (1, 8, 3) relabeled
        # Derive pred params from the permuted corners (so pred_c == gt_perm exactly)
        _, _, _, pred_c = _gt_to_pred_params(gt_perm)
        loss, _   = _symmetric_corner_loss(pred_c, gt, sym, beta=0.002)
        assert loss.item() < 1e-5, \
            f"Corner loss non-zero for SYM_PERM {k}: {loss.item():.2e}"


def test_full_loss_zero_for_any_sym_perm():
    """
    Full BBoxLoss ≈ 0 for pred matching GT under any SYM_PERM.
    Tests that center + size are permutation-invariant too.
    """
    rng     = np.random.default_rng(13)
    gt      = _random_gt_corners(2, rng)                     # (2, 8, 3)
    loss_fn = BBoxLoss(_cfg())

    for k in range(24):
        perm    = SYM_PERMS[k]
        gt_perm = gt[:, perm, :]                             # (2, 8, 3)
        center, size, _, pred_c = _gt_to_pred_params(gt_perm)
        total, _ = loss_fn(pred_c, center, size, gt)
        assert total.item() < 1e-4, \
            f"Full loss non-zero for SYM_PERM {k}: {total.item():.2e}"


# ---------------------------------------------------------------------------
# Loss — differentiability
# ---------------------------------------------------------------------------

def test_loss_differentiable():
    """Backward through total loss must succeed with all-finite gradients."""
    rng      = np.random.default_rng(99)
    gt       = _random_gt_corners(4, rng)
    center, size, R, pred_c = _gt_to_pred_params(gt)

    # Add small noise so pred ≠ gt
    pred_c   = pred_c  + 0.01 * torch.randn_like(pred_c)
    center   = center  + 0.01 * torch.randn_like(center)
    size     = size    * (1 + 0.1 * torch.randn_like(size)).abs()

    # Attach gradients
    pred_c   = pred_c.detach().requires_grad_(True)
    center   = center.detach().requires_grad_(True)
    size_pos = (size.detach().abs() + 1e-4).requires_grad_(True)

    loss_fn  = BBoxLoss(_cfg())
    total, _ = loss_fn(pred_c, center, size_pos, gt)
    total.backward()

    for t, name in [(pred_c, "pred_c"), (center, "center"), (size_pos, "size")]:
        assert t.grad is not None,                          f"{name}: grad is None"
        assert torch.isfinite(t.grad).all(),                f"{name}: non-finite grad"


def test_loss_end_to_end_differentiable():
    """Full model → loss → backward must produce finite gradients."""
    cfg   = _cfg()
    ED    = int(cfg.get("dataset", {}).get("extra_dim", 7))
    model = build_model(cfg).train()
    loss_fn = BBoxLoss(cfg)

    rng   = np.random.default_rng(55)
    B     = 4
    gt    = _random_gt_corners(B, rng)
    pts   = torch.randn(B, 512, 6)
    extra = torch.randn(B, ED)

    pred_center, pred_size, _, pred_corners = model.forward_decode(pts, extra)
    total, parts = loss_fn(pred_corners, pred_center, pred_size, gt)
    total.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


# ---------------------------------------------------------------------------
# Loss — non-negative components
# ---------------------------------------------------------------------------

def test_loss_components_nonneg():
    """All loss components must be ≥ 0."""
    rng     = np.random.default_rng(0)
    gt      = _random_gt_corners(8, rng)
    pred_c  = gt + 0.05 * torch.randn_like(gt)
    center  = gt.mean(dim=1) + 0.01 * torch.randn(8, 3)
    size    = torch.abs(torch.randn(8, 3)) * 0.1

    loss_fn = BBoxLoss(_cfg())
    total, parts = loss_fn(pred_c, center, size, gt)

    assert total.item()          >= 0, f"total loss < 0: {total.item()}"
    assert parts["corner"].item() >= 0
    assert parts["center"].item() >= 0
    assert parts["size"].item()   >= 0


# ---------------------------------------------------------------------------
# Utility: compute_log_size_prior
# ---------------------------------------------------------------------------

def test_compute_log_size_prior():
    """compute_log_size_prior returns a (3,) array of finite values."""
    import json
    from bbox3d.losses import compute_log_size_prior
    CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
    with open(CACHE_DIR / "split.json") as f:
        split = json.load(f)
    prior = compute_log_size_prior(split["train"], CACHE_DIR, n_samples=100)
    assert prior.shape == (3,), f"prior shape: {prior.shape}"
    assert np.isfinite(prior).all(), f"non-finite prior: {prior}"
    # log(size) should be roughly in [-5, 0] for objects 0.007–0.22 m
    assert (-6 < prior).all() and (prior < 0).all(), f"prior out of range: {prior}"
