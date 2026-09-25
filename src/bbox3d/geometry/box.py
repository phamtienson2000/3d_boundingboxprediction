"""
Geometry utilities for oriented 3D bounding boxes.

Corner convention (SPEC §1, verified on all 1917 GT boxes):
    u = c1-c0,  v = c3-c0,  w = c4-c0
    edges u,v,w are mutually orthogonal; det([u_hat, v_hat, w_hat]) = -1 (left-handed).
    Faces: bottom 0-1-2-3, top 4-5-6-7, c_{i+4} = c_i + w.

The canonical corner template (used by params_to_corners) is built from the unit box
    c_k = 0.5 * ( sx*sx_k, sy*sy_k, sz*sz_k )
where (sx_k, sy_k, sz_k) are the ±1 sign patterns for the 8 corners that reproduce
the same handedness.  The 24 symmetry permutations are the 24 proper rotations of the
cube applied to those 8 corner indices.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------

def _normalise(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def corners_to_params(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose 8×3 corner array into (center, size, R).

    center : (3,)   mean of 8 corners
    size   : (3,)   edge lengths [|u|, |v|, |w|]  (all positive)
    R      : (3,3)  proper rotation (det +1).  Columns are [u_hat, v_hat, -w_hat]
                    (third column negated) so that det(R)=+1 despite the GT
                    convention det([u_hat,v_hat,w_hat])=-1.  Paired with the
                    left-handed sign template (z-column negated), this gives:
                    params_to_corners_np(corners_to_params(c)) == c  exactly.

    Invariant: params_to_corners_np(corners_to_params(c)) == c  (max err < 1e-5)
    """
    c0, c1, c3, c4 = corners[0], corners[1], corners[3], corners[4]
    u = c1 - c0
    v = c3 - c0
    w = c4 - c0

    center = corners.mean(axis=0)
    su = float(np.linalg.norm(u))
    sv = float(np.linalg.norm(v))
    sw = float(np.linalg.norm(w))
    size = np.array([su, sv, sw], dtype=np.float64)

    nu = _normalise(u)
    nv = _normalise(v)
    # Negate w_hat so that R = [u_hat, v_hat, -w_hat] has det = -det([u,v,w]) = +1
    nw = -_normalise(w)

    R = np.stack([nu, nv, nw], axis=1)  # (3,3), det = +1 for all GT boxes
    return center, size, R


def params_to_corners_np(
    center: np.ndarray,
    size: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct 8×3 corners from (center, size, R) — numpy version.

    Uses the LEFT-HANDED sign template (z-column negated vs the naive template).
    Paired with corners_to_params (which uses R[:,2]=-w_hat), this achieves
    exact roundtrip for all GT boxes while R stays proper (det+1).

    Sign layout (z negated to make template left-handed):
        c0=(-,-,+), c1=(+,-,+), c2=(+,+,+), c3=(-,+,+)
        c4=(-,-,-), c5=(+,-,-), c6=(+,+,-), c7=(-,+,-)
    """
    signs = np.array([
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
    ], dtype=np.float64)  # (8,3) — left-handed z-negated template

    half = signs * (size / 2)[None, :]      # (8,3)
    corners = center[None, :] + half @ R.T  # (8,3)
    return corners


# ---------------------------------------------------------------------------
# Torch version (batched, differentiable)
# ---------------------------------------------------------------------------

# Left-handed sign template — z-column negated so that det([u,v,w])=-1 with proper R.
# Must match params_to_corners_np exactly.
_SIGNS = torch.tensor([
    [-1., -1.,  1.],
    [ 1., -1.,  1.],
    [ 1.,  1.,  1.],
    [-1.,  1.,  1.],
    [-1., -1., -1.],
    [ 1., -1., -1.],
    [ 1.,  1., -1.],
    [-1.,  1., -1.],
], dtype=torch.float32)  # (8,3)


def params_to_corners_torch(
    center: torch.Tensor,
    size: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """
    Batched differentiable corners from (center, size, R).

    center : (..., 3)
    size   : (..., 3)   positive
    R      : (..., 3, 3)  proper rotation (columns are axis directions)
    returns: (..., 8, 3)
    """
    signs = _SIGNS.to(center.device)          # (8,3)
    half = signs * (size / 2).unsqueeze(-2)   # (..., 8, 3)
    # corners = center + (R @ half[..., i])  for each i
    # Vectorised: half @ R^T  since R maps local->world
    corners = center.unsqueeze(-2) + half @ R.transpose(-1, -2)  # (..., 8, 3)
    return corners


# ---------------------------------------------------------------------------
# Gram-Schmidt rotation from 6D representation (Zhou et al. 2019)
# ---------------------------------------------------------------------------

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3×3 proper rotation matrix.

    x : (..., 6)  — first two columns of rotation matrix (not necessarily ortho-normal)
    returns: (..., 3, 3)
    """
    a1 = x[..., :3]
    a2 = x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.linalg.cross(b1, b2)
    R = torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3) columns are b1,b2,b3
    return R


# ---------------------------------------------------------------------------
# PCA frame
# ---------------------------------------------------------------------------

def pca_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA frame for a point cloud with deterministic axis signs.

    points : (N, 3)
    returns:
        R0 : (3,3)  proper rotation (det +1); columns = PCA axes, sorted eigenvalue descending
        t0 : (3,)   centroid (camera frame)

    Sign conventions (applied in order, axis 2 anchored first):
      - Axis 2 (~depth / view normal): always points away from camera (R0[:,2] @ t0 > 0).
      - Axis 0 (largest variance): sign fixed by skewness of projected distribution.
      - Axis 1: derived from det to preserve handedness, never touched independently.
    """
    t0 = points.mean(axis=0)
    centered = points - t0
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # eigvecs columns, ascending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    R0 = eigvecs[:, idx].copy()
    # Axis 2: always points away from camera (camera is at origin)
    if R0[:, 2] @ t0 < 0:
        R0[:, 2] *= -1
    # Axis 0: fix sign by skewness of projected coords
    proj = centered @ R0[:, 0]
    if (proj ** 3).mean() < 0:
        R0[:, 0] *= -1
    # Axis 1: fix handedness without touching axis 2
    if np.linalg.det(R0) < 0:
        R0[:, 1] *= -1
    return R0, t0


# ---------------------------------------------------------------------------
# Symmetry permutations
# ---------------------------------------------------------------------------

def _all_proper_axis_matrices() -> list[np.ndarray]:
    """
    Generate all 24 proper rotation matrices of the cube (axis-permutation + sign matrices).
    Each is a 3×3 matrix with exactly one ±1 per row and column, det = +1.
    """
    mats = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([-1, 1], repeat=3):
            M = np.zeros((3, 3), dtype=np.float64)
            for row, col in enumerate(perm):
                M[row, col] = signs[row]
            if round(np.linalg.det(M)) == 1:
                mats.append(M)
    return mats


def _corner_signs() -> np.ndarray:
    """Return the 8×3 left-handed sign template (same as _SIGNS above, numpy)."""
    return np.array([
        [-1., -1.,  1.],
        [ 1., -1.,  1.],
        [ 1.,  1.,  1.],
        [-1.,  1.,  1.],
        [-1., -1., -1.],
        [ 1., -1., -1.],
        [ 1.,  1., -1.],
        [-1.,  1., -1.],
    ], dtype=np.float64)


@lru_cache(maxsize=1)
def get_symmetry_perms() -> np.ndarray:
    """
    Compute the 24 corner-index permutations induced by the rotation group of the cube.

    Each proper axis-rotation matrix Q maps the unit-cube corners to a relabeling of the
    same corners.  We apply each Q to the sign template, match each transformed sign row
    back to its index in the original template, and collect the resulting permutation.

    Returns: (24, 8) int array — SYM_PERMS[k, i] = j means corner i in prediction
             corresponds to corner j in the permuted ground truth.
    """
    template = _corner_signs()  # (8, 3)
    proper_rots = _all_proper_axis_matrices()

    perms = []
    for Q in proper_rots:
        # Rotate all 8 corner sign-vectors: (8,3) @ Q.T = (8,3)
        rotated = template @ Q.T
        perm = np.zeros(8, dtype=np.int64)
        for i, r in enumerate(rotated):
            # Find which row of template matches r exactly
            diffs = np.abs(template - r[None, :]).max(axis=1)
            j = int(np.argmin(diffs))
            assert diffs[j] < 0.5, f"Corner match failed for Q={Q}, i={i}"
            perm[i] = j
        perms.append(perm)

    perms_arr = np.array(perms, dtype=np.int64)  # (24, 8)
    # Deduplicate (should already be 24 unique)
    unique = np.unique(perms_arr, axis=0)
    assert len(unique) == 24, f"Expected 24 unique perms, got {len(unique)}"
    return unique


SYM_PERMS: np.ndarray = get_symmetry_perms()  # (24, 8) — module-level constant

# Torch version for use in loss
_SYM_PERMS_TORCH: torch.Tensor | None = None


def get_sym_perms_torch(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return SYM_PERMS as a (24, 8) long tensor on the given device."""
    global _SYM_PERMS_TORCH
    t = torch.from_numpy(SYM_PERMS).long().to(device)
    return t
