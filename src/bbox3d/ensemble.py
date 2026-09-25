"""
Ensemble utilities for 3-seed box predictions (Step 5).

For each instance:
  1. Align corners of seed43, seed44 to seed42 using the best permutation from SYM_PERMS
     (permutation that minimises mean corner L2 distance to seed42).
  2. Average the three aligned corner arrays.
  3. Fit a valid oriented box from the averaged corners:
       center  = mean of 8 corners
       u,v,w   = edge vectors from c0 (averaged)
       R_raw   = stack [u_hat, v_hat, w_hat]  (may not be orthonormal)
       R       = closest proper rotation via SVD (Procrustes)
       size    = [|u|, |v|, |w|] averaged from all 3 pairwise directions
  4. Reconstruct corners from (center, size, R).
  5. Verify box: R orthogonal (max col-pair dot < 1e-6) and det(R) > 0.

Returns (ensemble_corners, stats).
"""
from __future__ import annotations

import numpy as np
from bbox3d.geometry.box import SYM_PERMS, corners_to_params, params_to_corners_np


def _align_to_reference(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Return target corners permuted so they best align to ref.
    ref, target: (8, 3).
    Returns (8, 3) permuted copy of target.
    """
    best_dist = float("inf")
    best_perm = SYM_PERMS[0]
    for perm in SYM_PERMS:
        dist = float(np.mean(np.linalg.norm(ref - target[perm], axis=-1)))
        if dist < best_dist:
            best_dist = dist
            best_perm = perm
    return target[best_perm].copy()


def _fit_box_from_corners(corners: np.ndarray) -> np.ndarray:
    """
    Fit a valid oriented box from (approximately) valid 8-corner array.

    Uses the exact corners_to_params decomposition (which already handles
    the left-handed convention), then rebuilds.  This is an exact round-trip
    so no Procrustes fitting is needed for valid inputs.

    Returns (8, 3) corners of the fitted box.
    """
    center, size, R = corners_to_params(corners)

    # Ensure proper rotation (det +1) via SVD
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt

    return params_to_corners_np(center, size, R_clean)


def _check_box(corners: np.ndarray) -> bool:
    """Return True if the box has orthogonal axes and det>0."""
    _, _, R = corners_to_params(corners)
    off_diag = abs(R.T @ R - np.eye(3)).max()
    det = np.linalg.det(R)
    return bool(off_diag < 1e-5 and det > 0)


def ensemble_three_seeds(
    corners_42: np.ndarray,   # (8, 3) seed 42
    corners_43: np.ndarray,   # (8, 3) seed 43
    corners_44: np.ndarray,   # (8, 3) seed 44
) -> tuple[np.ndarray, dict]:
    """
    Ensemble 3 sets of corners into one valid box.

    Returns (ensemble_corners (8,3), stats dict).
    stats: aligned_43_dist, aligned_44_dist, valid.
    """
    aligned_43 = _align_to_reference(corners_42, corners_43)
    aligned_44 = _align_to_reference(corners_42, corners_44)

    dist43 = float(np.mean(np.linalg.norm(corners_42 - aligned_43, axis=-1)))
    dist44 = float(np.mean(np.linalg.norm(corners_42 - aligned_44, axis=-1)))

    mean_corners = (corners_42 + aligned_43 + aligned_44) / 3.0

    fitted = _fit_box_from_corners(mean_corners)
    valid  = _check_box(fitted)

    return fitted, {
        "aligned_43_dist_mm": dist43 * 1000,
        "aligned_44_dist_mm": dist44 * 1000,
        "valid": valid,
    }
