"""
Estimate camera intrinsics (fx, fy, cx, cy) from an organized point cloud.

The organized cloud satisfies:
    u = fx * X/Z + cx
    v = fy * Y/Z + cy
for valid pixels (Z > 0).  We solve each axis independently by least squares.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def estimate_intrinsics(pc: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Least-squares estimate of (fx, fy, cx, cy) from an organized point cloud.

    pc : (3, H, W) float64  — XYZ in camera frame; Z==0 marks invalid pixels.
    returns: fx, fy, cx, cy  (all floats)

    Model:  u = fx*(X/Z) + cx   v = fy*(Y/Z) + cy
    Solved separately per axis with np.linalg.lstsq.
    """
    X, Y, Z = pc[0], pc[1], pc[2]   # (H, W) each
    H, W = Z.shape

    # Pixel coordinates
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))  # (H, W)

    valid = Z > 0
    if valid.sum() < 10:
        # Fallback: assume principal point at centre, approximate focal
        return float(W), float(W), float(W / 2), float(H / 2)

    XoZ = (X[valid] / Z[valid]).ravel()   # (M,)
    YoZ = (Y[valid] / Z[valid]).ravel()
    u_v = us[valid].ravel()
    v_v = vs[valid].ravel()

    # u = fx*(X/Z) + cx  →  [X/Z, 1] @ [fx, cx]^T = u
    A_u = np.stack([XoZ, np.ones_like(XoZ)], axis=1)
    params_u, _, _, _ = np.linalg.lstsq(A_u, u_v, rcond=None)
    fx, cx = float(params_u[0]), float(params_u[1])

    A_v = np.stack([YoZ, np.ones_like(YoZ)], axis=1)
    params_v, _, _, _ = np.linalg.lstsq(A_v, v_v, rcond=None)
    fy, cy = float(params_v[0]), float(params_v[1])

    return fx, fy, cx, cy


def reprojection_error(
    pc: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> Tuple[float, float]:
    """
    Compute reprojection error for estimated intrinsics.

    Returns (median_px, mean_px) pixel error over valid pixels.
    """
    X, Y, Z = pc[0], pc[1], pc[2]
    H, W = Z.shape
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    valid = Z > 0
    if valid.sum() == 0:
        return float("nan"), float("nan")

    u_pred = fx * X[valid] / Z[valid] + cx
    v_pred = fy * Y[valid] / Z[valid] + cy
    err = np.sqrt((u_pred - us[valid]) ** 2 + (v_pred - vs[valid]) ** 2)
    return float(np.median(err)), float(np.mean(err))
