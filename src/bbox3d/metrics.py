"""
Metrics for 3D oriented bounding box evaluation (Step 7 infrastructure, used from Step 4).

Key functions:
  box_iou_3d          — exact IoU via half-space intersection + ConvexHull
  symmetric_corner_distance — min over 24 SYM_PERMS of mean corner dist (mm)
  evaluate_instance   — all metrics for one (pred, gt) pair
  aggregate_metrics   — mean/median + thin/normal breakdown over a list

Half-space convention (scipy):  [n | d]  where  n·x + d <= 0  (i.e.  n·x <= -d).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

from bbox3d.geometry.box import SYM_PERMS, corners_to_params


# ---------------------------------------------------------------------------
# Corner distance
# ---------------------------------------------------------------------------

def symmetric_corner_distance(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[float, int]:
    """
    Min over 24 SYM_PERMS of mean L2 corner distance.

    Returns (distance_mm, argmin_perm_index).
    Both arrays are (8,3) in metres.
    """
    best_dist = float("inf")
    best_k = 0
    for k, perm in enumerate(SYM_PERMS):
        dist = float(np.mean(np.linalg.norm(pred - gt[perm], axis=-1)))
        if dist < best_dist:
            best_dist = dist
            best_k = k
    return best_dist * 1000.0, best_k   # mm


# ---------------------------------------------------------------------------
# Exact 3D IoU
# ---------------------------------------------------------------------------

def _box_halfspaces(
    center: np.ndarray, size: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """
    Build (6,4) half-space matrix [n | d] where n·x + d <= 0
    for the box {x : |R[:,i]·(x-c)| <= s_i/2  ∀i}.

    For each axis i:
      upper face:  R[:,i]·x  + (-R[:,i]·c - s_i/2) <= 0
      lower face: -R[:,i]·x  + ( R[:,i]·c - s_i/2) <= 0
    """
    rows = []
    for i in range(3):
        n  = R[:, i]
        nd = float(np.dot(n, center))
        half = size[i] / 2.0
        rows.append(np.append( n,  -(nd + half)))   # upper
        rows.append(np.append(-n,    nd - half))    # lower
    return np.array(rows, dtype=np.float64)  # (6,4)


def _chebyshev_center(
    halfspaces: np.ndarray,
) -> tuple[Optional[np.ndarray], float]:
    """
    Chebyshev center and radius for a polytope given as (M,4) half-spaces.

    Finds the largest ball of radius r centred at x0 that fits inside.
    Constraint for each halfspace [n|d]:
        n/|n| · x0 + r  <=  -d/|n|
    Variables: [x0_0, x0_1, x0_2, r].

    Returns (center_3d, radius).  radius <= 0 means infeasible / empty.
    """
    m = halfspaces.shape[0]
    norms = np.linalg.norm(halfspaces[:, :3], axis=1)
    norms = np.where(norms < 1e-12, 1e-12, norms)

    A_ub = np.column_stack([halfspaces[:, :3] / norms[:, None],
                             np.ones(m)])                      # (M,4)
    b_ub = -halfspaces[:, 3] / norms                           # (M,)
    c    = np.array([0.0, 0.0, 0.0, -1.0])                    # maximise r

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * 3 + [(None, None)],
                  method="highs")

    if res.status != 0:
        return None, -1.0
    return res.x[:3], float(res.x[3])


def box_iou_3d(pred_corners: np.ndarray, gt_corners: np.ndarray) -> float:
    """
    Exact 3D IoU between two oriented boxes given by (8,3) corner arrays.

    Each box is represented as 6 half-spaces; the intersection polytope
    volume is computed via scipy.spatial.HalfspaceIntersection + ConvexHull.
    Returns 0.0 for disjoint or degenerate boxes.
    """
    c_p, s_p, R_p = corners_to_params(pred_corners.astype(np.float64))
    c_g, s_g, R_g = corners_to_params(gt_corners.astype(np.float64))

    vol_p = float(np.prod(s_p))
    vol_g = float(np.prod(s_g))
    if vol_p < 1e-15 or vol_g < 1e-15:
        return 0.0

    hs = np.vstack([_box_halfspaces(c_p, s_p, R_p),
                    _box_halfspaces(c_g, s_g, R_g)])  # (12,4)

    interior, radius = _chebyshev_center(hs)
    if interior is None or radius <= 1e-9:
        return 0.0   # disjoint or only touching

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hs_int   = HalfspaceIntersection(hs, interior)
            hull     = ConvexHull(hs_int.intersections)
            vol_inter = hull.volume
    except Exception:
        return 0.0

    vol_union = vol_p + vol_g - vol_inter
    if vol_union < 1e-15:
        return 0.0
    return float(np.clip(vol_inter / vol_union, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(
    pred_corners: np.ndarray,   # (8,3) metres
    gt_corners: np.ndarray,     # (8,3) metres
) -> dict:
    """
    Compute all metrics for one (pred, gt) box pair.

    Returns dict with:
      corner_dist_mm, iou_3d,
      center_err_mm, size_err_mm, rot_err_deg,
      is_thin  (bool: GT min-dim < 2 cm)
    """
    pred = pred_corners.astype(np.float64)
    gt   = gt_corners.astype(np.float64)

    # Symmetric corner distance + best permutation index
    corner_dist_mm, k_best = symmetric_corner_distance(pred, gt)

    # 3D IoU
    iou = box_iou_3d(pred, gt)

    # Center error
    center_err_mm = float(
        np.linalg.norm(pred.mean(axis=0) - gt.mean(axis=0))
    ) * 1000.0

    # Rotation + size error (using the best-matching permutation)
    gt_perm = gt[SYM_PERMS[k_best]]
    _, s_pred, R_pred = corners_to_params(pred)
    _, s_gt_p, R_gt_p = corners_to_params(gt_perm)

    # Rotation error: geodesic angle between R_pred and R_gt_p
    dR    = R_pred.T @ R_gt_p
    cos_a = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.degrees(np.arccos(cos_a)))

    # Size error: mean absolute difference on matched axes
    size_err_mm = float(np.mean(np.abs(s_pred - s_gt_p))) * 1000.0

    # Thin flag: GT min edge < 2 cm
    _, s_gt_raw, _ = corners_to_params(gt)
    is_thin = bool(np.min(s_gt_raw) < 0.02)

    return dict(
        corner_dist_mm=corner_dist_mm,
        iou_3d=iou,
        center_err_mm=center_err_mm,
        size_err_mm=size_err_mm,
        rot_err_deg=rot_err_deg,
        is_thin=is_thin,
    )


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def _split_stats(data: list[dict], key: str) -> dict:
    vals = np.array([d[key] for d in data], dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(np.mean(finite)), "median": float(np.median(finite))}


def _group_summary(data: list[dict]) -> Optional[dict]:
    if not data:
        return None
    ious = [d["iou_3d"] for d in data]
    return {
        "n": len(data),
        "mean_iou":        float(np.mean(ious)),
        "acc_025":         float(np.mean([i >= 0.25 for i in ious])),
        "acc_050":         float(np.mean([i >= 0.50 for i in ious])),
        "corner_dist_mm":  _split_stats(data, "corner_dist_mm"),
        "center_err_mm":   _split_stats(data, "center_err_mm"),
        "size_err_mm":     _split_stats(data, "size_err_mm"),
        "rot_err_deg":     _split_stats(data, "rot_err_deg"),
    }


def aggregate_metrics(results: list[dict]) -> dict:
    """
    Compute mean/median statistics + thin/normal breakdown.

    Returns dict with keys 'all', 'normal', 'thin'.
    """
    normal = [r for r in results if not r["is_thin"]]
    thin   = [r for r in results if r["is_thin"]]
    return {
        "all":    _group_summary(results),
        "normal": _group_summary(normal),
        "thin":   _group_summary(thin),
    }


# ---------------------------------------------------------------------------
# Markdown table helper
# ---------------------------------------------------------------------------

def _fmt_summary(s: Optional[dict]) -> str:
    if s is None:
        return "—"
    def f(x): return f"{x:.4f}" if isinstance(x, float) else str(x)
    def fs(d, k): return f"{d[k]['mean']:.1f} / {d[k]['median']:.1f}"
    return (
        f"n={s['n']}  IoU={s['mean_iou']:.3f}  "
        f"Acc@0.25={s['acc_025']:.3f}  Acc@0.5={s['acc_050']:.3f}  "
        f"corner={fs(s,'corner_dist_mm')}mm  "
        f"center={fs(s,'center_err_mm')}mm  "
        f"size={fs(s,'size_err_mm')}mm  "
        f"rot={fs(s,'rot_err_deg')}deg"
    )


def metrics_markdown_table(agg: dict, title: str = "") -> str:
    """Return a markdown table string for one split's aggregated metrics."""
    rows = [
        f"## {title}\n" if title else "",
        "| Group | N | mean IoU | Acc@0.25 | Acc@0.5 | corner dist (mean/med mm) "
        "| center err (mean/med mm) | size err (mean/med mm) | rot err (mean/med deg) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group in ["all", "normal", "thin"]:
        s = agg.get(group)
        if s is None:
            continue
        def fs(k): return f"{s[k]['mean']:.1f} / {s[k]['median']:.1f}"
        rows.append(
            f"| {group} | {s['n']} | {s['mean_iou']:.4f} | {s['acc_025']:.3f} | "
            f"{s['acc_050']:.3f} | {fs('corner_dist_mm')} | "
            f"{fs('center_err_mm')} | {fs('size_err_mm')} | {fs('rot_err_deg')} |"
        )
    return "\n".join(rows)
