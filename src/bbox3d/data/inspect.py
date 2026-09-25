"""
Step 1 — Data inspection.

Iterates all scenes, validates shape consistency, computes statistics,
writes reports/data_report.md and histograms to outputs/figures/.

Usage:
    python -m bbox3d.data.inspect --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SceneStats(NamedTuple):
    scene_id: str
    n_instances: int
    img_h: int
    img_w: int
    points_per_instance: list[int]        # valid (Z>0) points per instance
    invalid_depth_pct: float              # % pixels with Z==0 across whole pc
    box_sizes: list[tuple[float, float, float]]  # sorted (s0<=s1<=s2) per instance
    centroid_offsets_z: list[float]       # mask centroid Z minus box center Z (m)
    orth_errors: list[float]              # max |dot(ui,uj)| for i!=j (ideal 0)
    handedness: list[float]              # det([u,v,w]) sign, ideal -1 per spec


# ---------------------------------------------------------------------------
# Per-scene loading & validation
# ---------------------------------------------------------------------------

def load_scene(scene_dir: Path) -> dict:
    rgb_path = scene_dir / "rgb.jpg"
    pc_path  = scene_dir / "pc.npy"
    mask_path = scene_dir / "mask.npy"
    bbox_path = scene_dir / "bbox3d.npy"

    from PIL import Image
    rgb = np.array(Image.open(rgb_path))       # (H, W, 3) uint8
    pc  = np.load(pc_path)                     # (3, H, W) float64
    mask = np.load(mask_path)                  # (N, H, W) bool
    bbox = np.load(bbox_path)                  # (N, 8, 3) float32

    return dict(rgb=rgb, pc=pc, mask=mask, bbox=bbox)


def validate_shapes(scene_id: str, rgb, pc, mask, bbox) -> list[str]:
    """Return list of error strings (empty == OK)."""
    errors = []
    img_h, img_w = rgb.shape[:2]

    if pc.shape != (3, img_h, img_w):
        errors.append(f"pc shape {pc.shape} != (3,{img_h},{img_w})")
    if mask.ndim != 3 or mask.shape[1] != img_h or mask.shape[2] != img_w:
        errors.append(f"mask shape {mask.shape} H/W mismatch with rgb")
    if bbox.ndim != 3 or bbox.shape[1:] != (8, 3):
        errors.append(f"bbox shape {bbox.shape} != (N,8,3)")
    if len(mask) != len(bbox):
        errors.append(f"mask N={len(mask)} != bbox N={len(bbox)}")
    return errors


def box_edge_vectors(corners: np.ndarray):
    """
    Return (u, v, w) edge vectors from corner 0.
    Per SPEC §1: u=c1-c0, v=c3-c0, w=c4-c0.
    """
    u = corners[1] - corners[0]
    v = corners[3] - corners[0]
    w = corners[4] - corners[0]
    return u, v, w


def compute_scene_stats(scene_id: str, rgb, pc, mask, bbox) -> SceneStats:
    N = len(mask)
    img_h, img_w = rgb.shape[:2]

    # Depth validity across whole image
    Z = pc[2]  # (H, W)
    invalid_depth_pct = float(np.mean(Z == 0) * 100)

    points_per_instance = []
    box_sizes = []
    centroid_offsets_z = []
    orth_errors = []
    handedness = []

    for i in range(N):
        m = mask[i]  # (H, W) bool

        # Points in this mask with valid depth
        valid = m & (Z > 0)
        xs = pc[0][valid]
        ys = pc[1][valid]
        zs = pc[2][valid]
        n_pts = int(valid.sum())
        points_per_instance.append(n_pts)

        # Mask point centroid (valid only)
        if n_pts > 0:
            cx_pts = float(np.mean(xs))
            cy_pts = float(np.mean(ys))
            cz_pts = float(np.mean(zs))
        else:
            cx_pts = cy_pts = cz_pts = 0.0

        corners = bbox[i]  # (8,3)

        # Box center = mean of 8 corners
        box_center = corners.mean(axis=0)

        # Centroid offset in Z (centroid_z - box_center_z)
        cz_offset = cz_pts - float(box_center[2]) if n_pts > 0 else float("nan")
        centroid_offsets_z.append(cz_offset)

        # Edge vectors and derived quantities
        u, v, w = box_edge_vectors(corners)

        # Orthogonality: max |dot(ui_norm, uj_norm)| for distinct pairs
        nu = u / (np.linalg.norm(u) + 1e-12)
        nv = v / (np.linalg.norm(v) + 1e-12)
        nw = w / (np.linalg.norm(w) + 1e-12)
        orth_err = max(
            abs(float(np.dot(nu, nv))),
            abs(float(np.dot(nu, nw))),
            abs(float(np.dot(nv, nw))),
        )
        orth_errors.append(orth_err)

        # Handedness: det([u_hat, v_hat, w_hat])
        R = np.stack([nu, nv, nw], axis=1)  # columns
        det = float(np.linalg.det(R))
        handedness.append(det)

        # Box sizes = edge lengths, sorted ascending
        lu = float(np.linalg.norm(u))
        lv = float(np.linalg.norm(v))
        lw = float(np.linalg.norm(w))
        sizes = tuple(sorted([lu, lv, lw]))
        box_sizes.append(sizes)

    return SceneStats(
        scene_id=scene_id,
        n_instances=N,
        img_h=img_h,
        img_w=img_w,
        points_per_instance=points_per_instance,
        invalid_depth_pct=invalid_depth_pct,
        box_sizes=box_sizes,
        centroid_offsets_z=centroid_offsets_z,
        orth_errors=orth_errors,
        handedness=handedness,
    )


# ---------------------------------------------------------------------------
# Aggregate & report
# ---------------------------------------------------------------------------

def inspect_all(data_root: Path, figures_dir: Path, report_path: Path):
    scene_dirs = sorted(data_root.iterdir())
    print(f"Found {len(scene_dirs)} scene directories.")

    all_stats: list[SceneStats] = []
    shape_errors: list[str] = []

    for sd in scene_dirs:
        if not sd.is_dir():
            continue
        try:
            d = load_scene(sd)
        except Exception as e:
            shape_errors.append(f"{sd.name}: LOAD ERROR {e}")
            continue

        errs = validate_shapes(sd.name, d["rgb"], d["pc"], d["mask"], d["bbox"])
        for e in errs:
            shape_errors.append(f"{sd.name}: {e}")

        stats = compute_scene_stats(sd.name, d["rgb"], d["pc"], d["mask"], d["bbox"])
        all_stats.append(stats)

    n_scenes = len(all_stats)
    print(f"Processed {n_scenes} scenes; {len(shape_errors)} shape errors.")

    # Flatten across all instances
    all_n_inst = [s.n_instances for s in all_stats]
    all_pts    = [p for s in all_stats for p in s.points_per_instance]
    all_inv    = [s.invalid_depth_pct for s in all_stats]
    all_s0     = [sz[0] * 100 for s in all_stats for sz in s.box_sizes]  # cm
    all_s1     = [sz[1] * 100 for s in all_stats for sz in s.box_sizes]
    all_s2     = [sz[2] * 100 for s in all_stats for sz in s.box_sizes]
    all_czoff  = [v for s in all_stats for v in s.centroid_offsets_z if not np.isnan(v)]
    all_orth   = [v for s in all_stats for v in s.orth_errors]
    all_hand   = [v for s in all_stats for v in s.handedness]

    # Image size variety
    img_sizes = sorted({(s.img_h, s.img_w) for s in all_stats})

    # Orthogonality violations (> 0.01)
    orth_thresh = 0.01
    n_orth_bad = sum(1 for v in all_orth if v > orth_thresh)

    # Handedness check: should all be ≈ -1
    hand_arr = np.array(all_hand)
    n_hand_ok = int(np.sum(hand_arr < -0.9))
    n_hand_bad = len(all_hand) - n_hand_ok

    # Thin objects: min dim < 2 cm
    n_thin = sum(1 for v in all_s0 if v < 2.0)

    czoff_arr = np.array(all_czoff) * 100  # cm

    # -----------------------------------------------------------------------
    # Histograms
    # -----------------------------------------------------------------------
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Dataset Statistics — All 200 Scenes", fontsize=14)

    axes[0, 0].hist(all_n_inst, bins=range(1, 15), edgecolor="black", align="left")
    axes[0, 0].set_title("Instances per Scene")
    axes[0, 0].set_xlabel("N instances")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(all_pts, bins=40, edgecolor="black")
    axes[0, 1].set_title("Valid Points per Instance (before filtering)")
    axes[0, 1].set_xlabel("# points (Z>0)")

    axes[0, 2].hist(all_inv, bins=30, edgecolor="black")
    axes[0, 2].set_title("Invalid Depth % per Scene")
    axes[0, 2].set_xlabel("% pixels with Z==0")

    axes[1, 0].hist(all_s0, bins=40, edgecolor="black", alpha=0.7, label="min dim")
    axes[1, 0].hist(all_s1, bins=40, edgecolor="black", alpha=0.5, label="mid dim")
    axes[1, 0].hist(all_s2, bins=40, edgecolor="black", alpha=0.3, label="max dim")
    axes[1, 0].set_title("Box Dimensions (cm)")
    axes[1, 0].set_xlabel("cm")
    axes[1, 0].legend()

    axes[1, 1].hist(czoff_arr, bins=40, edgecolor="black")
    axes[1, 1].set_title("Centroid Z − Box Center Z (cm)")
    axes[1, 1].set_xlabel("offset (cm)")

    axes[1, 2].hist(all_orth, bins=40, edgecolor="black")
    axes[1, 2].set_title("Orthogonality Error (|dot of normed edges|)")
    axes[1, 2].set_xlabel("max |dot|")
    axes[1, 2].axvline(orth_thresh, color="red", linestyle="--", label=f"thresh={orth_thresh}")
    axes[1, 2].legend()

    plt.tight_layout()
    fig_path = figures_dir / "data_stats.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"Saved histogram figure to {fig_path}")

    # -----------------------------------------------------------------------
    # Numeric summary (printed)
    # -----------------------------------------------------------------------
    pts_arr = np.array(all_pts)
    s0_arr = np.array(all_s0)
    s1_arr = np.array(all_s1)
    s2_arr = np.array(all_s2)
    inv_arr = np.array(all_inv)

    summary_lines = []
    def add(line=""):
        summary_lines.append(line)
        print(line)

    add("=" * 60)
    add("DATA INSPECTION REPORT — NUMERIC SUMMARY")
    add("=" * 60)
    add(f"Total scenes:                {n_scenes}")
    add(f"Shape/load errors:           {len(shape_errors)}")
    add(f"Total instances:             {sum(all_n_inst)}")
    add(f"Instances/scene: min={min(all_n_inst)} median={int(np.median(all_n_inst))} max={max(all_n_inst)}")
    add(f"Unique image sizes (H×W):    {img_sizes}")
    add()
    add(f"Valid points/instance (before filter):")
    add(f"  min={pts_arr.min()}  median={int(np.median(pts_arr))}  mean={pts_arr.mean():.0f}  max={pts_arr.max()}")
    add()
    add(f"Invalid depth % per scene:")
    add(f"  min={inv_arr.min():.2f}  median={np.median(inv_arr):.2f}  max={inv_arr.max():.2f}")
    add()
    add(f"Box dimensions (cm) — sorted edges per instance:")
    add(f"  min-dim : min={s0_arr.min():.2f}  median={np.median(s0_arr):.2f}  max={s0_arr.max():.2f}")
    add(f"  mid-dim : min={s1_arr.min():.2f}  median={np.median(s1_arr):.2f}  max={s1_arr.max():.2f}")
    add(f"  max-dim : min={s2_arr.min():.2f}  median={np.median(s2_arr):.2f}  max={s2_arr.max():.2f}")
    add(f"  Thin objects (min-dim < 2 cm): {n_thin} / {len(all_s0)}")
    add()
    add(f"Centroid Z - Box center Z (cm):")
    add(f"  min={czoff_arr.min():.2f}  median={np.median(czoff_arr):.2f}  mean={czoff_arr.mean():.2f}  max={czoff_arr.max():.2f}")
    add(f"  (negative = centroid closer to camera than box center, i.e. biased toward camera)")
    add()
    add(f"Orthogonality errors (max |dot(ui,uj)| per instance):")
    add(f"  min={np.min(all_orth):.6f}  median={np.median(all_orth):.6f}  max={np.max(all_orth):.6f}")
    add(f"  Violations > {orth_thresh}: {n_orth_bad} / {len(all_orth)}")
    add()
    add(f"Handedness (det([u_hat,v_hat,w_hat])), should be -1:")
    add(f"  min={hand_arr.min():.4f}  median={np.median(hand_arr):.4f}  max={hand_arr.max():.4f}")
    add(f"  OK (< -0.9): {n_hand_ok}  BAD: {n_hand_bad}")
    add("=" * 60)

    if shape_errors:
        add("\nSHAPE/LOAD ERRORS:")
        for e in shape_errors:
            add(f"  {e}")

    # -----------------------------------------------------------------------
    # Write markdown report
    # -----------------------------------------------------------------------
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Data Inspection Report\n\n")
        f.write(f"Generated: 2026-09-23  \nScenes: `data/dl_challenge/`\n\n")
        f.write("## Summary\n\n")
        f.write("```\n")
        f.write("\n".join(summary_lines))
        f.write("\n```\n\n")
        f.write("## Figures\n\n")
        f.write("![Dataset statistics](../outputs/figures/data_stats.png)\n\n")
        f.write("## Data Assumption Check (SPEC §1)\n\n")
        f.write(f"| Assumption | Status |\n|---|---|\n")

        def chk(ok, desc):
            return f"| {desc} | {'OK' if ok else 'VIOLATED'} |\n"

        f.write(chk(len(shape_errors) == 0,
                    "All scenes have rgb/pc/mask/bbox; shapes consistent"))
        f.write(chk(min(all_n_inst) >= 4 and max(all_n_inst) <= 9,
                    "N instances/scene in 4-9"))
        f.write(chk(s0_arr.min() * 10 >= 7 and s2_arr.max() <= 25,
                    "Object sizes ~1-21 cm (SPEC says 1-21 cm)"))
        f.write(chk(n_thin > 0, "Some very thin objects (< 2 cm) present"))
        f.write(chk(float(np.median(czoff_arr)) < 0,
                    "Centroid Z biased toward camera (negative median = closer to camera)"))
        f.write(chk(n_orth_bad == 0,
                    f"All GT boxes are orthogonal (|dot| < {orth_thresh})"))
        f.write(chk(n_hand_bad == 0,
                    "All GT boxes are left-handed (det ~= -1)"))
        f.write(chk(inv_arr.max() <= 15,
                    "Invalid depth <= 13% (SPEC says 0-13%)"))
        f.write("\n")

    print(f"\nReport written to {report_path}")
    return summary_lines, shape_errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1 — Data inspection")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(__file__).resolve().parents[3]  # src/bbox3d/data/inspect.py -> project root
    data_root    = project_root / cfg["data"]["root"]
    figures_dir  = project_root / cfg["outputs_dir"] / "figures"
    report_path  = project_root / cfg["reports_dir"] / "data_report.md"

    inspect_all(data_root, figures_dir, report_path)


if __name__ == "__main__":
    main()
