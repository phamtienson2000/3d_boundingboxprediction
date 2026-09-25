"""
Step 10 — Visualization (viz.py).

Produces:
    outputs/figures/training_curves.png
    outputs/figures/pred_<scene_id>.png   (≥8 test scenes, 2D overlay)
    outputs/figures/3d_<scene_id>.png     (3D matplotlib, all overlay scenes)
    outputs/figures/3d_<scene_id>.html    (plotly interactive, first scene only)
    outputs/figures/failure_cases.png     (4 lowest-IoU test instances)

Usage:
    python -m bbox3d.viz --config configs/default.yaml [--run main_run] [--n-scenes 8]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from torch.utils.data import DataLoader
import yaml

from bbox3d.data.dataset import BBox3DDataset
from bbox3d.geometry.intrinsics import estimate_intrinsics
from bbox3d.losses import compute_log_size_prior
from bbox3d.metrics import evaluate_instance
from bbox3d.models.pointnet_box import build_model


# 12 edges of a box defined by the GT corner convention (faces 0-1-2-3 / 4-5-6-7)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # pillars
]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_corners(
    corners: np.ndarray,             # (8, 3) camera frame, metres
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:                     # (8, 2) pixel coords
    X, Y, Z = corners[:, 0], corners[:, 1], corners[:, 2]
    Z = np.where(Z <= 0, 1e-6, Z)
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# Inference on the test split
# ---------------------------------------------------------------------------

def run_inference(
    cfg: dict,
    ckpt_path: Path,
    items: list[dict],
    cache_dir: Path,
) -> list[dict]:
    """
    Return per-instance result dicts:
      scene_id, inst_id, pred_corners_cam (8,3), gt_corners_cam (8,3), iou_3d
    """
    ckpt     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", cfg)

    with open(cache_dir / "split.json") as f:
        split_data = json.load(f)
    log_size_init = compute_log_size_prior(split_data["train"], cache_dir)

    model = build_model(ckpt_cfg, log_size_init)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = BBox3DDataset(
        items, cache_dir, ckpt_cfg, augment_data=False,
        seed=int(ckpt_cfg.get("train", {}).get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    results: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            pts      = batch["pts"]
            extra    = batch["extra"]
            gt_cam   = batch["gt_corners_cam"]
            R0       = batch["R0"]
            t0       = batch["t0"]
            sids     = batch["scene_id"]
            iids     = batch["inst_id"]

            _, _, _, pred_corners = model.forward_decode(pts, extra)
            pred_cam = (torch.bmm(pred_corners, R0.transpose(1, 2))
                        + t0.unsqueeze(1))  # (B,8,3)

            for b in range(pts.shape[0]):
                pred = pred_cam[b].numpy().astype(np.float64)
                gt   = gt_cam[b].numpy().astype(np.float64)
                m    = evaluate_instance(pred, gt)
                results.append({
                    "scene_id":         sids[b],
                    "inst_id":          int(iids[b]),
                    "pred_corners_cam": pred,
                    "gt_corners_cam":   gt,
                    "iou_3d":           m["iou_3d"],
                })
    return results


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(log_csv: Path, out_path: Path) -> None:
    """Loss components + val IoU/accuracy from log.csv → PNG."""
    rows: list[dict] = []
    with open(log_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", "nan") else float("nan"))
                         for k, v in r.items()})

    epochs     = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss   = [r["val_loss"]   for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: total loss
    ax = axes[0]
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss,   label="val", alpha=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Total Loss"); ax.legend(); ax.grid(True, alpha=0.4)

    # Panel 2: loss components (train)
    ax = axes[1]
    for key, label in [("train_corner", "corner"),
                       ("train_center", "center"),
                       ("train_size",   "size")]:
        if key in rows[0]:
            vals = [r[key] for r in rows]
            ax.plot(epochs, vals, label=label)
    ax.set_xlabel("epoch"); ax.set_ylabel("component loss")
    ax.set_title("Train Loss Components"); ax.legend(); ax.grid(True, alpha=0.4)

    # Panel 3: val IoU + accuracy (skip NaN rows)
    ax = axes[2]
    for key, label, color, marker in [
        ("val_iou",    "val IoU",   "green",  "o"),
        ("val_acc025", "Acc@0.25",  "blue",   "s"),
        ("val_acc050", "Acc@0.5",   "orange", "^"),
    ]:
        if key in rows[0]:
            xs = [r["epoch"] for r in rows if not np.isnan(r.get(key, float("nan")))]
            ys = [r[key]     for r in rows if not np.isnan(r.get(key, float("nan")))]
            ax.plot(xs, ys, f"{marker}-", color=color, label=label, markersize=4)
    ax.set_xlabel("epoch"); ax.set_ylabel("metric")
    ax.set_title("Val IoU / Accuracy"); ax.legend(); ax.grid(True, alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  curves   -> {out_path.name}")


# ---------------------------------------------------------------------------
# 2D overlay helpers
# ---------------------------------------------------------------------------

def _draw_box_2d(
    ax,
    corners_2d: np.ndarray,  # (8,2)
    color: str,
    lw: float = 1.5,
) -> None:
    for i, j in BOX_EDGES:
        ax.plot([corners_2d[i, 0], corners_2d[j, 0]],
                [corners_2d[i, 1], corners_2d[j, 1]],
                color=color, linewidth=lw)


def make_scene_overlay(
    scene_dir: Path,
    instances: list[dict],   # results for this scene
    out_path: Path,
) -> None:
    """GT (green) + predicted (red) box edges drawn over RGB, with IoU labels."""
    rgb_img = np.array(Image.open(scene_dir / "rgb.jpg"))
    pc      = np.load(scene_dir / "pc.npy")          # (3,H,W)
    fx, fy, cx, cy = estimate_intrinsics(pc)

    H, W = rgb_img.shape[:2]
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(rgb_img)
    ax.axis("off")

    for inst in instances:
        gt2d = project_corners(inst["gt_corners_cam"],   fx, fy, cx, cy)
        pd2d = project_corners(inst["pred_corners_cam"], fx, fy, cx, cy)
        _draw_box_2d(ax, gt2d, "lime", lw=1.5)
        _draw_box_2d(ax, pd2d, "red",  lw=1.5)
        ctr = gt2d.mean(axis=0)
        ax.text(ctr[0], ctr[1], f"#{inst['inst_id']} {inst['iou_3d']:.2f}",
                color="yellow", fontsize=6, ha="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, ec="none"))

    ax.legend(
        handles=[mpatches.Patch(color="lime", label="GT"),
                 mpatches.Patch(color="red",  label="Pred")],
        loc="upper right", fontsize=7,
    )
    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3D matplotlib PNG
# ---------------------------------------------------------------------------

def _box_wireframe_3d(ax, corners: np.ndarray, color: str, lw: float = 1.5) -> None:
    segs = [[corners[i], corners[j]] for i, j in BOX_EDGES]
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw))


def make_3d_scene_png(
    scene_dir: Path,
    instances: list[dict],
    out_path: Path,
    subsample: int = 2000,
) -> None:
    """Scene point cloud (RGB-colored) + GT/pred wireframes → matplotlib PNG."""
    pc      = np.load(scene_dir / "pc.npy")              # (3,H,W)
    rgb_img = np.array(Image.open(scene_dir / "rgb.jpg"))  # (H,W,3)

    X, Y, Z = pc[0].ravel(), pc[1].ravel(), pc[2].ravel()
    valid    = Z > 0
    xyz      = np.stack([X[valid], Y[valid], Z[valid]], axis=1)
    rgb_flat = rgb_img.reshape(-1, 3)[valid]

    if len(xyz) > subsample:
        idx      = np.random.default_rng(42).choice(len(xyz), subsample, replace=False)
        xyz      = xyz[idx]
        rgb_flat = rgb_flat[idx]

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=rgb_flat / 255.0, s=1.5, linewidths=0, depthshade=True)

    for inst in instances:
        _box_wireframe_3d(ax, inst["gt_corners_cam"],   "lime", lw=1.5)
        _box_wireframe_3d(ax, inst["pred_corners_cam"], "red",  lw=1.5)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"3D scene  n={len(instances)}  (green=GT, red=pred)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  3D PNG   -> {out_path.name}")


# ---------------------------------------------------------------------------
# 3D plotly HTML (one scene)
# ---------------------------------------------------------------------------

def make_3d_scene_plotly(
    scene_dir: Path,
    instances: list[dict],
    out_path: Path,
    subsample: int = 3000,
) -> None:
    """Plotly interactive 3D view → HTML."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed — skipping HTML")
        return

    pc      = np.load(scene_dir / "pc.npy")
    rgb_img = np.array(Image.open(scene_dir / "rgb.jpg"))

    X, Y, Z = pc[0].ravel(), pc[1].ravel(), pc[2].ravel()
    valid    = Z > 0
    xyz      = np.stack([X[valid], Y[valid], Z[valid]], axis=1)
    rgb_flat = rgb_img.reshape(-1, 3)[valid]

    if len(xyz) > subsample:
        idx      = np.random.default_rng(42).choice(len(xyz), subsample, replace=False)
        xyz      = xyz[idx]
        rgb_flat = rgb_flat[idx]

    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb_flat.tolist()]

    traces = [go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=colors, opacity=0.7),
        name="point cloud",
    )]

    def _edge_trace(corners: np.ndarray, color: str, name: str) -> go.Scatter3d:
        xs, ys, zs = [], [], []
        for i, j in BOX_EDGES:
            xs += [corners[i, 0], corners[j, 0], None]
            ys += [corners[i, 1], corners[j, 1], None]
            zs += [corners[i, 2], corners[j, 2], None]
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                            line=dict(color=color, width=3), name=name)

    for inst in instances:
        iid = inst["inst_id"]; iou = inst["iou_3d"]
        traces.append(_edge_trace(inst["gt_corners_cam"],   "lime", f"GT #{iid}"))
        traces.append(_edge_trace(inst["pred_corners_cam"], "red",  f"Pred #{iid} IoU={iou:.2f}"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Scene {scene_dir.name[:8]}… — 3D view",
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                   aspectmode="data"),
        legend=dict(itemsizing="constant"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"  3D HTML  -> {out_path.name}")


# ---------------------------------------------------------------------------
# Failure cases figure
# ---------------------------------------------------------------------------

def make_failure_cases(
    results: list[dict],
    items_test: list[dict],      # split test items (for pixel_bbox lookup)
    cache_dir: Path,
    data_root: Path,
    out_path: Path,
    n: int = 4,
) -> None:
    """Composite figure of the n lowest-IoU test instances, zoomed to the object."""
    # Build lookup: (scene_id, inst_id) → npz path
    npz_lookup = {(it["scene_id"], it["inst_id"]): cache_dir / it["npz"]
                  for it in items_test}

    worst = sorted(results, key=lambda r: r["iou_3d"])[:n]

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, inst in zip(axes, worst):
        sid  = inst["scene_id"]
        iid  = inst["inst_id"]
        iou  = inst["iou_3d"]

        scene_dir = data_root / sid
        rgb_img   = np.array(Image.open(scene_dir / "rgb.jpg"))
        pc        = np.load(scene_dir / "pc.npy")
        fx, fy, cx, cy = estimate_intrinsics(pc)

        gt2d = project_corners(inst["gt_corners_cam"],   fx, fy, cx, cy)
        pd2d = project_corners(inst["pred_corners_cam"], fx, fy, cx, cy)

        # zoom to the instance using pixel_bbox from npz
        npz_path = npz_lookup.get((sid, iid))
        pad = 40
        H, W = rgb_img.shape[:2]
        if npz_path and npz_path.exists():
            d   = np.load(npz_path, allow_pickle=True)
            bb  = d["pixel_bbox"]          # [r_min, r_max, c_min, c_max]
            r0, r1, c0, c1 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            xlim = (max(c0 - pad, 0), min(c1 + pad, W))
            ylim = (min(r1 + pad, H), max(r0 - pad, 0))   # matplotlib y-axis is flipped
        else:
            xlim, ylim = (0, W), (H, 0)

        ax.imshow(rgb_img)
        ax.axis("off")
        _draw_box_2d(ax, gt2d, "lime", lw=2.0)
        _draw_box_2d(ax, pd2d, "red",  lw=2.0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"#{iid}  IoU={iou:.3f}", fontsize=9)

    fig.legend(
        handles=[mpatches.Patch(color="lime", label="GT"),
                 mpatches.Patch(color="red",  label="Pred")],
        loc="upper center", ncol=2, fontsize=9,
    )
    fig.suptitle("Failure Cases (4 Lowest IoU)", fontsize=12, y=1.03)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  failures -> {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10 — Visualization")
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--run",      default="main_run")
    parser.add_argument("--n-scenes", type=int, default=8,
                        help="Number of test scenes to visualize (min 8)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root      = Path(__file__).resolve().parents[2]
    cache_dir = root / cfg["data"]["cache_dir"]
    data_root = root / cfg["data"]["root"]
    runs_dir  = root / cfg["outputs_dir"] / "runs"
    figs_dir  = root / cfg["outputs_dir"] / "figures"
    ckpt_path = runs_dir / args.run / "best.pt"
    log_csv   = runs_dir / args.run / "log.csv"

    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training curves
    print("=== Training curves ===")
    if log_csv.exists():
        plot_training_curves(log_csv, figs_dir / "training_curves.png")
    else:
        print(f"  log.csv not found at {log_csv}")

    # 2. Inference on test split
    with open(cache_dir / "split.json") as f:
        split_data = json.load(f)
    items_test = split_data["test"]

    print(f"\n=== Inference ({len(items_test)} test instances) ===")
    results = run_inference(cfg, ckpt_path, items_test, cache_dir)
    ious    = [r["iou_3d"] for r in results]
    print(f"  mean IoU={np.mean(ious):.4f}  median={np.median(ious):.4f}")

    # Group by scene; pick the n_scenes with most instances for richer figures
    scene_results: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        scene_results[r["scene_id"]].append(r)

    n_scenes  = max(args.n_scenes, 8)
    scene_ids = sorted(scene_results, key=lambda s: -len(scene_results[s]))[:n_scenes]
    print(f"\n=== 2D overlays ({len(scene_ids)} scenes) ===")

    for sid in scene_ids:
        out = figs_dir / f"pred_{sid}.png"
        make_scene_overlay(data_root / sid, scene_results[sid], out)
        mean_iou = np.mean([r["iou_3d"] for r in scene_results[sid]])
        print(f"  overlay  -> {out.name}  (n={len(scene_results[sid])}, "
              f"mean IoU={mean_iou:.3f})")

    # 3. 3D views: PNG for each overlay scene; plotly HTML for the first
    print(f"\n=== 3D views ===")
    for sid in scene_ids:
        make_3d_scene_png(data_root / sid, scene_results[sid],
                          figs_dir / f"3d_{sid}.png")

    make_3d_scene_plotly(data_root / scene_ids[0], scene_results[scene_ids[0]],
                         figs_dir / f"3d_{scene_ids[0]}.html")

    # 4. Failure cases
    print(f"\n=== Failure cases ===")
    make_failure_cases(results, items_test, cache_dir, data_root,
                       figs_dir / "failure_cases.png")

    print(f"\nAll figures saved to {figs_dir}")


if __name__ == "__main__":
    main()
