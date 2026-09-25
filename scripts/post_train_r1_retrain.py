"""
Post-training analysis for r1_full_s42_retrain.
Outputs (all NEW files, nothing overwritten):
  outputs/figures/loss_curve_r1_full_s42_retrain.png
  outputs/figures/inference_r1_full_s42_retrain_<scene8>_inst<id>.png  (5 instances)
Prints comparison table vs pca_sign_fix_full.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bbox3d.data.dataset import BBox3DDataset
from bbox3d.geometry.intrinsics import estimate_intrinsics
from bbox3d.losses import compute_log_size_prior
from bbox3d.metrics import aggregate_metrics, evaluate_instance
from bbox3d.models.pointnet_box import build_model
from bbox3d.viz import BOX_EDGES, project_corners, _draw_box_2d

FIGURES = ROOT / "outputs" / "figures"
CACHE   = ROOT / "outputs" / "cache"
DATA    = ROOT / "data" / "dl_challenge"

RUN_NEW = "r1_full_s42_retrain"
RUN_OLD = "pca_sign_fix_full"
CFG_NEW = "r1_full.yaml"
CFG_OLD = "pca_sign_fix_full.yaml"

# Scenes already used in 23/9 figures — do NOT pick these
USED_SCENES = {
    "878250cd-9915-11ee-9103-bbb8eae05561",
    "889a9fb5-9915-11ee-9103-bbb8eae05561",
    "8b061a8f-9915-11ee-9103-bbb8eae05561",
    "8c394190-9915-11ee-9103-bbb8eae05561",
    "9a7caa9a-9915-11ee-9103-bbb8eae05561",
    "9a7caa9b-9915-11ee-9103-bbb8eae05561",
    "9ce28687-9915-11ee-9103-bbb8eae05561",
    "9f50f3c0-9915-11ee-9103-bbb8eae05561",
}


# ---------------------------------------------------------------------------
# 1. Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(run_name: str, out_path: Path) -> None:
    log_csv = ROOT / "outputs" / "runs" / run_name / "log.csv"
    rows = []
    with open(log_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", "nan") else float("nan"))
                         for k, v in r.items()})

    epochs     = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss   = [r["val_loss"]   for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Training curves — {run_name}", fontsize=12)

    ax = axes[0]
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss,   label="val", alpha=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Total Loss"); ax.legend(); ax.grid(True, alpha=0.4)

    ax = axes[1]
    for key, label in [("train_corner","corner"),("train_center","center"),("train_size","size")]:
        if key in rows[0]:
            ax.plot(epochs, [r[key] for r in rows], label=label)
    ax.set_xlabel("epoch"); ax.set_ylabel("component loss")
    ax.set_title("Train Loss Components"); ax.legend(); ax.grid(True, alpha=0.4)

    ax = axes[2]
    for key, label, color, marker in [
        ("val_iou",    "val IoU",  "green",  "o"),
        ("val_acc025", "Acc@0.25", "blue",   "s"),
        ("val_acc050", "Acc@0.5",  "orange", "^"),
    ]:
        xs = [r["epoch"] for r in rows if not np.isnan(r.get(key, float("nan")))]
        ys = [r[key]     for r in rows if not np.isnan(r.get(key, float("nan")))]
        if xs:
            ax.plot(xs, ys, f"{marker}-", color=color, label=label, markersize=4)
    ax.set_xlabel("epoch"); ax.set_ylabel("metric")
    ax.set_title("Val IoU / Accuracy"); ax.legend(); ax.grid(True, alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[1] Loss curve -> {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 2. Load model + run inference on selected instances
# ---------------------------------------------------------------------------

def load_model(run_name: str, config_path: str):
    cfg = yaml.safe_load(open(ROOT / "configs" / config_path))
    with open(CACHE / "split.json") as f:
        split = json.load(f)
    log_size_init = compute_log_size_prior(split["train"], CACHE)
    ckpt = torch.load(ROOT / "outputs" / "runs" / run_name / "best.pt",
                      map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", cfg)
    model = build_model(ckpt_cfg, log_size_init)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt_cfg, split


def infer_instances(model, ckpt_cfg, items: list[dict]) -> list[dict]:
    ds = BBox3DDataset(items, CACHE, ckpt_cfg, augment_data=False, seed=42)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    results = []
    with torch.no_grad():
        for batch in loader:
            pts, extra = batch["pts"], batch["extra"]
            gt_cam = batch["gt_corners_cam"]
            R0, t0 = batch["R0"], batch["t0"]
            _, _, _, pred_corners = model.forward_decode(pts, extra)
            pred_cam = torch.bmm(pred_corners, R0.transpose(1, 2)) + t0.unsqueeze(1)
            for b in range(pts.shape[0]):
                pred = pred_cam[b].numpy().astype(np.float64)
                gt   = gt_cam[b].numpy().astype(np.float64)
                m    = evaluate_instance(pred, gt)
                results.append({
                    "scene_id":         batch["scene_id"][b],
                    "inst_id":          int(batch["inst_id"][b]),
                    "pred_corners_cam": pred,
                    "gt_corners_cam":   gt,
                    "iou_3d":           m["iou_3d"],
                })
    return results


# ---------------------------------------------------------------------------
# 3. Per-instance figure: 2D overlay (left) + 3D view (right)
#    GT = red, Pred = blue  (as requested)
# ---------------------------------------------------------------------------

def _box_wireframe_3d(ax, corners: np.ndarray, color: str, lw: float = 1.5) -> None:
    segs = [[corners[i], corners[j]] for i, j in BOX_EDGES]
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw))


def plot_instance(result: dict, out_path: Path) -> None:
    sid   = result["scene_id"]
    iid   = result["inst_id"]
    iou   = result["iou_3d"]
    pred  = result["pred_corners_cam"]
    gt    = result["gt_corners_cam"]

    scene_dir = DATA / sid
    rgb_img   = np.array(Image.open(scene_dir / "rgb.jpg"))
    pc        = np.load(scene_dir / "pc.npy")          # (3,H,W)
    fx, fy, cx, cy = estimate_intrinsics(pc)

    gt2d   = project_corners(gt,   fx, fy, cx, cy)
    pred2d = project_corners(pred, fx, fy, cx, cy)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"Scene {sid[:8]}…  inst #{iid}  |  IoU = {iou:.3f}", fontsize=11)

    # --- Left: 2D overlay ---
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.imshow(rgb_img)
    ax2d.axis("off")
    _draw_box_2d(ax2d, gt2d,   "red",  lw=1.8)
    _draw_box_2d(ax2d, pred2d, "blue", lw=1.8)
    ax2d.legend(handles=[
        mpatches.Patch(color="red",  label="GT"),
        mpatches.Patch(color="blue", label="Pred"),
    ], loc="upper right", fontsize=8)
    ax2d.set_title("2D projection", fontsize=9)

    # --- Right: 3D view ---
    X, Y, Z = pc[0].ravel(), pc[1].ravel(), pc[2].ravel()
    valid    = Z > 0
    xyz      = np.stack([X[valid], Y[valid], Z[valid]], axis=1)
    rgb_flat = rgb_img.reshape(-1, 3)[valid]
    if len(xyz) > 3000:
        idx      = np.random.default_rng(42).choice(len(xyz), 3000, replace=False)
        xyz      = xyz[idx]
        rgb_flat = rgb_flat[idx]

    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                 c=rgb_flat / 255.0, s=1.0, linewidths=0, depthshade=True)
    _box_wireframe_3d(ax3d, gt,   "red",  lw=2.0)
    _box_wireframe_3d(ax3d, pred, "blue", lw=2.0)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3D point cloud", fontsize=9)
    ax3d.legend(handles=[
        mpatches.Patch(color="red",  label="GT"),
        mpatches.Patch(color="blue", label="Pred"),
    ], loc="upper left", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[2] Inference fig -> {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 4. Evaluate test set
# ---------------------------------------------------------------------------

def eval_test(model, ckpt_cfg, split) -> dict:
    items = split["test"]
    ds = BBox3DDataset(items, CACHE, ckpt_cfg, augment_data=False, seed=42)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    all_m = []
    with torch.no_grad():
        for batch in loader:
            pts, extra = batch["pts"], batch["extra"]
            gt_cam = batch["gt_corners_cam"]
            R0, t0 = batch["R0"], batch["t0"]
            _, _, _, pred_corners = model.forward_decode(pts, extra)
            pred_cam = torch.bmm(pred_corners, R0.transpose(1, 2)) + t0.unsqueeze(1)
            for b in range(pts.shape[0]):
                all_m.append(evaluate_instance(
                    pred_cam[b].numpy().astype(float),
                    gt_cam[b].numpy().astype(float)))
    return aggregate_metrics(all_m)["all"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. Loss curve
    plot_loss_curve(RUN_NEW, FIGURES / "loss_curve_r1_full_s42_retrain.png")

    # Load models
    print("\nLoading models...")
    model_new, cfg_new, split_new = load_model(RUN_NEW, CFG_NEW)
    model_old, cfg_old, split_old = load_model(RUN_OLD, CFG_OLD)

    # 2. Pick 5 new test instances (not from USED_SCENES)
    with open(CACHE / "split.json") as f:
        split = json.load(f)
    new_items = [t for t in split["test"] if t["scene_id"] not in USED_SCENES][:5]
    print(f"\nSelected {len(new_items)} new test instances:")
    for it in new_items:
        print(f"  scene={it['scene_id'][:8]}  inst={it['inst_id']}")

    results = infer_instances(model_new, cfg_new, new_items)
    for r in results:
        sid8 = r["scene_id"][:8]
        iid  = r["inst_id"]
        out  = FIGURES / f"inference_r1_full_s42_retrain_{sid8}_inst{iid}.png"
        plot_instance(r, out)

    # 3. Comparison table
    print("\nEvaluating test set...")
    m_new = eval_test(model_new, cfg_new, split_new)
    m_old = eval_test(model_old, cfg_old, split_old)

    print("\n" + "="*70)
    print(f"{'Model':<30} {'Test IoU':>9} {'Acc@0.25':>9} {'Acc@0.5':>8} {'corner mm':>10}")
    print("-"*70)
    print(f"{'pca_sign_fix_full (23/9)':<30} {m_old['mean_iou']:>9.4f} {m_old['acc_025']:>9.4f} {m_old['acc_050']:>8.4f} {m_old['corner_dist_mm']['mean']:>10.1f}")
    print(f"{'r1_full_s42_retrain (new)':<30} {m_new['mean_iou']:>9.4f} {m_new['acc_025']:>9.4f} {m_new['acc_050']:>8.4f} {m_new['corner_dist_mm']['mean']:>10.1f}")
    delta_iou = m_new['mean_iou'] - m_old['mean_iou']
    delta_a50 = m_new['acc_050'] - m_old['acc_050']
    print(f"{'Delta (new - old)':<30} {delta_iou:>+9.4f} {m_new['acc_025']-m_old['acc_025']:>+9.4f} {delta_a50:>+8.4f}")
    print("="*70)

    verdict = "TOT HON" if delta_iou > 0 else "KEM HON"
    print(f"\nKet luan: model moi {verdict} ({delta_iou:+.4f} IoU, {delta_a50:+.4f} Acc@0.5)")


if __name__ == "__main__":
    main()
