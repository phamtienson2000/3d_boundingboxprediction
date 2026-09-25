"""
Oracle analysis for r1_full_s42_retrain on the VAL set.
Replaces one component at a time with GT to identify the biggest bottleneck.
Also reports canonical size errors per axis and thin/normal breakdown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bbox3d.data.dataset import BBox3DDataset
from bbox3d.geometry.box import corners_to_params, params_to_corners_np, pca_frame
from bbox3d.losses import compute_log_size_prior
from bbox3d.metrics import aggregate_metrics, evaluate_instance, box_iou_3d
from bbox3d.models.pointnet_box import build_model

CACHE   = ROOT / "outputs" / "cache"
RUN     = "r1_full_s42_retrain"
CFG_FILE = ROOT / "configs" / "r1_full.yaml"


def run_oracle():
    cfg = yaml.safe_load(open(CFG_FILE))
    with open(CACHE / "split.json") as f:
        split_data = json.load(f)

    items_val = split_data["val"]
    print(f"Val set: {len(items_val)} instances")

    log_size_init = compute_log_size_prior(split_data["train"], CACHE)
    ckpt = torch.load(ROOT / "outputs" / "runs" / RUN / "best.pt",
                      map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", cfg)
    model = build_model(ckpt_cfg, log_size_init)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = BBox3DDataset(items_val, CACHE, ckpt_cfg, augment_data=False, seed=42)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    records = []
    with torch.no_grad():
        for batch in loader:
            pts, extra = batch["pts"], batch["extra"]
            R0, t0 = batch["R0"], batch["t0"]       # (B,3,3), (B,3)
            gt_cam = batch["gt_corners_cam"]         # (B,8,3)
            gt_canon = batch["gt_corners_canon"]     # (B,8,3)

            _, _, _, pred_corners_canon = model.forward_decode(pts, extra)
            pred_cam = (torch.bmm(pred_corners_canon, R0.transpose(1, 2))
                        + t0.unsqueeze(1))           # (B,8,3) camera frame

            for b in range(pts.shape[0]):
                pred_c = pred_corners_canon[b].numpy().astype(np.float64)
                gt_c   = gt_canon[b].numpy().astype(np.float64)
                pred   = pred_cam[b].numpy().astype(np.float64)
                gt     = gt_cam[b].numpy().astype(np.float64)
                R0b    = R0[b].numpy().astype(np.float64)   # (3,3)
                t0b    = t0[b].numpy().astype(np.float64)   # (3,)

                # Decode pred params in canonical frame
                pred_center_c, pred_size, pred_R = corners_to_params(pred_c)
                gt_center_c,   gt_size,   gt_R   = corners_to_params(gt_c)

                # Helper: build corners in canonical frame, then back to camera
                def corners_cam(center, size, R):
                    co = params_to_corners_np(center, size, R)   # (8,3) canonical
                    return co @ R0b.T + t0b                       # (8,3) camera

                # --- Baseline ---
                iou_base = box_iou_3d(pred, gt)

                # --- Oracle center (replace pred center with GT center) ---
                corners_oc = corners_cam(gt_center_c, pred_size, pred_R)
                iou_oc = box_iou_3d(corners_oc, gt)

                # --- Oracle size ---
                corners_os = corners_cam(pred_center_c, gt_size, pred_R)
                iou_os = box_iou_3d(corners_os, gt)

                # --- Oracle rotation ---
                corners_or = corners_cam(pred_center_c, pred_size, gt_R)
                iou_or = box_iou_3d(corners_or, gt)

                # --- Size error per canonical axis (axis-0 = largest extent) ---
                # GT sizes are returned sorted descending by corners_to_params
                size_err = np.abs(pred_size - gt_size) * 1000.0  # mm

                records.append({
                    "iou_base": iou_base,
                    "iou_oracle_center": iou_oc,
                    "iou_oracle_size": iou_os,
                    "iou_oracle_rot": iou_or,
                    "size_err_ax0_mm": float(size_err[0]),
                    "size_err_ax1_mm": float(size_err[1]),
                    "size_err_ax2_mm": float(size_err[2]),
                    "gt_size": gt_size,
                })

    n = len(records)
    def m(k): return float(np.mean([r[k] for r in records]))

    print("\n=== Oracle Analysis (val set, N={}) ===".format(n))
    print(f"{'Component replaced':<30} {'Mean IoU':>10} {'Delta':>8}")
    print("-"*50)
    base = m("iou_base")
    print(f"{'(a) Original prediction':<30} {base:>10.4f} {'—':>8}")
    for label, key in [
        ("(b) Oracle center",    "iou_oracle_center"),
        ("(c) Oracle size",      "iou_oracle_size"),
        ("(d) Oracle rotation",  "iou_oracle_rot"),
    ]:
        v = m(key)
        print(f"{label:<30} {v:>10.4f} {v-base:>+8.4f}")

    print("\n=== Canonical Size Error per Axis (mm) ===")
    print(f"  axis-0 (largest extent):  mean={np.mean([r['size_err_ax0_mm'] for r in records]):.1f}  median={np.median([r['size_err_ax0_mm'] for r in records]):.1f}")
    print(f"  axis-1 (medium):          mean={np.mean([r['size_err_ax1_mm'] for r in records]):.1f}  median={np.median([r['size_err_ax1_mm'] for r in records]):.1f}")
    print(f"  axis-2 (smallest/depth):  mean={np.mean([r['size_err_ax2_mm'] for r in records]):.1f}  median={np.median([r['size_err_ax2_mm'] for r in records]):.1f}")

    # Check axis-0/1 swap analysis
    normal_err_vals = []
    swapped_err_vals = []
    for r in records:
        s_pred_sorted = np.array([r["size_err_ax0_mm"], r["size_err_ax1_mm"], r["size_err_ax2_mm"]])
        normal_err_vals.append(s_pred_sorted[:2].mean())
        swapped_err_vals.append(np.array([r["size_err_ax1_mm"], r["size_err_ax0_mm"], r["size_err_ax2_mm"]])[:2].mean())
    print(f"\n=== Axis-0/1 Swap ===")
    print(f"  Normal order mean err (mm):  {np.mean(normal_err_vals):.1f}")
    print(f"  Swapped order mean err (mm): {np.mean(swapped_err_vals):.1f}")
    frac_improved = np.mean([swapped_err_vals[i] < normal_err_vals[i] for i in range(n)])
    print(f"  Fraction improved by swap:   {frac_improved:.3f}")

    # thin vs normal breakdown on baseline
    full_results = []
    with torch.no_grad():
        ds2 = BBox3DDataset(items_val, CACHE, ckpt_cfg, augment_data=False, seed=42)
        loader2 = DataLoader(ds2, batch_size=64, shuffle=False, num_workers=0)
        for batch in loader2:
            pts, extra = batch["pts"], batch["extra"]
            R0, t0 = batch["R0"], batch["t0"]
            gt_cam = batch["gt_corners_cam"]
            _, _, _, pred_corners = model.forward_decode(pts, extra)
            pred_cam2 = (torch.bmm(pred_corners, R0.transpose(1, 2)) + t0.unsqueeze(1))
            for b in range(pts.shape[0]):
                full_results.append(evaluate_instance(
                    pred_cam2[b].numpy().astype(np.float64),
                    gt_cam[b].numpy().astype(np.float64),
                ))
    agg = aggregate_metrics(full_results)
    print("\n=== Thin vs Normal (val) ===")
    for grp in ["all", "normal", "thin"]:
        s = agg[grp]
        if s is None:
            continue
        print(f"  {grp:6s}: n={s['n']}  IoU={s['mean_iou']:.4f}  Acc@0.25={s['acc_025']:.3f}  Acc@0.5={s['acc_050']:.3f}  corner={s['corner_dist_mm']['mean']:.1f}mm")

    # Also compute test thin/normal
    print("\n=== Thin vs Normal (test) — for reference ===")
    items_test = split_data["test"]
    test_results = []
    with torch.no_grad():
        ds3 = BBox3DDataset(items_test, CACHE, ckpt_cfg, augment_data=False, seed=42)
        loader3 = DataLoader(ds3, batch_size=64, shuffle=False, num_workers=0)
        for batch in loader3:
            pts, extra = batch["pts"], batch["extra"]
            R0, t0 = batch["R0"], batch["t0"]
            gt_cam = batch["gt_corners_cam"]
            _, _, _, pred_corners = model.forward_decode(pts, extra)
            pred_cam3 = (torch.bmm(pred_corners, R0.transpose(1, 2)) + t0.unsqueeze(1))
            for b in range(pts.shape[0]):
                test_results.append(evaluate_instance(
                    pred_cam3[b].numpy().astype(np.float64),
                    gt_cam[b].numpy().astype(np.float64),
                ))
    agg_test = aggregate_metrics(test_results)
    for grp in ["all", "normal", "thin"]:
        s = agg_test[grp]
        if s is None:
            continue
        print(f"  {grp:6s}: n={s['n']}  IoU={s['mean_iou']:.4f}  Acc@0.25={s['acc_025']:.3f}  Acc@0.5={s['acc_050']:.3f}  corner={s['corner_dist_mm']['mean']:.1f}mm  center={s['center_err_mm']['mean']:.1f}mm  rot={s['rot_err_deg']['mean']:.1f}deg")

    # Mean IoU including skipped test instances as 0
    n_skipped_test = 3
    n_test = len(test_results)
    iou_vals = [r["iou_3d"] for r in test_results]
    mean_iou_incl_skipped = np.sum(iou_vals) / (n_test + n_skipped_test)
    print(f"\n=== Test mean IoU including {n_skipped_test} skipped instances as IoU=0 ===")
    print(f"  mean IoU (N={n_test}): {np.mean(iou_vals):.4f}")
    print(f"  mean IoU incl. skipped (N={n_test + n_skipped_test}): {mean_iou_incl_skipped:.4f}")


if __name__ == "__main__":
    run_oracle()
