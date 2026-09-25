"""
Generate 3D scene PNG figures for r1_full_s42_retrain on the 8 original scenes.

Saves to: outputs/figures/retrain_pre/3d_<scene_id>.png
Style: gray/RGB-colored point cloud + green=GT wireframe + red=Pred wireframe
       (same as viz.py make_3d_scene_png)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bbox3d.data.dataset import BBox3DDataset
from bbox3d.losses import compute_log_size_prior
from bbox3d.metrics import evaluate_instance
from bbox3d.models.pointnet_box import build_model
from bbox3d.viz import make_3d_scene_png
from torch.utils.data import DataLoader
import numpy as np

CACHE   = ROOT / "outputs" / "cache"
DATA    = ROOT / "data" / "dl_challenge"
OUT_DIR = ROOT / "outputs" / "figures" / "retrain_pre"
RUN     = "r1_full_s42_retrain"
CFG_FILE = "r1_full.yaml"

TARGET_SCENES = [
    "878250cd-9915-11ee-9103-bbb8eae05561",
    "889a9fb5-9915-11ee-9103-bbb8eae05561",
    "8b061a8f-9915-11ee-9103-bbb8eae05561",
    "8c394190-9915-11ee-9103-bbb8eae05561",
    "9a7caa9a-9915-11ee-9103-bbb8eae05561",
    "9a7caa9b-9915-11ee-9103-bbb8eae05561",
    "9ce28687-9915-11ee-9103-bbb8eae05561",
    "9f50f3c0-9915-11ee-9103-bbb8eae05561",
]


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / CFG_FILE))

    with open(CACHE / "split.json") as f:
        split_data = json.load(f)

    # Collect all items from target scenes across all splits
    all_items = split_data["train"] + split_data["val"] + split_data["test"]
    target_set = set(TARGET_SCENES)
    scene_items: dict[str, list[dict]] = defaultdict(list)
    for item in all_items:
        if item["scene_id"] in target_set:
            scene_items[item["scene_id"]].append(item)

    print(f"Found instances per scene:")
    for sid in TARGET_SCENES:
        print(f"  {sid[:8]}…  n={len(scene_items[sid])}")

    # Load model
    print(f"\nLoading {RUN}...")
    log_size_init = compute_log_size_prior(split_data["train"], CACHE)
    ckpt = torch.load(
        ROOT / "outputs" / "runs" / RUN / "best.pt",
        map_location="cpu", weights_only=False,
    )
    ckpt_cfg = ckpt.get("cfg", cfg)
    model = build_model(ckpt_cfg, log_size_init)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Run inference on all target-scene instances at once
    flat_items = [item for sid in TARGET_SCENES for item in scene_items[sid]]
    print(f"\nRunning inference on {len(flat_items)} instances...")
    ds = BBox3DDataset(flat_items, CACHE, ckpt_cfg, augment_data=False, seed=42)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    results: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            pts, extra = batch["pts"], batch["extra"]
            R0, t0 = batch["R0"], batch["t0"]
            gt_cam = batch["gt_corners_cam"]
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

    # Group results by scene
    scene_results: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        scene_results[r["scene_id"]].append(r)

    mean_ious = [r["iou_3d"] for r in results]
    print(f"Overall mean IoU: {np.mean(mean_ious):.4f}  (n={len(results)})")

    # Generate 3D PNG for each scene
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating 3D figures -> {OUT_DIR.relative_to(ROOT)}")
    for sid in TARGET_SCENES:
        insts = scene_results.get(sid, [])
        if not insts:
            print(f"  WARNING: no results for {sid[:8]}")
            continue
        out_path = OUT_DIR / f"3d_{sid}.png"
        scene_mean = np.mean([r["iou_3d"] for r in insts])
        make_3d_scene_png(DATA / sid, insts, out_path)
        print(f"  {sid[:8]}…  n={len(insts)}  mean IoU={scene_mean:.3f}  -> {out_path.name}")

    print(f"\nDone. {len(TARGET_SCENES)} figures saved to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
