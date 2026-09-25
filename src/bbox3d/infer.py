"""
Step 11 — Scene-level inference (infer.py).

Runs the full pipeline on a raw scene folder:
  rgb.jpg + pc.npy + mask.npy  ->  predicted 3D bounding box corners per instance.

Pre/post-processing (outlier filtering, point sampling, PCA canonicalization,
camera-frame transform) runs in numpy; the neural network step uses either the
PyTorch checkpoint or an ONNX Runtime session.

Usage:
    python -m bbox3d.infer \\
        --scene  data/dl_challenge/<uuid>/ \\
        --config configs/default.yaml \\
        --backend torch \\
        --model  outputs/runs/main_run/best.pt

    python -m bbox3d.infer \\
        --scene  data/dl_challenge/<uuid>/ \\
        --config configs/default.yaml \\
        --backend onnx \\
        --model  outputs/onnx/model_fp32.onnx

Outputs (written to --out-dir, default: outputs/infer/<scene_id>/):
    pred_corners.npy   (N, 8, 3) float32 — predicted box corners, camera frame
    pred_overlay.png   RGB image with predicted boxes (red, 12 edges each)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from bbox3d.benchmark import decode_raw_numpy
from bbox3d.data.preprocess import process_instance
from bbox3d.geometry.box import pca_frame
from bbox3d.geometry.intrinsics import estimate_intrinsics
from bbox3d.losses import compute_log_size_prior
from bbox3d.models.pointnet_box import build_model
from bbox3d.viz import BOX_EDGES, project_corners

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Per-instance preprocessing (no cache I/O; returns model-ready tensors)
# ---------------------------------------------------------------------------

def preprocess_instance(
    pc: np.ndarray,          # (3, H, W)
    mask_i: np.ndarray,      # (H, W) bool
    mask_all: np.ndarray,    # (N_inst, H, W) bool
    rgb_img: np.ndarray,     # (H, W, 3) uint8
    cfg: dict,
    num_points: int = 512,
    use_rgb: bool = True,
    use_pca: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Filter, sample, and PCA-canonicalize one instance's points.

    Returns (pts_P6, extra_4, R0_33, t0_3) or None if too few points.

    pts_P6  : (P, 6)  float32  xyz_canon + rgb
    extra_4 : (4,)    float32  [log1p(n_raw), ext_x, ext_y, ext_z]
    R0      : (3, 3)  float32  PCA rotation
    t0      : (3,)    float32  centroid in camera frame
    """
    pts_raw, stats, _ = process_instance(pc, mask_i, mask_all, rgb_img,
                                         cfg["preprocess"])
    if pts_raw is None:
        return None

    n_raw = len(pts_raw)
    xyz   = pts_raw[:, :3].astype(np.float64)
    rgb   = pts_raw[:, 3:].astype(np.float64)

    # Sample P points
    P   = num_points
    N   = len(xyz)
    rng = np.random.default_rng(seed)
    sel = rng.choice(N, P, replace=(N < P))
    xyz_s = xyz[sel]
    rgb_s = rgb[sel]

    # PCA canonicalization
    if use_pca:
        R0, t0 = pca_frame(xyz_s)
    else:
        R0 = np.eye(3, dtype=np.float64)
        t0 = xyz_s.mean(axis=0)

    xyz_c = (xyz_s - t0) @ R0       # (P, 3)

    extents = xyz_c.max(axis=0) - xyz_c.min(axis=0)
    view    = (t0 / (np.linalg.norm(t0) + 1e-12)) @ R0   # view direction in canonical frame
    extra   = np.array([np.log1p(float(n_raw)), *extents, *view], dtype=np.float32)

    if use_rgb:
        pts_out = np.concatenate([xyz_c, rgb_s], axis=1).astype(np.float32)
    else:
        pts_out = np.concatenate([xyz_c, np.zeros_like(rgb_s)], axis=1).astype(np.float32)

    return pts_out, extra, R0.astype(np.float32), t0.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def _load_torch(ckpt_path: Path, cfg: dict):
    """Return (model, ckpt_cfg)."""
    ckpt     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", cfg)

    root      = Path(__file__).resolve().parents[2]
    cache_dir = root / ckpt_cfg["data"]["cache_dir"]
    with open(cache_dir / "split.json") as f:
        split_data = json.load(f)
    log_size_init = compute_log_size_prior(split_data["train"], cache_dir)

    model = build_model(ckpt_cfg, log_size_init)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt_cfg


def _run_torch(model, pts_batch: np.ndarray, extra_batch: np.ndarray) -> np.ndarray:
    """
    pts_batch   (B, P, 6) float32
    extra_batch (B, 4)    float32
    Returns (B, 8, 3) float64 corners in canonical frame.
    """
    with torch.no_grad():
        _, _, _, corners = model.forward_decode(
            torch.from_numpy(pts_batch),
            torch.from_numpy(extra_batch),
        )
    return corners.numpy().astype(np.float64)


def _run_ort(sess, pts_batch: np.ndarray, extra_batch: np.ndarray) -> np.ndarray:
    """Returns (B, 8, 3) float64 corners in canonical frame (decoded from raw output)."""
    raw = sess.run(None, {"pts": pts_batch, "extra": extra_batch})[0]  # (B, 12)
    return decode_raw_numpy(raw)


# ---------------------------------------------------------------------------
# Visualization: predicted boxes on RGB (no GT)
# ---------------------------------------------------------------------------

def _save_pred_overlay(
    rgb_img: np.ndarray,         # (H, W, 3)
    pc: np.ndarray,              # (3, H, W)
    pred_list: list[np.ndarray], # list of (8,3) camera-frame corners
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fx, fy, cx, cy = estimate_intrinsics(pc)
    H, W = rgb_img.shape[:2]

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(rgb_img)
    ax.axis("off")

    for k, corners_cam in enumerate(pred_list):
        pts2d = project_corners(corners_cam, fx, fy, cx, cy)
        for i, j in BOX_EDGES:
            ax.plot([pts2d[i, 0], pts2d[j, 0]],
                    [pts2d[i, 1], pts2d[j, 1]],
                    color="red", linewidth=1.5)
        ctr = pts2d.mean(axis=0)
        ax.text(ctr[0], ctr[1], f"#{k}", color="yellow", fontsize=6, ha="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, ec="none"))

    ax.legend(handles=[mpatches.Patch(color="red", label="Predicted")],
              loc="upper right", fontsize=7)
    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11 — Scene inference")
    parser.add_argument("--scene",   required=True,
                        help="Path to raw scene folder (contains rgb.jpg, pc.npy, mask.npy)")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--model",   default=None,
                        help="Path to .pt checkpoint (torch) or .onnx model (onnx). "
                             "Defaults to outputs/runs/main_run/best.pt for torch, "
                             "or outputs/onnx/model_fp32.onnx for onnx.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. Default: outputs/infer/<scene_id>/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(__file__).resolve().parents[2]

    scene_dir = Path(args.scene)
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene not found: {scene_dir}")

    # Model path defaults
    if args.model is None:
        if args.backend == "torch":
            model_path = root / cfg["outputs_dir"] / "runs" / "main_run" / "best.pt"
        else:
            model_path = root / cfg["outputs_dir"] / "onnx" / "model_fp32.onnx"
    else:
        model_path = Path(args.model)

    out_dir = Path(args.out_dir) if args.out_dir else (
        root / cfg["outputs_dir"] / "infer" / scene_dir.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load scene data
    rgb_img = np.array(Image.open(scene_dir / "rgb.jpg"))
    pc      = np.load(scene_dir / "pc.npy")          # (3, H, W)
    masks   = np.load(scene_dir / "mask.npy")         # (N, H, W) bool
    N_inst  = len(masks)
    print(f"Scene {scene_dir.name}  ->  {N_inst} instances")

    # Load model
    if args.backend == "torch":
        model, ckpt_cfg = _load_torch(model_path, cfg)
        ds_cfg = ckpt_cfg.get("dataset", cfg.get("dataset", {}))
    else:
        import onnxruntime as ort
        sess    = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model   = None
        ckpt_cfg = cfg
        ds_cfg   = cfg.get("dataset", {})

    use_rgb  = bool(ds_cfg.get("use_rgb", True))
    use_pca  = bool(ds_cfg.get("use_pca", True))
    P        = int(ds_cfg.get("num_points", 512))

    # Preprocess all instances
    valid_instances: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    skipped = 0
    for i in range(N_inst):
        result = preprocess_instance(
            pc, masks[i], masks, rgb_img, cfg,
            num_points=P, use_rgb=use_rgb, use_pca=use_pca, seed=42 + i,
        )
        if result is None:
            skipped += 1
            continue
        pts, extra, R0, t0 = result
        valid_instances.append((i, pts, extra, R0, t0))

    print(f"  valid instances: {len(valid_instances)}  skipped: {skipped}")

    if not valid_instances:
        print("No valid instances found.")
        return

    # Batch inference
    inst_ids = [v[0] for v in valid_instances]
    pts_batch   = np.stack([v[1] for v in valid_instances])   # (M, P, 6)
    extra_batch = np.stack([v[2] for v in valid_instances])   # (M, 4)
    R0_batch    = np.stack([v[3] for v in valid_instances])   # (M, 3, 3)
    t0_batch    = np.stack([v[4] for v in valid_instances])   # (M, 3)

    if args.backend == "torch":
        corners_canon = _run_torch(model, pts_batch, extra_batch)  # (M, 8, 3)
    else:
        corners_canon = _run_ort(sess, pts_batch, extra_batch)

    # Transform canonical -> camera frame
    pred_cam = (corners_canon @ R0_batch.transpose(0, 2, 1)
                + t0_batch[:, None, :])  # (M, 8, 3)

    # Build (N_inst, 8, 3) output (skipped instances get NaN)
    pred_all = np.full((N_inst, 8, 3), np.nan, dtype=np.float32)
    for k, iid in enumerate(inst_ids):
        pred_all[iid] = pred_cam[k].astype(np.float32)

    # Save
    npy_path = out_dir / "pred_corners.npy"
    np.save(npy_path, pred_all)
    print(f"  Saved {npy_path}")

    # Visualization
    pred_list = [pred_cam[k] for k in range(len(inst_ids))]
    overlay_path = out_dir / "pred_overlay.png"
    _save_pred_overlay(rgb_img, pc, pred_list, overlay_path)
    print(f"  Saved {overlay_path}")


if __name__ == "__main__":
    main()
