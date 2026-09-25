"""
Step 4 — Geometric PCA baseline (baseline.py).

For each instance: take the filtered point cloud (xyz), compute its PCA
oriented bounding box (min/max extents along PCA axes), then evaluate
against the GT box using the Step-7 metrics.

The baseline intentionally has no amodal completion: it only sees the
top surface → biased Z centre + under-estimated depth.  This is the
number the DL model must beat.

Usage:
    python -m bbox3d.baseline --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import yaml

from bbox3d.geometry.box import pca_frame, params_to_corners_np
from bbox3d.metrics import evaluate_instance, aggregate_metrics, metrics_markdown_table


# ---------------------------------------------------------------------------
# PCA box prediction
# ---------------------------------------------------------------------------

def predict_pca_box(xyz: np.ndarray) -> np.ndarray:
    """
    Oriented bounding box of a point cloud via PCA.

    xyz   : (N,3) filtered point cloud in camera frame (metres)
    returns: (8,3) box corners using the left-handed template convention
             (same as GT, so corner-distance comparisons are valid).
    """
    R0, t0 = pca_frame(xyz)                      # proper rotation, centroid
    local  = (xyz - t0) @ R0                     # (N,3) in PCA frame
    lo, hi = local.min(axis=0), local.max(axis=0)
    size   = hi - lo                             # always positive
    center_cam = t0 + ((lo + hi) / 2.0) @ R0.T  # back to camera frame
    return params_to_corners_np(center_cam, size, R0)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    items: list[dict],
    cache_dir: Path,
    desc: str = "",
) -> list[dict]:
    """Evaluate the PCA baseline on a list of split items."""
    results = []
    skipped = 0
    for item in tqdm(items, desc=desc):
        npz_path = cache_dir / item["npz"]
        if not npz_path.exists():
            skipped += 1
            continue
        d = np.load(npz_path, allow_pickle=True)
        xyz = d["pts"][:, :3].astype(np.float64)
        gt  = d["gt_corners"].astype(np.float64)

        if len(xyz) < 3:
            skipped += 1
            continue

        pred = predict_pca_box(xyz)
        results.append(evaluate_instance(pred, gt))

    if skipped:
        print(f"  [{desc}] skipped {skipped} instances (missing npz or <3 pts)")
    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    val_results:  list[dict],
    test_results: list[dict],
    report_path:  Path,
) -> None:
    val_agg  = aggregate_metrics(val_results)
    test_agg = aggregate_metrics(test_results)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Geometric Baseline Report\n\n")
        f.write("**Method:** PCA oriented bounding box of the filtered visible "
                "surface point cloud (no amodal completion).  "
                "This is the number the DL model must beat.\n\n")
        f.write("**Limitation:** The baseline sees only the top surface of each "
                "object, so the predicted Z-centre is biased toward the camera "
                "and the predicted depth (third PCA dimension) is systematically "
                "under-estimated.\n\n")
        f.write(metrics_markdown_table(val_agg,  "Validation Set"))
        f.write("\n\n")
        f.write(metrics_markdown_table(test_agg, "Test Set"))
        f.write("\n")

    print(f"\nReport written to {report_path}")
    _print_summary("Val ", val_agg)
    _print_summary("Test", test_agg)


def _print_summary(prefix: str, agg: dict) -> None:
    s = agg.get("all")
    if s is None:
        return
    def fs(k): return f"{s[k]['mean']:.1f}/{s[k]['median']:.1f}"
    print(
        f"  {prefix}  n={s['n']:4d}  "
        f"IoU={s['mean_iou']:.4f}  "
        f"Acc@0.25={s['acc_025']:.3f}  Acc@0.5={s['acc_050']:.3f}  "
        f"corner={fs('corner_dist_mm')}mm  "
        f"center={fs('center_err_mm')}mm  "
        f"rot={fs('rot_err_deg')}deg"
    )
    tn = agg.get("thin")
    nm = agg.get("normal")
    if tn:
        print(f"    thin  (n={tn['n']:3d}): IoU={tn['mean_iou']:.4f}  "
              f"corner={tn['corner_dist_mm']['mean']:.1f}mm")
    if nm:
        print(f"    norm  (n={nm['n']:3d}): IoU={nm['mean_iou']:.4f}  "
              f"corner={nm['corner_dist_mm']['mean']:.1f}mm")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 4 — Geometric PCA baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root      = Path(__file__).resolve().parents[2]
    cache_dir = root / cfg["data"]["cache_dir"]
    report    = root / cfg["reports_dir"] / "baseline.md"

    with open(cache_dir / "split.json") as f:
        split = json.load(f)

    print("=== PCA Baseline evaluation ===")
    val_results  = run_eval(split["val"],  cache_dir, desc="val ")
    test_results = run_eval(split["test"], cache_dir, desc="test")

    write_report(val_results, test_results, report)


if __name__ == "__main__":
    main()
