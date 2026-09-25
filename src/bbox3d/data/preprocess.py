"""
Step 3 — Preprocessing (data/preprocess.py).

Per instance:
  1. Masked pixels -> XYZ + RGB (scaled [0,1]); drop Z <= 0.
  2. Z-outlier filter: |Z - median_Z| > max(3*MAD, 2 cm).
  3. Statistical outlier removal: kNN k=16, drop mean_dist > mu + 2*sigma.
  4. Always apply largest connected component (LCC) — no GT-based condition.
  5. Skip if < 30 points remain (logged). Skipped instances recorded for IoU=0 in eval.
  6. Compute gap_p50, gap_p90, ring_other_frac from pc+mask only (NO GT used).
  7. Save <cache_dir>/<scene_id>_<inst_id:02d>.npz.
  8. Write split.json: 80/10/10 by scene, seed 42.

Leak fix vs old code:
  - Old DBSCAN condition used box_center_z (GT) to decide whether to run DBSCAN.
  - New code always runs LCC; ring features computed from pc+mask only, never from GT corners.

Usage:
    python -m bbox3d.data.preprocess --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm
import yaml


RING_RADIUS = 15

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _mask_pixel_bbox(mask: np.ndarray) -> np.ndarray:
    """Return [r_min, r_max, c_min, c_max] (inclusive) for a boolean 2D mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    if len(r_idx) == 0 or len(c_idx) == 0:
        return np.array([0, 0, 0, 0], dtype=np.int32)
    return np.array([r_idx[0], r_idx[-1], c_idx[0], c_idx[-1]], dtype=np.int32)


def _largest_connected_component(xyz: np.ndarray, eps: float) -> np.ndarray:
    """
    Boolean mask for the largest connected component of points
    in 3-D space at radius eps, using cKDTree + scipy sparse graph.
    Returned mask has True for points in the largest component.
    """
    n = len(xyz)
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(eps)  # set of (i,j) with i<j

    if not pairs:
        return np.ones(n, dtype=bool)

    pairs_arr = np.array(list(pairs), dtype=np.int32)  # (M,2)
    rows = pairs_arr[:, 0]
    cols = pairs_arr[:, 1]
    data = np.ones(len(rows), dtype=np.float32)
    # symmetric sparse adjacency
    adj = csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n, n),
    )
    _, labels = connected_components(adj, directed=False)
    unique, counts = np.unique(labels, return_counts=True)
    best_label = unique[np.argmax(counts)]
    return labels == best_label


def _ring_features(z_map: np.ndarray, mask_i: np.ndarray,
                   mask_all: np.ndarray) -> dict:
    """
    Compute ring gap features for one instance.
    Uses only pc (z_map) and masks — no GT box info.

    Returns: gap_p50, gap_p90 (metres, positive = ring is farther than obj),
             ring_other_frac (fraction of ring occupied by other objects).
    """
    y, x = np.ogrid[-RING_RADIUS:RING_RADIUS+1, -RING_RADIUS:RING_RADIUS+1]
    disk = (x*x + y*y) <= RING_RADIUS**2

    dilated    = binary_dilation(mask_i, structure=disk)
    ring       = dilated & ~mask_i
    all_masks  = mask_all.any(0)
    ring_free  = ring & ~all_masks & (z_map > 0)
    ring_other = ring & all_masks & ~mask_i

    ring_total = int(ring.sum())
    ring_other_frac = (float(ring_other.sum()) / ring_total
                       if ring_total > 0 else float("nan"))

    obj_z_vals = z_map[mask_i & (z_map > 0)]
    if len(obj_z_vals) == 0:
        return dict(gap_p50=float("nan"), gap_p90=float("nan"),
                    ring_other_frac=ring_other_frac)

    median_obj_z = float(np.median(obj_z_vals))

    if ring_free.sum() < 5:
        return dict(gap_p50=float("nan"), gap_p90=float("nan"),
                    ring_other_frac=ring_other_frac)

    ring_z = z_map[ring_free]
    gap_p50 = float(np.median(ring_z))           - median_obj_z
    gap_p90 = float(np.percentile(ring_z, 90))   - median_obj_z
    return dict(gap_p50=gap_p50, gap_p90=gap_p90,
                ring_other_frac=ring_other_frac)


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def process_instance(
    pc: np.ndarray,         # (3, H, W)
    mask_i: np.ndarray,     # (H, W) bool
    mask_all: np.ndarray,   # (N_inst, H, W) bool — all instance masks for scene
    rgb: np.ndarray,        # (H, W, 3) uint8
    cfg: dict,
) -> tuple[np.ndarray | None, dict, dict]:
    """
    Filter one instance's points (no GT used).

    Returns (pts_Nx6 or None, stats_dict, ring_feats_dict).
    stats keys: n_raw, n_final, lcc_used, skipped, skip_reason.
    ring_feats: gap_p50, gap_p90, ring_other_frac (may be nan).
    """
    Z = pc[2]
    valid = mask_i & (Z > 0)

    ys, xs = np.where(valid)
    xyz = np.stack([pc[0][valid], pc[1][valid], pc[2][valid]], axis=1).astype(np.float64)
    rgb_pts = rgb[valid].astype(np.float32) / 255.0   # (N,3) in [0,1]

    n_raw = len(xyz)

    nan_ring = dict(gap_p50=float("nan"), gap_p90=float("nan"),
                    ring_other_frac=float("nan"))

    def skip(reason):
        return None, dict(
            n_raw=n_raw, n_final=0,
            lcc_used=False, skipped=True, skip_reason=reason,
        ), nan_ring

    if n_raw == 0:
        return skip("no_valid_depth")

    # Step 2 — Z-outlier filter
    z_vals = xyz[:, 2]
    z_med = float(np.median(z_vals))
    mad = float(np.median(np.abs(z_vals - z_med)))
    z_thresh = max(cfg["outlier_z_mad_factor"] * mad,
                   cfg["outlier_z_min_cm"] / 100.0)
    keep = np.abs(z_vals - z_med) <= z_thresh
    xyz = xyz[keep];  rgb_pts = rgb_pts[keep]

    if len(xyz) < cfg["min_points"]:
        return skip("too_few_after_z_filter")

    # Step 3 — Statistical outlier removal (kNN)
    if cfg["use_outlier_filter"] and len(xyz) > cfg["outlier_knn_k"]:
        tree = cKDTree(xyz)
        dists, _ = tree.query(xyz, k=cfg["outlier_knn_k"] + 1)
        mean_dists = dists[:, 1:].mean(axis=1)
        mu = float(mean_dists.mean())
        sigma = float(mean_dists.std())
        keep = mean_dists <= mu + cfg["outlier_knn_sigma"] * sigma
        xyz = xyz[keep];  rgb_pts = rgb_pts[keep]

    if len(xyz) < cfg["min_points"]:
        return skip("too_few_after_knn_filter")

    # Step 4 — Always apply LCC (no GT-based condition)
    keep = _largest_connected_component(xyz, eps=cfg["dbscan_eps"])
    lcc_used = True
    if keep.sum() >= cfg["min_points"]:
        xyz = xyz[keep];  rgb_pts = rgb_pts[keep]

    if len(xyz) < cfg["min_points"]:
        return skip("too_few_after_lcc")

    pts = np.concatenate([xyz.astype(np.float32), rgb_pts], axis=1)  # (N,6)

    # Compute ring features from pc+mask only
    ring_feats = _ring_features(pc[2], mask_i, mask_all)

    return pts, dict(
        n_raw=n_raw, n_final=len(xyz),
        lcc_used=lcc_used, skipped=False, skip_reason="",
    ), ring_feats


# ---------------------------------------------------------------------------
# Scene split
# ---------------------------------------------------------------------------

def make_scene_split(
    scene_ids: list[str],
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> dict[str, list[str]]:
    """Split scene IDs 80/10/10 by scene (never split instances of one scene)."""
    rng = np.random.default_rng(seed)
    ids = list(scene_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    return {
        "train": ids[:n_train],
        "val":   ids[n_train: n_train + n_val],
        "test":  ids[n_train + n_val:],
    }


# ---------------------------------------------------------------------------
# Main preprocessing loop
# ---------------------------------------------------------------------------

def preprocess_all(data_root: Path, cache_dir: Path, cfg: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_pre = cfg["preprocess"]
    cfg_pre.setdefault("dbscan_eps", 0.01)
    cfg_pre.setdefault("dbscan_min_samples", 5)

    scene_dirs = sorted(data_root.iterdir())
    scene_ids = [sd.name for sd in scene_dirs if sd.is_dir()]

    split_scenes = make_scene_split(
        scene_ids,
        seed=cfg["data"]["seed"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )
    scene_to_split = {}
    for spl, ids in split_scenes.items():
        for sid in ids:
            scene_to_split[sid] = spl

    all_stats: list[dict] = []
    split_instances: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    n_skipped = 0
    n_saved = 0
    n_lcc = 0

    for sd in tqdm(scene_dirs, desc="Preprocessing scenes"):
        if not sd.is_dir():
            continue
        scene_id = sd.name
        split = scene_to_split.get(scene_id, "train")

        pc    = np.load(sd / "pc.npy")
        rgb   = np.array(Image.open(sd / "rgb.jpg"))
        masks = np.load(sd / "mask.npy")
        bboxes = np.load(sd / "bbox3d.npy")

        for inst_id in range(len(masks)):
            pts, stats, ring_feats = process_instance(
                pc, masks[inst_id], masks, rgb, cfg_pre
            )
            stats["scene_id"] = scene_id
            stats["inst_id"]  = inst_id
            stats["split"]    = split
            all_stats.append(stats)

            if pts is None:
                n_skipped += 1
                continue

            if stats["lcc_used"]:
                n_lcc += 1

            npz_name = f"{scene_id}_{inst_id:02d}.npz"
            npz_path = cache_dir / npz_name
            np.savez_compressed(
                npz_path,
                pts=pts,
                gt_corners=bboxes[inst_id].astype(np.float32),
                scene_id=np.array(scene_id),
                inst_id=np.array(inst_id, dtype=np.int32),
                pixel_bbox=_mask_pixel_bbox(masks[inst_id]),
                gap_p50=np.array(ring_feats["gap_p50"], dtype=np.float32),
                gap_p90=np.array(ring_feats["gap_p90"], dtype=np.float32),
                ring_other_frac=np.array(ring_feats["ring_other_frac"], dtype=np.float32),
            )

            split_instances[split].append({
                "scene_id": scene_id,
                "inst_id": int(inst_id),
                "npz": npz_name,
            })
            n_saved += 1

    # Write split.json (saved instances only)
    split_json_path = cache_dir / "split.json"
    with open(split_json_path, "w") as f:
        json.dump(split_instances, f, indent=2)
    print(f"\nSplit written to {split_json_path}")

    # Write skipped_instances.json (instances with <30 pts, for IoU=0 in eval)
    skipped_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in all_stats:
        if s["skipped"]:
            spl = s.get("split", "train")
            skipped_by_split[spl].append({
                "scene_id": s["scene_id"],
                "inst_id":  int(s["inst_id"]),
                "reason":   s.get("skip_reason", "unknown"),
            })
    skipped_json_path = cache_dir / "skipped_instances.json"
    with open(skipped_json_path, "w") as f:
        json.dump(skipped_by_split, f, indent=2)
    n_skip_val  = len(skipped_by_split["val"])
    n_skip_test = len(skipped_by_split["test"])
    print(f"Skipped instances written: val={n_skip_val}  test={n_skip_test}")

    _print_stats(all_stats, n_saved, n_skipped, n_lcc, split_scenes)


def _print_stats(all_stats, n_saved, n_skipped, n_lcc, split_scenes):
    total = len(all_stats)
    skip_reasons = {}
    for s in all_stats:
        if s["skipped"]:
            r = s.get("skip_reason", "unknown")
            skip_reasons[r] = skip_reasons.get(r, 0) + 1

    print("=" * 60)
    print("PREPROCESSING REPORT (leak-fixed)")
    print("=" * 60)
    print(f"Total instances:              {total}")
    print(f"Saved .npz:                   {n_saved}")
    print(f"Skipped (total):              {n_skipped}")
    for reason, cnt in sorted(skip_reasons.items()):
        print(f"  - {reason}: {cnt}")
    print(f"LCC applied:                  {n_lcc}")
    print()
    print("Split (by scene):")
    for spl, ids in split_scenes.items():
        n_inst = sum(1 for s in all_stats if s["split"] == spl and not s["skipped"])
        n_skip = sum(1 for s in all_stats if s["split"] == spl and s["skipped"])
        print(f"  {spl:5s}: {len(ids):3d} scenes  {n_inst:4d} saved  {n_skip} skipped")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 3 — Preprocessing (leak-fixed)")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(__file__).resolve().parents[3]
    data_root  = project_root / cfg["data"]["root"]
    cache_dir  = project_root / cfg["data"]["cache_dir"]

    preprocess_all(data_root, cache_dir, cfg)


if __name__ == "__main__":
    main()
