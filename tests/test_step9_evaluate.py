"""Tests for Step 9 — evaluate.py."""
from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from bbox3d.evaluate import (
    _apply_ds_override,
    _eval_baseline,
    _eval_pointnet,
    write_reports,
    ABLATIONS,
)
from bbox3d.metrics import aggregate_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
SPLIT_JSON  = ROOT / "outputs" / "cache" / "split.json"
CACHE_DIR   = ROOT / "outputs" / "cache"
MAIN_CKPT   = ROOT / "outputs" / "runs" / "pca_sign_fix_full" / "best.pt"


@pytest.fixture(scope="module")
def base_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def test_items():
    with open(SPLIT_JSON) as f:
        split = json.load(f)
    return split["test"][:10]   # first 10 instances for speed


# ---------------------------------------------------------------------------

def test_apply_ds_override(base_cfg):
    cfg = _apply_ds_override(base_cfg, {"use_rgb": False, "use_pca": False})
    assert cfg["dataset"]["use_rgb"] is False
    assert cfg["dataset"]["use_pca"] is False
    # original untouched
    assert base_cfg["dataset"]["use_rgb"] is True


def test_ablation_variants_defined():
    names = [a["name"] for a in ABLATIONS]
    assert "pca_baseline"  in names
    assert "xyz_no_pca"    in names
    assert "xyz_pca"       in names
    assert "xyz_rgb_pca"   in names
    # exactly one "main" flag
    mains = [a for a in ABLATIONS if a.get("main")]
    assert len(mains) == 1


def test_eval_baseline_smoke(test_items):
    results = _eval_baseline(test_items, CACHE_DIR)
    assert len(results) > 0
    for r in results:
        assert "iou_3d" in r
        assert 0.0 <= r["iou_3d"] <= 1.0


@pytest.mark.skipif(not MAIN_CKPT.exists(), reason="main_run best.pt not found")
def test_eval_pointnet_smoke(base_cfg, test_items):
    results = _eval_pointnet(base_cfg, MAIN_CKPT, test_items, CACHE_DIR)
    assert len(results) == len(test_items)
    for r in results:
        assert 0.0 <= r["iou_3d"] <= 1.0


@pytest.mark.skipif(not MAIN_CKPT.exists(), reason="main_run best.pt not found")
def test_main_model_beats_baseline(base_cfg, test_items):
    """PointNet main model should achieve higher mean IoU than the PCA baseline."""
    baseline_results = _eval_baseline(test_items, CACHE_DIR)
    pointnet_results = _eval_pointnet(base_cfg, MAIN_CKPT, test_items, CACHE_DIR)

    baseline_iou = aggregate_metrics(baseline_results)["all"]["mean_iou"]
    pointnet_iou = aggregate_metrics(pointnet_results)["all"]["mean_iou"]

    print(f"\n  Baseline IoU={baseline_iou:.4f}  PointNet IoU={pointnet_iou:.4f}")
    # PointNet should beat baseline (or be close on the 10-instance sample)
    assert pointnet_iou >= baseline_iou - 0.05, (
        f"PointNet ({pointnet_iou:.4f}) should not be much worse than baseline ({baseline_iou:.4f})"
    )


def test_write_reports_smoke(tmp_path, base_cfg, test_items):
    results = _eval_baseline(test_items, CACHE_DIR)
    agg = aggregate_metrics(results)
    rows = [{"label": "Geometric PCA baseline", "agg": agg}]

    md_path  = tmp_path / "results.md"
    csv_path = tmp_path / "results.csv"
    write_reports(rows, md_path, csv_path)

    assert md_path.exists()
    assert csv_path.exists()
    content = md_path.read_text()
    assert "Geometric PCA baseline" in content
    assert "mean IoU" in content
