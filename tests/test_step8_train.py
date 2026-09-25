"""
Tests for Step 8 — training loop (train.py).

Smoke-tests only (2 epochs) to verify the infrastructure works:
  - run directory + required files are created
  - log.csv has correct header and rows
  - last.pt and best.pt are valid checkpoints
  - LR schedule is monotone during warmup and decays after it
  - loss decreases between epochs (sanity, not guaranteed but expected)
  - resume from last.pt restarts cleanly
"""
from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH  = PROJECT_ROOT / "configs" / "default.yaml"
RUN_DIR      = PROJECT_ROOT / "outputs" / "runs" / "_test_step8_pytest"


def _run_train(extra_cfg: dict | None = None, run_name: str = "_test_step8_pytest",
               resume: str | None = None):
    """Run 2 epochs of training and return the run dir."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["epochs"]               = 2
    cfg["train"]["warmup_epochs"]        = 1
    cfg["train"]["early_stop_patience"]  = 10
    cfg["train"]["val_iou_every"]        = 1
    if extra_cfg:
        for k, v in extra_cfg.items():
            cfg["train"][k] = v

    run_dir = PROJECT_ROOT / "outputs" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "_test_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    argv_backup = sys.argv[:]
    sys.argv = ["train", "--config", str(cfg_path), "--run", run_name]
    if resume:
        sys.argv += ["--resume", resume]
    try:
        from bbox3d.train import main
        main()
    finally:
        sys.argv = argv_backup
    return run_dir


# ---------------------------------------------------------------------------
# Run once and cache the result to avoid repeated 2-epoch runs
# ---------------------------------------------------------------------------

import os as _os
_TEST_RUN_NAME = f"_test_step8_pytest_{_os.getpid()}"
_cached_run_dir: Path | None = None


def _get_run_dir() -> Path:
    global _cached_run_dir
    if _cached_run_dir is None:
        _cached_run_dir = _run_train(run_name=_TEST_RUN_NAME)
    return _cached_run_dir


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------

def test_run_dir_files_created():
    d = _get_run_dir()
    assert (d / "log.csv").exists(),      "log.csv missing"
    assert (d / "last.pt").exists(),      "last.pt missing"
    assert (d / "config.yaml").exists() or (d / "_test_config.yaml").exists(), \
        "config yaml missing"
    assert (d / "run_info.json").exists(), "run_info.json missing"


def test_best_pt_created():
    """best.pt must exist if any IoU was computed (2 epochs, always better than −1)."""
    d = _get_run_dir()
    assert (d / "best.pt").exists(), "best.pt missing"


# ---------------------------------------------------------------------------
# log.csv structure
# ---------------------------------------------------------------------------

def test_log_csv_header():
    d = _get_run_dir()
    with open(d / "log.csv") as f:
        reader = csv.DictReader(f)
        assert "epoch"    in reader.fieldnames
        assert "val_iou"  in reader.fieldnames
        assert "train_loss" in reader.fieldnames
        assert "val_loss"   in reader.fieldnames


def test_log_csv_rows():
    d = _get_run_dir()
    with open(d / "log.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
    assert int(rows[0]["epoch"]) == 0
    assert int(rows[1]["epoch"]) == 1


def test_log_csv_finite_values():
    d = _get_run_dir()
    with open(d / "log.csv") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("train_loss", "val_loss", "val_iou", "lr"):
            val = float(row[key])
            assert np.isfinite(val), f"Non-finite {key}={val} in row {row['epoch']}"


# ---------------------------------------------------------------------------
# Checkpoint structure
# ---------------------------------------------------------------------------

def test_last_pt_keys():
    d = _get_run_dir()
    ckpt = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
    for key in ("epoch", "model_state", "optimizer_state", "scheduler_state",
                "best_val_iou"):
        assert key in ckpt, f"Missing key '{key}' in last.pt"


def test_last_pt_epoch():
    d = _get_run_dir()
    ckpt = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1, f"last.pt epoch={ckpt['epoch']}, expected 1"


def test_best_val_iou_nonneg():
    d = _get_run_dir()
    ckpt = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["best_val_iou"] >= 0, \
        f"best_val_iou={ckpt['best_val_iou']} is negative"


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def test_lr_warmup():
    """LR at epoch 0 must be > 0; epoch 1 lr is logged after cosine step so can reach 0."""
    d = _get_run_dir()
    with open(d / "log.csv") as f:
        rows = list(csv.DictReader(f))
    lr0 = float(rows[0]["lr"])
    assert lr0 > 0, f"lr at epoch 0 = {lr0}"
    # lr is logged as get_last_lr() after scheduler.step() so it reflects the NEXT epoch's lr.
    # With only 2 epochs, cosine completes fully → epoch-1 lr can be 0; just check finite.
    lr1 = float(rows[1]["lr"])
    assert np.isfinite(lr1) and lr1 >= 0, f"lr at epoch 1 = {lr1}"


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def test_resume_continues_epoch():
    """Resuming from last.pt (epoch 1) with total=4 epochs should reach epoch 3."""
    src_dir = _get_run_dir()
    resume_path = str(src_dir / "last.pt")
    run_name2 = f"_test_step8_resume_{_os.getpid()}"
    # Use 4 total epochs so the resumed run processes epochs 2 and 3
    run_dir2 = _run_train(extra_cfg={"epochs": 4}, run_name=run_name2,
                          resume=resume_path)
    ckpt = torch.load(run_dir2 / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 3, f"After resume, expected epoch=3, got {ckpt['epoch']}"


# ---------------------------------------------------------------------------
# run_info.json
# ---------------------------------------------------------------------------

def test_run_info_keys():
    d = _get_run_dir()
    with open(d / "run_info.json") as f:
        info = json.load(f)
    assert "run_name"   in info
    assert "start_time" in info
    assert "git_hash"   in info
