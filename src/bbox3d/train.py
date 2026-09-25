"""
Step 8 — Training loop (train.py).

Usage:
    python -m bbox3d.train --config configs/default.yaml [--run <name>] [--resume <last.pt>]

Saves to outputs/runs/<run>/: config.yaml, run_info.json, log.csv, last.pt, best.pt.
TensorBoard logs to the same run directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bbox3d.data.dataset import make_datasets
from bbox3d.losses import BBoxLoss, compute_log_size_prior
from bbox3d.metrics import aggregate_metrics, evaluate_instance
from bbox3d.models.pointnet_box import build_model


# ---------------------------------------------------------------------------
# Seeding + threading
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def _lr_lambda(warmup: int, total: int):
    def fn(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        t = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return fn


# ---------------------------------------------------------------------------
# One train epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    loss_fn:   BBoxLoss,
    optimizer: AdamW,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    acc = {"total": 0., "corner": 0., "center": 0., "size": 0.}
    n = 0

    for batch in loader:
        pts   = batch["pts"]
        extra = batch["extra"]
        gt    = batch["gt_corners_canon"]
        B     = pts.shape[0]

        optimizer.zero_grad()
        pred_c, pred_s, _, pred_corners = model.forward_decode(pts, extra)
        loss, parts = loss_fn(pred_corners, pred_c, pred_s, gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        acc["total"]  += loss.item()       * B
        acc["corner"] += parts["corner"].item() * B
        acc["center"] += parts["center"].item() * B
        acc["size"]   += parts["size"].item()   * B
        n += B

    return {k: v / n for k, v in acc.items()}


# ---------------------------------------------------------------------------
# One val epoch  (loss always; IoU metrics only when compute_iou=True)
# ---------------------------------------------------------------------------

def _val_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    loss_fn:     BBoxLoss,
    compute_iou: bool,
) -> tuple[float, Optional[dict]]:
    """
    Returns (val_loss_total, agg_or_None).
    agg is the aggregate_metrics dict when compute_iou=True, else None.
    """
    model.eval()
    total_loss = 0.
    n = 0
    instance_results: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            pts    = batch["pts"]
            extra  = batch["extra"]
            gt_can = batch["gt_corners_canon"]
            gt_cam = batch["gt_corners_cam"]
            R0     = batch["R0"]
            t0     = batch["t0"]
            B      = pts.shape[0]

            pred_c, pred_s, _, pred_corners = model.forward_decode(pts, extra)
            loss, _ = loss_fn(pred_corners, pred_c, pred_s, gt_can)
            total_loss += loss.item() * B
            n += B

            if compute_iou:
                # Transform predicted corners back to camera frame
                pred_cam = (torch.bmm(pred_corners,
                                      R0.transpose(1, 2))
                            + t0.unsqueeze(1))          # (B, 8, 3)
                for b in range(B):
                    instance_results.append(evaluate_instance(
                        pred_cam[b].numpy().astype(np.float64),
                        gt_cam[b].numpy().astype(np.float64),
                    ))

    val_loss = total_loss / n
    agg = aggregate_metrics(instance_results) if compute_iou else None
    return val_loss, agg


# ---------------------------------------------------------------------------
# Console summary line
# ---------------------------------------------------------------------------

def _console_line(
    epoch:     int,
    total:     int,
    lr:        float,
    t_loss:    float,
    v_loss:    float,
    agg:       Optional[dict],
    elapsed_s: float,
) -> str:
    iou_str = ""
    if agg and agg.get("all"):
        s = agg["all"]
        iou_str = (
            f"  IoU={s['mean_iou']:.4f}"
            f"  Acc@.25={s['acc_025']:.3f}"
            f"  Acc@.5={s['acc_050']:.3f}"
            f"  ctr={s['center_err_mm']['mean']:.1f}mm"
            f"  rot={s['rot_err_deg']['mean']:.1f}deg"
        )
    return (
        f"Ep {epoch:3d}/{total}"
        f"  lr={lr:.2e}"
        f"  trn={t_loss:.4f}"
        f"  val={v_loss:.4f}"
        f"{iou_str}"
        f"  {elapsed_s:.1f}s"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8 — Train PointNet box head")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run",    default=None,
                        help="Run name (default: timestamp)")
    parser.add_argument("--resume", default=None,
                        help="Path to last.pt to resume from")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- paths ----
    root       = Path(__file__).resolve().parents[2]
    cache_dir  = root / cfg["data"]["cache_dir"]
    split_json = root / cfg["data"]["split_json"]
    run_name   = args.run or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir    = root / cfg["outputs_dir"] / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- threading + seeding ----
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    tc   = cfg["train"]
    seed = int(tc["seed"])
    seed_everything(seed)

    # ---- save run metadata ----
    shutil.copy(args.config, run_dir / "config.yaml")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root
        ).decode().strip()[:8]
    except Exception:
        git_hash = "unknown"
    run_info = {
        "run_name":   run_name,
        "start_time": datetime.now().isoformat(),
        "git_hash":   git_hash,
        "config":     args.config,
        "n_threads":  n_threads,
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    # ---- datasets + loaders ----
    datasets = make_datasets(cfg, cache_dir, split_json)
    g_train  = torch.Generator(); g_train.manual_seed(seed)
    train_loader = DataLoader(
        datasets["train"], batch_size=tc["batch_size"],
        shuffle=True, num_workers=0, generator=g_train,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=tc["batch_size"],
        shuffle=False, num_workers=0,
    )

    # ---- model ----
    with open(split_json) as f:
        split = json.load(f)
    log_size_init = compute_log_size_prior(split["train"], cache_dir)
    print(f"Log-size prior: {np.round(log_size_init, 3)}")
    model   = build_model(cfg, log_size_init)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_param:,}  threads={n_threads}")

    # ---- loss / optimizer / scheduler ----
    loss_fn   = BBoxLoss(cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=float(tc["lr"]),
        weight_decay=float(tc["weight_decay"]),
    )
    scheduler = LambdaLR(
        optimizer,
        _lr_lambda(int(tc["warmup_epochs"]), int(tc["epochs"])),
    )

    # ---- resume ----
    start_epoch  = 0
    best_val_iou = -1.0
    no_improve   = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_iou = ckpt["best_val_iou"]
        no_improve   = ckpt.get("no_improve", 0)
        print(f"Resumed from epoch {start_epoch}, best IoU={best_val_iou:.4f}")

    # ---- TensorBoard + CSV ----
    writer   = SummaryWriter(str(run_dir))
    csv_path = run_dir / "log.csv"
    csv_header = [
        "epoch", "lr",
        "train_loss", "train_corner", "train_center", "train_size",
        "val_loss",
        "val_iou", "val_acc025", "val_acc050",
        "val_corner_mm", "val_center_mm", "val_rot_deg",
    ]
    # Fresh run: always overwrite; resume: append to existing log
    if not args.resume:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)
    elif not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    # ---- training loop ----
    patience       = int(tc["early_stop_patience"])
    max_epochs     = int(tc["epochs"])
    val_iou_every  = int(tc.get("val_iou_every", 5))
    grad_clip      = float(tc["grad_clip"])

    print(f"\nRun: {run_dir.name}  "
          f"epochs={max_epochs}  patience={patience}  "
          f"val_iou_every={val_iou_every}")
    print("=" * 72)

    for epoch in range(start_epoch, max_epochs):
        t_start = time.perf_counter()

        # train
        t_loss = _train_epoch(model, train_loader, loss_fn, optimizer, grad_clip)

        # val (full IoU every val_iou_every epochs, or on the last epoch)
        do_iou  = (epoch % val_iou_every == 0) or (epoch == max_epochs - 1)
        v_loss, agg = _val_epoch(model, val_loader, loss_fn, compute_iou=do_iou)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        elapsed = time.perf_counter() - t_start

        # ----- metrics for logging ----
        val_iou      = agg["all"]["mean_iou"]        if (agg and agg.get("all")) else float("nan")
        val_acc025   = agg["all"]["acc_025"]          if (agg and agg.get("all")) else float("nan")
        val_acc050   = agg["all"]["acc_050"]          if (agg and agg.get("all")) else float("nan")
        val_ctr_mm   = agg["all"]["center_err_mm"]["mean"] if (agg and agg.get("all")) else float("nan")
        val_rot_deg  = agg["all"]["rot_err_deg"]["mean"]   if (agg and agg.get("all")) else float("nan")
        val_cor_mm   = agg["all"]["corner_dist_mm"]["mean"] if (agg and agg.get("all")) else float("nan")

        # ----- console ----
        print(_console_line(epoch, max_epochs, lr, t_loss["total"], v_loss, agg, elapsed))

        # ----- TensorBoard ----
        writer.add_scalar("train/loss",        t_loss["total"],  epoch)
        writer.add_scalar("train/corner",      t_loss["corner"], epoch)
        writer.add_scalar("train/center",      t_loss["center"], epoch)
        writer.add_scalar("train/size",        t_loss["size"],   epoch)
        writer.add_scalar("val/loss",          v_loss,           epoch)
        writer.add_scalar("lr",                lr,               epoch)
        if not math.isnan(val_iou):
            writer.add_scalar("val/mean_iou",    val_iou,   epoch)
            writer.add_scalar("val/acc_025",     val_acc025, epoch)
            writer.add_scalar("val/acc_050",     val_acc050, epoch)
            writer.add_scalar("val/center_mm",   val_ctr_mm, epoch)
            writer.add_scalar("val/rot_deg",     val_rot_deg, epoch)

        # ----- CSV ----
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.6e}",
                f"{t_loss['total']:.6f}", f"{t_loss['corner']:.6f}",
                f"{t_loss['center']:.6f}", f"{t_loss['size']:.6f}",
                f"{v_loss:.6f}",
                f"{val_iou:.4f}", f"{val_acc025:.4f}", f"{val_acc050:.4f}",
                f"{val_cor_mm:.2f}", f"{val_ctr_mm:.2f}", f"{val_rot_deg:.2f}",
            ])

        # ----- checkpoints ----
        state = {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_iou":   best_val_iou,
            "no_improve":     no_improve,
            "cfg":            cfg,
        }
        torch.save(state, run_dir / "last.pt")

        if not math.isnan(val_iou):
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                state["best_val_iou"] = best_val_iou
                torch.save(state, run_dir / "best.pt")
                no_improve = 0
            else:
                # Count in epochs (val_iou_every epochs per IoU measurement)
                no_improve += val_iou_every

        if no_improve >= patience:
            print(f"Early stop: no IoU improvement for {no_improve} epochs.")
            break

    writer.close()
    print(f"\nDone. Best val IoU={best_val_iou:.4f}  run={run_dir}")


if __name__ == "__main__":
    main()
