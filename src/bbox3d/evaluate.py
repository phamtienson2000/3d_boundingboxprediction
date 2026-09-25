"""
Step 9 — Evaluation + ablations (evaluate.py).

Evaluates the trained PointNet model on the test split and produces an
ablation table comparing 4 variants:
  1. Geometric PCA baseline
  2. PointNet, xyz only, no PCA canonicalization
  3. PointNet, xyz only, with PCA
  4. PointNet, xyz + rgb, with PCA  (main model)

Usage:
    python -m bbox3d.evaluate --config configs/default.yaml [--main-run main_run]
    python -m bbox3d.evaluate --config configs/default.yaml --multi-seed

Outputs:
    reports/results.md    — ablation table in Markdown
    reports/results.csv   — one row per variant (machine-readable)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import yaml

from bbox3d.baseline import predict_pca_box
from bbox3d.data.dataset import BBox3DDataset, make_datasets
from bbox3d.losses import BBoxLoss, compute_log_size_prior
from bbox3d.metrics import aggregate_metrics, evaluate_instance
from bbox3d.models.pointnet_box import build_model
from bbox3d.train import (
    _lr_lambda,
    _train_epoch,
    _val_epoch,
    seed_everything,
)


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------

ABLATIONS = [
    dict(
        name="pca_baseline",
        label="Geometric PCA baseline",
        kind="baseline",
    ),
    dict(
        name="xyz_no_pca",
        label="PointNet xyz, no PCA",
        kind="pointnet",
        ds_override={"use_rgb": False, "use_pca": False},
    ),
    dict(
        name="xyz_pca",
        label="PointNet xyz + PCA",
        kind="pointnet",
        ds_override={"use_rgb": False, "use_pca": True},
    ),
    dict(
        name="xyz_rgb_pca",
        label="PointNet xyz+rgb + PCA (main)",
        kind="pointnet",
        ds_override={"use_rgb": True, "use_pca": True},
        main=True,
    ),
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _apply_ds_override(cfg: dict, ds_override: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["dataset"].update(ds_override)
    return c


# ---------------------------------------------------------------------------
# Training loop for ablation variants
# ---------------------------------------------------------------------------

def _train_variant(
    cfg: dict,
    run_dir: Path,
    split_json: Path,
    cache_dir: Path,
) -> None:
    """Train one variant from scratch; saves best.pt + log.csv into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tc   = cfg["train"]
    seed = int(tc["seed"])
    seed_everything(seed)

    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)

    with open(split_json) as f:
        split_data = json.load(f)

    datasets = make_datasets(cfg, cache_dir, split_json)
    g_train  = torch.Generator(); g_train.manual_seed(seed)
    train_loader = DataLoader(
        datasets["train"], batch_size=int(tc["batch_size"]),
        shuffle=True, num_workers=0, generator=g_train,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=int(tc["batch_size"]),
        shuffle=False, num_workers=0,
    )

    log_size_init = compute_log_size_prior(split_data["train"], cache_dir)
    model    = build_model(cfg, log_size_init)
    loss_fn  = BBoxLoss(cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=float(tc["lr"]),
        weight_decay=float(tc["weight_decay"]),
    )
    max_epochs     = int(tc["epochs"])
    warmup_epochs  = int(tc["warmup_epochs"])
    scheduler = LambdaLR(optimizer, _lr_lambda(warmup_epochs, max_epochs))

    patience      = int(tc["early_stop_patience"])
    val_iou_every = int(tc.get("val_iou_every", 5))
    grad_clip     = float(tc["grad_clip"])

    csv_path = run_dir / "log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr", "train_loss", "val_loss",
            "val_iou", "val_acc025", "val_acc050",
        ])

    best_val_iou = -1.0
    no_improve   = 0
    t0_run       = time.perf_counter()

    for epoch in range(max_epochs):
        t_loss = _train_epoch(model, train_loader, loss_fn, optimizer, grad_clip)
        do_iou  = (epoch % val_iou_every == 0) or (epoch == max_epochs - 1)
        v_loss, agg = _val_epoch(model, val_loader, loss_fn, compute_iou=do_iou)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        val_iou    = agg["all"]["mean_iou"] if (agg and agg.get("all")) else float("nan")
        val_acc025 = agg["all"]["acc_025"]  if (agg and agg.get("all")) else float("nan")
        val_acc050 = agg["all"]["acc_050"]  if (agg and agg.get("all")) else float("nan")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.6e}", f"{t_loss['total']:.6f}", f"{v_loss:.6f}",
                f"{val_iou:.4f}", f"{val_acc025:.4f}", f"{val_acc050:.4f}",
            ])

        if not math.isnan(val_iou):
            elapsed = time.perf_counter() - t0_run
            print(f"  ep {epoch:3d}  lr={lr:.2e}  trn={t_loss['total']:.4f}"
                  f"  val={v_loss:.4f}  IoU={val_iou:.4f}"
                  f"  Acc@.25={val_acc025:.3f}  {elapsed:.0f}s")
            state = {
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_iou":   best_val_iou,
                "cfg":            cfg,
            }
            torch.save(state, run_dir / "last.pt")
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                state["best_val_iou"] = best_val_iou
                torch.save(state, run_dir / "best.pt")
                no_improve = 0
            else:
                no_improve += val_iou_every

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}: {no_improve} epochs without improvement.")
            break

    print(f"  Done. best val IoU={best_val_iou:.4f}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_pointnet(
    cfg: dict,
    ckpt_path: Path,
    items: list[dict],
    cache_dir: Path,
) -> list[dict]:
    """Load a PointNet checkpoint and evaluate it on the given items list."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Use the config baked into the checkpoint if available; fall back to caller's cfg
    ckpt_cfg = copy.deepcopy(ckpt.get("cfg", cfg))

    # Auto-detect extra_dim from checkpoint state dict (handles old checkpoints with 4 features)
    head_w = ckpt["model_state"].get("head.0.weight")
    if head_w is not None:
        mlp_last = ckpt_cfg.get("model", {}).get("mlp_channels", [64, 128, 256, 512])[-1]
        inferred_extra_dim = int(head_w.shape[1]) - mlp_last * 2
        if "dataset" not in ckpt_cfg:
            ckpt_cfg["dataset"] = {}
        ckpt_cfg["dataset"]["extra_dim"] = inferred_extra_dim

    with open(cache_dir / "split.json") as f:
        split_data = json.load(f)
    log_size_init = compute_log_size_prior(split_data["train"], cache_dir)

    model = build_model(ckpt_cfg, log_size_init)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = BBox3DDataset(items, cache_dir, ckpt_cfg, augment_data=False,
                            seed=int(ckpt_cfg.get("train", {}).get("seed", 42)))
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    results: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            pts    = batch["pts"]
            extra  = batch["extra"]
            gt_cam = batch["gt_corners_cam"]
            R0     = batch["R0"]
            t0     = batch["t0"]

            _, _, _, pred_corners = model.forward_decode(pts, extra)
            # transform predicted corners (canonical → camera frame)
            pred_cam = (torch.bmm(pred_corners, R0.transpose(1, 2))
                        + t0.unsqueeze(1))    # (B, 8, 3)

            for b in range(pts.shape[0]):
                results.append(evaluate_instance(
                    pred_cam[b].numpy().astype(np.float64),
                    gt_cam[b].numpy().astype(np.float64),
                ))
    return results


def _eval_baseline(items: list[dict], cache_dir: Path) -> list[dict]:
    """PCA baseline evaluation on a list of split items."""
    results: list[dict] = []
    skipped = 0
    for item in items:
        npz_path = cache_dir / item["npz"]
        if not npz_path.exists():
            skipped += 1
            continue
        d   = np.load(npz_path, allow_pickle=True)
        xyz = d["pts"][:, :3].astype(np.float64)
        gt  = d["gt_corners"].astype(np.float64)
        if len(xyz) < 3:
            skipped += 1
            continue
        pred = predict_pca_box(xyz)
        results.append(evaluate_instance(pred, gt))
    if skipped:
        print(f"  baseline: skipped {skipped} items")
    return results


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def _row(label: str, agg: dict) -> tuple[list, str]:
    s = agg.get("all")
    if s is None:
        return [], ""
    csv_row = [
        label,
        s["n"],
        _fmt(s["mean_iou"]),
        _fmt(s["acc_025"]),
        _fmt(s["acc_050"]),
        _fmt(s["corner_dist_mm"]["mean"], 1),
        _fmt(s["center_err_mm"]["mean"],  1),
        _fmt(s["rot_err_deg"]["mean"],    1),
    ]
    md_row = (
        f"| {label} | {s['n']} | {s['mean_iou']:.4f} | {s['acc_025']:.3f} |"
        f" {s['acc_050']:.3f} | {s['corner_dist_mm']['mean']:.1f} |"
        f" {s['center_err_mm']['mean']:.1f} | {s['rot_err_deg']['mean']:.1f} |"
    )
    return csv_row, md_row


def write_reports(
    rows: list[dict],            # list of {label, agg, iou_list (optional)}
    report_md: Path,
    report_csv: Path,
) -> None:
    report_md.parent.mkdir(parents=True, exist_ok=True)

    header_csv = ["variant", "n", "mean_iou", "acc_025", "acc_050",
                  "corner_mm_mean", "center_mm_mean", "rot_deg_mean"]
    header_md  = (
        "| Variant | N | mean IoU | Acc@0.25 | Acc@0.5 |"
        " corner (mm) | center (mm) | rot (deg) |"
    )
    sep_md = "|---|---|---|---|---|---|---|---|"

    md_lines  = ["# Ablation Results\n",
                 header_md, sep_md]
    csv_rows  = [header_csv]

    for r in rows:
        csv_row, md_row = _row(r["label"], r["agg"])
        if csv_row:
            md_lines.append(md_row)
            csv_rows.append(csv_row)
        # multi-seed stats
        if r.get("multi_seed"):
            ms = r["multi_seed"]
            md_lines.append(
                f"| {r['label']} (3 seeds, mean±std) | {ms['n']} |"
                f" {ms['iou_mean']:.4f}±{ms['iou_std']:.4f} |"
                f" {ms['acc025_mean']:.3f}±{ms['acc025_std']:.3f} |"
                f" {ms['acc050_mean']:.3f}±{ms['acc050_std']:.3f} | — | — | — |"
            )

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\nReports written:")
    print(f"  {report_md}")
    print(f"  {report_csv}")


def _print_row(label: str, agg: dict) -> None:
    s = agg.get("all")
    if s is None:
        return
    print(
        f"  {label:<38s}  n={s['n']:4d}  "
        f"IoU={s['mean_iou']:.4f}  "
        f"Acc@0.25={s['acc_025']:.3f}  Acc@0.5={s['acc_050']:.3f}  "
        f"corner={s['corner_dist_mm']['mean']:.1f}mm  "
        f"rot={s['rot_err_deg']['mean']:.1f}deg"
    )


# ---------------------------------------------------------------------------
# Multi-seed helper
# ---------------------------------------------------------------------------

def _multi_seed_stats(
    cfg: dict,
    seeds: list[int],
    runs_dir: Path,
    split_json: Path,
    cache_dir: Path,
    items_test: list[dict],
) -> dict:
    ious_per_seed: list[list[float]] = []
    for seed in seeds:
        seed_cfg           = copy.deepcopy(cfg)
        seed_cfg["train"]["seed"] = seed
        run_name           = f"main_seed{seed}"
        run_dir            = runs_dir / run_name
        if not (run_dir / "best.pt").exists():
            print(f"\nTraining main model with seed={seed} ...")
            _train_variant(seed_cfg, run_dir, split_json, cache_dir)
        else:
            print(f"  seed={seed} checkpoint found, skipping training.")
        results = _eval_pointnet(seed_cfg, run_dir / "best.pt", items_test, cache_dir)
        ious_per_seed.append([r["iou_3d"] for r in results])
        agg = aggregate_metrics(results)
        print(f"  seed={seed}  IoU={agg['all']['mean_iou']:.4f}"
              f"  Acc@0.25={agg['all']['acc_025']:.3f}"
              f"  Acc@0.5={agg['all']['acc_050']:.3f}")

    # per-seed mean IoU + Acc metrics
    seed_iou_means   = [float(np.mean(iou_list)) for iou_list in ious_per_seed]
    seed_acc025_means = []
    seed_acc050_means = []
    for iou_list in ious_per_seed:
        seed_acc025_means.append(float(np.mean([i >= 0.25 for i in iou_list])))
        seed_acc050_means.append(float(np.mean([i >= 0.50 for i in iou_list])))

    n_per_seed = [len(x) for x in ious_per_seed]
    return {
        "n":            int(np.mean(n_per_seed)),
        "iou_mean":     float(np.mean(seed_iou_means)),
        "iou_std":      float(np.std(seed_iou_means)),
        "acc025_mean":  float(np.mean(seed_acc025_means)),
        "acc025_std":   float(np.std(seed_acc025_means)),
        "acc050_mean":  float(np.mean(seed_acc050_means)),
        "acc050_std":   float(np.std(seed_acc050_means)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 9 — Evaluation + ablations")
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--main-run",    default="main_run",
                        help="Name of the main run dir under outputs/runs/")
    parser.add_argument("--multi-seed",  action="store_true",
                        help="Also repeat main model with 3 seeds (takes ~3× longer)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Re-train ablation variants even if checkpoints exist")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    root       = Path(__file__).resolve().parents[2]
    cache_dir  = root / base_cfg["data"]["cache_dir"]
    split_json = root / base_cfg["data"]["split_json"]
    runs_dir   = root / base_cfg["outputs_dir"] / "runs"
    report_md  = root / base_cfg["reports_dir"] / "results.md"
    report_csv = root / base_cfg["reports_dir"] / "results.csv"

    with open(split_json) as f:
        split_data = json.load(f)
    items_test = split_data["test"]

    print(f"Test split: {len(items_test)} instances\n")

    report_rows: list[dict] = []

    for ab in ABLATIONS:
        label = ab["label"]

        if ab["kind"] == "baseline":
            print(f"Evaluating: {label}")
            results = _eval_baseline(items_test, cache_dir)
            agg = aggregate_metrics(results)
            _print_row(label, agg)
            report_rows.append({"label": label, "agg": agg})
            continue

        # PointNet variant
        ds_override = ab.get("ds_override", {})
        cfg         = _apply_ds_override(base_cfg, ds_override)
        is_main     = ab.get("main", False)

        if is_main:
            ckpt_path = runs_dir / args.main_run / "best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Main run checkpoint not found: {ckpt_path}\n"
                    "Run train.py first: python -m bbox3d.train"
                )
            print(f"Evaluating (main checkpoint): {label}")
        else:
            run_dir   = runs_dir / f"ablation_{ab['name']}"
            ckpt_path = run_dir / "best.pt"
            if args.force_retrain and ckpt_path.exists():
                ckpt_path.unlink()
                (run_dir / "last.pt").unlink(missing_ok=True)
            if not ckpt_path.exists():
                print(f"\nTraining ablation: {label}  -> {run_dir.name}")
                _train_variant(cfg, run_dir, split_json, cache_dir)
            else:
                print(f"Evaluating (existing checkpoint): {label}")

        results = _eval_pointnet(cfg, ckpt_path, items_test, cache_dir)
        agg = aggregate_metrics(results)
        _print_row(label, agg)
        report_rows.append({"label": label, "agg": agg})

    # Multi-seed stats for main model
    multi_seed_stats: Optional[dict] = None
    if args.multi_seed:
        print("\n=== Multi-seed evaluation (seeds 42, 123, 777) ===")
        multi_seed_stats = _multi_seed_stats(
            cfg=_apply_ds_override(base_cfg, {"use_rgb": True, "use_pca": True}),
            seeds=[42, 123, 777],
            runs_dir=runs_dir,
            split_json=split_json,
            cache_dir=cache_dir,
            items_test=items_test,
        )
        report_rows[-1]["multi_seed"] = multi_seed_stats
        print(f"\n  Main model (3 seeds): "
              f"IoU={multi_seed_stats['iou_mean']:.4f}±{multi_seed_stats['iou_std']:.4f}  "
              f"Acc@0.25={multi_seed_stats['acc025_mean']:.3f}±{multi_seed_stats['acc025_std']:.3f}  "
              f"Acc@0.5={multi_seed_stats['acc050_mean']:.3f}±{multi_seed_stats['acc050_std']:.3f}")

    write_reports(report_rows, report_md, report_csv)

    # Print final summary table
    print("\n=== Ablation Summary (test split) ===")
    for r in report_rows:
        _print_row(r["label"], r["agg"])


if __name__ == "__main__":
    main()
