"""
Step 11 — CPU inference benchmark (benchmark.py).

Measures latency (mean + p95 over 200 runs after 20 warmups) for batch sizes 1
and 16, and evaluates test-set mean IoU for every model variant.

Variants compared:
  PyTorch FP32  (torch model, no ONNX)
  ORT FP32      (model_fp32.onnx)
  ORT FP16      (model_fp16.onnx, inputs remain FP32 due to keep_io_types)
  ORT INT8      (model_int8.onnx, dynamic weight quantization)

Usage:
    python -m bbox3d.benchmark --config configs/default.yaml [--run main_run]

Outputs:
    reports/inference.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from torch.utils.data import DataLoader

from bbox3d.data.dataset import BBox3DDataset
from bbox3d.losses import compute_log_size_prior
from bbox3d.metrics import aggregate_metrics, evaluate_instance
from bbox3d.models.pointnet_box import build_model


# ---------------------------------------------------------------------------
# Numpy decode  (raw B×12 output -> corners B×8×3 in canonical frame)
# ---------------------------------------------------------------------------

_SIGNS_NP = np.array([
    [-1., -1.,  1.],
    [ 1., -1.,  1.],
    [ 1.,  1.,  1.],
    [-1.,  1.,  1.],
    [-1., -1., -1.],
    [ 1., -1., -1.],
    [ 1.,  1., -1.],
    [-1.,  1., -1.],
], dtype=np.float32)


def decode_raw_numpy(raw: np.ndarray) -> np.ndarray:
    """
    Decode raw model output (B, 12) to corners (B, 8, 3) in canonical frame.

    Mirrors the torch decode step: Gram-Schmidt rotation + params_to_corners.
    """
    center = raw[:, :3]              # (B, 3)
    size   = np.exp(raw[:, 3:6])     # (B, 3)

    a1 = raw[:, 6:9]
    a2 = raw[:, 9:12]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    proj = (b1 * a2).sum(axis=-1, keepdims=True)
    b2   = a2 - proj * b1
    b2   = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3   = np.cross(b1, b2)          # (B, 3)
    R    = np.stack([b1, b2, b3], axis=-1)   # (B, 3, 3)

    half    = _SIGNS_NP[None] * (size[:, None] / 2)          # (B, 8, 3)
    corners = center[:, None] + half @ R.transpose(0, 2, 1)  # (B, 8, 3)
    return corners.astype(np.float64)


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def _bench_torch(
    model: torch.nn.Module,
    batch: int,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> tuple[float, float]:
    """Returns (mean_ms, p95_ms) for PyTorch FP32 inference."""
    pts   = torch.zeros(batch, 512, 6)
    extra = torch.zeros(batch, model.head[0].in_features - 1024)
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(pts, extra)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(pts, extra)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def _bench_ort(
    sess: ort.InferenceSession,
    batch: int,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> tuple[float, float]:
    """Returns (mean_ms, p95_ms) for an ORT session."""
    extra_dim = sess.get_inputs()[1].shape[1]
    pts_np   = np.zeros((batch, 512, 6),       dtype=np.float32)
    extra_np = np.zeros((batch, extra_dim),    dtype=np.float32)
    feeds    = {"pts": pts_np, "extra": extra_np}

    for _ in range(n_warmup):
        sess.run(None, feeds)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


# ---------------------------------------------------------------------------
# Test-set IoU evaluation
# ---------------------------------------------------------------------------

def _eval_torch(
    model: torch.nn.Module,
    cfg: dict,
    items: list[dict],
    cache_dir: Path,
) -> float:
    dataset = BBox3DDataset(items, cache_dir, cfg, augment_data=False, seed=42)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    results = []
    with torch.no_grad():
        for batch in loader:
            _, _, _, pred_corners = model.forward_decode(batch["pts"], batch["extra"])
            pred_cam = (torch.bmm(pred_corners, batch["R0"].transpose(1, 2))
                        + batch["t0"].unsqueeze(1))
            for b in range(batch["pts"].shape[0]):
                results.append(evaluate_instance(
                    pred_cam[b].numpy().astype(np.float64),
                    batch["gt_corners_cam"][b].numpy().astype(np.float64),
                ))
    agg = aggregate_metrics(results)
    return float(agg["all"]["mean_iou"])


def _eval_ort(
    sess: ort.InferenceSession,
    cfg: dict,
    items: list[dict],
    cache_dir: Path,
) -> float:
    dataset = BBox3DDataset(items, cache_dir, cfg, augment_data=False, seed=42)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    results = []
    for batch in loader:
        pts_np   = batch["pts"].numpy()
        extra_np = batch["extra"].numpy()
        raw      = sess.run(None, {"pts": pts_np, "extra": extra_np})[0]  # (B, 12)

        corners_canon = decode_raw_numpy(raw)   # (B, 8, 3)
        R0  = batch["R0"].numpy()               # (B, 3, 3)
        t0  = batch["t0"].numpy()               # (B, 3)
        pred_cam = corners_canon @ R0.transpose(0, 2, 1) + t0[:, None, :]   # (B, 8, 3)

        for b in range(pts_np.shape[0]):
            results.append(evaluate_instance(
                pred_cam[b],
                batch["gt_corners_cam"][b].numpy().astype(np.float64),
            ))
    agg = aggregate_metrics(results)
    return float(agg["all"]["mean_iou"])


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_inference_report(rows: list[dict], report_path: Path) -> None:
    """Write inference.md with latency table and IoU table."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Inference Benchmark\n",
             "CPU-only (no CUDA). Latency over 200 runs (after 20 warmups).\n"]

    # Latency table
    lines += [
        "## Latency",
        "",
        "| Variant | File size (KB) | Batch 1 mean (ms) | Batch 1 p95 (ms) |"
        " Batch 16 mean (ms) | Batch 16 p95 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['size_kb']} |"
            f" {r['b1_mean']:.1f} | {r['b1_p95']:.1f} |"
            f" {r['b16_mean']:.1f} | {r['b16_p95']:.1f} |"
        )

    # Accuracy table
    lines += [
        "",
        "## Test-set Accuracy",
        "",
        "| Variant | Test mean IoU | Delta vs FP32 |",
        "|---|---|---|",
    ]
    fp32_iou = next((r["iou"] for r in rows if r["label"] == "PyTorch FP32"), None)
    for r in rows:
        delta = f"{r['iou'] - fp32_iou:+.4f}" if fp32_iou is not None else "—"
        lines.append(f"| {r['label']} | {r['iou']:.4f} | {delta} |")

    lines += [
        "",
        "## Notes",
        "- FP16 I/O kept in FP32 (`keep_io_types=True`); weights/activations in FP16.",
        "- INT8: dynamic weight quantization (OnnxRuntime `quantize_dynamic`, QInt8).",
        "- TensorRT: not supported on this hardware (Quadro M1200, Maxwell).",
        "  Next step: `trtexec --onnx=model_fp32.onnx --fp16` on a Pascal/Ampere GPU.",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport -> {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11 — Inference benchmark")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run",    default="main_run")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root      = Path(__file__).resolve().parents[2]
    cache_dir = root / cfg["data"]["cache_dir"]
    runs_dir  = root / cfg["outputs_dir"] / "runs"
    onnx_dir  = root / cfg["outputs_dir"] / "onnx"
    report_md = root / cfg["reports_dir"] / "inference.md"
    ckpt_path = runs_dir / args.run / "best.pt"

    with open(cache_dir / "split.json") as f:
        split_data = json.load(f)
    items_test = split_data["test"]

    # Load PyTorch model
    ckpt     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", cfg)
    log_size_init = compute_log_size_prior(split_data["train"], cache_dir)
    model = build_model(ckpt_cfg, log_size_init)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load ORT sessions
    fp32_path = onnx_dir / "model_fp32.onnx"
    fp16_path = onnx_dir / "model_fp16.onnx"
    int8_path = onnx_dir / "model_int8.onnx"
    for p in [fp32_path, fp16_path, int8_path]:
        if not p.exists():
            raise FileNotFoundError(f"ONNX not found: {p}\nRun export_onnx.py first.")

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_fp16 = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    variants = [
        ("PyTorch FP32", None,      model,     fp32_path.stat().st_size // 1024),
        ("ORT FP32",     sess_fp32, None,      fp32_path.stat().st_size // 1024),
        ("ORT FP16",     sess_fp16, None,      fp16_path.stat().st_size // 1024),
        ("ORT INT8",     sess_int8, None,      int8_path.stat().st_size // 1024),
    ]

    rows = []
    for label, sess, mdl, size_kb in variants:
        print(f"\n--- {label} ---")

        # Latency batch=1
        if mdl is not None:
            b1m, b1p = _bench_torch(mdl, 1)
            b16m, b16p = _bench_torch(mdl, 16)
        else:
            b1m,  b1p  = _bench_ort(sess, 1)
            b16m, b16p = _bench_ort(sess, 16)
        print(f"  batch=1:  mean={b1m:.1f}ms  p95={b1p:.1f}ms")
        print(f"  batch=16: mean={b16m:.1f}ms  p95={b16p:.1f}ms")

        # Test-set IoU
        print("  Evaluating test IoU ...")
        if mdl is not None:
            iou = _eval_torch(mdl, ckpt_cfg, items_test, cache_dir)
        else:
            iou = _eval_ort(sess, ckpt_cfg, items_test, cache_dir)
        print(f"  test mean IoU = {iou:.4f}")

        rows.append(dict(
            label=label, size_kb=size_kb,
            b1_mean=b1m, b1_p95=b1p,
            b16_mean=b16m, b16_p95=b16p,
            iou=iou,
        ))

    write_inference_report(rows, report_md)

    # Console summary
    print("\n=== Benchmark Summary ===")
    print(f"{'Variant':<18} {'Size(KB)':>8} {'B1 mean':>9} {'B1 p95':>8}"
          f" {'B16 mean':>9} {'IoU':>7}")
    for r in rows:
        print(f"{r['label']:<18} {r['size_kb']:>8}"
              f" {r['b1_mean']:>8.1f}ms {r['b1_p95']:>7.1f}ms"
              f" {r['b16_mean']:>8.1f}ms {r['iou']:>7.4f}")


if __name__ == "__main__":
    main()
