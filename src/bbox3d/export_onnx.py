"""
Step 11 — ONNX export (export_onnx.py).

Exports the PointNet model (raw forward, canonical frame) to three variants:
  FP32 — full precision
  FP16 — onnxconverter_common float16 conversion (keeps I/O in FP32)
  INT8  — OnnxRuntime dynamic quantization (weight-only QInt8)

Verification: onnx.checker + ORT vs PyTorch max abs diff < 1e-4 (FP32 only).

Usage:
    python -m bbox3d.export_onnx --config configs/default.yaml [--run main_run]

Outputs:
    outputs/onnx/model_fp32.onnx
    outputs/onnx/model_fp16.onnx
    outputs/onnx/model_int8.onnx
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic

from bbox3d.losses import compute_log_size_prior
from bbox3d.models.pointnet_box import build_model

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _load_model(cfg: dict, ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
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


def export_fp32(model: torch.nn.Module, onnx_path: Path, extra_dim: int = 4) -> bytes:
    """Export model to FP32 ONNX (opset 17, dynamic batch). Returns raw bytes."""
    pts   = torch.zeros(1, 512, 6)
    extra = torch.zeros(1, extra_dim)

    buf = io.BytesIO()
    torch.onnx.export(
        model, (pts, extra), buf,
        opset_version=17,
        input_names=["pts", "extra"],
        output_names=["raw_output"],
        dynamic_axes={"pts":  {0: "batch"},
                      "extra": {0: "batch"},
                      "raw_output": {0: "batch"}},
        dynamo=False,
    )
    raw = buf.getvalue()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(raw)
    return raw


def verify_fp32(fp32_bytes: bytes, model: torch.nn.Module, extra_dim: int = 4) -> float:
    """
    Check model validity and measure ORT vs PyTorch max abs diff on a random batch.
    Returns max abs diff (should be < 1e-4).
    """
    m = onnx.load_from_string(fp32_bytes)
    onnx.checker.check_model(m)

    sess = ort.InferenceSession(fp32_bytes, providers=["CPUExecutionProvider"])
    rng  = np.random.default_rng(0)
    pts_np   = rng.random((4, 512, 6),        dtype=np.float32)
    extra_np = rng.random((4, extra_dim),     dtype=np.float32)

    ort_out = sess.run(None, {"pts": pts_np, "extra": extra_np})[0]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(pts_np),
                          torch.from_numpy(extra_np)).numpy()
    return float(np.abs(ort_out - torch_out).max())


def export_fp16(fp32_bytes: bytes, onnx_path: Path) -> None:
    """Convert FP32 ONNX to FP16 (I/O kept in FP32 via keep_io_types=True)."""
    fp32_model = onnx.load_from_string(fp32_bytes)
    fp16_model = float16.convert_float_to_float16(fp32_model, keep_io_types=True)
    onnx_path.write_bytes(fp16_model.SerializeToString())


def export_int8(fp32_bytes: bytes, onnx_path: Path) -> None:
    """Dynamic INT8 quantization of weights using OnnxRuntime."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(fp32_bytes)
        tmp_in = f.name
    try:
        quantize_dynamic(tmp_in, str(onnx_path), weight_type=QuantType.QInt8)
    finally:
        os.unlink(tmp_in)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11 — ONNX export")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run",    default="main_run")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root      = Path(__file__).resolve().parents[2]
    runs_dir  = root / cfg["outputs_dir"] / "runs"
    onnx_dir  = root / cfg["outputs_dir"] / "onnx"
    ckpt_path = runs_dir / args.run / "best.pt"

    onnx_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model ...")
    model, ckpt_cfg = _load_model(cfg, ckpt_path)
    extra_dim = int(ckpt_cfg.get("dataset", {}).get("extra_dim", 4))

    # FP32
    fp32_path = onnx_dir / "model_fp32.onnx"
    print("Exporting FP32 ...")
    fp32_bytes = export_fp32(model, fp32_path, extra_dim=extra_dim)
    diff = verify_fp32(fp32_bytes, model, extra_dim=extra_dim)
    fp32_kb = fp32_path.stat().st_size // 1024
    print(f"  FP32  {fp32_kb:6d} KB   ORT-vs-PyTorch max diff = {diff:.2e}"
          f"  ({'PASS' if diff < 1e-4 else 'WARN'})")

    # FP16
    fp16_path = onnx_dir / "model_fp16.onnx"
    print("Exporting FP16 ...")
    export_fp16(fp32_bytes, fp16_path)
    fp16_kb = fp16_path.stat().st_size // 1024
    print(f"  FP16  {fp16_kb:6d} KB   ({fp16_kb / fp32_kb * 100:.0f}% of FP32)")

    # INT8
    int8_path = onnx_dir / "model_int8.onnx"
    print("Exporting INT8 ...")
    export_int8(fp32_bytes, int8_path)
    int8_kb = int8_path.stat().st_size // 1024
    print(f"  INT8  {int8_kb:6d} KB   ({int8_kb / fp32_kb * 100:.0f}% of FP32)")

    print(f"\nONNX models saved to {onnx_dir}")
    print(f"  model_fp32.onnx  {fp32_kb} KB")
    print(f"  model_fp16.onnx  {fp16_kb} KB")
    print(f"  model_int8.onnx  {int8_kb} KB")


if __name__ == "__main__":
    main()
