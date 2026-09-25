# Inference Benchmark

CPU-only (no CUDA). Latency over 200 runs (after 20 warmups).

## Latency

| Variant | File size (KB) | Batch 1 mean (ms) | Batch 1 p95 (ms) | Batch 16 mean (ms) | Batch 16 p95 (ms) |
|---|---|---|---|---|---|
| PyTorch FP32 | 3281 | 3.1 | 4.2 | 42.9 | 50.3 |
| ORT FP32 | 3281 | 1.0 | 1.3 | 16.9 | 22.8 |
| ORT FP16 | 1644 | 1.1 | 1.6 | 16.9 | 21.2 |
| ORT INT8 | 846 | 12.5 | 17.3 | 212.7 | 272.7 |

## Test-set Accuracy

| Variant | Test mean IoU | Delta vs FP32 |
|---|---|---|
| PyTorch FP32 | 0.4213 | +0.0000 |
| ORT FP32 | 0.4213 | +0.0000 |
| ORT FP16 | 0.4214 | +0.0000 |
| ORT INT8 | 0.3926 | -0.0288 |

## Notes
- FP16 I/O kept in FP32 (`keep_io_types=True`); weights/activations in FP16.
- INT8: dynamic weight quantization (OnnxRuntime `quantize_dynamic`, QInt8).
- TensorRT: not supported on this hardware (Quadro M1200, Maxwell).
  Next step: `trtexec --onnx=model_fp32.onnx --fp16` on a Pascal/Ampere GPU.
