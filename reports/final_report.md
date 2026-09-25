# Final Report — 3D Bounding Box Prediction (Honest Pipeline)

Generated: 2026-09-24 | Updated: 2026-09-25 (r1_full_s42_retrain)

## Pipeline Summary

This report presents the final honest evaluation of the 3D bounding box prediction
pipeline after fixing the GT-leakage bug in preprocessing (Step 2).

**Key changes vs. old `pca_sign_fix_full` run:**
- Preprocessing no longer uses GT box center for DBSCAN filtering
- 7-dim extra features: `[log1p(n_raw), extent_x, extent_y, extent_z, view_x, view_y, view_z]`
- Split: 1901 instances (1536 train / 207 val / 158 test; 16 skipped due to < 30 points)
- WINNER: R1 (extra_dim=7, use_rgb=False, use_pca=True)

## Decision Log

| Step | Decision | Basis |
|---|---|---|
| USE_NEW_FEATURES | True (ring gap features) | max |Pearson r| = 0.167 ≥ 0.10 |
| WINNER | R1 (extra_dim=7) | R2 IoU=0.3640 − R1 IoU=0.3568 = +0.0072 < 0.02 threshold |
| Seed | 42 | `r1_full_s42_retrain` |

## Step 3 — Quick Comparison (20 epochs)

| Model | Extra dim | use_rgb | Best val IoU (20 ep) |
|---|---|---|---|
| R1 (leak-fixed) | 7 | False | 0.3568 |
| R2 (new features) | 11 | False | 0.3640 |

## Step 4 — Full Training Results (80 epochs)

| Run | Best val IoU | Best epoch |
|---|---|---|
| `r1_full_s42_retrain` | 0.4012 | 75 |

## Step 5 — Best Model

| Split | Metric | `r1_full_s42_retrain` |
|---|---|---|
| Val | mean IoU | 0.4012 |
| Test | mean IoU | 0.4213 |
| Test | Acc@0.25 | 0.7215 |
| Test | Acc@0.5 | 0.4114 |

## Step 6 — Comparison with Baseline (test split)

| Model | N | mean IoU | Acc@0.25 | Acc@0.5 | corner (mm) |
|---|---|---|---|---|---|
| Geometric PCA baseline | 158 | 0.3407 | 0.6200 | 0.2278 | 49.8 |
| `r1_full_s42_retrain` (25 Sep) | 158 | **0.4213** | **0.7215** | **0.4114** | **44.0** |
| Delta vs baseline | | +0.081 | +0.102 | +0.184 | -5.8 |

## Oracle Analysis (val set, s42)

| Component replaced with GT | Mean IoU | Delta |
|---|---|---|
| (a) Original prediction | TBD | — |
| (b) Oracle center | TBD | TBD |
| (c) Oracle size | TBD | TBD |
| (d) Oracle rotation | TBD | TBD |

**Conclusion:** Replacing ___ gives the largest IoU gain (+TBD).

### Canonical size error per axis (mm)

| Axis | Description | Mean error (mm) |
|---|---|---|
| axis-0 | largest variance | TBD |
| axis-1 | medium variance | TBD |
| axis-2 | smallest (depth) | TBD |

### Axis-0/1 swap analysis

| | Normal | Swapped |
|---|---|---|
| Mean err (mm) | TBD | TBD |
| Fraction improved by swap | TBD | — |

## Ablation vs. Old Leaky Model

| Model | Val IoU | Test IoU | Notes |
|---|---|---|---|
| pca_sign_fix_full (LEAKY) | 0.441 | 0.396 | Old model — GT leaked in preprocessing |
| pca_sign_fix_full (honest re-eval) | 0.392 | 0.396 | Same checkpoint, clean eval |
| `r1_full_s42_retrain` (honest) | 0.401 | **0.421** | Leak-fixed; 7-dim extra; no RGB; 25 Sep |
| `r1_full_s42_retrain` (honest) | 0.401 | **0.421** | Leak-fixed; 7-dim extra; no RGB; 25 Sep |
