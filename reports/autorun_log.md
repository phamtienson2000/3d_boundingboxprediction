# Autorun Log — 2026-09-24

Pipeline: Step 0 → 7, fully autonomous.
Config baseline: `pca_sign_fix_full` (old leaky val IoU ≈ 0.441).

---

## Step 0 — IoU sanity checks
Status: DONE
- IoU(GT,GT)=1.000 PASS
- IoU(shift 0.5)=0.3333 PASS (expected 1/3)
- IoU(cube rot90z)=1.000 PASS
- All 3 tests pass. No IoU fix needed.

## Step 1 — Residual analysis (train split)
Status: DONE — USE_NEW_FEATURES = True
- Residual mean=23.8mm  std=39.2mm  (model under-predicts depth)
- t0[2]:            Pearson=-0.085  Spearman=-0.063
- gap_p50_clip:     Pearson=+0.146  Spearman=+0.181
- gap_p90_clip:     Pearson=+0.167  Spearman=+0.203
- ring_other_frac:  Pearson=+0.108  Spearman=+0.082
- max |r| = 0.167 >= 0.1 → USE_NEW_FEATURES = True

## Step 2 — Leak fix + re-preprocess
Status: DONE
- Removed GT-based DBSCAN condition; always apply LCC (no GT used)
- gap_p50, gap_p90, ring_other_frac computed from pc+mask only, saved to .npz
- 1901 total saved (train=1536, val=207, test=158)
- Skipped: train=13, val=0, test=3 (IoU=0 in eval)
- OLD pca_sign_fix_full val IoU (leaky):  0.441
- NEW pca_sign_fix_full val IoU (honest): 0.392 (delta=-4.9pp, leakage confirmed)
- git commit: 5d6e9c8

## Step 3 — Train 20ep R1 and R2
Status: DONE — WINNER = R1
- R1 (extra_dim=7, leak-fixed, use_rgb=False): best val IoU=0.3568 (epoch 15)
- R2 (extra_dim=11, use_new_features=True): best val IoU=0.3640 (epoch 19)
- Delta R2-R1 = +0.0072 < 0.02 threshold => WINNER = R1 (r1_full.yaml)

## Step 4 — Full train WINNER (R1, 80 epochs)
Status: DONE
- r1_full_s42_retrain (25 Sep): DONE — val IoU=0.4012, epoch 75

## Step 5 — Best model
Status: DONE — r1_full_s42_retrain (single model)

## Step 6 — Final eval + oracle
Status: PARTIAL
- r1_full_s42_retrain: test IoU=0.4213, Acc@0.25=0.7215, Acc@0.5=0.4114, corner=44.0mm
- Oracle analysis: not run

## Step 7 — Report
Status: DONE (25 Sep 2026)
- README_r1.md: results, comparison with 23 Sep baseline, failure cases, inference viz
- reports/final_report.md, results.md, inference.md: updated with r1_full_s42_retrain
- ONNX re-exported: FP32 IoU=0.4213, FP16=0.4214, INT8=0.3926 (delta -0.029)
