# Ablation Results

> ⚠️ Rows below the horizontal rule were evaluated on data with DBSCAN leakage (23 Sep 2026).
> Absolute numbers are inflated; relative rankings are expected to hold.
> See `README_r1.md` for details and clean re-evaluation.

| Variant | N | mean IoU | Acc@0.25 | Acc@0.5 | corner (mm) | center (mm) | rot (deg) | Data |
|---|---|---|---|---|---|---|---|---|
| Geometric PCA baseline | 158 | 0.3407 | 0.620 | 0.228 | 49.8 | 33.2 | 18.0 | clean |
| **r1_full_s42_retrain** | **158** | **0.4213** | **0.7215** | **0.4114** | **44.0** | — | — | **clean** |
|---|---|---|---|---|---|---|---|---|
| PointNet xyz, no PCA *(leaky)* | 158 | 0.4005 | 0.741 | 0.373 | 46.0 | 26.8 | 20.2 | leaky |
| PointNet xyz + PCA *(leaky)* | 158 | 0.4118 | 0.734 | 0.418 | 44.2 | 26.8 | 17.4 | leaky |
| PointNet xyz+rgb + PCA *(leaky)* | 158 | 0.4085 | 0.741 | 0.418 | 45.5 | 27.6 | 18.2 | leaky |
