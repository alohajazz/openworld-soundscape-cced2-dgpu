# CCED2 Component Overlap Analysis (2026-05-15)

Analysis script archive supporting Discussion §3.2 + §3.4 Limitation #5 +
Supplementary Table S4 + Supplementary Figure S4 added in revision 2 of
Noda et al. (Scientific Reports).

## Question addressed

How much does the CCED2 score (= z(d_knn) + z(d_maha)) overlap with each
single-distance score in selecting the top-1% most-extreme windows? Is the
relative balance between the kNN and Mahalanobis components encoder-dependent?

## Symbol notation

- `d_knn` : per-window kNN distance (mean Euclidean distance to k=50 nearest
            in-distribution neighbours; Methods §4.3).
- `d_maha`: per-window Mahalanobis distance (with Ledoit–Wolf-shrunk
            covariance over in-distribution embeddings; Methods §4.3).
- `z(x)`  : z-score standardisation of `x` using the in-distribution
            background set's mean and standard deviation (Methods §4.3).
- `cced2` : per-window CCED2 score, defined as `z(d_knn) + z(d_maha)`.

## Files

- `analyze_168.py`        : First overlap analysis (BEATs+DAPT, kNN vs CCED2
                            only). Generated Supplementary Figure S2 Venn data
                            (461 shared / 84 each unique out of 545 per score).
- `perch_overlap.py`      : Pairwise overlap stats (kNN vs Maha vs CCED2) for
                            both BEATs+DAPT and Perch 2.0 on HICEAS OP
                            (n = 54,419 windows). Source data for
                            Supplementary Table S4.
- `make_perch_venn3.py`   : Generated Supplementary Figure S4
                            (Perch 2.0 3-way Venn).

## Key numerical findings

Top-1% selection overlap (n_select = 545 per score):

| Encoder    | kNN vs Maha | kNN vs CCED2 | Maha vs CCED2 |
|------------|-------------|--------------|----------------|
| BEATs+DAPT | 81.8%       | 84.6%        | 97.1%          |
| Perch 2.0  | 60.0%       | 74.5%        | 84.4%          |

Pearson r over all 54,419 windows (raw distances, not the top 1% only):

| Encoder    | r(d_knn, d_maha) | r(d_maha, cced2) | r(d_knn, cced2) |
|------------|------------------|------------------|-----------------|
| BEATs+DAPT | 0.698            | 0.995            | 0.767           |
| Perch 2.0  | 0.942            | 0.991            | 0.979           |

The Mahalanobis vs CCED2 overlap dropping from 97.1% (BEATs+DAPT) to 84.4%
(Perch) on the same dataset, and the kNN vs Mahalanobis overlap dropping from
81.8% to 60.0%, demonstrate that the relative balance between the two CCED2
components is encoder-dependent.

## How to re-run

```bash
# Inside the docker environment with embeddings + ind models prepared:
python3 perch_overlap.py     # numerical comparison (BEATs+DAPT and Perch)
python3 make_perch_venn3.py  # regenerate Supplementary Fig S4 PNG
```

Required inputs (paths used in the scripts):
- `/workspace/embeddings/hiceas_op_fulldata_winaware/` — BEATs+DAPT HICEAS OP embeddings
- `/workspace/embeddings/perch_hiceas_op2s_10s_win10/` — Perch 2.0 HICEAS OP embeddings
- `/workspace/embeddings/cced2_fulldata/{knn_cced2.pkl,maha_cced2.pkl,cced2_norm.json}` — BEATs+DAPT in-distribution kNN + Mahalanobis models + CCED2 normalisation constants
- `/workspace/embeddings/perch_ind_models/{knn_perch.pkl,maha_perch.pkl,cced2_norm_perch.json}` — Perch 2.0 counterparts
