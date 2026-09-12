# CCED2 component-overlap archive

The scripts in this directory include historical May 2026 analyses. Their
previous BEATs+DAPT numerical findings are superseded and must not be used as
retained manuscript results. In particular, older values such as 81.8%, 84.6%,
97.1% and correlations 0.698/0.995/0.767 do not describe the final
step-127,641 analysis.

## Historical question addressed

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

## Historical files

- `analyze_168.py`        : First overlap analysis (BEATs+DAPT, kNN vs CCED2
                            only). Generated Supplementary Figure S2 Venn data
                            (461 shared / 84 each unique out of 545 per score).
- `perch_overlap.py`      : Pairwise overlap stats (kNN vs Maha vs CCED2) for
                            both BEATs+DAPT and Perch 2.0 on HICEAS OP
                            (n = 54,419 windows). Source data for
                            Supplementary Table S4.
- `make_perch_venn3.py`   : Generated Supplementary Figure S4
                            (Perch 2.0 3-way Venn).

## Final frozen results

The authoritative frozen BEATs+DAPT aggregate outputs are in
`paper_artifacts/minor_revision_2026-09/`:

- The original `supp_fig_s2_s3_data_step127641.npz` contains 54,419 OP-window
  score rows but is not included in this aggregate-result release.
- `supp_fig_s2_s3_summary_step127641.json` records 501 shared, 44 kNN-only,
  and 44 CCED2-only top-1% windows (545 per score; Jaccard 0.8505942275).
- `plot_fixed_step127641.py` is a plotting-only consumer requiring that
  separately supplied NPZ; load trusted numeric arrays with `allow_pickle=False`.

For final Supplementary Table S4, the BEATs+DAPT top-1% overlap counts are
kNN/Mahalanobis 495/545 (90.8%), kNN/CCED2 501/545 (91.9%), and
Mahalanobis/CCED2 539/545 (98.9%); the three-way intersection is 495. The
corresponding correlations `r(kNN, Mahalanobis)`, `r(kNN, CCED2)`, and
`r(Mahalanobis, CCED2)` are 0.805600559486, 0.857098132341, and
0.995685899853.

The BEATs and Perch S4 scores share the same 54,419 OP windows, but they do
not share fitted reference pools: BEATs uses 1,623 validation-reference
embeddings, whereas Perch uses 6,781 reference embeddings (6,489 train + 292
dev). Therefore these results are not an encoder-only comparison. Perch S4
remains a separate baseline; all final score labels use positive
higher-is-more-unknown direction.

## Verification and rerun scope

Run `python scripts/verify_minor_revision_artifacts.py` from the repository
root to hash-check public frozen artifacts. Recomputing the S2/S3 summary
requires the original NPZ via `--private-artifact-dir`; this check is explicitly
not performed without that optional input. Full reruns require corrected embeddings and
encoder-specific reference inputs. The historical scripts have hard-coded
execution paths and are retained for provenance, not as a claimed complete
public reproduction route.

The legacy scripts refer to these original execution paths:
- `/workspace/embeddings/hiceas_op_fulldata_winaware/` — BEATs+DAPT HICEAS OP embeddings
- `/workspace/embeddings/perch_hiceas_op2s_10s_win10/` — Perch 2.0 HICEAS OP embeddings
- `/workspace/embeddings/cced2_fulldata/{knn_cced2.pkl,maha_cced2.pkl,cced2_norm.json}` — BEATs+DAPT in-distribution kNN + Mahalanobis models + CCED2 normalisation constants
- `/workspace/embeddings/perch_ind_models/{knn_perch.pkl,maha_perch.pkl,cced2_norm_perch.json}` — Perch 2.0 counterparts
