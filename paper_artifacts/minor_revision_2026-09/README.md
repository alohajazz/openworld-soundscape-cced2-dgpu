# Frozen source artifacts for the September 2026 minor revision

> **Historical frozen record.** This directory remains the `v3.0.4`
> historical artifact set. The frozen `v3.0.5` FRDR-only correction is at
> `scripts/frdr_correction_2026-09-13/` and
> `paper_artifacts/frdr_correction_2026-09-13/`. Do not mix its correction
> outputs with the hashes or CSVs in this historical directory.

These files are source outputs used to populate the revised tables and figures.
They are verification artifacts, not substitutes for the original evaluation
datasets.

For FRDR, reported values are discrete selector outputs: Table 3 chooses the
point nearest 10 FP/h on a 0.5-percentile grid; Table 2 Quiet and Promoter
choose the point nearest 10 FP/h, Union chooses maximum recall at FP/h ≤ 10.5,
and Fusion is alpha=1 per-file CCED2 on a finer 0.1 grid. No values are
interpolated, and Fusion is not complementary-signal fusion.

| File | SHA-256 |
|---|---|
| `table2_fixed_step127641.csv` | `7e4c9041b975b3e40ca0824456ab67207e94c454ab86c3706aa806f6e2b50609` |
| `table2_fusion_fixed_step127641.csv` | `2abd81431e6f09a4c393e5d85c43d6a61967caa14d746d36f773260dfe9772b6` |
| `table3_fixed_step127641.csv` | `8ccb32e9014be780353c416f182d09e0e13cc8c75010606b3b8912de40680ce0` |
| `supp_table_s3_fixed_step127641_unknown_high.csv` | `4483c7b0af2bb208bf6e217855ec08adc229e7b692c369f7e0a3ff03c1586139` |
| `supp_table_s3_perch_unknown_high.csv` | `727c22c42e5c6c2ca9e24c25904fb682f6dc4fba5c38e7b7f1e954925e48bfc0` |
| `fig4_label_efficiency_fixed.csv` | `57c4256272f2faf3e21f4fd819647a078aadc441e5f28a31a3e8027f1ab2a5c0` |
| `fig4_label_efficiency_fixed_agg.csv` | `585286b8285e83ba3c7403c058ab8258c8dc1378ddeb177f3db6588c670ec86d` |
| `supp_fig_s1_summary_step127641.json` | `ba45360039edab0facd80fafa988e67c554b05596440d229700d8b06d48468a6` |
| `supp_fig_s2_s3_summary_step127641.json` | `a42dbd3621858c6f737159c7c2f62b5d751ad067d6f396a871bbd7272bb0e142` |

`supp_table_s3_fixed_step127641_unknown_high.csv` uses positive distances as
higher-is-more-unknown. The reproduction-only output preserving the submitted
negated-score/upper-tail mismatch is intentionally not a reported artifact.

Both final S3 CSVs contain 12 rows (three scores × two tolerances × macro/micro
averaging) and seven species. They are diagnostic evaluations on
species-event-positive recording sets. Their FP/h value divides by the number
of scored recordings times nominal 60 s, including short recordings, so it is
not a continuous negative-recording false-alarm estimate.
Candidate matching uses clipped annotation start times (not event midpoints),
counts at most one TP per recording, and emits no candidate from a recording
with fewer than five windows.

The window-level score/membership NPZ is not included in this aggregate-result
release. Its original SHA-256 is
`2cb139ede7353531188c95aa4653251d64aa35a9e3326dcfa2194d2448c41f28`.
Only the frozen S2/S3 aggregate summary is public; recomputing that summary
or its score-space plot needs separately supplied window-level scores. Run
`python scripts/verify_minor_revision_artifacts.py` from the repository root
to validate public frozen hashes, S3 schemas, Table 1/Table 4 aggregates,
HICEAS provenance and CCED2 normalisation/threshold hashes. Optional
`--private-artifact-dir` and `--private-reference-dir` inputs enable the
additional NPZ-summary and fitted-model byte checks. Missing optional inputs
are explicitly reported as not checked. This does not claim a full
rerun because the required embeddings, event mappings, and reference inputs
are not all public.

`manifests/input_provenance.json` records the compressed and uncompressed
hashes, window counts, and unique-recording counts for the three archived
input manifests. `manifests/hiceas_recording_labels.zip` contains the eight
recording-level label CSVs used for the availability audit. These are
provenance records for the frozen results, not redistributions of audio or
embeddings.

The Figure 4 CSVs are frozen outputs of an exploratory partial-positive-label
masking analysis. At finite budgets, unselected positive records remain with
label 0; the procedure is not standard record-subsampling label efficiency.
The all-days result is unaffected. `scripts/plot_minor_revision_fig4.py`
reads `fig4_label_efficiency_fixed_agg.csv` only and produces
`fig4_partial_positive_labels.png`; it does not rerun a model or alter the
frozen CSV.
