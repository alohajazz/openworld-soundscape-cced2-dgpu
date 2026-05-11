# Window-aware extraction analysis scripts (R2.1, Phase 2 v5)

Analysis scripts and output CSVs for the Sci Rep R2.1 revision (window-aware embedding
extraction; ~32× speed-up via `per_file_pre_cache + np.load slice`).

## Scripts

| Script | Purpose | Output dir |
|---|---|---|
| `build_16k_cache.py` | Pre-cache 16 kHz mono audio (Phase 2 v5 speed-up enabler) | (caches) |
| `build_part2_manifest.py` | Construct HICEAS 1706 Part2 species manifest | (manifest) |
| `frdr_label_efficiency.py` | FRDR Promoter label-efficiency curve (N=10/30/100/300/1000/all) | `frdr_label_efficiency_2026-05-09/` |
| `hiceas_label_efficiency_postpart2.py` | HICEAS Promoter sample-efficiency under Cross-day GroupKFold (post-Part2 expansion) | `hiceas_label_efficiency_postpart2_2026-05-09/` |
| `unified_promoter_audit.py` | 12-condition × 7-species Promoter comparison (LR / MLP / DANN × raw / z-norm / rank-norm × per-day weighting) | `unified_promoter_audit_2026-05-09/` |
| `perday_calibration_promoter.py` | Per-day rank-normalization Promoter (Pattern C principled fix) | `perday_calibration_2026-05-09/` |
| `day_adversarial_promoter.py` | DANN (Ganin 2016) Promoter for Cross-day generalization | `day_adversarial_promoter_2026-05-09/` |
| `frdr_pr_curve.py` | FRDR Promoter PR-curve sweep | `frdr_pr_curve_2026-05-09/` |
| `frdr_pr_curve_extreme.py` | FRDR Top-K precision analysis (Top-1/5/10) | `frdr_pr_curve_extreme_2026-05-09/` |
| `eval_table4_post_part2.py` | Table 4 / 5-fold stratified CV + GroupKFold side-by-side | `table4_post_part2_2026-05-09/` |
| `hiceas_label_efficiency.py` / `_strict.py` | HICEAS pre-Part2 + strict variants | corresponding dirs |
| `dump_winaware_cached.py` | Window-aware extraction from cached audio (Phase 2 v5 core) | (embeddings/) |

## Outputs

All output CSVs reflect Phase 2 v5 (window-aware) re-extraction values that
populate Tables 2/3/4 and Supplementary Table S3 + Figure S1 of the
R2.1 manuscript.

## Pre-Phase 2 v5 caveats

Earlier Phase 2 v4 output (before this re-extraction) suffered from a
`SegDataset` instance re-use bug that caused 19 evaluation directories
to receive file-level constant (not window-level) embeddings. The
scripts in this directory replace the in-DataLoader resampling with a
`per_file_pre_cache + np.load slice` approach, eliminating the bug
and yielding ~32× extraction speed-up.
