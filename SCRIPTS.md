# Script index for the September 2026 minor revision

This file documents the role of each script in this repository relative to
the revised Noda et al. (*Scientific Reports*) results. Use this as a
quick map from manuscript Tables/Figures to the source code that generated
them. For paths and runtime context see the docstrings in each file.

## Top-level entry points

| Script | Manuscript artifact | Purpose |
|---|---|---|
| `dapt_train.py` | Table 1 (DAPT) — encoder | Compatibility entry point for the audited corrected script `scripts/dapt_train_beats_mam_fixed.py`. The final one-pass endpoint is `BEATs_DAPT_MAM_fixed_step127641.pt`. |
| `dump_known56_features.py` | Table 1 / 56-class SED head | Extracts BEATs+DAPT embeddings + 56-class SED logits for any labelled manifest; feeds into both SED head training and CCED2 fitting. |
| `run_frdr_cced2_op_eval.py` | Table 2 (FRDR Quiet/Union/Fusion) | Original FRDR continuous-recording evaluation (Quiet detector vs CCED2-only thresholding vs OR-style Union). |
| `run_frdr_fp_recall.py` | Table 3 (FRDR FP/h–Recall, binary search variant) | Per-model recall at a target FP/h (binary search on threshold). |
| `run_hiceas_canon_promoter.py` | Historical Table 4 | Earlier canon-level Promoter CLI; not the final minor-revision protocol. Use `scripts/run_table4_fixed.py` with the exact corrected inputs. |

## Helper / support scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `dapt_make_manifest_all.py`, `dapt_make_shards.py`, `dapt_qc_manifest.py` | World-DAPT manifest construction & quality control |
| `dapt_extract_kmeans_labels.py` | Compute k-means k=1024 cluster labels on PRETRAIN BEATs patch features (used as the MAM tokeniser in `dapt_train.py`) |
| `dapt_train_beats_mam_fixed.py` | Exact executed frozen-teacher/input-mask DAPT implementation for the final step-127,641 encoder |
| `dapt_train_beats_mam_545h_legacy.py` | Earlier 545-h variant of MAM DAPT (kept as reference; superseded by fulldata in `dapt_train.py`) |
| `train_sed_beats_weak_plus.py` | Compatible 56-class SED training workflow; the manuscript head identity is `sed_head_fixed_s42_ep7.pt`, not the legacy epoch-8 file. Exact reproduction requires restricted data and splits. |
| `eval_b2_canon_promoter_compare.py` | Historical paired-encoder comparison; not the final Table 4 route. |
| `run_table4_fixed.py` | Final Table 4 wrapper; validates corrected embedding/index alignment and calls archived `minor_revision_2026-09/groupkfold_table4_eval.py`. |
| `verify_minor_revision_artifacts.py` | Public aggregate/provenance hash and schema checks; optional authorised NPZ and reference-file checks are reported separately. No full model rerun is implied. |
| `plot_minor_revision_fig4.py` | Replots the exploratory partial-positive-label masking results from the frozen aggregate CSV. |
| `run_frdr_fp_recall_interp.py` | Interpolation variant of `run_frdr_fp_recall.py` (alternative interpolation rule near the FP/h target) |
| `export_hiceas_table6.py` | Legacy: exports the original event-level Table 6 (replaced by Table 4 canon-level results in revision 2.2) |

## Legacy (`legacy/`)

See `legacy/README.md` for the per-file rationale. In short: scripts and
weights from the original December 2025 submission that have been superseded.
The May 2026 post-encoder-mask DAPT implementation is also superseded; see
`MINOR_REVISION_2026-09.md` for the distinction.

## Path conventions

Scripts that originated from the GPU-server analysis snapshot retain
hardcoded `/workspace/...` paths that were used in the actual run. To
reproduce on a different host, edit the constants at the top of each
script (or — for `run_hiceas_canon_promoter.py` — pass paths via the CLI
arguments documented in its `--help`). Future updates will progressively
parameterise the remaining scripts.

## Final Table 4 verification and rerun boundary

The public frozen `FIXED` audit and fold CSV in
`paper_artifacts/minor_revision_2026-09/` support checking the final Table 4
means, population SDs and record counts. Run:

```bash
python scripts/verify_minor_revision_artifacts.py
```

A full re-analysis additionally requires exact step-127,641 corrected
embeddings aligned to the three archived HICEAS manifests and the recording
labels. The corrected encoder/head hashes are recorded, but their public
distribution is not yet claimed. The legacy OceanBEATs files are not substitutes.
Once the required inputs are available, configure `scripts/run_table4_fixed.py`
with `--op-dir`, `--species-dir`, `--part2-dir`, `--canon-dir`, and `--out-json`.
Compare its FIXED stratified results against the minor-revision audit, not the
older `paper_artifacts/table4_hiceas_canon_promoter/` CSV.
