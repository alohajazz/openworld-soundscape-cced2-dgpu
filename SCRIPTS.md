# Script index for manuscript revision 2.2

This file documents the role of each script in this repository relative to
the published Noda et al. (Sci. Rep., revision 2.2) results. Use this as a
quick map from manuscript Tables/Figures to the source code that generated
them. For paths and runtime context see the docstrings in each file.

## Top-level entry points (canonical for revision 2.2)

| Script | Manuscript artifact | Purpose |
|---|---|---|
| `dapt_train.py` | Table 1 (DAPT) — encoder | Domain-Adaptive PreTraining (Masked Audio Modeling with k-means k=1024 tokenisation) on the World-DAPT 5,673-h corpus. Produces `BEATs_DAPT_MAM_step120000.pt` (= published `weights/beats_dapt_mam_step120000.pt`). Replaces the earlier SimCLR/InfoNCE training in `legacy/dapt_train_simclr_buggy.py`. |
| `dump_known56_features.py` | Table 1 / 56-class SED head | Extracts BEATs+DAPT embeddings + 56-class SED logits for any labelled manifest; feeds into both SED head training and CCED2 fitting. |
| `run_frdr_cced2_op_eval.py` | Table 2 (FRDR Quiet/Union/Fusion) | Original FRDR continuous-recording evaluation (Quiet detector vs CCED2-only thresholding vs OR-style Union). |
| `run_frdr_fp_recall.py` | Table 3 (FRDR FP/h–Recall, binary search variant) | Per-model recall at a target FP/h (binary search on threshold). |
| `run_hiceas_canon_promoter.py` | **Table 4 (HICEAS canon-level Promoter, 7 species)** | Canon-level positive/negative labelling + StratifiedKFold logistic-regression Promoter on canon-mean BEATs+DAPT embeddings. The canonical reproduction script for the new manuscript Table 4. |

## Helper / support scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `dapt_make_manifest_all.py`, `dapt_make_shards.py`, `dapt_qc_manifest.py` | World-DAPT manifest construction & quality control |
| `dapt_extract_kmeans_labels.py` | Compute k-means k=1024 cluster labels on PRETRAIN BEATs patch features (used as the MAM tokeniser in `dapt_train.py`) |
| `dapt_train_beats_mam_545h_legacy.py` | Earlier 545-h variant of MAM DAPT (kept as reference; superseded by fulldata in `dapt_train.py`) |
| `train_sed_beats_weak_plus.py` | 56-class SED head training (produces `sed_head_56_fulldata_ep8.pt`) |
| `eval_b2_canon_promoter_compare.py` | Earlier paired-encoder comparison script (PRETRAIN vs DAPT) used during analysis; the published Table 4 uses `run_hiceas_canon_promoter.py` |
| `run_frdr_fp_recall_interp.py` | Interpolation variant of `run_frdr_fp_recall.py` (alternative interpolation rule near the FP/h target) |
| `export_hiceas_table6.py` | Legacy: exports the original event-level Table 6 (replaced by Table 4 canon-level results in revision 2.2) |

## Legacy (`legacy/`)

See `legacy/README.md` for the per-file rationale. In short: scripts and
weights from the original December 2026 submission that have been superseded
by the revision 2.2 changes (DCLDE 2013 removal, MAM DAPT replacing buggy
SimCLR DAPT, canon-level Promoter replacing OP event-level grid search).

## Path conventions

Scripts that originated from the GPU-server analysis snapshot retain
hardcoded `/workspace/...` paths that were used in the actual run. To
reproduce on a different host, edit the constants at the top of each
script (or — for `run_hiceas_canon_promoter.py` — pass paths via the CLI
arguments documented in its `--help`). Future updates will progressively
parameterise the remaining scripts.

## End-to-end reproduction sketch

For the manuscript Table 4 reproduction starting from raw HICEAS audio
and the published encoder + SED head:

1. Resample HICEAS FLACs to 16 kHz mono (Methods §4.2.1).
2. Build per-species canon manifests (`pos_<Species>.csv`, `neg_all.csv`)
   using the `canon = first_three_underscore_tokens(basename)` rule.
3. Extract BEATs+DAPT embeddings with `dump_known56_features.py` using
   `weights/beats_dapt_mam_step120000.pt`.
4. Run `run_hiceas_canon_promoter.py --emb_dirs ... --canon_dir ...
   --out_json results/hiceas_canon_promoter.json`.
5. Compare reported per-species AUC (mean ± std across 5 folds) against
   `paper_artifacts/table4_hiceas_canon_promoter/`.
