# Revision 2.2 change log

This document describes the substantive changes between the original
December 2026 manuscript submission (paired with `README.md.v1_dec2026_backup`)
and the May 2026 revision 2.2.

## Summary

Revision 2.2 corrects two material problems in the original submission and
restructures the species-wise HICEAS evaluation accordingly:

1. **DAPT training bug**: original SimCLR/InfoNCE training under AMP fp16
   experienced loss = NaN at every step → `GradScaler` skipped every
   update → BEATs encoder weights were never modified. The "BEATs+DAPT"
   encoder shipped with v1 was therefore equivalent to PRETRAIN.
2. **DCLDE 2013 bandwidth mismatch**: the DCLDE 2013 OOD benchmark used
   in v1 contains essentially no signal above 1 kHz, while the InD
   reference (56-class) spans the full 0–8 kHz analysis band. The
   resulting "OOD detection" largely reflected silence-vs-content rather
   than meaningful out-of-distribution discrimination.

Revision 2.2 replaces both with a corrected pipeline:

1. **Masked Audio Modeling (MAM) DAPT** under bfloat16 precision, trained
   on the full 5,673-h World-DAPT corpus for 126,365 steps. The MAM
   target uses k-means k=1024 cluster labels computed on PRETRAIN BEATs
   patch features. Stage 2 continual-DAPT on a 287-h PALAOA polar subset
   was added to integrate the polar acoustic environment described in
   the original preprint while preserving Stage 1 representations
   (Stage 2 vs Stage 1: cosine similarity 0.999995, no significant SED
   metric change at p > 0.67 across n=10 random seeds; statistically
   indistinguishable).
2. **HICEAS as the OOD benchmark**, using a held-out 56-class validation
   InD reference (n = 1,623) and the HICEAS multi-species archive as OOD
   (n = 2,973). All bandwidth-matched at the 0–8 kHz analysis band.

## Per-table impact (manuscript)

| Manuscript table | v1 | v2.2 |
|---|---|---|
| Table 1 (SED) | "DAPT (Stage-1)" Event F1 = 0.485, "DAPT (Top-up)" = 0.495 — both implicitly = PRETRAIN due to bug | Single DAPT row: Event F1 = 0.487 (best epoch). Single-seed headline 0.483; n=10 mean ± std 0.475 ± 0.017 |
| Table 2 | DCLDE 2013 InD/OOD detection benchmark | **Removed** (renumbered; old Table 4 FRDR Quiet/Union/Fusion is now Table 2) |
| Table 3 | DCLDE 2013 ablation (kNN_z, Maha_z, CCED2) | **Removed** (renumbered; old Table 5 FRDR FP/h-recall is now Table 3) |
| Table 4 | FRDR Quiet/Union/Fusion (was) | Now: HICEAS canon-level Promoter discrimination (7 species, AUC 0.89–1.00, 5-fold CV). Replaces the old event-level OP grid-search Table 6 (10 species, including beaked whales whose echolocation clicks lie above the 0–8 kHz analysis band). |
| Table 5 | FRDR FP/h-recall at FP/h=10 (Perch comparison) | Renumbered: now Table 3 |
| Table 6 | OP event-level Quiet/Quiet_perclass_best/Promoter/Union grid search per species (10 species) | **Removed** (replaced by canon-level Table 4) |

## Per-file impact (this repository)

| Path (v1) | Path (v2.2) | Notes |
|---|---|---|
| `dapt_train.py` (SimCLR/AMP fp16, buggy) | `legacy/dapt_train_simclr_buggy.py` | Replaced by canonical MAM training |
| (new) | `dapt_train.py` (MAM, bfloat16, k=1024 tokeniser) | Canonical |
| `run_dclde2013_cced2_eval.py` | `legacy/run_dclde2013_cced2_eval.py` | DCLDE 2013 removed |
| `scripts/make_dclde2013_manifests.py` | `legacy/scripts/...` | Same |
| `scripts/dclde_table3_ablation.py` | `legacy/scripts/...` | Same |
| `run_hiceas_multi_species_eval.py` (OP grid search) | `legacy/run_hiceas_multi_species_eval_op_event_level.py` | Replaced by canon-level Promoter |
| (new) | `run_hiceas_canon_promoter.py` | Canonical Table 4 entry point, fully parameterised CLI |
| (new) | `run_frdr_fp_recall.py` | FP/h-recall sweep (Table 3) |
| (new) | `scripts/run_frdr_fp_recall_interp.py` | Interpolation variant |
| (new) | `scripts/dapt_extract_kmeans_labels.py` | k-means k=1024 tokeniser |
| (new) | `scripts/dapt_train_beats_mam_545h_legacy.py` | Earlier 545-h variant, kept for reference |
| (new) | `scripts/eval_b2_canon_promoter_compare.py` | Original analysis snapshot for Table 4 (PRETRAIN vs DAPT comparison) |
| `weights/beats_dapt_topup_encoder.pt` (Dec 2026, buggy = PRETRAIN) | `legacy/weights/beats_dapt_topup_encoder_buggy.pt` (gitignored) | Replaced by `weights/beats_dapt_mam_step120000.pt` |
| `weights/sed_head_56_topup_ep8.pt` (Dec 2026) | `legacy/weights/sed_head_56_topup_ep8_buggy.pt` (gitignored) | Replaced by `weights/sed_head_56_fulldata_ep8.pt` |
| `weights/cced2/{cced2_norm.json,knn_dapt.pkl,maha_dapt.pkl,theta_cced.json,theta_cced2.json}` | `legacy/weights/cced2/...` | Replaced by new fit against fulldata DAPT encoder; new `theta_cced2.json` = 3.287 (was 3.4576) |
| `paper_artifacts/dclde_table3.csv` | `legacy/paper_artifacts/dclde_table3.csv` | DCLDE 2013 removed |
| `paper_artifacts/frdr_table4/` | `paper_artifacts/table2_frdr_quiet_union_fusion/` | Renumbered |
| `paper_artifacts/frdr_table5/` | `paper_artifacts/table3_frdr_fp_recall/` | Renumbered |
| `paper_artifacts/hiceas_table6/` | `legacy/paper_artifacts/hiceas_table6_old_event_level/` | Replaced |
| (new) | `paper_artifacts/table4_hiceas_canon_promoter/` | Canon-level Promoter results, with primary-literature anchor README |

## CCED2 normalisation values (encoder change impact)

The new fulldata DAPT encoder produces an embedding distribution slightly
different from the old (buggy/PRETRAIN-equivalent) encoder. The CCED2
normalisation factors had to be refitted:

| Factor | v1 (buggy) | v2.2 (fulldata) |
|---|---|---|
| `mk` (kNN mean on InD)        | 2.1384 | 2.5438 |
| `sk` (kNN std on InD)         | 0.8336 | 1.0810 |
| `mm` (Maha mean on InD)       | 22.7901 | 21.0938 |
| `sm` (Maha std on InD)        | 7.8321 | 5.7382 |
| `theta_cced2` (q95 of InD CCED2) | 3.4576 | **3.2874** |

The new InD CCED2 distribution: mean 4.82 × 10⁻⁷ ± std 1.947, q5 = -2.97,
q95 = 3.287 (n = 1,623 56-class held-out validation segments).

## Reproducing the v1 numbers (legacy)

If you specifically need to reproduce the original December 2026 numbers
(e.g. for a delta comparison), the legacy weights and scripts are
preserved under `legacy/`:

- `legacy/weights/{beats_dapt_topup_encoder_buggy.pt,sed_head_56_topup_ep8_buggy.pt}`
  *(local only — gitignored due to size; available on request)*
- `legacy/weights/cced2/{cced2_norm.json,knn_dapt.pkl,maha_dapt.pkl,theta_cced*.json}`
- `legacy/paper_artifacts/dclde_table3.csv`
- `legacy/paper_artifacts/hiceas_table6_old_event_level/`

The legacy scripts (`legacy/dapt_train_simclr_buggy.py`,
`legacy/run_dclde2013_cced2_eval.py`,
`legacy/run_hiceas_multi_species_eval_op_event_level.py`) are kept as
historical record. **They should not be used for any new analysis.**

## What did NOT change

- The DGPU framework (Detect–Group–Promote–Union) and its motivation.
- The mathematical definition of CCED2.
- The 56-class internal SED dataset and the SED head architecture.
- The FRDR North Atlantic right whale upcall continuous-detection setup.
- The general philosophy of FP/h-constrained operating-point design.

## 2026-05-09 follow-up: window-aware extraction + bandwidth-consistent 7-species HICEAS Supplementary Table S3

While preparing revision 2.1 of the manuscript, two further refinements were applied to ensure that the published numbers are bandwidth-consistent with the 0–8 kHz analysis band and that per-window scores are computed from the manifest-specified window boundaries:

1. **Window-aware extraction**. A sliding-window evaluation manifest (e.g., FRDR continuous detection at 2 s hop) requires each row's `center_sec` to be wired through to the audio slicer. The previous extractor (`dump_known56_features.py`, designed for the known56 1-clip-per-row setting) loaded the first `target_seconds` of the file regardless of `center_sec`, which produced per-file constant per-window embeddings when reused on sliding-window manifests. The new `scripts/winaware_2026-05-09/dump_winaware_features.py` reads `center_sec` and slices `[start_sec, start_sec + target_seconds]` of each row. An optional 16 kHz mono pre-cache (`scripts/winaware_2026-05-09/build_16k_cache.py` + `dump_winaware_cached.py`) eliminates redundant decode/resample for high-sample-rate sources (e.g., 500 kHz × 6 ch HICEAS hydrophone FLAC).

2. **Bandwidth-consistent seven-species HICEAS Supplementary Table S3**. Supplementary Table S3 was previously evaluated on a ten-species OP subset that included beaked whales (Cuvier's, Longman's, unidentified beaked). The echolocation clicks of all classified Hawaiian odontocete types peak at ≥ 12.5 kHz (Ziegenhorn et al. 2024) and therefore lie above the 8 kHz Nyquist frequency of the 0–8 kHz analysis band. To match Table 4 (per-recording Promoter discrimination on the seven-species PR set, all of which have lower-frequency in-band signal classes within 0–8 kHz), Supplementary Table S3 was re-evaluated on the same seven-species PR set with a per-recording top-1 cap. New paper_artifacts under `paper_artifacts/supp_table_s3_winaware_2026-05-09/` contain the re-evaluated values for BEATs+DAPT and Perch 2.0.

### Per-table impact (2026-05-09)

| Manuscript | v2.2 (Apr 2026 reorganisation) | v2.1 / 2026-05-09 (window-aware re-extraction) |
|---|---|---|
| Table 2 (FRDR Quiet/Union/Fusion) | Quiet 0.069 / Union 0.126 / Fusion 0.084 (FP/h ≈ 10) | Quiet 0.074 / Union 0.427 / Fusion 0.070 (FP/h ≈ 10) |
| Table 3 (FRDR ablation kNN_z / Maha_z / CCED2_z) | BEATs 0.077 / 0.049 / 0.072 ; Perch 0.109 / 0.095 / 0.104 | BEATs 0.052 / 0.078 / 0.074 ; Perch 0.090 / 0.081 / 0.084 |
| Table 4 (HICEAS canon Promoter, 7 species AUC range) | 0.893–0.997 | 0.919–0.996 |
| Supplementary Table S3 (HICEAS unknownness-only) | 10-species OP subset; per-file constant CCED2 ⇒ degenerate single-FP/h cluster | 7-species PR set; per-recording top-1; tolerance-sensitive |
| Fig. 3 caption | Quiet 0.069 / Union 0.126 / Fusion 0.084 | Quiet 0.074 / Union 0.427 / Fusion 0.070 |

Updated CSVs are in `paper_artifacts/table2_frdr_quiet_union_fusion/`, `paper_artifacts/table3_frdr_fp_recall/`, `paper_artifacts/table4_hiceas_canon_promoter/` (in-place updates), with full sweep curves and per-encoder breakdowns under `paper_artifacts/winaware_2026-05-09/` and `paper_artifacts/supp_table_s3_winaware_2026-05-09/`.

### Reproducing the 2026-05-09 numbers

```
# 1. Pre-cache 16 kHz mono npy files for high-sample-rate sources (one-time)
python scripts/winaware_2026-05-09/build_16k_cache.py

# 2. Extract window-aware embeddings using cached audio
python scripts/winaware_2026-05-09/dump_winaware_cached.py \
    --csv <manifest_winsafe.csv> \
    --ckpt_beats <beats_dapt.pt> \
    --outdir <emb_dir> \
    --batch_size 32 --num_workers 4 --shard_size 10000

# 3. Run per-task evaluations
python scripts/winaware_2026-05-09/frdr_supervised_promoter.py        # Table 2 + Fig 3
python scripts/winaware_2026-05-09/run_frdr_table3_winaware.py        # Table 3
python scripts/winaware_2026-05-09/groupkfold_table4_eval_winaware.py # Table 4
python scripts/winaware_2026-05-09/supp_s3_7sp_full.py                # Supp Table S3 (BEATs+DAPT)
python scripts/winaware_2026-05-09/supp_s3_perch.py                   # Supp Table S3 (Perch)
```

## Acknowledgement

The bug discovery and the choice of DCLDE 2013 replacement were prompted
by reviewer comments on the December 2026 submission. We thank the
reviewers for the rigorous read.
