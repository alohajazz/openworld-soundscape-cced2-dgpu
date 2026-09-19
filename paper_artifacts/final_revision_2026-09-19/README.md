# Final revision artifacts (19 September 2026)

Results reported in the final revision of the manuscript. The two-stage DAPT design of the manuscript is kept:

| Stage | Checkpoint | Steps | SHA-256 | Used for |
|---|---|---:|---|---|
| 1 | `BEATs_DAPT_MAM_fixed_step127641.pt` | 127,641 | `2a2d1d93f53ec29227bdd52da087fd0abcf0ce797c3c4a8629cd1435a314a6f9` | Table 1; initialisation of Stage 2 |
| 2 | `BEATs_DAPT_MAM_fixed_palaoa_step6385.pt` | 6,385 | `4f7869751d7f15e3a806fb062902654597ca5566be610fedc1762c440d5c2a89` | FRDR and HICEAS analyses and their CCED2 reference |

Both stages mask 75% of the input patch embeddings before the transformer encoder and use a frozen BEATs AS-2M
teacher. Stage 2 continues from the Stage 1 checkpoint on the 2021 PALAOA subset (102,168 training windows, 1,032 held
out by `random_split` with seed 42, learning rate 1e-5). The executed trainer and its log are
`scripts/final_revision_2026-09-19/dapt_train_beats_mam_fixed_STAGE2_COPY.py` and `encoder/stage2_training.log`.

Stage 2 does not preserve the Stage 1 embedding space (mean cosine similarity 0.913, minimum 0.793 on the 1,623
validation clips; maximum absolute weight change 0.0108). The CCED2 reference was therefore refitted on Stage 2
embeddings. Values are in `encoder/encoder_and_embedding_diagnostics.json`.

## Contents

| Manuscript item | File |
|---|---|
| Table 2, Fig. 3 | `frdr/table2/frdr_quiet_promoter_union_supervised.csv`, `frdr/fusion/fusion.csv`, `figures/Fig3.png` |
| Table 3 | `frdr/table3/table3_interp_at_fph10.json` (recall at FP/h = 10 by linear interpolation of the sweep in `frdr/table3/all_evaluations.csv`; script `scripts/final_revision_2026-09-19/table3_interpolate_fph10.py`). `frdr/table3/frdr_table3.csv` holds the nearest evaluated points. |
| 300-label result | `frdr/labels/label_efficiency_agg.csv` |
| Share of labels in the 300-label result (~32%) | `frdr/labels/training_fold_event_counts.json` (script `scripts/final_revision_2026-09-19/frdr_fold_event_counts.py`) |
| Table 1, Methods 4.6 (eight-seed means and paired differences) | `table1/summary.csv` (per-seed values written by the executed launchers), `table1/table1_seed_summary.json` (script `scripts/final_revision_2026-09-19/table1_seed_summary.py`); seed-42 detail in `paper_artifacts/minor_revision_2026-09/table1_*_s42_metrics.json`; executed launchers, trainer and evaluator in `scripts/final_revision_2026-09-19/executed/` |
| CCED2 reference (Stage 2) | executed `dump_known56_features.py`, `cced2_utils.py` and `run_stage2_downstream_commands.sh` in `scripts/final_revision_2026-09-19/executed/` |
| Table 4 (within-day) | `hiceas/table4_stage2.json` (`stratified`) |
| Cross-day results, Fig. 4 | `hiceas/table4_stage2.json` (`group`), `hiceas/crossday_classifier_normalisation_grid.csv` (exploratory 3 x 3 grid), `hiceas/fig4_label_efficiency_agg.csv`, `figures/Fig4.png` |
| Supplementary Table S3 | `hiceas/supp_table_s3_stage2_unknown_high.csv`; Perch block unchanged from `paper_artifacts/minor_revision_2026-09/supp_table_s3_perch_unknown_high.csv` |
| Supplementary Table S4, Figs S2-S3 | `hiceas/supp_table_s4_overlap_stage2.json`, `hiceas/supp_fig_s2_s3_summary_stage2.json`, `figures/SuppFigS2.png`, `figures/SuppFigS3.png` |
| Supplementary Fig. S1 | `ood/supp_fig_s1_summary_stage2.json`, `figures/SuppFigS1.png` |
| Supplementary Fig. S4 | `figures/SuppFigS4.png` (Perch only; counts unchanged; set labels no longer carry a minus sign) |
| In-text InD/OOD diagnostics and weight changes | `encoder/encoder_and_embedding_diagnostics.json` |

FRDR false positives are counted over all 50 recordings, as in release `v3.0.5-sr-minor-2026-09-13`; the executed
FRDR sources are the files in `scripts/frdr_correction_2026-09-13/source/` (hashes in `frdr/execution_receipt.json`).

## Reproducibility limits

The limits stated in `MANUSCRIPT_ARTIFACT_MAP.md` apply unchanged. The Stage 2 embeddings, the fitted CCED2 reference
models and the window-level score arrays are not distributed, so `figures/SuppFigS1.png` and `figures/SuppFigS3.png`
cannot be redrawn from this directory alone; Fig. 3, Fig. 4 and Supplementary Fig. S2 can
(`scripts/final_revision_2026-09-19/make_figures_main.py`, `scripts/final_revision_2026-09-19/make_figures_supp.py`). File hashes are in `release_manifest.json`.
