# Manuscript–artifact map

Frozen release: `v3.0.4-sr-minor-2026-09-13` (model-availability update;
scientific code and numerical outputs unchanged from `v3.0.3-sr-minor-2026-09-13`)

This map distinguishes public-data reproduction, verification from frozen
source artifacts, analyses requiring the non-public 56-class corpus, and
conceptual figures.

| Manuscript item | Code or artifact | Required data | Reproducibility scope |
|---|---|---|---|
| Figure 1 | Conceptual schematic | None | Conceptual; not generated from numerical data |
| Figure 2 | Conceptual schematic | None | Conceptual; not generated from numerical data |
| Table 1 | `paper_artifacts/minor_revision_2026-09/table1_beats_as2m_s42_metrics.json`; `table1_dapt_fixed_s42_metrics.json`; exact model identities/hashes | Restricted 56-class audio, labels, splits | Aggregate values can be hash- and schema-verified. The corrected encoder/head are public on Hugging Face (see `MODEL_AVAILABILITY.json`), but exact retraining/evaluation is not independently reproducible without restricted data. |
| Table 2 | `paper_artifacts/minor_revision_2026-09/table2_fixed_step127641.csv`; `table2_fusion_fixed_step127641.csv`; archived executed sources `scripts/minor_revision_2026-09/frdr_supervised_promoter_COPY.py` and `run_fusion_winaware_COPY.py` | Public FRDR plus exact corrected embeddings, manifest, and fixed parameters | Source outputs are frozen; archived environment-specific sources document execution but are not a generic top-level rerun route. Quiet/Promoter select the discrete sweep point nearest 10 FP/h; Union selects maximum recall at FP/h ≤ 10.5; Fusion is alpha=1 per-file CCED2 on a finer 0.1 grid, not complementary-signal fusion. |
| Table 3 | `paper_artifacts/minor_revision_2026-09/table3_fixed_step127641.csv`; archived executed source `scripts/minor_revision_2026-09/run_frdr_table3_winaware_COPY.py` | Public FRDR plus exact corrected embeddings/manifests and fitted reference inputs | Frozen source output can be hash-verified. Final values select the discrete point nearest 10 FP/h from a 0.5-percentile grid; no interpolation is applied. The archived executed generator preserves its original environment and is not a generic public-input-only regeneration route. |
| Table 4 | `paper_artifacts/minor_revision_2026-09/table4_fixed_and_submitted_audit.json` (the `FIXED` arm), `table4_fold_auc_audit.csv`, and `scripts/run_table4_fixed.py` | Public HICEAS plus the exact corrected embeddings and canon manifests | Frozen audit data can verify the reported stratified-fold means, population SDs, and record counts. The wrapper validates embedding/index alignment before calling the archived audited evaluator. Each species uses all 6,135 common available negative records, not a twice-positive negative sample. The older `paper_artifacts/table4_hiceas_canon_promoter/` CSV is not the final minor-revision table. |
| Figure 3 | Final frozen Table 2 and Fusion source artifacts | Same as Table 2 | Plot is reproducible from the frozen source outputs; the historical May directories are not the final source route. |
| Figure 4 | `scripts/plot_minor_revision_fig4.py`; frozen final CSV and aggregate CSV | Frozen aggregate CSV (for plotting); a full re-analysis needs HICEAS plus the exact corrected embeddings/manifests | Plotting reads the frozen aggregate CSV only and does not rerun a model. It is exploratory partial-positive-label masking, not standard record-subsampling efficiency: finite budgets retain unselected positives with label 0. The all-days result is unaffected. |
| Supplementary Table S1 | `dapt_train.py`; checkpoint hashes | World-DAPT source recordings and manifest | Configuration/provenance supplied; exact retraining requires obtaining source recordings |
| Supplementary Table S2 | Supplementary Information | Restricted 56-class corpus | Complete taxonomy and aggregate counts/durations supplied; clip-level data remain non-public |
| Supplementary Table S3 | Two corrected positive-direction CSVs in `paper_artifacts/minor_revision_2026-09/`; current guarded entrypoints `scripts/winaware_2026-05-09/supp_s3_7sp_full.py` and `supp_s3_perch.py` | Corrected cruise-1705 OP and cruise-1706 species BEATs embeddings; cruise-1705 Perch OP embeddings; encoder-specific reference fits; event mappings | Final CSVs can be hash/schema-verified. Current entrypoints default to final input locations, write outside frozen artifacts, and fail on embedding/index/manifest row or order mismatch. Required embeddings, reference pools, and mappings are not released, so no complete public rerun is claimed. The S3 FP/h denominator is a nominal 60-s-per-positive-recording convention, including short files; it is a diagnostic, not a continuous negative-recording false-alarm estimate. |
| Supplementary Table S4 | Frozen S2/S3 aggregate summary in `paper_artifacts/minor_revision_2026-09/`; `scripts/verify_minor_revision_artifacts.py` | Recalculation requires the separately supplied window-level NPZ; full rerun also requires encoder-specific reference pools and HICEAS embeddings | The public summary can be hash-verified, not independently recomputed without the withheld NPZ. It uses 54,419 shared OP windows, but BEATs reference fitting used 1,623 validation embeddings and Perch used 6,781 train/reference embeddings (6,489 train + 292 dev); this is not an encoder-only comparison. |
| Supplementary Table S5 | Supplementary Information | None beyond Methods | Descriptive protocol table |
| Supplementary Figure S1 | `supp_fig_s1_summary_step127641.json` | Restricted 56-class reference plus public HICEAS | Frozen derived summary supplied; independent reference refit is impossible without the 56-class corpus |
| Supplementary Figures S2–S3 | `analysis/cced2_component_overlap/plot_fixed_step127641.py`; frozen S2/S3 aggregate summary | Plotting requires the separately supplied window-level NPZ; full re-analysis needs restricted BEATs reference plus HICEAS embeddings | Only aggregate outputs are public; memberships and score-space inputs are not included. Final aggregate counts are 501 shared, 44 kNN-only, and 44 CCED2-only. |
| Supplementary Figure S4 | Historical Perch-overlap scripts and frozen reporting record | Perch OP embeddings and its encoder-specific reference fit | Perch-only labels retain positive higher-is-more-unknown direction. The shared OP window universe does not make its reference fit interchangeable with BEATs+DAPT. |

## Frozen model identities

- Encoder: `BEATs_DAPT_MAM_fixed_step127641.pt`
- Encoder SHA-256: `2a2d1d93f53ec29227bdd52da087fd0abcf0ce797c3c4a8629cd1435a314a6f9`
- Seed-42 SED head: `sed_head_fixed_s42_ep7.pt`
- SED-head SHA-256: `9b2b202ab3e52b0d1efe4cd3479ee479db7646b0f42ab5b0e32f1e3ca551f119`
- Corrected CCED2 aggregate normalisation/threshold parameters: `weights/cced2_step127641/`; fitted kNN/Mahalanobis pickle files are not included.
- Corrected public model location: `https://huggingface.co/BiologgingSolutions/OceanBEATs/tree/v3.0.3-sr-minor-2026-09-13`; immutable revision and file hashes are recorded in `MODEL_AVAILABILITY.json`.
- Legacy revision `dbb29a3dfc4fe1605c9fdd87079723db12903849` contains the old step-120,000 encoder and epoch-8 head. Those files remain unchanged for provenance and must not be substituted for the corrected pair.
