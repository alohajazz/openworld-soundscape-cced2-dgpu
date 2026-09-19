# Manuscript–artifact map

Frozen release for the final revision: `v3.1.0-sr-minor-2026-09-19`.
Preceding public correction release: `v3.0.5-sr-minor-2026-09-13`.

> **Final revision.** Table 1 uses the Stage 1 encoder and reports the mean of eight SED-head seeds
> (`paper_artifacts/final_revision_2026-09-19/table1/`). Tables 2-4, Figures 3-4, Supplementary
> Tables S3-S4 and Supplementary Figures S1-S3 of the final revision use the Stage 2 encoder
> (`BEATs_DAPT_MAM_fixed_palaoa_step6385.pt`, SHA-256 `4f7869751d7f15e3a806fb062902654597ca5566be610fedc1762c440d5c2a89`);
> their files are listed item by item in `paper_artifacts/final_revision_2026-09-19/README.md`. The executed Table 1 launcher, trainer and
> evaluator and the executed CCED2-reference sources are in `scripts/final_revision_2026-09-19/executed/`. The rows below keep the
> code paths and the reproducibility limits, which are unchanged; the Stage 1 field values they point to are
> superseded for those items.
The preceding `v3.0.4-sr-minor-2026-09-13` FRDR outputs remain preserved as
historical artifacts.

This map distinguishes public-data reproduction, verification from frozen
source artifacts, analyses requiring the non-public 56-class corpus, and
conceptual figures.

> **Frozen FRDR correction release.**
> `scripts/frdr_correction_2026-09-13/` and
> `paper_artifacts/frdr_correction_2026-09-13/` contain the reviewed correction
> over all 50 manifest files, including six without annotations. These are
> the sources/results used in the corrected manuscript and frozen public
> release `v3.0.5-sr-minor-2026-09-13`.
> The old `minor_revision_2026-09` FRDR outputs are retained as historical,
> superseded results and must not be used for the corrected Tables 2/3 or Figure 3.

| Manuscript item | Code or artifact | Required data | Reproducibility scope |
|---|---|---|---|
| Figure 1 | Conceptual schematic | None | Conceptual; not generated from numerical data |
| Figure 2 | Conceptual schematic | None | Conceptual; not generated from numerical data |
| Table 1 | `paper_artifacts/final_revision_2026-09-19/table1/summary.csv` (per-seed values) and `table1_seed_summary.json` (eight-seed means and paired differences); seed-42 detail in `paper_artifacts/minor_revision_2026-09/table1_beats_as2m_s42_metrics.json` and `table1_dapt_fixed_s42_metrics.json`; exact model identities/hashes | Restricted 56-class audio, labels, splits | Aggregate values can be hash- and schema-verified. The corrected encoder/head are public on Hugging Face (see `MODEL_AVAILABILITY.json`), but exact retraining/evaluation is not independently reproducible without restricted data. |
| Table 2 | `paper_artifacts/frdr_correction_2026-09-13/results/table2/frdr_quiet_promoter_union_supervised.csv`; `results/fusion/fusion.csv` under the same correction root; executed sources in `scripts/frdr_correction_2026-09-13/source/` | Public FRDR plus exact corrected embeddings, manifest, and fixed parameters | Accepted local all-50-file correction. Quiet/Promoter select nearest 10 FP/h; Union maximum recall at FP/h ≤ 10.5; Fusion highest recall across its alpha grid after nearest-10-FP/h selection at each alpha on the 0.1-percentile grid (selected alpha=1.0). Exact embeddings/reference fits remain unavailable, so no public-input-only rerun is claimed. |
| Table 3 | `paper_artifacts/frdr_correction_2026-09-13/results/table3/frdr_table3.csv`; `scripts/frdr_correction_2026-09-13/source/run_frdr_table3_winaware_COPY.py` | Public FRDR plus exact corrected embeddings/manifests and fitted reference inputs | Accepted local all-50-file correction. Discrete nearest-10-FP/h selector on the 0.5-percentile grid, without interpolation. It is not a generic public-input-only regeneration route. |
| Table 4 | `paper_artifacts/minor_revision_2026-09/table4_fixed_and_submitted_audit.json` (the `FIXED` arm), `table4_fold_auc_audit.csv`, and `scripts/run_table4_fixed.py` | Public HICEAS plus the exact corrected embeddings and canon manifests | Frozen audit data can verify the reported stratified-fold means, population SDs, and record counts. The wrapper validates embedding/index alignment before calling the archived audited evaluator. Each species uses all 6,135 common available negative records, not a twice-positive negative sample. The older `paper_artifacts/table4_hiceas_canon_promoter/` CSV is not the final minor-revision table. |
| Figure 3 | `scripts/frdr_correction_2026-09-13/plot_frdr_correction.py`; `paper_artifacts/frdr_correction_2026-09-13/manuscript/Fig3_FRDR_all50.png` | Frozen Table 2 and Fusion aggregate CSVs only | The corrected FRDR-only plot is regenerable from hash-verified aggregates without model fitting. This does not reproduce the underlying scores from audio. |
| FRDR 300-label claim | `paper_artifacts/frdr_correction_2026-09-13/results/labels/label_efficiency.csv` and `label_efficiency_agg.csv`; executed `frdr_label_efficiency_param.py` | Exact corrected FRDR embeddings, manifests and annotations | Five selection seeds 0–4; mean recall 0.3960242 and sample SD 0.0229521. All label budgets were rerun with unchanged folds, labels and selectors. Raw inputs are not supplied by this correction bundle. |
| Figure 4 | `scripts/plot_minor_revision_fig4.py`; frozen final CSV and aggregate CSV | Frozen aggregate CSV (for plotting); a full re-analysis needs HICEAS plus the exact corrected embeddings/manifests | Unchanged by the FRDR correction. Plotting reads the frozen aggregate CSV only and does not rerun a model. It is exploratory partial-positive-label masking, not standard record-subsampling efficiency: finite budgets retain unselected positives with label 0. The all-days result is unaffected. |
| Supplementary Table S1 | `dapt_train.py`; checkpoint hashes | World-DAPT source recordings and manifest | Configuration/provenance supplied; exact retraining requires obtaining source recordings |
| Supplementary Table S2 | Supplementary Information | Restricted 56-class corpus | Complete taxonomy and aggregate counts/durations supplied; clip-level data remain non-public |
| Supplementary Table S3 | Two corrected positive-direction CSVs in `paper_artifacts/minor_revision_2026-09/`; current guarded entrypoints `scripts/winaware_2026-05-09/supp_s3_7sp_full.py` and `supp_s3_perch.py` | Corrected cruise-1705 OP and cruise-1706 species BEATs embeddings; cruise-1705 Perch OP embeddings; encoder-specific reference fits; event mappings | Final CSVs can be hash/schema-verified. Current entrypoints default to final input locations, write outside frozen artifacts, and fail on embedding/index/manifest row or order mismatch. Required embeddings, reference pools, and mappings are not released, so no complete public rerun is claimed. The S3 FP/h denominator is a nominal 60-s-per-positive-recording convention, including short files; it is a diagnostic, not a continuous negative-recording false-alarm estimate. |
| Supplementary Table S4 | Frozen S2/S3 aggregate summary in `paper_artifacts/minor_revision_2026-09/`; `scripts/verify_minor_revision_artifacts.py` | Recalculation requires the separately supplied window-level NPZ; full rerun also requires encoder-specific reference pools and HICEAS embeddings | The public summary can be hash-verified, not independently recomputed without the withheld NPZ. It uses 54,419 shared OP windows, but BEATs reference fitting used 1,623 validation embeddings and Perch used 6,781 train/reference embeddings (6,489 train + 292 dev); this is not an encoder-only comparison. |
| Supplementary Table S5 | Supplementary Information | None beyond Methods | Descriptive protocol table |
| Supplementary Figure S1 | `supp_fig_s1_summary_step127641.json` | Restricted 56-class reference plus public HICEAS | Frozen derived summary supplied; independent reference refit is impossible without the 56-class corpus |
| Supplementary Figures S2–S3 | `analysis/cced2_component_overlap/plot_fixed_step127641.py`; frozen S2/S3 aggregate summary | Plotting requires the separately supplied window-level NPZ; full re-analysis needs restricted BEATs reference plus HICEAS embeddings | Only aggregate outputs are public; memberships and score-space inputs are not included. Final aggregate counts are 501 shared, 44 kNN-only, and 44 CCED2-only. |
| Supplementary Figure S4 | Historical Perch-overlap scripts and frozen reporting record | Perch OP embeddings and its encoder-specific reference fit | Perch-only labels retain positive higher-is-more-unknown direction. The shared OP window universe does not make its reference fit interchangeable with BEATs+DAPT. |

## Frozen model identities

- Stage 1 encoder (Table 1): `BEATs_DAPT_MAM_fixed_step127641.pt`
- Encoder SHA-256: `2a2d1d93f53ec29227bdd52da087fd0abcf0ce797c3c4a8629cd1435a314a6f9`
- Stage 2 encoder (FRDR, HICEAS, CCED2 reference): `BEATs_DAPT_MAM_fixed_palaoa_step6385.pt`
- Stage 2 encoder SHA-256: `4f7869751d7f15e3a806fb062902654597ca5566be610fedc1762c440d5c2a89`
- Seed-42 SED head: `sed_head_fixed_s42_ep7.pt`
- SED-head SHA-256: `9b2b202ab3e52b0d1efe4cd3479ee479db7646b0f42ab5b0e32f1e3ca551f119`
- Corrected CCED2 aggregate normalisation/threshold parameters: `weights/cced2_step127641/`; fitted kNN/Mahalanobis pickle files are not included.
- Corrected public model location: `https://huggingface.co/BiologgingSolutions/OceanBEATs/tree/v3.0.3-sr-minor-2026-09-13`; immutable revision and file hashes are recorded in `MODEL_AVAILABILITY.json`.
- Legacy revision `dbb29a3dfc4fe1605c9fdd87079723db12903849` contains the old step-120,000 encoder and epoch-8 head. Those files remain unchanged for provenance and must not be substituted for the corrected pair.
