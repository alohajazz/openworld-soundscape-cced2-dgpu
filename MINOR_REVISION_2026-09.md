# Scientific Reports minor revision freeze (September 2026)

This record distinguishes invalid or superseded analyses from the corrected
analyses used in the revised manuscript.

> **FRDR correction frozen (2026-09-13).** Release
> `v3.0.5-sr-minor-2026-09-13` package at
> `scripts/frdr_correction_2026-09-13/` completes the annotation dictionary
> across all 50 FRDR manifest files, assigning empty lists to files without
> annotations and failing closed for annotations outside the manifest. It
> retains the 25.055555555555554-h nominal-duration convention and all event
> processing, references, encoders, folds, labels, grids, and selectors.
> The corrected numerical outputs have passed independent and parent review
> and are frozen in `paper_artifacts/frdr_correction_2026-09-13/` as
> `v3.0.5-sr-minor-2026-09-13`. The old `v3.0.4` FRDR outputs remain historical
> and are superseded for the corrected manuscript. This has no effect on HICEAS
> Figure 4 or Table 1's descriptive 6,489-train/1,623-validation head
> selection (not an independent test set).

## DAPT checkpoint history

| Manuscript stage | Status | Audit result |
|---|---|---|
| Original SimCLR-style DAPT, step 20,000 | Invalid | Exported encoder and all 250 floating-point tensors are identical to public BEATs AS-2M; the raw optimizer state is empty and GradScaler scale is zero. |
| May 2026 MAM revision, step 120,000 | Superseded | Encoder weights changed, but masking was applied only after unmasked audio had passed through the encoder; this is not the stated masked-input objective. |
| September 2026 input-mask MAM, step 127,641 | Final | Frozen teacher; 75% patch masking before the student transformer; fixed 1,024-cluster targets; cross-entropy only at masked positions. |

The final run used 2,042,268 non-overlapping 10-s windows, batch size 16,
`drop_last=True`, and one complete pass through the shuffled manifest. This
gives `floor(2,042,268 / 16) = 127,641` optimizer steps; the final 12 rows were
the incomplete batch. No DAPT validation split, diel-balanced runtime sampler,
or downstream checkpoint selection was used.

Final encoder:

- filename: `BEATs_DAPT_MAM_fixed_step127641.pt`
- SHA-256: `2a2d1d93f53ec29227bdd52da087fd0abcf0ce797c3c4a8629cd1435a314a6f9`
- matching seed-42 SED head: `sed_head_fixed_s42_ep7.pt`
- SED-head SHA-256: `9b2b202ab3e52b0d1efe4cd3479ee479db7646b0f42ab5b0e32f1e3ca551f119`

## Extraction correction

The superseded extractor could return the first 10 s of a source file even
when the manifest requested a different centre. The corrected extractor uses
`start_sec` when present and otherwise begins at
`max(0, center_sec - 5)`, reading an exact 10-s window and padding only at a
file boundary. All retained FRDR and HICEAS embeddings were regenerated with
this rule and the final encoder.

## Supplementary Table S3 correction

> **Stage 1 values (release `v3.0.5-sr-minor-2026-09-13`).** The numbers and statements in this section describe the
> Stage 1 field analyses. For the final revision they are superseded by the section "Final revision" below: FRDR and
> HICEAS use the Stage 2 encoder, Table 3 is obtained by linear interpolation at 10 FP/h, and the best Fusion weight is
> alpha = 0.


The submitted S3 generator negated distance scores and then selected their
upper 1% tail. That combination selects the low-unknownness direction. The
final analysis treats `kNN_z`, `Mahalanobis_z`, and `CCED2` as
higher-is-more-unknown and selects the upper tail for both BEATs+DAPT and
Perch 2.0. The corrected source CSVs are
`paper_artifacts/minor_revision_2026-09/supp_table_s3_fixed_step127641_unknown_high.csv`
and `paper_artifacts/minor_revision_2026-09/supp_table_s3_perch_unknown_high.csv`.

At macro averaging and ±20-s tolerance, BEATs+DAPT F1 is 0.290 for kNN,
0.305 for Mahalanobis, and 0.309 for CCED2. Perch F1 is 0.139, 0.128 and
0.127, respectively. The ordering is therefore configuration-dependent. These are
standalone score diagnostics; they are not evidence that CCED2 surfacing or
Group caused the supervised Promoter gain.

All retained BEATs+DAPT embedding analyses use the final step-127,641 encoder
and corrected window-aware extraction. This does not make BEATs-versus-Perch
diagnostics encoder-only comparisons: although the final S4 scores use the
same 54,419 OP windows, BEATs uses a 1,623-embedding validation reference
pool whereas Perch uses 6,781 reference embeddings (6,489 train + 292 dev).
For S3, BEATs evaluates cruise-1705 OP plus cruise-1706 species embeddings,
while Perch uses only its cruise-1705 OP baseline. S3 FP/h is a nominal 60-s-per-positive-recording
denominator (including short recordings), not a continuous negative-recording
false-alarm estimate.

## Interpretation boundaries

> **Stage 1 values (release `v3.0.5-sr-minor-2026-09-13`).** The numbers and statements in this section describe the
> Stage 1 field analyses. For the final revision they are superseded by the section "Final revision" below: FRDR and
> HICEAS use the Stage 2 encoder, Table 3 is obtained by linear interpolation at 10 FP/h, and the best Fusion weight is
> alpha = 0.


- FRDR operating points are discrete selector outputs. Table 3 uses the point
  nearest 10 FP/h on a 0.5-percentile grid; Table 2 Quiet and Promoter use the
  point nearest 10 FP/h, Union uses maximum recall at FP/h ≤ 10.5, and Fusion
  is alpha=1 per-file CCED2 on a finer 0.1 grid. No interpolation is reported,
  and the Fusion row is not complementary-signal fusion.
- The post-hoc 3-classifier × 3-normalisation cross-day matrix is exploratory.
  The manuscript reports fixed raw-embedding logistic regression as the
  descriptive primary analysis and does not report per-species best-of-nine
  maxima as generalisation estimates. Deployment-day GroupKFold requests five
  folds, but a single-class train or test partition is omitted; four or five
  usable folds remain by species.
- Figure 4 is an exploratory partial-positive-label masking analysis. At each
  finite budget it retains unselected positive records with label 0, rather
  than removing records as in standard label-budget subsampling. The all-days
  result is unaffected.
- AUROC 1.000 is dataset-membership discrimination between fixed,
  bandwidth-matched 56-class-reference and HICEAS samples. It is not semantic
  unknown-source discovery.
- Group assignments and CCED2-surfaced candidates were not used to train the
  reported Promoter. The FRDR gain is attributed to the supervised
  proxy-label protocol, not to the full DGPU loop.

## Data boundary

The current 56-class corpus expands and reorganises the taxonomy associated
with the earlier 52-class study: 46 semantic classes are shared, six earlier
classes are absent, and ten classes are new. The clip units and split
memberships also differ. The earlier Dryad dataset is not the present
56-class training corpus.

The 56-class raw clips and row-level metadata remain non-public because of
sensitive location and operational information and permissions held by the
original collaborating organisations. See `MANUSCRIPT_ARTIFACT_MAP.md` for
the exact reproducibility boundary of every reported item.

## Final revision (19 September 2026, release `v3.1.0-sr-minor-2026-09-19`)

The manuscript keeps the two-stage DAPT design of the previous revision. Stage 2 was rerun with the corrected
input-mask trainer from the Stage 1 endpoint (2021 PALAOA subset; 102,168 training windows, 1,032 held out by
`random_split` with seed 42; 6,385 steps; learning rate 1e-5; final checkpoint SHA-256
`4f7869751d7f15e3a806fb062902654597ca5566be610fedc1762c440d5c2a89`).

- The corrected Stage 2 changes the embeddings relative to Stage 1: for the same 1,623 validation clips the mean
  cosine similarity between Stage 1 and Stage 2 embeddings is 0.913 (minimum 0.793; maximum absolute weight change
  0.0108), whereas the superseded encoders had given 0.999995, that is, practically identical embeddings. The CCED2
  reference of the final revision was therefore refitted on Stage 2 embeddings. The earlier statements that Stage 2
  preserves Stage 1, and the ten-seed Stage 1 versus Stage 2 comparison, belonged to the superseded encoders and
  were removed.
- Table 1 uses Stage 1 and reports the mean of eight SED-head seeds (42, 123, 777, 101, 202, 303, 404, 505):
  Event/Clip/2-s-segment F1 = 0.478/0.746/0.499 for BEATs AS-2M and 0.493/0.739/0.523 for BEATs+DAPT. Paired
  differences: +0.015 (event) and +0.023 (segment), positive for all eight seeds; -0.007 (clip), positive for three.
  The seed-42 values given earlier in this file (0.483/0.784/0.506 and 0.493/0.749/0.518) are one of the eight seeds.
  FRDR, HICEAS and the CCED2 reference use Stage 2.
- FRDR near 10 FP/h: Quiet 0.027, Promoter 0.390, Union 0.276, Fusion 0.044. Union is below the Promoter alone
  because both thresholds are raised to keep the combined detections near 10 FP/h.
- Table 3 reports recall at FP/h = 10 by linear interpolation of the threshold sweep, as its note states.
- Cross-day: the fixed logistic regression on raw embeddings is the primary estimate; the 3 x 3
  classifier/normalisation grid is exploratory and not nested-validated.

Files: `paper_artifacts/final_revision_2026-09-19/` and `scripts/final_revision_2026-09-19/`.
