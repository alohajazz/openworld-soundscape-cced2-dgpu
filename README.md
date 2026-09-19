# CCED2 and DGPU for Unknownness-aware Underwater Acoustic Monitoring

This repository accompanies *"Discovery and promotion of unknown sounds into operational detection
targets for underwater passive acoustic monitoring under false alarm
constraints"* (Noda et al., *Scientific Reports*, revision in
review).
It provides a minimal implementation of:

- **BEATs+DAPT** — a self-supervised audio encoder adapted to underwater
  soundscapes via Masked Audio Modeling (MAM) on a 5,673-h multi-site corpus.
- **CCED2** — an embedding-space unknownness score combining z-normalised
  k-nearest-neighbour and Mahalanobis distances on InD reference statistics.
- **DGPU** — a proposed Detect–Group–Promote–Union architecture. The manuscript
  evaluates candidate scoring, clustering diagnostics, and supervised
  Promoter/Union performance as separate components; Group and CCED2-based
  candidate selection were not inputs to the reported Promoter gain.

The repository provides code, recorded model identities, and source artifacts for the
public-data analyses in the manuscript. Its reproducibility scope is stated
explicitly below; in particular, the exact numerical results in Table 1 cannot
be reproduced without the non-public 56-class dataset. The repository does not
include product-deployment workflows.

The framework addresses practical PAM operational constraints — limited
annotation budgets and multi-day deployments under variable recording
conditions — through an exploratory partial-positive-label masking analysis of
the Promoter stage (paper §2.2) and cross-day generalisation analysis under
5-fold GroupKFold by deployment day (paper §2.3; four or five usable folds
after omissions for single-class train/test folds).

> **Correction history.** The original December 2025 SimCLR-style DAPT run is
> invalid because AMP fp16 skipped encoder updates. The model used in the May
> 2026 revision did update its encoder, but masking was applied only after the
> unmasked waveform had passed through the encoder; it therefore did not
> implement the stated masked-input objective. Both analyses are superseded.
> The September 2026 revision uses a frozen teacher and replaces 75% of student
> patch embeddings with a trainable mask token before transformer encoding.
> Table 1 uses the one-pass Stage 1 endpoint at step 127,641. The FRDR and
> HICEAS results of the final revision use the Stage 2 endpoint (step 6,385,
> continued on the 2021 PALAOA subset) and the corrected window-aware
> extractor. See `MINOR_REVISION_2026-09.md`.

> **Final revision (release `v3.1.0-sr-minor-2026-09-19`).** The manuscript keeps its two-stage
> DAPT design. The field results reported in the final revision are in
> `paper_artifacts/final_revision_2026-09-19/`, with the executed sources in `scripts/final_revision_2026-09-19/`.
> Where the sections below describe the Stage 1 field results of release
> `v3.0.5-sr-minor-2026-09-13`, those values are superseded for Tables 2-4,
> Figures 3-4, Supplementary Tables S3-S4 and Supplementary Figures S1-S3.
> Table 3 of the final revision reports recall at FP/h = 10 by linear
> interpolation of the threshold sweep.

> **FRDR correction status.** The frozen correction package at
> `scripts/frdr_correction_2026-09-13/` completes annotations over all 50
> FRDR manifest files (including empty-annotation files). It is release
> `v3.0.5-sr-minor-2026-09-13`; corrected frozen results are in
> `paper_artifacts/frdr_correction_2026-09-13/`.
> The `v3.0.4` FRDR outputs are historical and superseded for the corrected
> manuscript; Figure 4/HICEAS is unchanged.

## Companion artifacts

- **Corrected BEATs+DAPT encoder**
  (`BEATs_DAPT_MAM_fixed_step127641.pt`; 361 MB) — SHA-256
  `2a2d1d93f53ec29227bdd52da087fd0abcf0ce797c3c4a8629cd1435a314a6f9`
- **Stage 2 BEATs+DAPT encoder**
  (`BEATs_DAPT_MAM_fixed_palaoa_step6385.pt`; 361 MB) — SHA-256
  `4f7869751d7f15e3a806fb062902654597ca5566be610fedc1762c440d5c2a89`
- **Matching seed-42 56-class SED head**
  (`sed_head_fixed_s42_ep7.pt`; 18 MB) —
  SHA-256 `9b2b202ab3e52b0d1efe4cd3479ee479db7646b0f42ab5b0e32f1e3ca551f119`
- **Aggregate CCED2 parameters** (`weights/cced2_step127641/`) — normalisation
  factors and thresholds from the fixed n=1,623 held-out 56-class reference.
  The fitted pickle files are not included: the kNN fit retains individual
  restricted-reference embeddings, and model-file distribution is separate.
- **Bundled BEATs source code** (`beats_core/`)

The source code is released under the MIT License. CCED2 parameters ship
inside this repository under CC BY 4.0. The corrected encoder and matching
SED head are publicly distributed on
[Hugging Face](https://huggingface.co/BiologgingSolutions/OceanBEATs/tree/v3.0.3-sr-minor-2026-09-13),
not as GitHub release attachments. Their exact model revision, download links
and verification evidence are recorded in `MODEL_AVAILABILITY.json`.
The legacy step-120,000 and epoch-8-head files remain unchanged for provenance;
do not use them for corrected results. The Detect-Group-Promote-Union (DGPU) framework
and the CCED2 unknownness score are subject to patent applications filed by
Biologging Solutions Inc.; these copyright licences do not grant patent
rights.

> **⚠️ Data Availability.** The internally curated 56-class underwater SED
> dataset is **not publicly available** because its raw clips and row-level
> metadata contain sensitive location and operational information and are
> governed by permissions held by the original collaborating organisations.
> Consequently, the exact Table 1 training and evaluation results, and a refit
> of the 56-class in-distribution reference statistics, cannot be reproduced
> from public materials alone. Supplementary Table S2 provides aggregate
> per-class statistics and the complete label taxonomy. The repository records
> the corrected model identities and includes frozen aggregate outputs and
> CCED2 normalisation/threshold parameters, not the fitted reference models;
> the corrected encoder/head are available separately on Hugging Face and
> must be hash-matched before use. Requests specifically
> to verify reported results and proposals for new academic collaborations may
> be considered individually by the data manager, subject to approval by the
> original collaborating organisations and an appropriate Data Use Agreement.
> Approval is not guaranteed; any approved access would prohibit redistribution
> of raw audio, commercial use, and attempts to infer sensitive information.

> **Perch baseline (optional).** Perch 2.0 embeddings are not redistributed.
> The frozen positive-direction Supplementary Table S3 output is included at
> `paper_artifacts/minor_revision_2026-09/supp_table_s3_perch_unknown_high.csv`;
> a full rerun requires users to supply Perch embeddings and matching fitted
> Perch reference models in the documented "win10" manifest format.

## Evaluation conventions

### FRDR continuous evaluation (manuscript Methods §4.5.1)

- Predictions and references are sorted and greedily matched one-to-one within
  ±tol seconds. Each matched pair is one TP; unmatched predictions are FPs and
  unmatched references are FNs.

### HICEAS canon-level Promoter (manuscript Methods §4.1.4 + §4.4.4 + Table 4)

- **Canon = 60-s FLAC recording**. Canon ID is `first_three_underscore_tokens(basename)`
  (e.g. `1705_20171008_191500` from `1705_20171008_191500_5400.flac`).
- **Positives**: any canon overlapping an annotated DetectionTimeStart-End
  interval for the species.
- **Negatives**: all 6,135 common available canons without an annotation for
  the species; they are not sampled at twice the positive count.
- **Promoter**: scikit-learn `LogisticRegression(C=1.0, max_iter=1000)`
  on canon-mean BEATs+DAPT embeddings (768-dim).
- **Cross-validation**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
  The separate deployment-day GroupKFold cross-day analysis requests five
  folds; a fold with a single-class train or test partition is omitted, leaving
  four or five usable folds by species.

## Repository structure

```text
.
├── README.md                              # This file
├── REVISION2.md                           # Historical change log through May 2026
├── MINOR_REVISION_2026-09.md              # Final correction/provenance record
├── MANUSCRIPT_ARTIFACT_MAP.md             # Item-by-item reproducibility scope
├── SCRIPTS.md                             # Per-script index with manuscript role
├── requirements.txt
├── LICENSE_CODE.txt
├── LICENSE_WEIGHTS.txt
│
├── beats_core/                            # Bundled BEATs source
│
├── cced2_utils.py                         # CCED2 score computation
├── dapt_dataset.py                        # DAPT dataloader
├── fp_recall_helpers.py                   # FP/h-recall sweep utilities
│
├── dapt_train.py                          # Corrected input-mask MAM DAPT training
├── dump_known56_features.py               # Embedding + 56-class SED logits dump
├── run_frdr_fp_recall.py                  # Historical FRDR FP/h-recall sweep
├── run_hiceas_canon_promoter.py           # Historical HICEAS canon-level Promoter CLI
│
├── scripts/                               # Helper / variant scripts
│   ├── dapt_make_manifest_all.py
│   ├── dapt_make_shards.py
│   ├── dapt_qc_manifest.py
│   ├── dapt_extract_kmeans_labels.py      # k=1024 k-means tokeniser
│   ├── dapt_train_beats_mam_545h_legacy.py  # Earlier 545-h MAM variant
│   ├── eval_b2_canon_promoter_compare.py  # Original analysis snapshot for Table 4
│   ├── run_frdr_fp_recall_interp.py       # FRDR FP/h-recall, interpolation variant
│   ├── run_table4_fixed.py                 # Final Table 4 wrapper around archived audited evaluator
│   ├── verify_minor_revision_artifacts.py # Public frozen-artifact verifier; optional private-input audit
│   ├── train_sed_beats_weak_plus.py       # 56-class SED head training
│   └── export_hiceas_table6.py            # (legacy) original event-level Table 6 export
│
├── weights/
│   ├── BEATs_DAPT_MAM_fixed_step127641.pt # Corrected encoder when obtained from a hash-verified release
│   ├── sed_head_fixed_s42_ep7.pt          # Matching head when obtained from a hash-verified release
│   └── cced2_step127641/
│       ├── cced2_norm.json                # kNN/Maha mean+std for z-normalisation
│       ├── README.md                     # Withheld fit identities and distribution limits
│       └── theta_cced2.json               # Decision threshold (InD q95)
│
├── paper_artifacts/                       # Reproducible Table CSVs
│   ├── table2_frdr_quiet_union_fusion/
│   ├── table3_frdr_fp_recall/
│   └── table4_hiceas_canon_promoter/
│       ├── README.md                      # Method + per-species primary-literature anchors
│       └── table4_canon_promoter_7species.csv
│
└── legacy/                                # Pre-revision-2.2 scripts and artifacts
    ├── README.md                          # What is here and why
    ├── dapt_train_simclr_buggy.py
    ├── run_dclde2013_cced2_eval.py
    ├── run_hiceas_multi_species_eval_op_event_level.py
    ├── scripts/
    │   ├── dclde_table3_ablation.py
    │   └── make_dclde2013_manifests.py
    ├── paper_artifacts/
    │   ├── dclde_table3.csv
    │   └── hiceas_table6_old_event_level/
    └── weights/
        └── cced2/                         # Old buggy CCED2 fits (PRETRAIN-equivalent)
```

## Setup

### Python environment

```bash
pip install -r requirements.txt
```

Tested with PyTorch 2.7.1 + CUDA 12.8 (GPU recommended for DAPT training
and embedding extraction; CCED2 / Promoter eval runs on CPU).

### Obtaining corrected weights

Obtain the corrected encoder and head from the immutable Hugging Face revision
in `MODEL_AVAILABILITY.json` (tag `v3.0.3-sr-minor-2026-09-13`). Use the exact
case-sensitive filenames in **Companion artifacts**; the retained legacy
filenames are not substitutes. Place verified files in `weights/` if running
code that requires them, then check them with `shasum -a 256` against the
recorded values.

Only CCED2 normalisation and threshold JSON files are tracked in
`weights/cced2_step127641/`. The fitted kNN/Mahalanobis models are not supplied;
exact CCED2 inference requires separately authorised reference inputs.

## Reproducibility scope

This repository does **not** reproduce every table and figure in the
manuscript from public materials alone.

- **Table 1 (56-class SED):** the training code, configuration, recorded
  corrected-model identities, and aggregate results are provided, but exact
  numerical reproduction requires the non-public 56-class dataset despite
  the public availability of the corrected encoder/head.
- **Tables 2 and 3 (FRDR):** the `v3.0.4` outputs are historical frozen
  artifacts. The all-50-file `v3.0.5` source package is a frozen correction,
  rerun route; exact corrected embeddings, manifests, annotations, and
  reference inputs remain unavailable.
- **Table 4:** frozen source artifacts are provided. Full re-analysis
  requires the public datasets plus the exact corrected embeddings, manifests,
  and other inputs described in the artifact map.
- **Supplementary Tables S3–S4:** the final corrected outputs are
  frozen verification artifacts. The retained historical scripts document prior
  environment-specific generation, but the complete corrected input manifests,
  embeddings, and reference pools are not all public; these results are not
  claimed to be fully reproducible from this repository alone.
- **Figures 1–2:** conceptual schematics rather than computational outputs.
- **Figure 3:** the generating script and source artifacts are provided under
  `scripts/winaware_2026-05-09/` and `paper_artifacts/winaware_2026-05-09/`.
- **Figure 4 and supplementary computational outputs:** the frozen release
  identifies the exact scripts and source artifacts in
  `MANUSCRIPT_ARTIFACT_MAP.md`. Figure 4 is exploratory partial-positive-label
  masking: at finite budgets, unselected positive records remain in the data
  with label 0 rather than being removed. It is not standard record-subsampling
  label efficiency; the all-days result is unaffected.

## Reproducing manuscript analyses

### Table 1 — SED Performance (56-class)

`scripts/train_sed_beats_weak_plus.py` documents the training and evaluation
workflow for a compatible 56-class SED head. The manuscript reports the mean
of eight SED-head seeds: Event/Clip/2-s-segment F1 = 0.478/0.746/0.499 for
BEATs AS-2M and 0.493/0.739/0.523 for the corrected BEATs+DAPT encoder
(seed 42 alone: 0.483/0.784/0.506 and 0.493/0.749/0.518; per-seed values in
`paper_artifacts/final_revision_2026-09-19/table1/`). These exact values
cannot be independently regenerated from the public repository because the
underlying clips, labels, and splits are unavailable.

### Table 2 — FRDR: Quiet / Union / Fusion

The paths and values below are the historical `v3.0.4` frozen record.
The frozen all-50-file correction is documented separately in
`scripts/frdr_correction_2026-09-13/`; the accepted local replacement outputs
are in `paper_artifacts/frdr_correction_2026-09-13/results/`.

The reported operating points are discrete selector outputs, not interpolated
values: Quiet and Promoter use the sweep point nearest 10 FP/h, Union uses the
maximum-recall point with FP/h ≤ 10.5, and Fusion uses the alpha=1 per-file
CCED2 selector on its finer 0.1 grid. Fusion is therefore not a combination of
complementary signals.

The final source outputs are
`paper_artifacts/minor_revision_2026-09/table2_fixed_step127641.csv` and
`table2_fusion_fixed_step127641.csv`. The retained executed provenance is
`scripts/minor_revision_2026-09/frdr_supervised_promoter_COPY.py` and
`run_fusion_winaware_COPY.py`; these archived, environment-specific scripts
are not claimed as a generic top-level regeneration recipe.

### Table 3 — FRDR: FP/h–Recall

The paths and values below are the historical `v3.0.4` frozen record.
The frozen correction preserves the discrete selector and is in
`paper_artifacts/frdr_correction_2026-09-13/results/table3/`.

The final values are in
`paper_artifacts/minor_revision_2026-09/table3_fixed_step127641.csv`.
Each score uses the nearest operating point to 10 FP/h on the discrete
0.5-percentile sweep grid; no interpolation is applied.
`scripts/minor_revision_2026-09/run_frdr_table3_winaware_COPY.py` preserves
the executed generator and its original environment assumptions. The older
`scripts/winaware_2026-05-09/run_frdr_table3_winaware.py` and
`run_frdr_fp_recall.py` cover historical paths/parameters, not the final
corrected recipe. The exact embedding/manifests and fitted reference inputs
are still required; no public-input-only rerun is claimed.
Use `scripts/verify_minor_revision_artifacts.py` to verify the frozen outputs.

### Table 4 — HICEAS canon-level Promoter (7 species)

The final values are supported by the `FIXED` arm of
`paper_artifacts/minor_revision_2026-09/table4_fixed_and_submitted_audit.json`
and its fold-level audit CSV. The older
`paper_artifacts/table4_hiceas_canon_promoter/table4_canon_promoter_7species.csv`
and `run_hiceas_canon_promoter.py` are historical and do not represent the
final minor-revision table. The final audit can be verified from frozen
artifacts; a full rerun needs the exact corrected embeddings and canon
manifests. `scripts/run_table4_fixed.py` is the current path-configuring
wrapper: it validates embedding/index row counts before calling the archived
audited evaluator and writes a new result outside frozen artifacts by default.

The `In-band signal (0–8 kHz)` column in Table 4 is anchored on primary
literature for each species; see
`paper_artifacts/table4_hiceas_canon_promoter/README.md` for the full
citation list and the rationale (Hawaiian odontocete echolocation clicks
all peak at ≥ 12.5 kHz per Ziegenhorn et al. 2022, so the in-band
discrimination is anchored on whistles, tonal calls, the minke boing,
and the sperm-whale low-frequency p0 pulse + IPI structure).

## DAPT training (advanced — for re-training)

The exact executed implementation is
`scripts/dapt_train_beats_mam_fixed.py`; `dapt_train.py` is a compatibility
entry point to that audited script. Its path constants correspond to the
Adelie execution environment and should be adapted when rerunning elsewhere.

```bash
python dapt_train.py
```

Pre-requisites: BEATs PRETRAIN encoder (`BEATs_iter3_plus_AS2M.pt`) +
k-means k=1024 cluster centroids on PRETRAIN BEATs patch features (use
`scripts/dapt_extract_kmeans_labels.py`).

Training of the manuscript model (corrected weights available on Hugging Face):
- 2,042,268 non-overlapping 10-s windows (approximately 5,673 h)
- One pass, batch size 16, `drop_last=True`: 127,641 optimiser steps and 12
  unused rows from the incomplete final batch
- AdamW, encoder learning rate 1e-4, predictor/mask-token learning rate 1e-3,
  5% warm-up then cosine decay, bfloat16 autocast
- No DAPT validation split, no diel-balanced runtime sampler, and no downstream
  checkpoint selection; the final one-pass endpoint is the reported checkpoint

## Citation

If you use this code or weights, please cite:

```
Noda, T. et al. Discovery and promotion of unknown sounds into
operational detection targets for underwater passive acoustic monitoring
under false alarm constraints. Scientific Reports (revision, in review).
```

## Licence

- Code under `LICENSE_CODE.txt` (MIT License — permissive, commercial use
  permitted).
- Weights and CCED2 fits under `LICENSE_WEIGHTS.txt` (CC BY 4.0 — open,
  commercial use permitted with attribution).
- The DGPU framework and CCED2 unknownness score are subject to patent
  applications filed by Biologging Solutions Inc.; the open licenses on
  the code and weights do not grant rights under those patents.

## Contact / issues

For reproducibility issues, open a GitHub issue at
<https://github.com/alohajazz/openworld-soundscape-cced2-dgpu/issues>.
