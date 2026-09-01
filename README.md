# CCED2 and DGPU for Open-world Discovery in Underwater Soundscapes

This repository accompanies *"Discovery and promotion of unknown sounds into operational detection
targets for underwater passive acoustic monitoring under false alarm
constraints"* (Noda et al., Sci. Rep., **revision** in review).
It provides a minimal implementation of:

- **BEATs+DAPT** — a self-supervised audio encoder adapted to underwater
  soundscapes via Masked Audio Modeling (MAM) on a 5,673-h multi-site corpus.
- **CCED2** — an embedding-space unknownness score combining z-normalised
  k-nearest-neighbour and Mahalanobis distances on InD reference statistics.
- **DGPU** — the Detect–Group–Promote–Union pipeline that surfaces candidate
  unknown events for triage and incorporates them into FP/h-constrained
  detection policies.

The repository provides code, released weights, and source artifacts for the
public-data analyses in the manuscript. Its reproducibility scope is stated
explicitly below; in particular, the exact numerical results in Table 1 cannot
be reproduced without the non-public 56-class dataset. The repository does not
include product-deployment workflows.

The framework addresses practical PAM operational constraints — limited
annotation budgets and multi-day deployments under variable recording
conditions — through label-efficiency analysis of the Promoter stage
(paper §2.2) and cross-day generalisation analysis under 5-fold
GroupKFold by deployment day (paper §2.3).

> **About this revision** (May 2026). The original December 2025 submission
> contained a numerical-instability bug in the SimCLR DAPT training (AMP
> fp16 prevented BEATs encoder weight updates) and used an evaluation
> dataset (DCLDE 2013) with a bandwidth mismatch against the InD reference.
> This revision corrects both: DAPT is now Masked Audio Modeling on a
> 5,673-h corpus with bfloat16 precision, and the species-wise HICEAS
> evaluation is reformulated as canon-level Promoter discrimination on
> seven cetacean species. See `REVISION2.md` for the detailed change log
> and `legacy/` for the superseded scripts and weights.

## Companion artifacts

- **Pretrained BEATs+DAPT encoder** (`weights/beats_dapt_mam_step120000.pt`,
  361 MB) — SHA-256 `0fe9f7dd92780c2e564f1df06a192482dbcb9a56bdab4202f4d94862b9168f89`
- **56-class SED head** (`weights/sed_head_56_fulldata_ep8.pt`, 18 MB) —
  SHA-256 `135d11738a6619a57769955468ce5cb6eee3f07044fa45e6c950bf25ac4f8f60`
- **Pre-fitted CCED2 model** (`weights/cced2/`) — kNN + Mahalanobis pickled
  models, normalisation factors, and decision thresholds (q95 of InD
  CCED2 = 3.287, computed from the n=1,623 56-class held-out validation set
  using the new fulldata DAPT encoder)
- **Bundled BEATs source code** (`beats_core/`)

The source code is released under the MIT License. The current encoder, SED
head, and CCED2 parameters are released under CC BY 4.0. Encoder + SED head
are distributed via HuggingFace (`BiologgingSolutions/OceanBEATs`); CCED2 fits
ship inside this repository. The Detect-Group-Promote-Union (DGPU) framework
and the CCED2 unknownness score are subject to patent applications filed by
Biologging Solutions Inc.; these copyright licences do not grant patent
rights.

> **⚠️ Data Availability.** The internally curated 56-class underwater SED
> dataset is **not publicly available** because its raw clips and row-level
> metadata contain sensitive location and operational information, and no
> controlled-access mechanism has been approved by the original collaboration.
> Consequently, the exact Table 1 training and evaluation results, and a refit
> of the 56-class in-distribution reference statistics, cannot be reproduced
> from public materials alone. Supplementary Table S2 provides aggregate
> per-class statistics and the complete label taxonomy. Released weights,
> pre-fitted CCED2 parameters, scripts, and public evaluation datasets support
> inference and the public-data analyses described below.

> **Perch baseline (optional).** Perch 2.0 baseline rows in Tables 2–3 are
> not included in this repository. Users can supply Perch embeddings in
> the same manifest format (see Methods §4.2 for the "win10" window
> definition).

## Evaluation conventions

### FRDR continuous evaluation (manuscript Methods §4.5.1)

- **TP (reference-based)**: each ground-truth event is a TP if at least
  one detected event falls within ±tol seconds.
- **FP (prediction-based)**: a detection is an FP only if it falls outside
  the tolerance window of all reference events.
- Multiple detections near the same reference event do not increase TP
  and are not counted as FPs.

### HICEAS canon-level Promoter (manuscript Methods §4.1.4 + §4.4.4 + Table 4)

- **Canon = 60-s FLAC recording**. Canon ID is `first_three_underscore_tokens(basename)`
  (e.g. `1705_20171008_191500` from `1705_20171008_191500_5400.flac`).
- **Positives**: any canon overlapping an annotated DetectionTimeStart-End
  interval for the species.
- **Negatives**: canons without any annotation for the species, sampled
  at twice the positive count.
- **Promoter**: scikit-learn `LogisticRegression(C=1.0, max_iter=1000)`
  on canon-mean BEATs+DAPT embeddings (768-dim).
- **Cross-validation**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

## Repository structure

```text
.
├── README.md                              # This file
├── REVISION2.md                           # Change log from v1 (Dec 2025) to v2.2 (May 2026)
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
├── dapt_train.py                          # MAM-based DAPT training (canonical, fulldata)
├── dump_known56_features.py               # Embedding + 56-class SED logits dump
├── run_frdr_cced2_op_eval.py              # FRDR Quiet/Union/Fusion (Table 2)
├── run_frdr_fp_recall.py                  # FRDR FP/h-recall sweep (Table 3)
├── run_hiceas_canon_promoter.py           # HICEAS canon-level Promoter (Table 4) — fully parameterised CLI
│
├── scripts/                               # Helper / variant scripts
│   ├── dapt_make_manifest_all.py
│   ├── dapt_make_shards.py
│   ├── dapt_qc_manifest.py
│   ├── dapt_extract_kmeans_labels.py      # k=1024 k-means tokeniser
│   ├── dapt_train_beats_mam_545h_legacy.py  # Earlier 545-h MAM variant
│   ├── eval_b2_canon_promoter_compare.py  # Original analysis snapshot for Table 4
│   ├── run_frdr_fp_recall_interp.py       # FRDR FP/h-recall, interpolation variant
│   ├── train_sed_beats_weak_plus.py       # 56-class SED head training
│   └── export_hiceas_table6.py            # (legacy) original event-level Table 6 export
│
├── weights/
│   ├── beats_dapt_mam_step120000.pt       # Encoder (download separately, 361 MB)
│   ├── sed_head_56_fulldata_ep8.pt        # SED head (download separately, 18 MB)
│   └── cced2/
│       ├── cced2_norm.json                # kNN/Maha mean+std for z-normalisation
│       ├── knn_cced2.pkl                  # Pre-fit kNN model
│       ├── maha_cced2.pkl                 # Pre-fit Mahalanobis (Ledoit–Wolf shrinkage)
│       └── theta_cced2.json               # Decision threshold (InD q95 = 3.287)
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

### Downloading weights

Encoder + SED head live on HuggingFace (`BiologgingSolutions/OceanBEATs`).
After cloning this repo:

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="BiologgingSolutions/OceanBEATs",
                filename="beats_dapt_mam_step120000.pt",
                local_dir="weights/")
hf_hub_download(repo_id="BiologgingSolutions/OceanBEATs",
                filename="sed_head_56_fulldata_ep8.pt",
                local_dir="weights/")
PY

# Verify
shasum -a 256 weights/beats_dapt_mam_step120000.pt
# expect 0fe9f7dd92780c2e564f1df06a192482dbcb9a56bdab4202f4d94862b9168f89
shasum -a 256 weights/sed_head_56_fulldata_ep8.pt
# expect 135d11738a6619a57769955468ce5cb6eee3f07044fa45e6c950bf25ac4f8f60
```

CCED2 fits (`weights/cced2/`) are tracked directly in this repository and
do not require separate download.

## Reproducibility scope

This repository does **not** reproduce every table and figure in the
manuscript from public materials alone.

- **Table 1 (56-class SED):** the training code, configuration, released model
  weights, and aggregate results are provided, but exact numerical
  reproduction requires the non-public 56-class dataset.
- **Tables 2–4:** evaluation code and source result artifacts are provided.
  Users must obtain the public FRDR and HICEAS data from their respective
  providers and construct the manifests described in the Methods and repository
  documentation.
- **Figures 1–2:** conceptual schematics rather than computational outputs.
- **Figure 3:** the generating script and source artifacts are provided under
  `scripts/winaware_2026-05-09/` and `paper_artifacts/winaware_2026-05-09/`.
- **Figure 4 and supplementary computational outputs:** the final frozen
  release accompanying the revised manuscript will identify the exact scripts
  and artifacts corresponding to the final analyses.

## Reproducing manuscript analyses

### Table 1 — SED Performance (56-class)

`scripts/train_sed_beats_weak_plus.py` documents the training and evaluation
workflow for a compatible 56-class SED head. The manuscript headline value
(Stage 1 single-seed Event F1 = 0.483) and ten-seed variance (mean ± std =
0.475 ± 0.017) were obtained from the non-public dataset described in Methods
§4.1.2. These exact values cannot be independently regenerated from the public
repository because the underlying clips, labels, and splits are unavailable.

### Table 2 — FRDR: Quiet / Union / Fusion

```bash
python run_frdr_cced2_op_eval.py \
  --emb_dir   /path/to/frdr_fulldata_embeddings \
  --manifest  /path/to/frdr_continuous_hop2s.csv \
  --ann_csv   /path/to/annotations_B_cont.csv \
  --out_dir   results/table2/
```

Compare against `paper_artifacts/table2_frdr_quiet_union_fusion/`.

### Table 3 — FRDR: FP/h–Recall

```bash
python run_frdr_fp_recall.py
# (paths in script header — adapt to your environment)
```

Compare against `paper_artifacts/table3_frdr_fp_recall/`.

### Table 4 — HICEAS canon-level Promoter (7 species)

```bash
python run_hiceas_canon_promoter.py \
  --emb_dirs  /path/to/hiceas_op_embeddings  /path/to/hiceas_1706_embeddings \
  --canon_dir /path/to/per_species_canon_manifests \
  --out_json  results/table4_hiceas_canon_promoter.json
```

Per-species manifests must follow:
- `pos_<Species>.csv` with a `canon` column
- `neg_all.csv` with a `canon` column

Compare against `paper_artifacts/table4_hiceas_canon_promoter/table4_canon_promoter_7species.csv`.

The `In-band signal (0–8 kHz)` column in Table 4 is anchored on primary
literature for each species; see
`paper_artifacts/table4_hiceas_canon_promoter/README.md` for the full
citation list and the rationale (Hawaiian odontocete echolocation clicks
all peak at ≥ 12.5 kHz per Ziegenhorn et al. 2022, so the in-band
discrimination is anchored on whistles, tonal calls, the minke boing,
and the sperm-whale low-frequency p0 pulse + IPI structure).

## DAPT training (advanced — for re-training)

```bash
python dapt_train.py \
  --train_manifest /path/to/world_dapt_train.csv \
  --val_manifest   /path/to/world_dapt_val.csv \
  --kmeans_labels  /path/to/labels_k1024.npy \
  --kmeans_ckpt    /path/to/centroids_k1024.npy \
  --out_dir        ckpts_dapt_mam_fulldata/
```

Pre-requisites: BEATs PRETRAIN encoder (`BEATs_iter3_plus_AS2M.pt`) +
k-means k=1024 cluster centroids on PRETRAIN BEATs patch features (use
`scripts/dapt_extract_kmeans_labels.py`).

Training of the published model:
- 5,673-h World-DAPT corpus (SanctSound, US Navy USWTR, NOAA NRS/ONMS,
  ICListen / ONC, NPS Glacier Bay, PALAOA — see Methods §4.1.1)
- One epoch, 126,365 steps, batch size 16, learning rate 1e-4 (cosine),
  bfloat16 precision
- Selected checkpoint: `BEATs_DAPT_MAM_step120000.pt`

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
