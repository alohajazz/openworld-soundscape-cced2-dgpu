# CCED2 and DGPU for Open-world Discovery in Underwater Soundscapes

This repository provides a minimal implementation of the **CCED2 unknownness score** and the **Detect–Group–Promote–Union (DGPU) pipeline**, developed for our paper *"A stethoscope for the ocean: Open-world discovery in underwater soundscapes"*. This repository is intended for research reproducibility and does not include product deployment workflows (e.g., OTA pipelines, production monitoring, or operational tooling).

The repository includes evaluation scripts for public datasets (DCLDE2013, FRDR, HICEAS) and supports the pretrained BEATs+DAPT encoder and 56-class SED head (downloaded separately; see **Setup: Downloading Weights**). Lightweight CCED2 parameters in `weights/cced2/` are included to reproduce the reported CCED2 configurations under the non-commercial weights license.

## Overview

In addition to the source code, this repository provides:

* **Pretrained BEATs+DAPT encoder** (World-DAPT, Top-up version) (downloaded separately; see **Setup: Downloading Weights**).
* **56-class SED head** trained on our internal underwater SED dataset (downloaded separately; see **Setup: Downloading Weights**).
* **Pre-fitted CCED2 model parameters** (kNN index, Mahalanobis statistics, normalisation factors, and decision thresholds) (included in this repository).
* **Bundled BEATs source code** (`beats_core/`) for easy setup without external dependencies.

These weights and parameters are released **for non-commercial research use only**. They match the exact configuration used in the paper, enabling reproduction of the main results on DCLDE2013, FRDR, and HICEAS using only public datasets.

> **⚠️ Important Note on Data Availability**
>
> The 56-class underwater SED training dataset used to train the SED head contains sensitive operational data and is **not publicly available**. However, the provided weights allow you to reproduce the inference and evaluation steps described in the paper. Reference training scripts are also provided for those wishing to train their own models on their own datasets.

> **Perch baseline (optional)**. The Perch 2.0 baseline reported in the paper is optional and is not included in this repository due to external dependencies and distribution constraints. Users can reproduce the main BEATs+DAPT results with the provided weights; Perch-based evaluations can be run by supplying Perch embeddings in the same manifest format (see the paper/Methods for the “win10” window definition).

## Evaluation conventions (important)

Unless otherwise stated, operational evaluation follows the paper's event-matching definition (Methods 4.5.1):

- **TP (reference-based)**: each ground-truth (reference) event is counted as a TP if **at least one** detected event falls within ±tol seconds.
- **FP (prediction-based)**: a detected event is counted as an FP only if it falls **outside** the tolerance window of **all** reference events.
- Multiple detections near the same reference event do **not** increase TP and are **not** counted as FP.

For HICEAS operational evaluation, we apply **canon-level DNS** per species:
- only recordings (canons) that contain at least one annotation for the target species are scored;
- within scored canons, we do not further mask unlabelled time spans.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE_CODE.txt
├── LICENSE_WEIGHTS.txt
│
├── beats_core/                        # Bundled BEATs source code
│   ├── BEATs.py
│   ├── backbone.py
│   ├── modules.py
│   ├── LICENSE.txt
│   └── README.md
│
├── dapt_train.py                      # DAPT training script (SimCLR/InfoNCE)
├── dapt_dataset.py                    # DAPT dataset loader
├── dump_known56_features.py           # Embedding/logit extractor (template)
├── cced2_utils.py                     # CCED2 fit/score implementation
├── fp_recall_helpers.py               # FP/h and Recall utilities
│
├── run_dclde2013_cced2_eval.py        # DCLDE2013 InD/OOD benchmark (Table 2)
├── run_frdr_cced2_op_eval.py          # FRDR CCED2-only continuous detection (Table 5 support)
├── run_hiceas_multi_species_eval.py   # HICEAS multi-species sanity check using unknownness scores (Methods-aligned matching; supports --tail high/low)
│
├── configs/
│   └── hiceas_table6/                 # Table 6 policy configs (ops_*.json, fewshot labels, lookup table)
│
├── notes/
│   └── HICEAS_run_ops_note.txt        # Note on HICEAS ops-point script input requirements
│
├── paper_artifacts/
│   ├── dclde_table3.csv               # Table 3 (distance-score ablation) results (CSV)
│   ├── frdr_table4/                   # Table 4 picks + exported CSV
│   ├── frdr_table5/                   # Table 5 final CSVs (interp at FP/h=10; beats/perch)
│   └── hiceas_table6/                 # Table 6 final outputs (truth table: _FINAL_SPECIES_TABLE.csv)
│
├── scripts/
│   ├── dapt_make_manifest_all.py
│   ├── dapt_qc_manifest.py
│   ├── dapt_make_shards.py
│   ├── train_sed_beats_weak_plus.py
│   ├── make_dclde2013_manifests.py
│   ├── dclde_table3_ablation.py       # Table 3 ablation (kNN_z / Mahalanobis_z / CCED2)
│   ├── export_hiceas_table6.py        # Regenerate FINAL_summary/tex from _FINAL_SPECIES_TABLE
│   ├── frdr_table4/
│   │   └── export_frdr_table4_from_picks.py
│   ├── frdr_table5/
│   │   ├── run_frdr_table5_tol10.py
│   │   ├── run_frdr_table5_tol10_grid.py
│   │   ├── run_frdr_table5_tol10_interp.py
│   │   ├── run_frdr_table5_tol10_beats_interp_anymatch.py
│   │   ├── run_frdr_table5_tol10_perch_interp.py
│   │   └── run_frdr_ablate_scores.py
│   └── hiceas_table6/
│       └── precision80_rule_ext/
│           ├── make_label_pack.sh
│           ├── ops_rule.json
│           ├── ops_strict.json
│           ├── README_ops.md
│           ├── run_ops.sh
│           ├── run_review_once.sh
│           └── sweep_kq_ops.py
│
└── weights/
    ├── beats_dapt_topup_encoder.pt    # (Download from Hugging Face)
    ├── sed_head_56_topup_ep8.pt       # (Download from Hugging Face)
    └── cced2/
        ├── knn_dapt.pkl
        ├── maha_dapt.pkl
        ├── cced2_norm.json
        └── theta_cced2.json
```

## Setup: Downloading Weights

The pretrained model weights (300MB+) are hosted on Hugging Face due to GitHub's file size limits.
Before running the scripts, please download the weights and place them in the `weights/` directory.

Visit the Hugging Face repository: **BiologgingSolutions/OceanBEATs**

Download the following files:
* `beats_dapt_topup_encoder.pt`
* `sed_head_56_topup_ep8.pt`

Place them in the local `weights/` directory. Ensure the directory structure looks like this:

```text
openworld-soundscape-cced2-dgpu/
└── weights/
    ├── beats_dapt_topup_encoder.pt  <-- Place here
    ├── sed_head_56_topup_ep8.pt     <-- Place here
    └── cced2/                       <-- (Already included in this repo)
        ├── knn_dapt.pkl
        └── ...
```

> **Note:** The lightweight parameters for CCED2 (inside `weights/cced2/`) are included in this GitHub repository, so you only need to download the `.pt` files.

## Requirements

* Python ≥ 3.9
* PyTorch (tested on 1.12+)
* torchaudio, numpy, pandas, scikit-learn, umap-learn, hdbscan

Install dependencies via:

```bash
pip install -r requirements.txt
```

> **Note:** The official BEATs source code is bundled in the `beats_core/` directory. You do not need to clone the external Microsoft repository.

## Usage Examples

### 1. DAPT Training (SimCLR)
To run the domain-adaptive pretraining loop using the bundled script:

> **World-DAPT manifest defaults (log-aligned):**
> - 10-s windows with **50% overlap** (stride 5 s)
> - `diel` uses **4 bins**: `00-06`, `06-12`, `12-18`, `18-24`
>
> You can override these via environment variables (e.g., `SEG_S`, `STRIDE_S`).

```bash
# 1. Create a manifest
python scripts/dapt_make_manifest_all.py

# 2. Run training (Adjust paths/batch size via env vars)
export TSV="./data/dapt_manifest.tsv"
export BEATS_CKPT="./weights/beats_dapt_topup_encoder.pt" # Start from provided weights
export CKPT_DIR="./ckpts_dapt"

python dapt_train.py
```

### 2. DCLDE2013: Unknown Detection Benchmark (Tables 2–3)

* **Table 2:** `run_dclde2013_cced2_eval.py` (AUROC / AUPR for CCED2 on DCLDE2013)
* **Table 3:** `scripts/dclde_table3_ablation.py` (kNN_z / Mahalanobis_z / CCED2 ablation; results saved to `paper_artifacts/dclde_table3.csv`)

```bash
# 1. Generate manifests from raw DCLDE data
python scripts/make_dclde2013_manifests.py --input_root /data/DCLDE2013 --out_dir ./manifests

# 2. Extract embeddings for DCLDE test set
python dump_known56_features.py \
  --csv ./manifests/dclde_test.csv \
  --ckpt_beats ./weights/beats_dapt_topup_encoder.pt \
  --outdir ./embeddings/dclde_test \
  --dump_embeddings

# 3. Table 2: run CCED2 evaluation
python run_dclde2013_cced2_eval.py \
  --ind-emb-dir ./embeddings/ind_train_dapt \
  --ind-eval-emb-dir ./embeddings/ind_val_dapt \
  --ood-emb-dir ./embeddings/dclde_test \
  --model-dir ./weights/cced2 \
  --out-prefix ./results/dclde_eval \
  --k 50

# 4. Table 3: distance-score ablation
python scripts/dclde_table3_ablation.py \
  --ind-emb-dir ./embeddings/ind_val_dapt \
  --ood-emb-dir ./embeddings/dclde_test \
  --model-dir ./weights/cced2 \
  --out-csv paper_artifacts/dclde_table3.csv
```

> **Note:** Table 3 includes optional Perch 2.0 rows. Reproducing the Perch entries requires providing Perch embeddings computed on explicit **10-s windows (“win10”)**, defined as `start_sec = center_sec - 5` and `duration_sec = 10`. We then compute distance-based unknownness scores (kNN_z, Mahalanobis_z, CCED2) and report AUROC/AUPR on the DCLDE2013 Test split, treating OOD segments as the positive class (as in the paper).

### 3. FRDR: Quiet / Union / Fusion (Table 4)
Table 4 is reproduced from the published operating-point pick files under `paper_artifacts/frdr_table4/`.

```bash
python scripts/frdr_table4/export_frdr_table4_from_picks.py \
  --quiet_cmp   paper_artifacts/frdr_table4/comparison_quiet_vs_promoted.csv \
  --union_pick  paper_artifacts/frdr_table4/union_quiet_promoted_pick.csv \
  --fusion_pick paper_artifacts/frdr_table4/fusion_pick.csv \
  --out_csv     paper_artifacts/frdr_table4/table4_frdr.csv
```

### 4. FRDR: CCED2-only Continuous Detection (Table 5 and supporting analyses)
The core CCED2-only continuous operational evaluation script is:

```bash
python run_frdr_cced2_op_eval.py \
  --manifest-csv ./manifests/frdr_continuous.csv \
  --score-npy ./results/frdr_cced2_scores.npy \
  --ann-csv ./data/frdr_annotations.csv \
  --target-fp-per-hour 10.0 \
  --k 2 --gap-sec 3.0 --tol-sec 10.0 --smooth
```
Scripts used to produce the Table 5 CSVs are provided under `scripts/frdr_table5/`, and the exported results are under `paper_artifacts/frdr_table5/`.

**Paper Table 5 (truth CSVs):**
- `paper_artifacts/frdr_table5/table5_beats_tol10_interp_at_fp10.csv`
- `paper_artifacts/frdr_table5/table5_perch_tol10_interp_at_fp10.csv`

> Note: `run_frdr_cced2_op_eval.py` uses `fp_recall_helpers.py`, which follows the event-matching definition described above (TP per reference; FP only outside all reference windows).

### 5. HICEAS: Species-wise Hybrid Policies (Table 6)
The final Table 6 outputs are provided under `paper_artifacts/hiceas_table6/`:

* `_FINAL_SPECIES_TABLE.csv` (truth; includes TP/FP/FN and P/R/F1/FP/h)
* `FINAL_table_15s.tex`
* `FINAL_summary.md`

You can regenerate the markdown/tex from the truth table:

```bash
python scripts/export_hiceas_table6.py
```

Policy configurations used for Table 6 are under `configs/hiceas_table6/`, and the ops-point rule scripts are under `scripts/hiceas_table6/precision80_rule_ext/`.
Note that `precision80_rule_ext/run_ops.sh` requires an annotation CSV (see `notes/HICEAS_run_ops_note.txt`).

These files are the **exact outputs used in the paper**. If you re-run the ops-point scripts on your environment, ensure that your evaluation conventions (matching, DNS, tolerance) are consistent with the paper; otherwise numbers may differ slightly.


### 6. HICEAS sanity-check (Table S3-style CAP selection)
This script corresponds to the Table S3 "unknownness-only" sanity-check protocol in the paper.

`run_hiceas_multi_species_eval.py` applies CAP-style selection:
- species-wise quantile threshold `q` on window-level `s_hat`
- then keeps top-K events per hour per canon

(Added in this repository revision; if your local copy predates this option, please pull the latest changes.)

**Tail direction matters.**
- Use `--tail high` when larger `s_hat` means more unknown/extreme (default; typical Quiet-style scores).
- Use `--tail low` when smaller `s_hat` means more unknown/extreme (e.g., if you saved `s_hat = -kNN_z`, `-Mahalanobis_z`, or `-CCED2`).

Examples:

```bash
# Quiet-style score (higher is more extreme)
python run_hiceas_multi_species_eval.py \
  --manifest manifest.csv --predictions pred_quiet.csv \
  --out-summary out/hiceas_summary.csv --out-macro out/hiceas_macro.csv \
  --q 0.99 --K 2 --tail high

# Negative distance score (lower is more extreme)
python run_hiceas_multi_species_eval.py \
  --manifest manifest.csv --predictions pred_neg_knn.csv \
  --out-summary out/hiceas_summary.csv --out-macro out/hiceas_macro.csv \
  --q 0.99 --K 2 --tail low
```  

## Citation

If you use this code or the provided weights in your research, please cite:

```bibtex
@article{noda2026stethoscope,
  title={A stethoscope for the ocean: Unknownness-aware monitoring under false-positives-per-hour constraints in underwater soundscapes},
  author={Noda, Takuji and Koizumi, Takuya},
  journal={Scientific Reports},
  note={Under Review},
  year={2026}
}
```

## License

This repository contains materials under two different licenses:

* **Source Code (.py files):** Released under the **MIT License**. See `LICENSE_CODE.txt` for details.
* **Model Weights (`weights/` directory):** Released under **CC BY-NC 4.0** (Attribution-NonCommercial). Commercial use is strictly prohibited without prior permission. See `LICENSE_WEIGHTS.txt` for details.

> **Note:** The `beats_core` directory contains code from the official BEATs implementation (Microsoft), which is licensed under the MIT License.

## Patent notice (important)

No patent rights are granted under this repository, whether expressly or by implication. Commercial use of the methods described here may require a separate patent license from Biologging Solutions Inc.

## Commercial use

The source code is released under the MIT License. The pretrained model weights and parameters are released under CC BY-NC 4.0 and are not permitted for commercial use. For commercial licensing of the weights and/or patent licensing, please contact Biologging Solutions Inc.
