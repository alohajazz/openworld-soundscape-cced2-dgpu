# CCED2 and DGPU for Open-world Discovery in Underwater Soundscapes

This repository provides a minimal implementation of the **CCED2 unknownness score** and the **Detect–Group–Promote–Union (DGPU) pipeline**, developed for our paper *"A stethoscope for the ocean: Open-world discovery in underwater soundscapes"*. This repository is intended for research reproducibility and does not include product deployment workflows (e.g., OTA pipelines, production monitoring, or operational tooling).

The repository includes evaluation scripts for public datasets (DCLDE2013, FRDR, HICEAS) and supports the pretrained BEATs+DAPT encoder and 56-class SED head (downloaded separately; see “**Setup: Downloading Weights**”). Lightweight CCED2 parameters in weights/cced2/ are included to reproduce the reported CCED2 configurations under the non-commercial weights license.

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

## Repository Structure

```text
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE_CODE                       # License for source code (MIT)
├── LICENSE_WEIGHTS                    # License for weights (CC BY-NC 4.0)
│
├── beats_core/                        # Bundled BEATs Source Code
│   ├── BEATs.py
│   ├── backbone.py
│   ├── modules.py
│   ├── LICENSE
│   └── README.md
│
├── dapt_train.py                      # DAPT training script (SimCLR/InfoNCE)
├── dapt_dataset.py                    # DAPT Dataset loader
├── dump_known56_features.py           # Feature extractor (Embeddings & Logits)
├── cced2_utils.py                     # CCED2 score implementation
├── fp_recall_helpers.py               # FP/h and Recall utilities
│
├── run_dclde2013_cced2_eval.py        # DCLDE2013 InD/OOD benchmark
├── run_frdr_cced2_op_eval.py          # FRDR continuous detection benchmark
├── run_hiceas_multi_species_eval.py   # HICEAS multi-species policy optimization
│
├── scripts/                           # Helper scripts
│   ├── dapt_make_manifest_all.py      # Manifest generator for massive datasets
│   ├── dapt_qc_manifest.py            # Manifest Quality Control tool
│   ├── dapt_make_shards.py            # (Optional) Tar sharding tool
│   ├── train_sed_beats_weak_plus.py   # Main SED training script
│   └── make_dclde2013_manifests.py    # Manifest generator for DCLDE2013
│
└── weights/                           # Pretrained Models & Parameters
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

1.  Visit the Hugging Face repository: **[BiologgingSolutions/OceanBEATs](https://huggingface.co/BiologgingSolutions/OceanBEATs)**
2.  Download the following files:
    * `beats_dapt_topup_encoder.pt`
    * `sed_head_56_topup_ep8.pt`
3.  Place them in the local `weights/` directory.

Ensure the directory structure looks like this:

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

## Model Details: 56-class SED Head

The 56-class SED head provided in `weights/sed_head_56_topup_ep8.pt` was trained on weakly labelled 10-s clips recorded mainly in coastal and lagoon environments around Ishigaki Island (Okinawa, Japan).

### Intended Scope and Taxonomy

The label taxonomy follows the category scheme introduced in **Noda et al. (2024)** (Biophony, Geophony, Anthrophony).

**Note:** Several species appear multiple times in the label set (e.g., "type A", "type B"). These correspond to dataset-internal call types and should be interpreted as internal acoustic prototypes specific to this dataset. Classes 50–52 correspond to sounds recorded from green sea turtles (*Chelonia mydas*).

| idx | Level 1 | Level 2 | Description |
| :--- | :--- | :--- | :--- |
| 0 | Biophony | Fish | *Myripristis berndti* (soldierfish) |
| 1 | Biophony | Fish | *Pseudanthias dispar* (anthias) |
| 2 | Geophony | Noise | Background noise |
| 3 | Biophony | Fish | *Lutjanus decussatus* (snapper) |
| 4 | Biophony | Fish | *Chromis viridis* (blue-green chromis), type A |
| 5 | Biophony | Fish | *Chromis viridis* (blue-green chromis), type B |
| 6 | Anthrophony | Human | Diver breathing |
| 7 | Anthrophony | Vessel | Diver support boat |
| 8 | Biophony | Fish | *Gymnothorax javanicus* (giant moray) |
| 9 | Anthrophony | Vessel | Small workboat / drone-like vessel |
| 10 | Biophony | Mammal | *Dugong dugon* (dugong), type A |
| 11 | Biophony | Mammal | *Dugong dugon* (dugong), type B |
| 12 | Biophony | Fish | *Pseudanthias evansi* (anthias) |
| 13 | Anthrophony | Vessel | Ferry |
| 14 | Biophony | Fish | *Dascyllus aruanus* (humbug damselfish), type A |
| 15 | Biophony | Fish | *Plotosus spp.* (striped eel catfish) |
| 16 | Biophony | Fish | *Amphiprion frenatus* (tomato clownfish) |
| 17 | Anthrophony | Construction | Underwater hammer |
| 18 | Biophony | Fish | *Chaetodon unimaculatus* (teardrop butterflyfish) |
| 19 | Anthrophony | Vessel | Patrol boat |
| 20 | Anthrophony | Vessel | Cargo ship |
| 21 | Biophony | Fish | *Zoramia leptacanthus* (glass cardinalfish) |
| 22 | Biophony | Mammal | *Globicephala macrorhynchus* (short-finned pilot whale), type A |
| 23 | Anthrophony | Vessel | Pleasure / small coastal boat |
| 24 | Biophony | Fish | *Myripristis kuntee* (soldierfish), type A |
| 25 | Biophony | Fish | *Myripristis kuntee* (soldierfish), type B |
| 26 | Biophony | Fish | *Stegastes nigricans* (dusky damselfish), type A |
| 27 | Biophony | Fish | *Stegastes nigricans* (dusky damselfish), type B |
| 28 | Biophony | Mammal | *Globicephala macrorhynchus* (short-finned pilot whale), type B |
| 29 | Biophony | Fish | *Neoniphon sammara* (squirrelfish), type A |
| 30 | Biophony | Mammal | *Tursiops aduncus* (Indo-Pacific bottlenose dolphin) |
| 31 | Biophony | Fish | *Dascyllus aruanus* (humbug damselfish), type B |
| 32 | Biophony | Fish | *Dascyllus aruanus* (humbug damselfish), type C |
| 33 | Biophony | Fish | *Dascyllus aruanus* (humbug damselfish), type D |
| 34 | Biophony | Fish | *Centropyge vrolikii* (angelfish) |
| 35 | Biophony | Fish | *Acanthurus dussumieri* (surgeonfish) |
| 36 | Biophony | Fish | *Pomacentrus amboinensis* (amber damsel), type A |
| 37 | Biophony | Fish | *Pomacentrus amboinensis* (amber damsel), type B |
| 38 | Geophony | Rain | Rain |
| 39 | Biophony | Fish | *Abudefduf sexfasciatus* (sergeant major), type A |
| 40 | Biophony | Fish | *Abudefduf sexfasciatus* (sergeant major), type B |
| 41 | Biophony | Fish | *Chrysiptera cyanea* (blue damselfish), type A |
| 42 | Biophony | Fish | *Chrysiptera cyanea* (blue damselfish), type B |
| 43 | Biophony | Mammal | *Tursiops truncatus* (common bottlenose dolphin) |
| 44 | Biophony | Mammal | *Globicephala macrorhynchus* (short-finned pilot whale), type C |
| 45 | Anthrophony | Vessel | Tanker |
| 46 | Biophony | Mammal | *Pseudorca crassidens* (false killer whale) |
| 47 | Anthrophony | Vessel | Regular service vessel |
| 48 | Anthrophony | Vessel | Tugboat |
| 49 | Biophony | Fish | *Neoniphon sammara* (squirrelfish), type B |
| 50 | Biophony | Reptile | Green sea turtle (*C. mydas*), breath |
| 51 | Biophony | Reptile | Green sea turtle (*C. mydas*), flipper stroke |
| 52 | Biophony | Reptile | Green sea turtle (*C. mydas*), collision/impact |
| 53 | Geophony | Wave | Wave |
| 54 | Biophony | Mammal | *Megaptera novaeangliae* (humpback whale), type A |
| 55 | Biophony | Mammal | *Megaptera novaeangliae* (humpback whale), type B |

## Scripts and Modules Overview

### 1. DAPT (Domain-Adaptive Pretraining)
* `scripts/dapt_make_manifest_all.py`: Scans underwater recordings and generates a TSV manifest. Automatically infers timestamps and site IDs from filenames.
* `dapt_train.py`: Self-supervised training script using SimCLR/InfoNCE loss. Loads the bundled BEATs encoder and fine-tunes it on the target underwater dataset.
* `dapt_dataset.py`: Handles loading of 10-s audio segments and time-balanced sampling.

### 2. 56-class SED Training
* `scripts/train_sed_beats_weak_plus.py`: The main script used to train the 56-class model. Freezes the BEATs encoder and trains a 1D-Conv head using "smart target" labels.

### 3. Embedding & CCED2 Utilities
* `dump_known56_features.py`: The main inference engine. Extracts Embeddings (Raw BEATs features) and Logits (56-class scores) from a CSV manifest.
* `cced2_utils.py`: Implements the CCED2 score. Fits kNN/Mahalanobis models on InD data and computes the combined unknownness score.

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

```bash
# 1. Create a manifest
python scripts/dapt_make_manifest_all.py

# 2. Run training (Adjust paths/batch size via env vars)
export TSV="./data/dapt_manifest.tsv"
export BEATS_CKPT="./weights/beats_dapt_topup_encoder.pt" # Start from provided weights
export CKPT_DIR="./ckpts_dapt"

python dapt_train.py
```

### 2. DCLDE2013: Unknown Detection Benchmark
Evaluate how well CCED2 separates In-Distribution (InD) sounds from Out-of-Distribution (OOD) sounds.

```bash
# 1. Generate manifests from raw DCLDE data
python scripts/make_dclde2013_manifests.py --input_root /data/DCLDE2013 --out_dir ./manifests

# 2. Extract embeddings for DCLDE test set
python dump_known56_features.py \
  --csv ./manifests/dclde_test.csv \
  --ckpt_beats ./weights/beats_dapt_topup_encoder.pt \
  --outdir ./embeddings/dclde_test \
  --dump_embeddings

# 3. Run evaluation (Using pre-fitted CCED2 params)
python run_dclde2013_cced2_eval.py \
  --ood-emb-dir ./embeddings/dclde_test \
  --model-dir ./weights/cced2 \
  --out-prefix ./results/dclde_eval
```

### 3. FRDR: Continuous Detection (FP/h vs Recall)
Evaluate the "Quiet" vs "Union" performance under strict FP/h constraints.

```bash
python run_frdr_cced2_op_eval.py \
  --manifest-csv ./manifests/frdr_continuous.csv \
  --score-npy ./results/frdr_cced2_scores.npy \
  --ann-csv ./data/frdr_annotations.csv \
  --target-fp-per-hour 10.0 \
  --k 2 --gap-sec 3.0 --tol-sec 10.0
```

### 4. HICEAS: Multi-Species Policy Optimization
Reproduce the selection of hybrid operating points (e.g., switching between Quiet and Promoter per species).

```bash
python run_hiceas_multi_species_eval.py \
  --manifest ./data/hiceas_manifest.csv \
  --predictions ./results/hiceas_raw_preds.csv \
  --out-summary ./results/hiceas_optimal_ops.csv \
  --constraint-precision 0.9 \
  --constraint-fph 0.5
```

## Citation

If you use this code or the provided weights in your research, please cite:

```bibtex
@article{noda2025stethoscope,
  title={A stethoscope for the ocean: Open-world discovery in underwater soundscapes},
  author={Noda, Takuji and Koizumi, Takuya},
  journal={npj Artificial Intelligence (Special Collection: Sensing Intelligence and Machine Learning)},
  note={Under Review},
  year={2025}
}
```

## License

This repository contains materials under two different licenses:

* **Source Code (.py files):** Released under the **MIT License**. See `LICENSE_CODE` for details.
* **Model Weights (`weights/` directory):** Released under **CC BY-NC 4.0** (Attribution-NonCommercial). Commercial use is strictly prohibited without prior permission. See `LICENSE_WEIGHTS` for details.

> **Note:** The `beats_core` directory contains code from the official BEATs implementation (Microsoft), which is licensed under the MIT License.

## Patent notice (important)

No patent rights are granted under this repository, whether expressly or by implication. Commercial use of the methods described here may require a separate patent license from Biologging Solutions Inc.

## Commercial use

The source code is released under the MIT License. The pretrained model weights and parameters are released under CC BY-NC 4.0 and are not permitted for commercial use. For commercial licensing of the weights and/or patent licensing, please contact Biologging Solutions Inc.
