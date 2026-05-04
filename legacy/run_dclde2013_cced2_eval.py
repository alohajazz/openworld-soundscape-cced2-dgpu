#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_dclde2013_cced2_eval.py

Evaluate the CCED2 (Cross-Class Embedding Distance 2) unknownness score
on the DCLDE2013 dataset in an InD/OOD setting.

Assumptions:
    - In-distribution (InD) embeddings have been extracted using
      `dump_known56_features.py` for some training set
      (e.g., 56-class SED training data plus HICEAS validation).
    - CCED2 has been fitted on these InD embeddings using:
        python cced2_utils.py fit \\
          --embeddings_dir /path/to/ind_train_embeddings \\
          --out_dir        /path/to/cced2_model \\
          --k 50
    - Evaluation embeddings are available for:
        * InD evaluation set (e.g. HICEAS validation) and
        * OOD set (DCLDE2013 Test)
      and CCED2 scores can be computed with:
        python cced2_utils.py score \\
          --model_dir      /path/to/cced2_model \\
          --embeddings_dir /path/to/some_embeddings \\
          --out_path       /path/to/score_cced2.npy

This script then:
    1) Applies the fitted CCED2 model to InD evaluation and OOD embeddings.
    2) Concatenates the scores and assigns labels (0 for InD, 1 for OOD).
    3) Computes AUROC and AUPR (for OOD as positive, and optionally InD as positive),
       and writes them to a JSON file.

Example usage:

    python run_dclde2013_cced2_eval.py \\
      --ind-emb-dir      /path/to/ind_train_embeddings \\
      --ind-eval-emb-dir /path/to/ind_val_embeddings \\
      --ood-emb-dir      /path/to/dclde2013_test_embeddings \\
      --model-dir        /path/to/cced2_model \\
      --out-prefix       /path/to/results/dclde2013_cced2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure local modules are importable
sys.path.append(str(Path(__file__).parent))
try:
    from cced2_utils import fit_cced2, score_cced2
except ImportError:
    print("Error: Could not import 'cced2_utils'. Please ensure it is in the same directory.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CCED2 performance on DCLDE2013 (InD vs OOD).")
    parser.add_argument(
        "--ind-emb-dir",
        required=True,
        help="Directory containing embeddings_*.npy for the InD training set used to fit CCED2."
    )
    parser.add_argument(
        "--ind-eval-emb-dir",
        required=True,
        help="Directory containing embeddings_*.npy for the InD evaluation set "
             "(e.g., HICEAS validation)."
    )
    parser.add_argument(
        "--ood-emb-dir",
        required=True,
        help="Directory containing embeddings_*.npy for the OOD set (e.g., DCLDE2013 Test)."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory where CCED2 models (kNN, Mahalanobis, cced2_norm.json) will be stored."
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output files (e.g., /path/to/results/dclde2013_cced2)."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of neighbours for kNN (default: 50)."
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_prefix = Path(args.out_prefix)

    # 1) Fit CCED2 on InD training embeddings
    print("Fitting CCED2 model on InD training data...")
    fit_cced2(args.ind_emb_dir, str(model_dir), k=args.k)

    # 2) Apply CCED2 to InD evaluation and OOD embeddings
    print("Scoring InD evaluation data...")
    _, _, s_ind = score_cced2(
        str(model_dir),
        args.ind_eval_emb_dir,
        out_path=str(out_prefix.with_name(out_prefix.name + "_ind_cced2.npy")),
    )
    
    print("Scoring OOD evaluation data...")
    _, _, s_ood = score_cced2(
        str(model_dir),
        args.ood_emb_dir,
        out_path=str(out_prefix.with_name(out_prefix.name + "_ood_cced2.npy")),
    )

    # 3) Compute AUROC and AUPR
    y = np.concatenate([np.zeros_like(s_ind), np.ones_like(s_ood)])  # 0: InD, 1: OOD
    scores = np.concatenate([s_ind, s_ood])

    auroc = roc_auc_score(y, scores)
    aupr_out = average_precision_score(y, scores)         # OOD as positive
    aupr_in = average_precision_score(1 - y, -scores)     # InD as positive

    metrics = {
        "n_in": int(len(s_ind)),
        "n_ood": int(len(s_ood)),
        "AUROC": float(auroc),
        "AUPR_out": float(aupr_out),
        "AUPR_in": float(aupr_in),
    }

    print(json.dumps(metrics, indent=2))

    metrics_path = out_prefix.with_name(out_prefix.name + "_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[run_dclde2013_cced2_eval] wrote metrics to: {metrics_path}")


if __name__ == "__main__":
    main()