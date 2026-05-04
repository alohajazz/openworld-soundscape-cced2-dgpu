#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/dclde_table3_ablation.py

Table 3 (DCLDE2013): Ablation of distance-based unknownness scores
- kNN_z
- Mahalanobis_z
- CCED2 = kNN_z + Mahalanobis_z

Outputs AUROC / AUPR_out / AUPR_in (OOD as positive; AUPR_in computed on -score).
"""

import argparse
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score


def load_embeddings_dir(d: Path, pattern: str = "embeddings_*.npy") -> np.ndarray:
    ps = sorted(glob.glob(str(d / pattern)))
    if not ps:
        raise FileNotFoundError(f"No {pattern} under {d}")
    return np.concatenate([np.load(p) for p in ps], axis=0).astype("float32")


def metrics_ood_positive(score_ind: np.ndarray, score_ood: np.ndarray) -> dict:
    y = np.concatenate([np.zeros(len(score_ind)), np.ones(len(score_ood))]).astype(int)  # OOD=1
    s = np.concatenate([score_ind, score_ood]).astype("float32")
    auroc = roc_auc_score(y, s)
    aupr_out = average_precision_score(y, s)      # OOD positive
    aupr_in  = average_precision_score(1 - y, -s) # InD positive (flip)
    return {"AUROC": float(auroc), "AUPR_out": float(aupr_out), "AUPR_in": float(aupr_in)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ind-emb-dir", required=True, help="InD embeddings dir (embeddings_*.npy)")
    ap.add_argument("--ood-emb-dir", required=True, help="OOD embeddings dir (embeddings_*.npy) e.g., DCLDE2013 test")
    ap.add_argument("--pattern", default="embeddings_*.npy")

    # model-dir: default to repo weights (portable)
    ap.add_argument("--model-dir", default="weights/cced2",
                    help="Directory containing knn_dapt.pkl / maha_dapt.pkl / cced2_norm.json (default: weights/cced2)")
    ap.add_argument("--knn-pkl", default="knn_dapt.pkl")
    ap.add_argument("--maha-pkl", default="maha_dapt.pkl")

    # allow norm-json override (for legacy workspace paths)
    ap.add_argument("--norm-json", default="",
                    help="Optional path to cced2_norm.json. If empty, uses <model-dir>/cced2_norm.json")

    ap.add_argument("--out-csv", default="paper_artifacts/dclde_table3.csv")
    ap.add_argument("--out-json", default="", help="Optional. Default: alongside out-csv with .json extension")
    args = ap.parse_args()

    ind_dir = Path(args.ind_emb_dir)
    ood_dir = Path(args.ood_emb_dir)
    model_dir = Path(args.model_dir)

    knn_path  = model_dir / args.knn_pkl
    maha_path = model_dir / args.maha_pkl

    if args.norm_json:
        norm_path = Path(args.norm_json)
    else:
        norm_path = model_dir / "cced2_norm.json"

    # Friendly checks
    missing = [p for p in [knn_path, maha_path, norm_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model files:\n"
            + "\n".join([f"  - {p}" for p in missing])
            + "\nTip: run from repo root, or pass --model-dir / --norm-json explicitly."
        )

    # Load embeddings
    E_ind = load_embeddings_dir(ind_dir, args.pattern)
    E_ood = load_embeddings_dir(ood_dir, args.pattern)

    # Load models + norm
    knn_obj  = joblib.load(knn_path)
    maha_obj = joblib.load(maha_path)
    cfg = json.loads(norm_path.read_text())
    mk, sk = float(cfg["mk"]), float(cfg["sk"])
    mm, sm = float(cfg["mm"]), float(cfg["sm"])
    sk = sk if sk > 0 else 1e-8
    sm = sm if sm > 0 else 1e-8

    # kNN raw -> z
    k_eff = int(knn_obj.get("k", 50))
    d_ind, _ = knn_obj["knn"].kneighbors(E_ind, n_neighbors=k_eff)
    d_ood, _ = knn_obj["knn"].kneighbors(E_ood, n_neighbors=k_eff)
    knn_raw_ind = d_ind.mean(axis=1).astype("float32")
    knn_raw_ood = d_ood.mean(axis=1).astype("float32")
    knn_z_ind = (knn_raw_ind - mk) / sk
    knn_z_ood = (knn_raw_ood - mk) / sk

    # Mahalanobis raw -> z
    mu = np.asarray(maha_obj["mu"]).astype("float32")
    P  = np.asarray(maha_obj["precision"]).astype("float32")
    D_ind = E_ind - mu
    D_ood = E_ood - mu
    maha_raw_ind = np.sqrt((D_ind @ P * D_ind).sum(axis=1)).astype("float32")
    maha_raw_ood = np.sqrt((D_ood @ P * D_ood).sum(axis=1)).astype("float32")
    maha_z_ind = (maha_raw_ind - mm) / sm
    maha_z_ood = (maha_raw_ood - mm) / sm

    cced2_ind = (knn_z_ind + maha_z_ind).astype("float32")
    cced2_ood = (knn_z_ood + maha_z_ood).astype("float32")

    rows = []
    r = metrics_ood_positive(knn_z_ind, knn_z_ood); r["score"] = "kNN_z"; rows.append(r)
    r = metrics_ood_positive(maha_z_ind, maha_z_ood); r["score"] = "Mahalanobis_z"; rows.append(r)
    r = metrics_ood_positive(cced2_ind, cced2_ood); r["score"] = "CCED2"; rows.append(r)

    df = pd.DataFrame(rows)[["score","AUROC","AUPR_out","AUPR_in"]].copy()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")
    out_json.write_text(df.to_json(orient="records", indent=2))

    print(df.to_string(index=False))
    print("[OK] wrote:", out_csv)
    print("[OK] wrote:", out_json)


if __name__ == "__main__":
    main()
