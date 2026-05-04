#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_hiceas_canon_promoter.py

Canon-level Promoter discrimination evaluation on HICEAS for cetacean species
(Noda et al., manuscript Table 4).

Per-species evaluation:
  - Positives: 60-s FLAC recordings (canons) overlapping any annotated
    DetectionTimeStart-DetectionTimeEnd interval for the species.
  - Negatives: FLACs without any annotation for the species, sampled at
    a configurable ratio (default 2x positives) from the same time-of-day
    distribution.
  - Promoter: logistic regression (regularisation C, default 1.0) on
    canon-level mean embeddings (768-dim BEATs+DAPT).
  - Cross-validation: 5-fold StratifiedKFold (shuffle=True, configurable seed).
  - Metrics: ROC AUC mean ± std, F1, Precision, Recall.

Inputs (CLI):
  --emb_dirs     : one or more directories each containing embeddings_*.npy
                   and index_*.csv (sharded). Embeddings are concatenated
                   in the order given.
  --canon_dir    : directory containing pos_<Species>.csv and neg_all.csv
                   per-species manifests; canon column required.
  --species      : optional list to restrict evaluation; default = all
                   pos_*.csv files in canon_dir.
  --out_json     : path to write per-species results.

Each pos_/neg_ CSV must contain a "canon" column whose values match
canon_from_path(path) = first three underscore-separated tokens of the
basename (e.g. "1705_20171008_191500" from
"1705_20171008_191500_5400.flac").

Example:
  python run_hiceas_canon_promoter.py \\
    --emb_dirs /path/to/emb_op /path/to/emb_1706 \\
    --canon_dir /path/to/revision1_species_canons \\
    --out_json results/hiceas_canon_promoter.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ----- helpers -----


def canon_from_path(path: str) -> str:
    """Extract canon ID = first three underscore-tokens of basename.

    Example: "1705_20171008_191500_5400.flac" -> "1705_20171008_191500"
    """
    base = os.path.basename(path).replace(".flac", "").replace(".FLAC", "")
    parts = base.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return base


def load_emb_dir(emb_dir: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load all embeddings_*.npy + index_*.csv shards in a directory.

    Returns (embeddings [N, D], index_df with at least a "path" column).
    """
    emb_files = sorted(glob.glob(os.path.join(emb_dir, "embeddings_*.npy")))
    idx_files = sorted(glob.glob(os.path.join(emb_dir, "index_*.csv")))
    if not emb_files:
        raise FileNotFoundError(f"No embeddings_*.npy in {emb_dir}")
    if len(emb_files) != len(idx_files):
        raise ValueError(
            f"Shard mismatch in {emb_dir}: {len(emb_files)} embeddings vs {len(idx_files)} indices"
        )
    embs = np.concatenate([np.load(f) for f in emb_files])
    idxs = pd.concat([pd.read_csv(f) for f in idx_files], ignore_index=True)
    if "path" not in idxs.columns:
        raise ValueError(f"index_*.csv in {emb_dir} must contain 'path' column")
    return embs, idxs


def build_canon_embeddings(embeddings: np.ndarray, index_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Group embeddings by canon ID and average within each canon."""
    df = index_df.copy()
    df["canon"] = df["path"].apply(canon_from_path)
    out: dict[str, np.ndarray] = {}
    for canon, group in df.groupby("canon"):
        idxs = group.index.values
        valid = idxs[idxs < len(embeddings)]
        if len(valid) > 0:
            out[canon] = embeddings[valid].mean(axis=0)
    return out


def evaluate_species(
    canon_embs: dict[str, np.ndarray],
    pos_csv: str,
    neg_csv: str,
    n_splits: int = 5,
    C: float = 1.0,
    seed: int = 42,
) -> dict | None:
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    if "canon" not in pos_df.columns or "canon" not in neg_df.columns:
        raise ValueError(f"pos/neg CSVs must contain 'canon' column")

    X_list, y_list = [], []
    for _, row in pos_df.iterrows():
        canon = row["canon"]
        if canon in canon_embs:
            X_list.append(canon_embs[canon])
            y_list.append(1)
    for _, row in neg_df.iterrows():
        canon = row["canon"]
        if canon in canon_embs:
            X_list.append(canon_embs[canon])
            y_list.append(0)
    if not X_list or sum(y_list) == 0:
        return None

    X = np.array(X_list)
    y = np.array(y_list)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    aucs, f1s, ps, rs = [], [], [], []
    for tr_idx, te_idx in skf.split(X, y):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        clf.fit(X[tr_idx], y[tr_idx])
        y_prob = clf.predict_proba(X[te_idx])[:, 1]
        y_pred = clf.predict(X[te_idx])
        if len(np.unique(y[te_idx])) > 1:
            aucs.append(roc_auc_score(y[te_idx], y_prob))
        f1s.append(f1_score(y[te_idx], y_pred, zero_division=0))
        ps.append(precision_score(y[te_idx], y_pred, zero_division=0))
        rs.append(recall_score(y[te_idx], y_pred, zero_division=0))
    if not aucs:
        return None
    return {
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "p_mean": float(np.mean(ps)),
        "r_mean": float(np.mean(rs)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Canon-level Promoter eval on HICEAS (manuscript Table 4)"
    )
    ap.add_argument(
        "--emb_dirs", nargs="+", required=True,
        help="One or more directories with embeddings_*.npy and index_*.csv",
    )
    ap.add_argument(
        "--canon_dir", required=True,
        help="Directory with pos_<Species>.csv and neg_all.csv per-species manifests",
    )
    ap.add_argument(
        "--species", nargs="*", default=None,
        help="Restrict to listed species (matches pos_<S>.csv filename); default = all",
    )
    ap.add_argument("--n_splits", type=int, default=5, help="StratifiedKFold splits")
    ap.add_argument("--C", type=float, default=1.0, help="LogReg regularisation strength")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (for KFold shuffle)")
    ap.add_argument("--out_json", required=True, help="Output JSON path")
    args = ap.parse_args()

    print("=== HICEAS canon-level Promoter evaluation (Table 4) ===", flush=True)
    print(f"Embedding dirs: {args.emb_dirs}", flush=True)

    emb_list, idx_list = [], []
    for d in args.emb_dirs:
        e, i = load_emb_dir(d)
        emb_list.append(e)
        idx_list.append(i)
        print(f"  loaded {d}: emb shape {e.shape}, index rows {len(i)}", flush=True)
    embeddings = np.concatenate(emb_list)
    index_df = pd.concat(idx_list, ignore_index=True)
    canon_embs = build_canon_embeddings(embeddings, index_df)
    print(f"Total canons: {len(canon_embs)}", flush=True)

    if args.species is None:
        species = sorted(
            os.path.basename(p).replace("pos_", "").replace(".csv", "")
            for p in glob.glob(os.path.join(args.canon_dir, "pos_*.csv"))
        )
    else:
        species = list(args.species)
    if not species:
        raise SystemExit(f"No pos_*.csv files in {args.canon_dir}")
    print(f"Species ({len(species)}): {species}", flush=True)

    print(
        f"\n{'Species':<32} {'N_pos':>6} {'AUC mean ± std':>18} {'F1':>7} {'P':>7} {'R':>7}",
        flush=True,
    )
    print("-" * 80, flush=True)

    results = []
    for sp in species:
        pos_csv = os.path.join(args.canon_dir, f"pos_{sp}.csv")
        neg_csv = os.path.join(args.canon_dir, "neg_all.csv")
        if not os.path.exists(pos_csv):
            print(f"  {sp}: pos_{sp}.csv not found", flush=True)
            continue
        if not os.path.exists(neg_csv):
            raise SystemExit(f"neg_all.csv missing in {args.canon_dir}")
        r = evaluate_species(
            canon_embs, pos_csv, neg_csv,
            n_splits=args.n_splits, C=args.C, seed=args.seed,
        )
        if r is None:
            print(f"  {sp}: insufficient data", flush=True)
            continue
        print(
            f"{sp:<32} {r['n_pos']:>6} "
            f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}  "
            f"{r['f1_mean']:.3f}  {r['p_mean']:.3f}  {r['r_mean']:.3f}",
            flush=True,
        )
        results.append({"species": sp, **r})

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "emb_dirs": args.emb_dirs,
                    "canon_dir": args.canon_dir,
                    "n_splits": args.n_splits,
                    "C": args.C,
                    "seed": args.seed,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
