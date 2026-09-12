"""Table 4 Promoter re-evaluation with GroupKFold (deployment-day grouping).

Compares:
  (a) Original StratifiedKFold (paper Table 4) — 60-s recording-level split
  (b) GroupKFold with deployment-day key (e.g. '1705_20170930') — encounter-level proxy

Uses identical embeddings + Promoter (logistic regression, C=1.0).
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

CANON_DIR = "/workspace/revision1_species_canons"
DAPT_OP_DIR  = "/workspace/embeddings/hiceas_op_fulldata"
DAPT_1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata"

SPECIES = [
    ("Minke whale",                 "pos_Minke_whale.csv"),
    ("Sperm whale",                 "pos_Sperm_whale.csv"),
    ("False killer whale",          "pos_False_killer_whale.csv"),
    ("Short-finned pilot whale",    "pos_Short-finned_pilot_whale.csv"),
    ("Rough-toothed dolphin",       "pos_Rough-toothed_dolphin.csv"),
    ("Offshore spotted dolphin",    "pos_Offshore_spotted_dolphin.csv"),
    ("Striped dolphin",             "pos_Striped_dolphin.csv"),
]

def load_sharded(emb_dir):
    paths = sorted(glob.glob(os.path.join(emb_dir, "embeddings_*.npy")))
    idxs  = sorted(glob.glob(os.path.join(emb_dir, "index_*.csv")))
    embs = np.concatenate([np.load(p) for p in paths]).astype("float32")
    idx  = pd.concat([pd.read_csv(p) for p in idxs], ignore_index=True)
    return embs, idx

def canon_from_path(path):
    base = os.path.basename(path).replace(".flac", "").replace(".FLAC", "")
    parts = base.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return base

def deployment_day_group(canon):
    """Extract deployment-day key (first 2 underscore segments)."""
    parts = canon.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return canon

def build_canon_embeddings(embeddings, index_df):
    index_df = index_df.copy()
    index_df["canon"] = index_df["path"].apply(canon_from_path)
    canon_embs = {}
    for canon, group in index_df.groupby("canon"):
        idxs = group.index.values
        valid = idxs[idxs < len(embeddings)]
        if len(valid) > 0:
            canon_embs[canon] = embeddings[valid].mean(axis=0)
    return canon_embs

def evaluate(canon_embs, pos_csv, neg_csv, mode, seed=42, n_splits=5):
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    X, y, g = [], [], []
    for _, row in pos_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs:
            X.append(canon_embs[c]); y.append(1); g.append(deployment_day_group(c))
    for _, row in neg_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs:
            X.append(canon_embs[c]); y.append(0); g.append(deployment_day_group(c))
    if not X or sum(y) == 0:
        return None
    X = np.array(X); y = np.array(y); g = np.array(g)
    if mode == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)
    elif mode == "group":
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups=g)
    else:
        raise ValueError(mode)
    aucs, f1s, ps, rs, n_train_groups, n_test_groups = [], [], [], [], [], []
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        y_prob = clf.predict_proba(X[test_idx])[:, 1]
        y_pred = clf.predict(X[test_idx])
        aucs.append(roc_auc_score(y[test_idx], y_prob))
        f1s.append(f1_score(y[test_idx], y_pred, zero_division=0))
        ps.append(precision_score(y[test_idx], y_pred, zero_division=0))
        rs.append(recall_score(y[test_idx], y_pred, zero_division=0))
        n_train_groups.append(len(np.unique(g[train_idx])))
        n_test_groups.append(len(np.unique(g[test_idx])))
    if not aucs:
        return None
    return {
        "n_pos": int(sum(y == 1)), "n_neg": int(sum(y == 0)),
        "n_unique_groups": int(len(np.unique(g))),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "p_mean": float(np.mean(ps)),   "r_mean": float(np.mean(rs)),
        "n_folds_used": len(aucs),
        "median_train_groups": int(np.median(n_train_groups)) if n_train_groups else None,
        "median_test_groups": int(np.median(n_test_groups)) if n_test_groups else None,
    }

def main():
    print("=== Loading fulldata DAPT embeddings ===", flush=True)
    op_emb, op_idx = load_sharded(DAPT_OP_DIR)
    sp_emb, sp_idx = load_sharded(DAPT_1706_DIR)
    emb = np.concatenate([op_emb, sp_emb])
    idx = pd.concat([op_idx, sp_idx], ignore_index=True)
    canon_embs = build_canon_embeddings(emb, idx)
    print(f"  Loaded {len(canon_embs)} canons (= 60-s recordings)", flush=True)
    neg_csv = os.path.join(CANON_DIR, "neg_all.csv")
    rows = []
    print(f"\n{'Species':<28} {'Strat AUC':>13} {'Group AUC':>13} {'Delta':>8} {'Groups':>7} {'Folds':>5}", flush=True)
    print("-" * 90, flush=True)
    for sp_name, pos_file in SPECIES:
        pos_csv = os.path.join(CANON_DIR, pos_file)
        sk = evaluate(canon_embs, pos_csv, neg_csv, "stratified")
        gk = evaluate(canon_embs, pos_csv, neg_csv, "group")
        if sk is None or gk is None:
            print(f"  {sp_name:<28} (skip)", flush=True)
            continue
        delta = gk["auc_mean"] - sk["auc_mean"]
        print(f"  {sp_name:<28} {sk['auc_mean']:.3f}+/-{sk['auc_std']:.3f}  {gk['auc_mean']:.3f}+/-{gk['auc_std']:.3f}  {delta:+.3f}  {gk['n_unique_groups']:>5}  {gk['n_folds_used']:>3}", flush=True)
        rows.append({"species": sp_name, **{f"strat_{k}": v for k,v in sk.items()}, **{f"group_{k}": v for k,v in gk.items()}})
    out_csv = "/workspace/revision1_outputs_groupkfold/table4_groupkfold_compare.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}", flush=True)

if __name__ == "__main__":
    main()
