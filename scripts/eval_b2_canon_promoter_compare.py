#!/usr/bin/env python3
"""
B2 Promoter full evaluation: PRETRAIN vs DAPT-B 3ep
Using OP + 1706 embeddings (5,316 canons) for both.
"""
import numpy as np
import pandas as pd
import os, json, glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

CANON_DIR = "/workspace/revision1_species_canons"

# PRETRAIN embeddings (旧DAPT = PRETRAIN)
PRETRAIN_OP_EMB = "/workspace/embeddings/hiceas_op_pretrain/op2s_dapt/embeddings_000.npy"
PRETRAIN_OP_IDX = "/workspace/embeddings/hiceas_op_pretrain/op2s_dapt/index_000.csv"
PRETRAIN_1706_EMB = "/workspace/embeddings/hiceas_1706_species/embeddings_000.npy"
PRETRAIN_1706_IDX = "/workspace/embeddings/hiceas_1706_species/index_000.csv"

# DAPT-B 3ep embeddings
DAPT_OP_DIR = "/workspace/embeddings/hiceas_op_fulldata"
DAPT_1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata"

SPECIES = [
    "Sperm_whale",
    "False_killer_whale",
    "Minke_whale",
    "Striped_dolphin",
    "Offshore_spotted_dolphin",
    "Short-finned_pilot_whale",
    "Rough-toothed_dolphin",
]

def load_single(emb_path, idx_path):
    return np.load(emb_path), pd.read_csv(idx_path)

def load_sharded(emb_dir):
    emb_files = sorted(glob.glob(os.path.join(emb_dir, "embeddings_*.npy")))
    idx_files = sorted(glob.glob(os.path.join(emb_dir, "index_*.csv")))
    embs = np.concatenate([np.load(f) for f in emb_files])
    idxs = pd.concat([pd.read_csv(f) for f in idx_files], ignore_index=True)
    return embs, idxs

def canon_from_path(path):
    base = os.path.basename(path).replace(".flac", "").replace(".FLAC", "")
    parts = base.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return base

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

def evaluate_species(canon_embs, pos_csv, neg_csv, n_splits=5, seed=42):
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    X_list, y_list = [], []
    for _, row in pos_df.iterrows():
        canon = row.get("canon", "")
        if canon in canon_embs:
            X_list.append(canon_embs[canon])
            y_list.append(1)
    for _, row in neg_df.iterrows():
        canon = row.get("canon", "")
        if canon in canon_embs:
            X_list.append(canon_embs[canon])
            y_list.append(0)
    if len(X_list) == 0 or sum(y_list) == 0:
        return None
    X = np.array(X_list)
    y = np.array(y_list)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, f1s, ps, rs = [], [], [], []
    for train_idx, test_idx in skf.split(X, y):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        y_prob = clf.predict_proba(X[test_idx])[:, 1]
        y_pred = clf.predict(X[test_idx])
        if len(np.unique(y[test_idx])) > 1:
            aucs.append(roc_auc_score(y[test_idx], y_prob))
        f1s.append(f1_score(y[test_idx], y_pred, zero_division=0))
        ps.append(precision_score(y[test_idx], y_pred, zero_division=0))
        rs.append(recall_score(y[test_idx], y_pred, zero_division=0))
    if not aucs:
        return None
    return {
        "n_pos": int(sum(y == 1)), "n_neg": int(sum(y == 0)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "p_mean": float(np.mean(ps)), "r_mean": float(np.mean(rs)),
    }

def main():
    print("=== B2 Promoter Full Evaluation (OP + 1706) ===", flush=True)

    # PRETRAIN
    print("Loading PRETRAIN embeddings...", flush=True)
    pre_op_emb, pre_op_idx = load_single(PRETRAIN_OP_EMB, PRETRAIN_OP_IDX)
    pre_1706_emb, pre_1706_idx = load_single(PRETRAIN_1706_EMB, PRETRAIN_1706_IDX)
    pre_emb = np.concatenate([pre_op_emb, pre_1706_emb])
    pre_idx = pd.concat([pre_op_idx, pre_1706_idx], ignore_index=True)
    pre_canons = build_canon_embeddings(pre_emb, pre_idx)
    print(f"  PRETRAIN: {len(pre_canons)} canons", flush=True)

    # DAPT-B 3ep
    print("Loading DAPT-B 3ep embeddings...", flush=True)
    dpt_op_emb, dpt_op_idx = load_sharded(DAPT_OP_DIR)
    dpt_1706_emb, dpt_1706_idx = load_sharded(DAPT_1706_DIR)
    dpt_emb = np.concatenate([dpt_op_emb, dpt_1706_emb])
    dpt_idx = pd.concat([dpt_op_idx, dpt_1706_idx], ignore_index=True)
    dpt_canons = build_canon_embeddings(dpt_emb, dpt_idx)
    print(f"  DAPT-B: {len(dpt_canons)} canons", flush=True)

    # Evaluate
    print(f"\n{'Species':<30} {'N_pos':>6} {'PRE AUC':>12} {'DAPT AUC':>12} {'Delta':>8} {'PRE F1':>8} {'DAPT F1':>8}", flush=True)
    print("-" * 90, flush=True)

    results = []
    for sp in SPECIES:
        pos_csv = os.path.join(CANON_DIR, f"pos_{sp}.csv")
        neg_csv = os.path.join(CANON_DIR, "neg_all.csv")
        if not os.path.exists(pos_csv):
            print(f"  {sp}: not found", flush=True)
            continue
        r_pre = evaluate_species(pre_canons, pos_csv, neg_csv)
        r_dpt = evaluate_species(dpt_canons, pos_csv, neg_csv)
        if r_pre and r_dpt:
            delta = r_dpt["auc_mean"] - r_pre["auc_mean"]
            print(f"{sp:<30} {r_pre['n_pos']:>6} "
                  f"{r_pre['auc_mean']:.3f}±{r_pre['auc_std']:.3f} "
                  f"{r_dpt['auc_mean']:.3f}±{r_dpt['auc_std']:.3f} "
                  f"{delta:+.4f} "
                  f"{r_pre['f1_mean']:.2f}   {r_dpt['f1_mean']:.2f}",
                  flush=True)
            results.append({
                "species": sp,
                "n_pos": r_pre["n_pos"],
                "pretrain_auc": r_pre["auc_mean"], "pretrain_std": r_pre["auc_std"],
                "daptb_auc": r_dpt["auc_mean"], "daptb_std": r_dpt["auc_std"],
                "delta": delta,
                "pretrain_f1": r_pre["f1_mean"], "daptb_f1": r_dpt["f1_mean"],
                "pretrain_p": r_pre["p_mean"], "pretrain_r": r_pre["r_mean"],
                "daptb_p": r_dpt["p_mean"], "daptb_r": r_dpt["r_mean"],
            })

    print(f"\n--- Summary ---", flush=True)
    if results:
        avg_delta = np.mean([r["delta"] for r in results])
        improved = sum(1 for r in results if r["delta"] > 0.001)
        degraded = sum(1 for r in results if r["delta"] < -0.001)
        print(f"Average AUC delta: {avg_delta:+.4f}", flush=True)
        print(f"Improved (>0.001): {improved}/{len(results)}", flush=True)
        print(f"Degraded (<-0.001): {degraded}/{len(results)}", flush=True)

    with open("/workspace/logs/b2_promoter_fulldata_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to /workspace/logs/b2_promoter_fulldata_results.json", flush=True)

if __name__ == "__main__":
    main()
