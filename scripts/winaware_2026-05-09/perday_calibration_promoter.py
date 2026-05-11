"""Per-day score calibration — Tier 1 (d) cross-day fix.

Compares 3 protocols on 7 HICEAS species under 5-fold GroupKFold by day:
  (1) LR_raw    : LogReg C=1.0, raw scores, AUC over all test samples
                  (= paper Table 4 GroupKFold protocol)
  (2) LR_z      : same LR, but per-day z-score within test fold before AUC
  (3) LR_rank   : same LR, but per-day rank-normalize (uniform [0,1]) before AUC

Rationale (確実):
  - AUC over the union of multiple test days is sensitive to cross-day
    score-distribution differences. If the LR's calibration is day-specific
    (e.g. day A scores all near +0.7, day B all near +0.3, both with same
    rank-order within day), pooled AUC is hurt even when within-day rankings
    are correct.
  - Per-day z-score and rank-normalize remove day-level offset/scale.
  - This is a TEST-TIME ADAPTATION (TENT lineage, Wang et al. 2021 ICLR
    arXiv:2006.10726): adapt only at inference, encoder + classifier frozen.

Implementation note:
  - Calibration uses ONLY the test fold's own day groupings (no train info
    leaks). Within each test day, compute mean/std (or rank), apply.
  - Fully transparent: per-day calibration does not change within-day AUC,
    only cross-day pooled AUC. So if pooled AUC improves > within-day AUC,
    the day-level confound is the culprit.
"""
import sys, glob, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base

OP_DIR = "/workspace/embeddings/hiceas_op_fulldata_winaware"
S1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata_winaware"
PART2_DIR = "/workspace/embeddings/hiceas_1706_part2_fulldata_winaware"
OUTDIR = Path("/workspace/scripts/winaware_2026-05-09/perday_calibration_2026-05-09")
OUTDIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5

def load_dir(emb_dir):
    paths = sorted(glob.glob(f"{emb_dir}/embeddings_*.npy"))
    idxs = sorted(glob.glob(f"{emb_dir}/index_*.csv"))
    embs = np.concatenate([np.load(p) for p in paths]).astype("float32")
    idx = pd.concat([pd.read_csv(p) for p in idxs], ignore_index=True)
    return embs, idx

def per_day_zscore(scores, days):
    """Within each unique day in `days`, z-normalize `scores`.
    Days with <2 samples skip normalization (kept as-is)."""
    out = np.array(scores, dtype=np.float64).copy()
    for d in np.unique(days):
        mask = days == d
        if mask.sum() < 2:
            continue
        s = out[mask]
        sd = s.std()
        if sd < 1e-9:
            continue
        out[mask] = (s - s.mean()) / sd
    return out

def per_day_rank(scores, days):
    """Within each unique day, rank-transform to [0,1]."""
    out = np.array(scores, dtype=np.float64).copy()
    for d in np.unique(days):
        mask = days == d
        n = mask.sum()
        if n < 2:
            continue
        ranks = rankdata(out[mask], method="average")
        out[mask] = (ranks - 1.0) / (n - 1.0)
    return out

def within_day_auc(scores, y, days):
    """Mean of per-day AUC (only days with both classes counted)."""
    aucs = []
    for d in np.unique(days):
        mask = days == d
        if len(np.unique(y[mask])) < 2:
            continue
        aucs.append(roc_auc_score(y[mask], scores[mask]))
    return float(np.mean(aucs)) if aucs else float("nan")

print("Loading combined embeddings (OP + 1706 + Part2)...", flush=True)
op_e, op_i = load_dir(OP_DIR)
s1_e, s1_i = load_dir(S1706_DIR)
p2_e, p2_i = load_dir(PART2_DIR)
emb = np.concatenate([op_e, s1_e, p2_e])
idx = pd.concat([op_i, s1_i, p2_i], ignore_index=True)
canon_embs = base.build_canon_embeddings(emb, idx)
print(f"Total canons: {len(canon_embs)}", flush=True)

neg_csv = os.path.join(base.CANON_DIR, "neg_all.csv")
results = []
print(f"\n{'Species':<28} {'LR_raw':>14} {'LR_z':>14} {'LR_rank':>14} {'wd_AUC':>10} folds",
      flush=True)
print("-" * 100, flush=True)

for sp_name, pos_file in base.SPECIES:
    pos_csv = os.path.join(base.CANON_DIR, pos_file)
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    X, y, g = [], [], []
    for _, row in pos_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs:
            X.append(canon_embs[c]); y.append(1); g.append(base.deployment_day_group(c))
    for _, row in neg_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs:
            X.append(canon_embs[c]); y.append(0); g.append(base.deployment_day_group(c))
    X = np.array(X, dtype=np.float32); y = np.array(y); g = np.array(g)

    splitter = GroupKFold(n_splits=N_FOLDS)
    aucs_raw, aucs_z, aucs_rank, aucs_wd = [], [], [], []
    for train_idx, test_idx in splitter.split(X, y, groups=g):
        if len(np.unique(y[test_idx])) < 2 or sum(y[train_idx] == 1) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        scores = clf.predict_proba(X[test_idx])[:, 1]
        days_test = g[test_idx]; y_test = y[test_idx]

        aucs_raw.append(roc_auc_score(y_test, scores))
        aucs_z.append(roc_auc_score(y_test, per_day_zscore(scores, days_test)))
        aucs_rank.append(roc_auc_score(y_test, per_day_rank(scores, days_test)))
        aucs_wd.append(within_day_auc(scores, y_test, days_test))

    if not aucs_raw:
        continue
    res = {
        "species": sp_name,
        "lr_raw_mean": float(np.mean(aucs_raw)),
        "lr_raw_std": float(np.std(aucs_raw)),
        "lr_z_mean": float(np.mean(aucs_z)),
        "lr_z_std": float(np.std(aucs_z)),
        "lr_rank_mean": float(np.mean(aucs_rank)),
        "lr_rank_std": float(np.std(aucs_rank)),
        "within_day_auc_mean": float(np.nanmean(aucs_wd)),
        "delta_z_minus_raw": float(np.mean(aucs_z) - np.mean(aucs_raw)),
        "delta_rank_minus_raw": float(np.mean(aucs_rank) - np.mean(aucs_raw)),
        "n_folds": len(aucs_raw),
    }
    results.append(res)
    print(f"{sp_name:<28} {res['lr_raw_mean']:.3f}±{res['lr_raw_std']:.3f}  "
          f"{res['lr_z_mean']:.3f}±{res['lr_z_std']:.3f}  "
          f"{res['lr_rank_mean']:.3f}±{res['lr_rank_std']:.3f}  "
          f"{res['within_day_auc_mean']:.3f}     {res['n_folds']:>2}", flush=True)

df = pd.DataFrame(results)
df.to_csv(OUTDIR/"perday_calibration_compare.csv", index=False)
print(f"\nSaved {OUTDIR}/perday_calibration_compare.csv", flush=True)
