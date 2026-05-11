"""HICEAS sample-efficiency post-Part2 with deployment-day sub-sampling.

Protocol:
  - Outer split: 5-fold GroupKFold by deployment-day (cross-day generalization)
  - Sub-sampling: select N_days from training fold's positive days
  - Within selected days, use ALL positive recordings (realistic encounter labelling
    scenario where an expert labels every vocalisation of an attended encounter)
  - N_days values: 1, 2, 4, 8, 16, "all" (capped at species' available days)
  - Multiple seeds: 10 (for variance estimation; "all" uses 1 seed)

Output:
  hiceas_label_efficiency_postpart2.csv (per-(species, N, seed))
  hiceas_label_efficiency_postpart2_agg.csv (mean ± std over seeds)
  hiceas_label_efficiency_postpart2.png/pdf (per-species curve panel)
"""
import sys, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base

OP_DIR = "/workspace/embeddings/hiceas_op_fulldata_winaware"
S1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata_winaware"
PART2_DIR = "/workspace/embeddings/hiceas_1706_part2_fulldata_winaware"
OUTDIR = Path("/workspace/scripts/winaware_2026-05-09/hiceas_label_efficiency_postpart2_2026-05-09")
OUTDIR.mkdir(parents=True, exist_ok=True)

N_DAYS_VALUES = [1, 2, 4, 8, 16, "all"]
N_SEEDS = 10
N_FOLDS = 5

def load_dir(emb_dir):
    paths = sorted(glob.glob(f"{emb_dir}/embeddings_*.npy"))
    idxs = sorted(glob.glob(f"{emb_dir}/index_*.csv"))
    embs = np.concatenate([np.load(p) for p in paths]).astype("float32")
    idx = pd.concat([pd.read_csv(p) for p in idxs], ignore_index=True)
    return embs, idx

def evaluate_day_budget(canon_embs, pos_csv, neg_csv, N_days, sub_seed):
    """Sub-sample N_days from training fold's positive days; use all
    positive recordings within selected days. Cross-day GroupKFold."""
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
    X = np.array(X); y = np.array(y); g = np.array(g)
    if sum(y == 1) == 0:
        return None
    pos_days_total = len(np.unique(g[y == 1]))

    splitter = GroupKFold(n_splits=N_FOLDS)
    rng = np.random.default_rng(sub_seed)
    aucs = []
    n_days_used_per_fold = []
    for train_idx, test_idx in splitter.split(X, y, groups=g):
        train_pos_idx = train_idx[y[train_idx] == 1]
        if len(train_pos_idx) == 0:
            continue
        train_pos_days = np.unique(g[train_pos_idx])
        if N_days == "all" or N_days >= len(train_pos_days):
            selected_days = set(train_pos_days)
        else:
            selected_days = set(rng.choice(train_pos_days, int(N_days), replace=False))
        # Use ALL positive recordings within selected days
        keep_mask = np.zeros(len(train_idx), dtype=bool)
        for li, ti in enumerate(train_idx):
            if y[ti] == 1 and g[ti] in selected_days:
                keep_mask[li] = True
            elif y[ti] == 0:
                keep_mask[li] = True  # negatives all kept
        y_train_mod = np.where(
            np.isin(g[train_idx], list(selected_days)) & (y[train_idx] == 1), 1, 0
        ).astype(np.int8)
        if y_train_mod.sum() < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y_train_mod)
        y_prob = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], y_prob))
        n_days_used_per_fold.append(len(selected_days))
    if not aucs:
        return None
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_folds_used": len(aucs),
        "n_pos_days_total": pos_days_total,
        "n_days_used_median": int(np.median(n_days_used_per_fold)),
    }

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
for sp_name, pos_file in base.SPECIES:
    pos_csv = os.path.join(base.CANON_DIR, pos_file)
    print(f"\n--- {sp_name} ---", flush=True)
    for N in N_DAYS_VALUES:
        n_seeds = 1 if N == "all" else N_SEEDS
        for seed in range(n_seeds):
            r = evaluate_day_budget(canon_embs, pos_csv, neg_csv, N, seed)
            if r is None:
                continue
            results.append({"species": sp_name, "N_days": str(N), "seed": seed, **r})
            print(f"  N={N} seed={seed}: AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f} "
                  f"(days_total={r['n_pos_days_total']}, days_used={r['n_days_used_median']})",
                  flush=True)

df_r = pd.DataFrame(results)
df_r.to_csv(OUTDIR/"hiceas_label_efficiency_postpart2.csv", index=False)
agg = (df_r.groupby(["species", "N_days"], sort=False)
       .agg(auc_mean=("auc_mean", "mean"), auc_std=("auc_mean", "std"),
            n_seeds=("seed", "count"), n_pos_days_total=("n_pos_days_total", "first"),
            n_days_used_median=("n_days_used_median", "median"))
       .reset_index())
agg.to_csv(OUTDIR/"hiceas_label_efficiency_postpart2_agg.csv", index=False)
print(agg)

order_idx = ["1", "2", "4", "8", "16", "all"]
species_list = [s for s, _ in base.SPECIES]
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
axes = axes.flatten()
for i, sp_name in enumerate(species_list):
    ax = axes[i]
    sub = (agg[agg["species"] == sp_name].set_index("N_days").reindex(order_idx).reset_index())
    xs = list(range(len(sub)))
    ax.errorbar(xs, sub["auc_mean"], yerr=sub["auc_std"].fillna(0),
                fmt="o-", capsize=3, color="tab:green")
    ax.set_xticks(xs); ax.set_xticklabels(sub["N_days"], rotation=45)
    n_days = sub["n_pos_days_total"].dropna().iloc[0] if not sub["n_pos_days_total"].dropna().empty else "?"
    ax.set_title(f"{sp_name}\n(N_days_total = {int(n_days) if n_days != '?' else '?'})", fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3); ax.axhline(0.9, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)
    if i % 4 == 0:
        ax.set_ylabel("AUC (5-fold GroupKFold by day)")
axes[-1].set_visible(False)
fig.suptitle("HICEAS Promoter — Post-Part2 day-level label-efficiency\n"
             "(all positive recordings within selected days; cross-day GroupKFold)",
             y=1.02, fontsize=11)
fig.supxlabel("N_days = number of independent deployment-days labelled")
fig.tight_layout()
fig.savefig(OUTDIR/"hiceas_label_efficiency_postpart2.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTDIR/"hiceas_label_efficiency_postpart2.pdf", bbox_inches="tight")
print(f"Saved to {OUTDIR}", flush=True)
