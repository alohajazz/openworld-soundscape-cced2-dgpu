#!/usr/bin/env python3
"""HICEAS Promoter label-efficiency curve — STRICT deployment-day version.

Addresses the within-day correlation concern of the recording-level
sub-sampling: each acoustic encounter (operationally proxied by a
deployment-day, e.g. "1705_20170930") may produce many highly correlated
60-s recordings. Random sub-sampling at the recording level therefore
overestimates few-shot performance because multiple "labels" can come
from the same encounter.

Strict protocol implemented here:
  1. Outer split: 5-fold GroupKFold by deployment-day (train/test days
     are disjoint).
  2. Within the training fold, list unique positive deployment-days.
  3. Sub-sample N_days unique days (without replacement).
  4. From each selected day, take exactly ONE random positive recording
     as the labelled positive (y=1).
  5. All other training recordings (positives from unselected days,
     positives from selected days that were not chosen, and negatives)
     are y=0 — i.e. noisy negative supervision.
  6. Train LogReg, evaluate on held-out fold (different days), report AUC.
  7. Repeat 10 random seeds for stable variance at small N.

This simulates the realistic deployment scenario in which an expert
attends N independent encounters during a survey and labels one
representative example per encounter, then deploys the trained Promoter
on different days.

Inputs (winaware embeddings):
  /workspace/embeddings/hiceas_op_fulldata_winaware
  /workspace/embeddings/hiceas_1706_fulldata_winaware
  /workspace/revision1_species_canons/

Output:
  /workspace/scripts/winaware_2026-05-09/hiceas_label_efficiency_strict_2026-05-09/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base  # noqa: E402

DAPT_OP_DIR = "/workspace/embeddings/hiceas_op_fulldata_winaware"
DAPT_1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata_winaware"
CANON_DIR = "/workspace/revision1_species_canons"
OUTDIR = Path(
    "/workspace/scripts/winaware_2026-05-09/hiceas_label_efficiency_strict_2026-05-09"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

SPECIES_PR = base.SPECIES
N_DAYS_VALUES = [1, 2, 4, 8, "all"]
N_SEEDS = 10
N_FOLDS = 5


def evaluate_day_budget(canon_embs, pos_csv, neg_csv, N_days, sub_seed):
    """Train on N_days × 1 random positive recording per day; evaluate AUC
    via 5-fold GroupKFold by deployment-day."""
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)

    X, y, g = [], [], []
    for _, row in pos_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs:
            X.append(canon_embs[c])
            y.append(1)
            g.append(base.deployment_day_group(c))
    for _, row in neg_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs:
            X.append(canon_embs[c])
            y.append(0)
            g.append(base.deployment_day_group(c))
    X = np.array(X)
    y = np.array(y)
    g = np.array(g)
    if sum(y == 1) == 0:
        return None

    n_unique_days_total = len(np.unique(g[y == 1]))

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
            selected_days = train_pos_days
        else:
            selected_days = rng.choice(train_pos_days, int(N_days), replace=False)

        # 1 random positive per selected day
        selected_idx = []
        for day in selected_days:
            day_pos = train_pos_idx[g[train_pos_idx] == day]
            if len(day_pos) > 0:
                selected_idx.append(int(rng.choice(day_pos)))

        if len(selected_idx) < 2:
            continue

        y_train_mod = np.zeros(len(train_idx), dtype=int)
        selected_set = set(selected_idx)
        for i, ti in enumerate(train_idx):
            if ti in selected_set:
                y_train_mod[i] = 1

        if len(np.unique(y[test_idx])) < 2:
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
        "n_days_total": n_unique_days_total,
        "n_days_used_median": int(np.median(n_days_used_per_fold)),
    }


# --------------------------------------------------------------------------
print("[1/4] Loading WINAWARE fulldata DAPT embeddings...", flush=True)
op_emb, op_idx = base.load_sharded(DAPT_OP_DIR)
sp_emb, sp_idx = base.load_sharded(DAPT_1706_DIR)
print(f"  OP={op_emb.shape}  1706={sp_emb.shape}", flush=True)
emb = np.concatenate([op_emb, sp_emb])
idx = pd.concat([op_idx, sp_idx], ignore_index=True)
canon_embs = base.build_canon_embeddings(emb, idx)
print(f"  Loaded {len(canon_embs)} canons", flush=True)

neg_csv = f"{CANON_DIR}/neg_all.csv"

print("[2/4] Running STRICT day-level label-efficiency sweep...", flush=True)
results = []
for sp_name, pos_file in SPECIES_PR:
    pos_csv = f"{CANON_DIR}/{pos_file}"
    print(f"\n--- {sp_name} ---", flush=True)
    for N_days in N_DAYS_VALUES:
        n_seeds = 1 if N_days == "all" else N_SEEDS
        for seed in range(n_seeds):
            r = evaluate_day_budget(canon_embs, pos_csv, neg_csv, N_days, seed)
            if r is None:
                print(f"  N_days={N_days} seed={seed}: SKIP")
                continue
            results.append(
                {
                    "species": sp_name,
                    "N_days": str(N_days),
                    "seed": seed,
                    **r,
                }
            )
            print(
                f"  N_days={N_days} seed={seed}: "
                f"AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f} "
                f"(days_total={r['n_days_total']}, "
                f"days_used_median={r['n_days_used_median']}, "
                f"folds={r['n_folds_used']})",
                flush=True,
            )

print("\n[3/4] Saving CSVs...", flush=True)
df_r = pd.DataFrame(results)
df_r.to_csv(OUTDIR / "hiceas_label_efficiency_strict.csv", index=False)

agg = (
    df_r.groupby(["species", "N_days"], sort=False)
    .agg(
        auc_mean=("auc_mean", "mean"),
        auc_std=("auc_mean", "std"),
        n_seeds=("seed", "count"),
        n_days_total=("n_days_total", "first"),
        n_days_used_median=("n_days_used_median", "median"),
    )
    .reset_index()
)
agg.to_csv(OUTDIR / "hiceas_label_efficiency_strict_agg.csv", index=False)
print(agg)

print("\n[4/4] Plotting per-species panel...", flush=True)
order_idx = ["1", "2", "4", "8", "all"]
species_list = [s for s, _ in SPECIES_PR]

fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
axes = axes.flatten()
for i, sp_name in enumerate(species_list):
    ax = axes[i]
    sub = (
        agg[agg["species"] == sp_name]
        .set_index("N_days")
        .reindex(order_idx)
        .reset_index()
    )
    xs = list(range(len(sub)))
    ax.errorbar(
        xs,
        sub["auc_mean"],
        yerr=sub["auc_std"].fillna(0),
        fmt="o-",
        capsize=3,
        color="tab:purple",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(sub["N_days"], rotation=45)
    n_days = sub["n_days_total"].dropna().iloc[0] if not sub["n_days_total"].dropna().empty else "?"
    ax.set_title(f"{sp_name}\n(N_days_total = {int(n_days) if n_days != '?' else '?'})", fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)  # chance
    if i % 4 == 0:
        ax.set_ylabel("AUC (5-fold GroupKFold by day)")

axes[-1].set_visible(False)
fig.suptitle(
    "HICEAS Promoter — STRICT day-level label-efficiency (1 recording per day)",
    y=1.02,
    fontsize=11,
)
fig.supxlabel("N_days = number of independent deployment-days labelled")
fig.tight_layout()
fig.savefig(OUTDIR / "hiceas_label_efficiency_strict.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTDIR / "hiceas_label_efficiency_strict.pdf", bbox_inches="tight")
print(f"Saved to {OUTDIR}", flush=True)
