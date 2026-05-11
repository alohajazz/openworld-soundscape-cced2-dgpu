#!/usr/bin/env python3
"""HICEAS Promoter label-efficiency curve (per-species, 7-species PR set).

Mirrors the FRDR label-efficiency analysis: simulates a sparse-label
scenario by sub-sampling N positive 60-s recordings for Promoter training,
under the same 5-fold StratifiedKFold protocol used in Table 4
(matching paper §4.4.3 / §4.5).

For each species and each N in [10, 30, 100, 300, "all"], with 5 random
seeds (1 seed for "all"), train an L2-regularised LogReg on the BEATs+DAPT
recording-mean embeddings and report AUC mean ± std (cross-fold mean).

Realistic sparse-label simulation:
  - Sub-sample N positives from the training fold's positive recordings.
  - Unsampled training positives are treated as negatives (noisy negative
    supervision), matching FRDR experiment treatment.
  - Test fold uses the full ground-truth labels (pos=1, neg=0).

Inputs (must match groupkfold_table4_eval_winaware.py):
  DAPT_OP_DIR   = /workspace/embeddings/hiceas_op_fulldata_winaware
  DAPT_1706_DIR = /workspace/embeddings/hiceas_1706_fulldata_winaware
  CANON_DIR     = /workspace/revision1_species_canons

Output (under user-writable /workspace/scripts/winaware_2026-05-09/hiceas_label_efficiency_2026-05-09):
  hiceas_label_efficiency.csv      — per-(species, N, seed) row
  hiceas_label_efficiency_agg.csv  — mean ± std aggregated over seeds
  hiceas_label_efficiency.png/pdf  — 7-species curve panel
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Reuse base evaluator's data loaders + canon utilities
sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base  # noqa: E402

# --------------------------------------------------------------------------
# Paths (winaware) — match groupkfold_table4_eval_winaware.py
# --------------------------------------------------------------------------
DAPT_OP_DIR = "/workspace/embeddings/hiceas_op_fulldata_winaware"
DAPT_1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata_winaware"
CANON_DIR = "/workspace/revision1_species_canons"
OUTDIR = Path(
    "/workspace/scripts/winaware_2026-05-09/hiceas_label_efficiency_2026-05-09"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# Species: 7 PR set (matches paper Table 4)
SPECIES_PR = base.SPECIES  # 7-species list

N_SAMPLES = [10, 30, 100, 300, 1000, "all"]
N_SEEDS = 5  # 1 seed for "all"
N_FOLDS = 5
SPLIT_SEED = 42  # fixed StratifiedKFold seed (so folds are stable across N/seed)


# --------------------------------------------------------------------------
def evaluate_label_budget(canon_embs, pos_csv, neg_csv, N, sub_seed):
    """Train Promoter on N sub-sampled positive recordings (rest treated as
    negatives), evaluate AUC on the held-out fold via 5-fold StratifiedKFold.

    Returns (auc_mean, auc_std, n_folds_used, n_pos_total) or None.
    """
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)

    X, y = [], []
    for _, row in pos_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs:
            X.append(canon_embs[c])
            y.append(1)
    for _, row in neg_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs:
            X.append(canon_embs[c])
            y.append(0)
    X = np.array(X)
    y = np.array(y)
    if len(X) == 0 or sum(y) == 0:
        return None

    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED
    )
    rng = np.random.default_rng(sub_seed)

    aucs = []
    for train_idx, test_idx in splitter.split(X, y):
        train_pos_idx = train_idx[y[train_idx] == 1]
        if N == "all" or N >= len(train_pos_idx):
            selected_pos = set(train_pos_idx)
        else:
            selected_pos = set(rng.choice(train_pos_idx, N, replace=False))

        y_train_mod = np.zeros(len(train_idx), dtype=int)
        for i, ti in enumerate(train_idx):
            if ti in selected_pos:
                y_train_mod[i] = 1

        if y_train_mod.sum() < 2:
            continue
        if len(np.unique(y[test_idx])) < 2:
            continue

        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y_train_mod)
        y_prob = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], y_prob))

    if not aucs:
        return None
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_folds_used": len(aucs),
        "n_pos_total": int(sum(y == 1)),
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

print("[2/4] Running label-efficiency sweep across 7 species...", flush=True)
results = []
for sp_name, pos_file in SPECIES_PR:
    pos_csv = f"{CANON_DIR}/{pos_file}"
    print(f"\n--- {sp_name} ---", flush=True)
    for N in N_SAMPLES:
        n_seeds = 1 if N == "all" else N_SEEDS
        for seed in range(n_seeds):
            r = evaluate_label_budget(canon_embs, pos_csv, neg_csv, N, seed)
            if r is None:
                print(f"  N={N} seed={seed}: SKIP (no valid folds)")
                continue
            results.append(
                {
                    "species": sp_name,
                    "N": str(N),
                    "seed": seed,
                    **r,
                }
            )
            print(
                f"  N={N} seed={seed}: AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f} "
                f"(n_pos_total={r['n_pos_total']}, folds={r['n_folds_used']})",
                flush=True,
            )

print("\n[3/4] Saving CSVs...", flush=True)
df_r = pd.DataFrame(results)
df_r.to_csv(OUTDIR / "hiceas_label_efficiency.csv", index=False)

agg = (
    df_r.groupby(["species", "N"], sort=False)
    .agg(
        auc_mean=("auc_mean", "mean"),
        auc_std=("auc_mean", "std"),  # std across seeds
        n_seeds=("seed", "count"),
        n_pos_total=("n_pos_total", "first"),
    )
    .reset_index()
)
agg.to_csv(OUTDIR / "hiceas_label_efficiency_agg.csv", index=False)
print(agg)

print("\n[4/4] Plotting per-species panel...", flush=True)
order_idx = ["10", "30", "100", "300", "1000", "all"]
species_list = [s for s, _ in SPECIES_PR]

fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
axes = axes.flatten()
for i, sp_name in enumerate(species_list):
    ax = axes[i]
    sub = agg[agg["species"] == sp_name].set_index("N").reindex(order_idx).reset_index()
    xs = list(range(len(sub)))
    ax.errorbar(
        xs,
        sub["auc_mean"],
        yerr=sub["auc_std"].fillna(0),
        fmt="o-",
        capsize=3,
        color="tab:blue",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(sub["N"], rotation=45)
    n_pos = sub["n_pos_total"].dropna().iloc[0] if not sub["n_pos_total"].dropna().empty else "?"
    ax.set_title(f"{sp_name}\n(N_pos = {int(n_pos) if n_pos != '?' else '?'})", fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.4)
    if i % 4 == 0:
        ax.set_ylabel("AUC (5-fold CV mean)")

# Hide 8th panel
axes[-1].set_visible(False)
fig.suptitle("HICEAS Promoter label-efficiency curve (7-species PR set)", y=1.02, fontsize=11)
fig.supxlabel("N labelled positive recordings used for Promoter training")
fig.tight_layout()
fig.savefig(OUTDIR / "hiceas_label_efficiency.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTDIR / "hiceas_label_efficiency.pdf", bbox_inches="tight")
print(f"Saved to {OUTDIR}", flush=True)
