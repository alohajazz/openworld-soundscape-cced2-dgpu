#!/usr/bin/env python3
"""FRDR Promoter label-efficiency curve.

Simulates a sparse-label scenario by sub-sampling N positive GT events
for Promoter training (vs. the full-GT baseline) under the same 5-fold
GroupKFold-by-file protocol used in frdr_supervised_promoter.py.

For each N in [10, 30, 100, 300, 1000, "all"], with 5 random seeds (1 seed
for "all"), train an L2-regularised LogReg on BEATs+DAPT embeddings, sweep
the threshold over the per-file MAD-normalised + 3-point smoothed score, and
record Recall at FP/h ≈ 10.

Inputs (assume same as frdr_supervised_promoter.py):
  EMB_DIR, MAN_CSV, ANN_CSV (loaded from /workspace).

Outputs (written to OUTDIR):
  label_efficiency.csv      — per-(N, seed) result row
  label_efficiency_agg.csv  — mean ± std aggregated over seeds
  label_efficiency.png/pdf  — recall vs N plot

Run inside the analysis container:
  python /workspace/release_repo/scripts/winaware_2026-05-09/frdr_label_efficiency.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

# --------------------------------------------------------------------------
# Paths (same as frdr_supervised_promoter.py)
# --------------------------------------------------------------------------
import os as _os
EMB_DIR = _os.environ.get("LE_EMB_DIR","/workspace/embeddings/frdr_fulldata_winaware")
MAN_CSV = "/workspace/data/externaldata/frdr_upcall/manifests/frdr_B_continuous_hop2s.csv"
ANN_CSV = "/workspace/data/externaldata/frdr_upcall/raw/data/continuous/dataset_B/annotations_B_cont.csv"
OUTDIR = Path("/workspace/logs/labeleff_"+_os.environ.get("LE_TAG","x"))
OUTDIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Hyperparameters (matched to frdr_supervised_promoter.py)
# --------------------------------------------------------------------------
K_CONSEC = 2
GAP_SEC = 3.0
TOL_SEC = 10.0
SMOOTH_WIN = 3
TARGET_FP = 10.0
N_FOLDS = 5

# Label-efficiency sweep
N_SAMPLES = [10, 30, 100, 300, 1000, "all"]
N_SEEDS = 5  # 1 seed for "all" (deterministic baseline)


# --------------------------------------------------------------------------
# Helpers (verbatim from frdr_supervised_promoter.py)
# --------------------------------------------------------------------------
def mad(x):
    return float(np.median(np.abs(x - np.median(x))) + 1e-6)


def smooth_same(x, win):
    if win <= 1:
        return x.astype(np.float32)
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same").astype(np.float32)


def per_file_norm(m, S, smooth_win):
    out = np.empty_like(S, dtype=np.float32)
    for base, idx in m.groupby("base").groups.items():
        idx = np.array(list(idx))
        v = S[idx].astype(np.float32)
        v = (v - np.median(v)) / mad(v)
        v = smooth_same(v, smooth_win)
        out[idx] = v
    return out


def extract_events(centers, scores, theta, k_consec=K_CONSEC, gap_sec=GAP_SEC):
    hi = scores >= theta
    n = len(scores)
    evts = []
    i = 0
    while i < n:
        if not hi[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hi[j + 1] and (centers[j + 1] - centers[j]) <= gap_sec:
            j += 1
        if j - i + 1 >= k_consec:
            kmax = i + int(np.argmax(scores[i : j + 1]))
            evts.append({"center": float(centers[kmax]), "peak": float(scores[kmax])})
        i = j + 1
    return evts


def match_1to1(pred_times, gt_times, tol):
    pred = np.array(sorted(pred_times), dtype=float)
    gt = np.array(sorted(gt_times), dtype=float)
    i = j = TP = 0
    while i < len(gt) and j < len(pred):
        if pred[j] < gt[i] - tol:
            j += 1
        elif pred[j] > gt[i] + tol:
            i += 1
        else:
            TP += 1
            i += 1
            j += 1
    return TP, len(pred) - TP, len(gt) - TP


def eval_at(cache, gt_by_file, theta, hours):
    TP = FP = FN = 0
    pred_total = 0
    for base, gt_times in gt_by_file.items():
        if base in cache:
            cc, ss = cache[base]
            evts = extract_events(cc, ss, theta)
            pred_times = [e["center"] for e in evts]
        else:
            pred_times = []
        tp, fp, fn = match_1to1(pred_times, gt_times, TOL_SEC)
        TP += tp
        FP += fp
        FN += fn
        pred_total += len(pred_times)
    return TP / max(1, TP + FN), FP / hours, TP, FP, FN, pred_total


# --------------------------------------------------------------------------
# Load embeddings, manifest, annotations
# --------------------------------------------------------------------------
print("[1/4] Loading embeddings, manifest, annotations...", flush=True)
E = np.concatenate(
    [np.load(p) for p in sorted(glob.glob(f"{EMB_DIR}/embeddings_*.npy"))]
).astype("float32")
m = pd.read_csv(MAN_CSV)
m["base"] = m["path"].apply(lambda p: Path(p).name)
m["center_sec"] = m["center_sec"].astype(float)
ann = pd.read_csv(ANN_CSV, sep=";")
ann.columns = [c.strip().lower() for c in ann.columns]
fc = next(c for c in ann.columns if "file" in c)
tc = next(c for c in ann.columns if "time" in c)
ann["base"] = ann[fc].apply(lambda p: Path(str(p)).name)
gt_by_file = {b: g[tc].astype(float).tolist() for b, g in ann.groupby("base")}
total_GT = sum(len(v) for v in gt_by_file.values())
hours = sum((g["center_sec"].max() + 5.0) / 3600.0 for _, g in m.groupby("base"))
print(f"  E={E.shape}, manifest={len(m)}, GT events={total_GT}, hours={hours:.2f}")

groups = m["base"].values
gkf = GroupKFold(n_splits=N_FOLDS)
fold_indices = list(gkf.split(E, np.zeros(len(E)), groups=groups))


def build_y_for_train_fold(tr_idx, selected_ann):
    """Build y array of length len(tr_idx) where y=1 for windows within
    ±TOL_SEC of any event in selected_ann (a subset of full annotations)."""
    y_train = np.zeros(len(tr_idx), dtype=np.int8)
    tr_pos_lookup = {gi: li for li, gi in enumerate(tr_idx)}
    for _, row in selected_ann.iterrows():
        base = row["base"]
        gt_t = float(row[tc])
        file_idx = m[m["base"] == base].index.values
        if len(file_idx) == 0:
            continue
        centers = m.loc[file_idx, "center_sec"].values
        within_tol = file_idx[np.abs(centers - gt_t) <= TOL_SEC]
        for gi in within_tol:
            li = tr_pos_lookup.get(gi)
            if li is not None:
                y_train[li] = 1
    return y_train


# --------------------------------------------------------------------------
# Sweep label budget × seed
# --------------------------------------------------------------------------
print("[2/4] Running label-efficiency sweep...", flush=True)
results = []
for N in N_SAMPLES:
    n_seeds = 1 if N == "all" else N_SEEDS
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        s_p_oof = np.zeros(len(m), dtype=np.float32)
        skipped_folds = 0
        for fold, (tr, te) in enumerate(fold_indices):
            train_files = set(np.unique(groups[tr]))
            train_ann = ann[ann["base"].isin(train_files)].reset_index(drop=True)
            n_train_events = len(train_ann)

            if N == "all" or N >= n_train_events:
                selected = train_ann
            else:
                idx = rng.choice(n_train_events, N, replace=False)
                selected = train_ann.iloc[idx].reset_index(drop=True)

            y_train = build_y_for_train_fold(tr, selected)
            if y_train.sum() < 2:
                skipped_folds += 1
                continue

            lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
            lr.fit(E[tr], y_train)
            s_p_oof[te] = lr.predict_proba(E[te])[:, 1]

        if skipped_folds == N_FOLDS:
            print(f"  N={N} seed={seed}: all folds skipped, no positives.")
            continue

        S_p = per_file_norm(m, s_p_oof, SMOOTH_WIN)
        cache_p = {}
        for base, idx in m.groupby("base").groups.items():
            idx = np.array(list(idx))
            cc = m.loc[idx, "center_sec"].to_numpy()
            order = np.argsort(cc)
            cache_p[base] = (cc[order], S_p[idx[order]])

        prom_curve = []
        for pctl in np.arange(50, 99.91, 0.5):
            th = float(np.percentile(S_p, pctl))
            rec, fph, TP, FP, FN, npred = eval_at(cache_p, gt_by_file, th, hours)
            prom_curve.append({"pctl": pctl, "theta": th, "Recall": rec, "FP_h": fph})
        df = pd.DataFrame(prom_curve)
        pick = df.iloc[(df["FP_h"] - TARGET_FP).abs().argsort().iloc[0]]
        results.append(
            {
                "N": str(N),
                "seed": seed,
                "recall_at_fp10": float(pick["Recall"]),
                "fp_h": float(pick["FP_h"]),
                "theta": float(pick["theta"]),
                "skipped_folds": skipped_folds,
            }
        )
        print(
            f"  N={N} seed={seed}: recall@FP/h={pick['FP_h']:.2f} = {pick['Recall']:.4f}"
            + (f" (skipped {skipped_folds} folds)" if skipped_folds else "")
        )

# --------------------------------------------------------------------------
# Save + plot
# --------------------------------------------------------------------------
print("[3/4] Saving CSVs...", flush=True)
df_r = pd.DataFrame(results)
df_r.to_csv(OUTDIR / "label_efficiency.csv", index=False)

agg = (
    df_r.groupby("N", sort=False)
    .agg(
        recall_mean=("recall_at_fp10", "mean"),
        recall_std=("recall_at_fp10", "std"),
        fp_h_mean=("fp_h", "mean"),
        n_seeds=("seed", "count"),
    )
    .reset_index()
)
agg.to_csv(OUTDIR / "label_efficiency_agg.csv", index=False)
print(agg)

print("[4/4] Plotting...", flush=True)
order_idx = ["10", "30", "100", "300", "1000", "all"]
agg_plot = agg.set_index("N").reindex(order_idx).reset_index()
fig, ax = plt.subplots(figsize=(6, 4))
xs = list(range(len(agg_plot)))
ax.errorbar(
    xs,
    agg_plot["recall_mean"],
    yerr=agg_plot["recall_std"].fillna(0),
    fmt="o-",
    capsize=3,
    color="tab:orange",
    label="Promoter (sub-sampled labels)",
)
ax.set_xticks(xs)
ax.set_xticklabels(agg_plot["N"])
ax.set_xlabel("N labelled events used for Promoter training")
ax.set_ylabel("Recall @ FP/h ≈ 10")
ax.set_title("FRDR Promoter — label-efficiency curve\n(5 seeds × 5-fold GroupKFold by file)")
ax.set_ylim(0, max(0.5, agg_plot["recall_mean"].max() + 0.05))
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUTDIR / "label_efficiency.png", dpi=150)
fig.savefig(OUTDIR / "label_efficiency.pdf")
print(f"Saved to {OUTDIR}")
