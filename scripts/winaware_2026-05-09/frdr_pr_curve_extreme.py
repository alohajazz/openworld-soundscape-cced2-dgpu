#!/usr/bin/env python3
"""FRDR PR curve — extreme threshold + top-N precision analysis.

Goal: empirically determine whether Precision can approach 100%.

Adds to frdr_pr_curve.py:
  - Sweeps theta up to 20 (vs 6) to see Precision asymptote
  - Computes top-N precision: among top-K most confident events,
    what fraction are TP? K ∈ {1, 5, 10, 25, 50, 100, 200, 500}
  - Uses class_weight=None (matches original frdr_label_efficiency.py protocol)

Output:
  pr_curve_extreme.csv         — extended threshold sweep
  pr_curve_extreme_summary.csv — top-N precision table
  pr_curve_extreme.png/pdf     — asymptote + top-N panels
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

EMB_DIR = "/workspace/embeddings/frdr_fulldata_winaware"
MAN_CSV = "/workspace/data/externaldata/frdr_upcall/manifests/frdr_B_continuous_hop2s.csv"
ANN_CSV = "/workspace/data/externaldata/frdr_upcall/raw/data/continuous/dataset_B/annotations_B_cont.csv"
OUTDIR = Path("/workspace/scripts/winaware_2026-05-09/frdr_pr_curve_extreme_2026-05-09")
OUTDIR.mkdir(parents=True, exist_ok=True)

K_CONSEC = 2
GAP_SEC = 3.0
TOL_SEC = 10.0
SMOOTH_WIN = 3
N_FOLDS = 5

THETAS = np.concatenate([
    np.linspace(0.0, 6.0, 31),
    np.linspace(6.5, 20.0, 28),
])
THETAS = np.unique(np.round(THETAS, 2))

TOP_KS = [1, 5, 10, 25, 50, 100, 200, 500]


def mad(x):
    return float(np.median(np.abs(x - np.median(x))) + 1e-6)

def smooth_same(x, win):
    if win <= 1: return x.astype(np.float32)
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

def extract_events(centers, scores, theta):
    hi = scores >= theta
    n = len(scores)
    evts = []
    i = 0
    while i < n:
        if not hi[i]:
            i += 1; continue
        j = i
        while j + 1 < n and hi[j + 1] and (centers[j + 1] - centers[j]) <= GAP_SEC:
            j += 1
        if j - i + 1 >= K_CONSEC:
            kmax = i + int(np.argmax(scores[i : j + 1]))
            evts.append((float(centers[kmax]), float(scores[kmax])))
        i = j + 1
    return evts

def match_1to1(pred_times, gt_times, tol):
    pred = np.array(sorted(pred_times), dtype=float)
    gt = np.array(sorted(gt_times), dtype=float)
    i = j = TP = 0
    while i < len(gt) and j < len(pred):
        if pred[j] < gt[i] - tol: j += 1
        elif pred[j] > gt[i] + tol: i += 1
        else:
            TP += 1; i += 1; j += 1
    return TP, len(pred) - TP, len(gt) - TP


print("[1/4] Loading...", flush=True)
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
print(f"  E={E.shape}, GT={total_GT}, hours={hours:.2f}")

groups = m["base"].values
gkf = GroupKFold(n_splits=N_FOLDS)
fold_indices = list(gkf.split(E, np.zeros(len(E)), groups=groups))

print("[2/4] 5-fold Promoter (no class_weight)...", flush=True)
s_oof = np.zeros(len(m), dtype=np.float32)
for fold, (tr, te) in enumerate(fold_indices):
    train_ann = ann[ann["base"].isin(set(np.unique(groups[tr])))].reset_index(drop=True)
    y_train = np.zeros(len(tr), dtype=np.int8)
    tr_pos_lookup = {gi: li for li, gi in enumerate(tr)}
    for _, row in train_ann.iterrows():
        base = row["base"]; gt_t = float(row[tc])
        file_idx = m[m["base"] == base].index.values
        if len(file_idx) == 0: continue
        centers = m.loc[file_idx, "center_sec"].values
        within_tol = file_idx[np.abs(centers - gt_t) <= TOL_SEC]
        for gi in within_tol:
            li = tr_pos_lookup.get(gi)
            if li is not None: y_train[li] = 1
    if y_train.sum() == 0: continue
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")  # no class_weight
    clf.fit(E[tr], y_train)
    s_oof[te] = clf.predict_proba(E[te])[:, 1].astype(np.float32)

s_norm = per_file_norm(m, s_oof, SMOOTH_WIN)
m["s_norm"] = s_norm
np.save(OUTDIR / "s_oof.npy", s_oof)
np.save(OUTDIR / "s_norm.npy", s_norm)

cache = {b: (g["center_sec"].values, g["s_norm"].values) for b, g in m.groupby("base")}

print("[3/4] Threshold sweep up to theta=20...", flush=True)
records = []
for theta in THETAS:
    TP_total = FP_total = FN_total = 0
    pred_total = 0
    for base, gt_times in gt_by_file.items():
        if base in cache:
            cc, ss = cache[base]
            evts = extract_events(cc, ss, theta)
            pred_times = [e[0] for e in evts]
        else:
            pred_times = []
        tp, fp, fn = match_1to1(pred_times, gt_times, TOL_SEC)
        TP_total += tp; FP_total += fp; FN_total += fn
        pred_total += len(pred_times)
    R = TP_total / max(1, TP_total + FN_total)
    P = TP_total / max(1, TP_total + FP_total)
    fp_h = FP_total / hours
    records.append({"theta": theta, "TP": TP_total, "FP": FP_total, "FN": FN_total,
                    "n_pred": pred_total, "recall": R, "precision": P, "fp_h": fp_h})
df = pd.DataFrame(records)
df.to_csv(OUTDIR / "pr_curve_extreme.csv", index=False)
print(f"  Saved {len(df)} thresholds")
print(f"  Max Precision: {df['precision'].max():.3f} at theta={df.loc[df['precision'].idxmax(),'theta']:.2f}")

print("[4/4] Top-K precision (across all extracted events)...", flush=True)
# Extract events at very low theta (theta=0), then sort by peak score, top-K precision
all_events = []
for base, gt_times in gt_by_file.items():
    if base not in cache: continue
    cc, ss = cache[base]
    evts = extract_events(cc, ss, theta=0.0)  # low theta, harvest all candidates
    for c, s in evts:
        all_events.append({"base": base, "center": c, "peak": s})
all_events_df = pd.DataFrame(all_events).sort_values("peak", ascending=False).reset_index(drop=True)
print(f"  Total candidate events at theta=0: {len(all_events_df)}")

# For each top-K, compute precision against GT
topk_results = []
gt_lookup = {b: sorted(t) for b, t in gt_by_file.items()}
for K in TOP_KS + [len(all_events_df)]:
    if K > len(all_events_df): K = len(all_events_df)
    top = all_events_df.iloc[:K]
    # Match top-K to GT (per-file 1-to-1 with tolerance)
    TP = 0
    used_gt = {b: [False] * len(g) for b, g in gt_lookup.items()}
    for _, ev in top.iterrows():
        base = ev["base"]; c = ev["center"]
        gt_list = gt_lookup.get(base, [])
        used = used_gt.get(base, [])
        for gi, gt_t in enumerate(gt_list):
            if used[gi]: continue
            if abs(c - gt_t) <= TOL_SEC:
                TP += 1
                used[gi] = True
                break
    P = TP / max(1, K)
    R = TP / max(1, total_GT)
    topk_results.append({"K": K, "TP": TP, "FP": K - TP, "precision": P, "recall": R})

topk_df = pd.DataFrame(topk_results)
topk_df.to_csv(OUTDIR / "pr_curve_extreme_topk.csv", index=False)
print("\n=== Top-K Precision ===")
print("  K       TP      FP    Precision  Recall")
for r in topk_results:
    print(f"  {r['K']:>5}  {r['TP']:>5}  {r['FP']:>5}  {r['precision']:.3f}      {r['recall']:.3f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(df["theta"], df["precision"], "o-", color="tab:blue", markersize=3)
ax.set_xlabel("Threshold (theta)"); ax.set_ylabel("Precision")
ax.set_title("Precision asymptote vs threshold")
ax.set_ylim(0, 1); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(df["recall"], df["precision"], "o-", color="tab:green", markersize=3)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Full PR curve (no class_weight)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)

ax = axes[2]
ax.semilogx(topk_df["K"], topk_df["precision"], "o-", color="tab:red", markersize=5)
ax.set_xlabel("Top-K events (log)"); ax.set_ylabel("Precision @ K")
ax.set_title("Top-K precision (sorted by peak score)")
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="P=1.0")
ax.legend()

fig.tight_layout()
fig.savefig(OUTDIR / "pr_curve_extreme.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTDIR / "pr_curve_extreme.pdf", bbox_inches="tight")
print(f"\nSaved plots to {OUTDIR}")
