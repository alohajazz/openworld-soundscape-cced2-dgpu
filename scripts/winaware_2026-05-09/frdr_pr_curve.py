#!/usr/bin/env python3
"""FRDR Promoter Precision-Recall (PR) curve.

Trains Promoter on full GT (5-fold GroupKFold by file), gets out-of-fold
scores, then sweeps threshold over a wide range to produce:
  - Precision-Recall curve
  - FP/h vs Recall curve

Identical preprocessing/post-processing to frdr_label_efficiency.py:
  - L2-regularised LogReg on BEATs+DAPT 768-dim winaware embeddings
  - Per-file MAD-normalised + 3-point smoothed score
  - Event extraction: K_CONSEC=2 consecutive ≥theta, GAP_SEC=3.0
  - Match: ±TOL_SEC=10s 1-to-1 against GT

Output:
  pr_curve_full.csv  — per (theta) row: TP, FP, FN, recall, fp_h, precision
  pr_curve.png/pdf   — PR curve + FP/h vs Recall curve (2 panel)
  pr_curve_summary.csv — interpolated values at FP/h ∈ {1, 5, 10, 20, 50, 100}
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
OUTDIR = Path("/workspace/scripts/winaware_2026-05-09/frdr_pr_curve_2026-05-09")
OUTDIR.mkdir(parents=True, exist_ok=True)

K_CONSEC = 2
GAP_SEC = 3.0
TOL_SEC = 10.0
SMOOTH_WIN = 3
N_FOLDS = 5

# Threshold sweep — log-spaced to cover both tails
THETAS = np.concatenate([
    np.linspace(-2.0, 0.0, 11),
    np.linspace(0.5, 6.0, 56),
])
THETAS = np.unique(np.round(THETAS, 3))


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

def extract_events(centers, scores, theta, k_consec=K_CONSEC, gap_sec=GAP_SEC):
    hi = scores >= theta
    n = len(scores)
    evts = []
    i = 0
    while i < n:
        if not hi[i]:
            i += 1; continue
        j = i
        while j + 1 < n and hi[j + 1] and (centers[j + 1] - centers[j]) <= gap_sec:
            j += 1
        if j - i + 1 >= k_consec:
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

print("[2/4] Training 5-fold Promoter and assembling OOF scores...", flush=True)
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
    if y_train.sum() == 0:
        print(f"  fold {fold}: no positives in train, skip", flush=True)
        continue
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced")
    clf.fit(E[tr], y_train)
    s_oof[te] = clf.predict_proba(E[te])[:, 1].astype(np.float32)
    print(f"  fold {fold}: train+={y_train.sum()} test={len(te)}", flush=True)

print("[3/4] Per-file MAD-normalize + smooth, then sweep thresholds...", flush=True)
s_norm = per_file_norm(m, s_oof, SMOOTH_WIN)
m["s_norm"] = s_norm

cache = {}
for base, g in m.groupby("base"):
    cache[base] = (g["center_sec"].values, g["s_norm"].values)

records = []
for theta in THETAS:
    TP_total = FP_total = FN_total = pred_total = 0
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
    recall = TP_total / max(1, TP_total + FN_total)
    fp_h = FP_total / hours
    precision = TP_total / max(1, TP_total + FP_total)
    records.append({
        "theta": theta, "TP": TP_total, "FP": FP_total, "FN": FN_total,
        "n_pred": pred_total, "recall": recall, "fp_h": fp_h, "precision": precision,
    })
    print(f"  theta={theta:.2f}  TP={TP_total:>4} FP={FP_total:>5} "
          f"FN={FN_total:>4}  R={recall:.3f}  FP/h={fp_h:.2f}  P={precision:.3f}",
          flush=True)

df = pd.DataFrame(records)
df.to_csv(OUTDIR / "pr_curve_full.csv", index=False)
print(f"\nSaved {OUTDIR}/pr_curve_full.csv ({len(df)} thresholds)", flush=True)

print("[4/4] Plotting + interpolating at fixed FP/h...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel 1: PR curve
ax = axes[0]
df_s = df.sort_values("recall").reset_index(drop=True)
ax.plot(df_s["recall"], df_s["precision"], "o-", color="tab:blue", markersize=4)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("FRDR Promoter PR curve\n(BEATs+DAPT, full GT, 5-fold GroupKFold by file)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

# Panel 2: FP/h vs Recall
ax = axes[1]
df_s = df.sort_values("fp_h").reset_index(drop=True)
ax.semilogx(df_s["fp_h"].clip(lower=0.01), df_s["recall"], "o-", color="tab:red", markersize=4)
ax.axvline(10, color="gray", linestyle="--", alpha=0.5, label="paper FP/h=10")
ax.set_xlabel("FP per hour (log scale)"); ax.set_ylabel("Recall")
ax.set_title("Recall vs FP/h trade-off")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3); ax.legend()

fig.tight_layout()
fig.savefig(OUTDIR / "pr_curve.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTDIR / "pr_curve.pdf", bbox_inches="tight")

# Interpolate at fixed FP/h
target_fps = [1, 5, 10, 20, 50, 100]
df_uniq = df.drop_duplicates(subset="fp_h").sort_values("fp_h")
summary = []
for tfp in target_fps:
    if tfp < df_uniq["fp_h"].min() or tfp > df_uniq["fp_h"].max():
        summary.append({"fp_h_target": tfp, "recall": float("nan"),
                        "precision": float("nan"), "note": "out of range"})
        continue
    r_int = float(np.interp(tfp, df_uniq["fp_h"], df_uniq["recall"]))
    p_int = float(np.interp(tfp, df_uniq["fp_h"], df_uniq["precision"]))
    summary.append({"fp_h_target": tfp, "recall": r_int, "precision": p_int, "note": ""})
pd.DataFrame(summary).to_csv(OUTDIR / "pr_curve_summary.csv", index=False)
print("\n=== Interpolated at fixed FP/h ===", flush=True)
for r in summary:
    print(f"  FP/h={r['fp_h_target']:>4}  Recall={r['recall']:.3f}  "
          f"Precision={r['precision']:.3f}  {r['note']}", flush=True)
print(f"\nSaved {OUTDIR}", flush=True)
