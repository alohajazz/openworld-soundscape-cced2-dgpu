#!/usr/bin/env python3
import os as _os  # 2026-07-31 コピー用（原本無変更）
"""FRDR Quiet/Promoter/Union with CORRECT paper protocol:
- Quiet = CCED2 (per-file MAD norm + smooth, percentile threshold)
- Promoter = supervised LogReg trained with GT annotations (Pos = window within ±10s of GT)
- Union = logical OR of Quiet ∪ Promoter at separate thresholds
- 5-fold cross-validation for Promoter (to avoid train/test overlap on file basis)
"""
import os, json, glob, joblib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

# 2026-07-31 コピー。原本 = scripts/winaware_2026-05-09/frdr_supervised_promoter.py（無変更）。
# 環境変数で encoder と InD 参照のみ差し替え可能にした。他は一切変えていない。
_EMB = _os.environ.get("FR_EMB", "/workspace/embeddings/frdr_fulldata_winaware")
EMB_DIR = _EMB
MAN_CSV = "/workspace/data/externaldata/frdr_upcall/manifests/frdr_B_continuous_hop2s.csv"
ANN_CSV = "/workspace/data/externaldata/frdr_upcall/raw/data/continuous/dataset_B/annotations_B_cont.csv"
_REF = _os.environ.get("FR_REF", "")
KNN_PKL = (_REF + "/knn_cced2.pkl") if _REF else "/workspace/embeddings/known56/models/knn_dapt.pkl"
MAHA_PKL = (_REF + "/maha_cced2.pkl") if _REF else "/workspace/embeddings/known56/models/maha_dapt.pkl"
NORM_JSON = (_REF + "/cced2_norm.json") if _REF else "/workspace/embeddings/cced2_norm.json"
OUTDIR = Path(_os.environ.get("FR_OUT", "/workspace/scripts/audit_2026-07-30/t2_rerun"))
OUTDIR.mkdir(parents=True, exist_ok=True)

K_CONSEC = 2
GAP_SEC = 3.0
TOL_SEC = 10.0
SMOOTH_WIN = 3
TARGET_FP = 10.0
N_FOLDS = 5

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
    n = len(scores); evts = []; i = 0
    while i < n:
        if not hi[i]: i += 1; continue
        j = i
        while j+1 < n and hi[j+1] and (centers[j+1] - centers[j]) <= gap_sec:
            j += 1
        if j - i + 1 >= k_consec:
            kmax = i + int(np.argmax(scores[i:j+1]))
            evts.append({"center": float(centers[kmax]), "peak": float(scores[kmax])})
        i = j + 1
    return evts

def match_1to1(pred_times, gt_times, tol):
    pred = np.array(sorted(pred_times), dtype=float)
    gt = np.array(sorted(gt_times), dtype=float)
    i = j = TP = 0
    while i < len(gt) and j < len(pred):
        if pred[j] < gt[i] - tol: j += 1
        elif pred[j] > gt[i] + tol: i += 1
        else: TP += 1; i += 1; j += 1
    return TP, len(pred) - TP, len(gt) - TP

def eval_at(cache, gt_by_file, theta, hours):
    TP=FP=FN=0; pred_total=0
    for base, gt_times in gt_by_file.items():
        if base in cache:
            cc, ss = cache[base]
            evts = extract_events(cc, ss, theta)
            pred_times = [e["center"] for e in evts]
        else:
            pred_times = []
        tp, fp, fn = match_1to1(pred_times, gt_times, TOL_SEC)
        TP += tp; FP += fp; FN += fn; pred_total += len(pred_times)
    return TP / max(1, TP+FN), FP / hours, TP, FP, FN, pred_total

def eval_union(cache_q, cache_p, gt_by_file, theta_q, theta_p, hours):
    TP=FP=FN=0; pred_total=0
    for base, gt_times in gt_by_file.items():
        q_evts = extract_events(*cache_q[base], theta_q) if base in cache_q else []
        p_evts = extract_events(*cache_p[base], theta_p) if base in cache_p else []
        # Combine with NMS within gap_sec
        combined = sorted([(e["center"], e["peak"]) for e in q_evts] + [(e["center"], e["peak"]) for e in p_evts])
        kept = []
        for t, s in combined:
            if not kept or t - kept[-1][0] >= GAP_SEC:
                kept.append((t, s))
        pred_times = [t for t, _ in kept]
        tp, fp, fn = match_1to1(pred_times, gt_times, TOL_SEC)
        TP += tp; FP += fp; FN += fn; pred_total += len(pred_times)
    return TP / max(1, TP+FN), FP / hours, TP, FP, FN, pred_total

# ---- Load ----
print("[1/7] Loading embeddings, manifest, annotations...", flush=True)
E = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{EMB_DIR}/embeddings_*.npy"))]).astype("float32")
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
print(f"  E={E.shape}, manifest={len(m)}, GT={total_GT}")
hours = sum((g["center_sec"].max() + 5.0) / 3600.0 for _, g in m.groupby("base"))
print(f"  total hours: {hours:.2f}")

# ---- 1. CCED2 + per-file norm ----
print("[2/7] Computing CCED2 + per-file MAD norm...", flush=True)
KNN = joblib.load(KNN_PKL); MAHA = joblib.load(MAHA_PKL); cfg = json.load(open(NORM_JSON))
dists, _ = KNN["knn"].kneighbors(E)
knn_raw = dists.mean(1)
D = E - np.asarray(MAHA["mu"])
maha_raw = np.sqrt((D @ MAHA["precision"] * D).sum(1))
cced2 = (knn_raw - cfg["mk"])/cfg["sk"] + (maha_raw - cfg["mm"])/cfg["sm"]
S_q = per_file_norm(m, cced2, SMOOTH_WIN)

cache_q = {}
for base, idx in m.groupby("base").groups.items():
    idx = np.array(list(idx))
    cc = m.loc[idx, "center_sec"].to_numpy()
    order = np.argsort(cc)
    cache_q[base] = (cc[order], S_q[idx[order]])

# ---- 2. GT supervised labels ----
print("[3/7] Building supervised labels (window within ±10s of GT)...", flush=True)
y = np.zeros(len(m), dtype=np.int8)
for base, gts in ann.groupby("base"):
    file_idx = m[m["base"] == base].index.values
    if len(file_idx) == 0: continue
    centers = m.loc[file_idx, "center_sec"].values
    for gt in gts[tc].astype(float).values:
        y[file_idx[np.abs(centers - gt) <= TOL_SEC]] = 1
print(f"  Pos windows: {y.sum()}, Neg windows: {len(y) - y.sum()}, Pos rate: {y.mean():.3f}")

# ---- 3. Train Promoter via 5-fold GroupKFold (groups = file base) ----
print("[4/7] Training Promoter LogReg with GroupKFold (5-fold by file)...", flush=True)
groups = m["base"].values
gkf = GroupKFold(n_splits=N_FOLDS)
s_p = np.zeros(len(m), dtype=np.float32)  # out-of-fold predictions
for fold, (tr, te) in enumerate(gkf.split(E, y, groups=groups)):
    lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    lr.fit(E[tr], y[tr])
    s_p[te] = lr.predict_proba(E[te])[:, 1]
    n_files_test = len(np.unique(groups[te]))
    print(f"  Fold {fold}: train={len(tr)}, test={len(te)}, test files={n_files_test}, "
          f"AUROC={(np.corrcoef(y[te], s_p[te])[0,1] if y[te].sum()>0 else float('nan')):.3f}")

# Per-file norm Promoter
S_p = per_file_norm(m, s_p, SMOOTH_WIN)
cache_p = {}
for base, idx in m.groupby("base").groups.items():
    idx = np.array(list(idx))
    cc = m.loc[idx, "center_sec"].to_numpy()
    order = np.argsort(cc)
    cache_p[base] = (cc[order], S_p[idx[order]])

# ---- 4. Quiet sweep ----
print("[5/7] Quiet (CCED2) FP/h-Recall sweep...", flush=True)
quiet_curve = []
for pctl in np.arange(80, 99.51, 0.5):
    th = float(np.percentile(S_q, pctl))
    rec, fph, TP, FP, FN, npred = eval_at(cache_q, gt_by_file, th, hours)
    quiet_curve.append({"pctl": pctl, "theta": th, "Recall": rec, "FP_h": fph, "TP": TP, "FP": FP, "Pred": npred})
df_q = pd.DataFrame(quiet_curve)
df_q.to_csv(OUTDIR/"quiet_sweep.csv", index=False)
quiet_pick = df_q.iloc[(df_q["FP_h"] - TARGET_FP).abs().argsort().iloc[0]]
print(f"  Quiet @ FP/h≈10: Recall={quiet_pick['Recall']:.4f}, FP/h={quiet_pick['FP_h']:.2f}, theta={quiet_pick['theta']:.3f}")

# ---- 5. Promoter sweep ----
print("[6/7] Promoter FP/h-Recall sweep...", flush=True)
prom_curve = []
for pctl in np.arange(50, 99.91, 0.5):
    th = float(np.percentile(S_p, pctl))
    rec, fph, TP, FP, FN, npred = eval_at(cache_p, gt_by_file, th, hours)
    prom_curve.append({"pctl": pctl, "theta": th, "Recall": rec, "FP_h": fph, "TP": TP, "FP": FP, "Pred": npred})
df_p = pd.DataFrame(prom_curve)
df_p.to_csv(OUTDIR/"promoter_sweep.csv", index=False)
prom_pick = df_p.iloc[(df_p["FP_h"] - TARGET_FP).abs().argsort().iloc[0]]
print(f"  Promoter @ FP/h≈10: Recall={prom_pick['Recall']:.4f}, FP/h={prom_pick['FP_h']:.2f}, theta={prom_pick['theta']:.3f}")

# ---- 6. Union sweep ----
print("[7/7] Union (logical OR) sweep...", flush=True)
union_pts = []
for pctl_q in [98, 97, 96, 95, 94, 93, 92, 90]:
    th_q = float(np.percentile(S_q, pctl_q))
    for pctl_p in np.arange(50, 99.91, 0.5):
        th_p = float(np.percentile(S_p, pctl_p))
        rec, fph, TP, FP, FN, npred = eval_union(cache_q, cache_p, gt_by_file, th_q, th_p, hours)
        union_pts.append({"pctl_q": pctl_q, "theta_q": th_q, "pctl_p": pctl_p, "theta_p": th_p,
                          "Recall": rec, "FP_h": fph, "TP": TP, "FP": FP, "Pred": npred})
df_u = pd.DataFrame(union_pts)
df_u.to_csv(OUTDIR/"union_sweep.csv", index=False)
# Best union with FP/h ≤ 10.5 (slightly above target)
df_u_in = df_u[df_u["FP_h"] <= 10.5]
union_pick = df_u_in.iloc[df_u_in["Recall"].argmax()] if len(df_u_in) > 0 else None
if union_pick is not None:
    print(f"  Union @ FP/h≤10.5: Recall={union_pick['Recall']:.4f}, FP/h={union_pick['FP_h']:.2f}, "
          f"theta_q={union_pick['theta_q']:.3f} (pctl={union_pick['pctl_q']}), "
          f"theta_p={union_pick['theta_p']:.3f} (pctl={union_pick['pctl_p']})")

# ---- Summary table ----
rows = [
    {"mode": "Quiet (CCED2)", **{k: float(quiet_pick[k]) for k in ["Recall","FP_h","TP","FP","theta"]}},
    {"mode": "Promoter (supervised LogReg, 5-fold OOF)", **{k: float(prom_pick[k]) for k in ["Recall","FP_h","TP","FP","theta"]}},
]
if union_pick is not None:
    rows.append({"mode": "Union (Quiet ∪ Promoter)",
                 "Recall": float(union_pick["Recall"]), "FP_h": float(union_pick["FP_h"]),
                 "TP": int(union_pick["TP"]), "FP": int(union_pick["FP"]),
                 "theta": f"q={union_pick['theta_q']:.3f}/p={union_pick['theta_p']:.3f}"})
summary = pd.DataFrame(rows)
summary.to_csv(OUTDIR/"frdr_quiet_promoter_union_supervised.csv", index=False)
print("\n=== Summary (winaware, supervised Promoter, paper §4.4.3 protocol) ===")
print(summary.to_string(index=False))

# ---- Plot Fig 3 ----
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
ax.plot(df_q["FP_h"], df_q["Recall"], "o-", label="Quiet (CCED2)", color="tab:blue", markersize=3)
ax.plot(df_p["FP_h"], df_p["Recall"], "s-", label="Promoter (supervised)", color="tab:orange", markersize=3)
df_u_sorted = df_u.sort_values("FP_h")
pareto = []
best = -1
for _, row in df_u_sorted.iterrows():
    if row["Recall"] > best:
        best = row["Recall"]
        pareto.append((row["FP_h"], row["Recall"]))
if pareto:
    px, py = zip(*pareto)
    ax.plot(px, py, "^-", label="Union (logical OR, Pareto)", color="tab:green", markersize=3)
ax.axvline(TARGET_FP, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("FP/h"); ax.set_ylabel("Recall")
ax.set_title("FRDR right whale upcall: Quiet/Promoter/Union (winaware, supervised Promoter)")
ax.set_xlim(0, 30); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUTDIR/"fig3_frdr_supervised.png")
fig.savefig(OUTDIR/"fig3_frdr_supervised.pdf")
plt.close(fig)
print(f"\nWrote outputs to {OUTDIR}")
