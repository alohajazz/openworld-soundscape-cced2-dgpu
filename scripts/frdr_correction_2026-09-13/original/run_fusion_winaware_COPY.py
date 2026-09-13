#!/usr/bin/env python3
"""Fusion = simple linear fusion of per-file MAD-normalised CCED2 + raw CCED2 ŝ.
Sweep α and theta to find operating point at FP/h ≈ 10."""
import os, json, glob, joblib
import numpy as np, pandas as pd
from pathlib import Path

import os as _os  # 2026-07-31 コピー（原本無変更）
EMB_DIR = _os.environ.get("FU_EMB","/workspace/embeddings/frdr_fulldata_winaware")
_REF=_os.environ.get("FU_REF","")
MAN_CSV = "/workspace/data/externaldata/frdr_upcall/manifests/frdr_B_continuous_hop2s.csv"
ANN_CSV = "/workspace/data/externaldata/frdr_upcall/raw/data/continuous/dataset_B/annotations_B_cont.csv"
KNN_PKL = (_REF+"/knn_cced2.pkl") if _REF else "/workspace/embeddings/known56/models/knn_dapt.pkl"
MAHA_PKL = (_REF+"/maha_cced2.pkl") if _REF else "/workspace/embeddings/known56/models/maha_dapt.pkl"
NORM_JSON = (_REF+"/cced2_norm.json") if _REF else "/workspace/embeddings/cced2_norm.json"
OUT_CSV = Path(_os.environ.get("FU_OUT","/workspace/scripts/audit_2026-07-30/fusion_rerun.csv"));OUT_CSV.parent.mkdir(parents=True,exist_ok=True)

K_CONSEC = 2; GAP_SEC = 3.0; TOL_SEC = 10.0; SMOOTH_WIN = 3; TARGET_FP = 10.0

def mad(x): return float(np.median(np.abs(x - np.median(x))) + 1e-6)
def smooth_same(x, w):
    if w <= 1: return x.astype(np.float32)
    k = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, k, mode="same").astype(np.float32)
def per_file_norm(m, S, sw):
    out = np.empty_like(S, dtype=np.float32)
    for b, idx in m.groupby("base").groups.items():
        idx = np.array(list(idx)); cc = m.loc[idx, "center_sec"].to_numpy(); o = np.argsort(cc); idx_s = idx[o]
        v = S[idx_s].astype(np.float32); v = (v - np.median(v)) / mad(v); v = smooth_same(v, sw)
        out[idx_s] = v
    return out
def extract_events(c, s, th, k=K_CONSEC, gp=GAP_SEC):
    hi = s >= th; n = len(s); o = []; i = 0
    while i < n:
        if not hi[i]: i += 1; continue
        j = i
        while j+1 < n and hi[j+1] and (c[j+1]-c[j]) <= gp: j += 1
        if j-i+1 >= k:
            km = i + int(np.argmax(s[i:j+1])); o.append(float(c[km]))
        i = j + 1
    return o
def match(p, g, tol):
    p = np.array(sorted(p), dtype=float); g = np.array(sorted(g), dtype=float); i = j = TP = 0
    while i < len(g) and j < len(p):
        if p[j] < g[i] - tol: j += 1
        elif p[j] > g[i] + tol: i += 1
        else: TP += 1; i += 1; j += 1
    return TP, len(p) - TP, len(g) - TP
def build_cache(m, S):
    c = {}
    for b, idx in m.groupby("base").groups.items():
        idx = np.array(list(idx)); cc = m.loc[idx, "center_sec"].to_numpy(); o = np.argsort(cc)
        c[b] = (cc[o], S[idx[o]])
    return c
def eval_at(c, gt, th, hr):
    TP=FP=FN=0
    for b, gts in gt.items():
        if b in c:
            cc, ss = c[b]; pt = extract_events(cc, ss, th)
        else: pt = []
        tp, fp, fn = match(pt, gts, TOL_SEC); TP += tp; FP += fp; FN += fn
    return TP / max(1, TP+FN), FP / hr, TP, FP

E = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{EMB_DIR}/embeddings_*.npy"))]).astype("float32")
m = pd.read_csv(MAN_CSV); m["base"] = m["path"].apply(lambda p: Path(p).name); m["center_sec"] = m["center_sec"].astype(float)
ann = pd.read_csv(ANN_CSV, sep=";"); ann.columns = [c.strip().lower() for c in ann.columns]
fc = next(c for c in ann.columns if "file" in c); tc = next(c for c in ann.columns if "time" in c)
ann["base"] = ann[fc].apply(lambda p: Path(str(p)).name)
gt_by = {b: g[tc].astype(float).tolist() for b, g in ann.groupby("base")}
hours = sum((g["center_sec"].max()+5.0)/3600.0 for _, g in m.groupby("base"))

KNN = joblib.load(KNN_PKL); MAHA = joblib.load(MAHA_PKL); cfg = json.load(open(NORM_JSON))
dists, _ = KNN["knn"].kneighbors(E)
knn_raw = dists.mean(1)
D = E - np.asarray(MAHA["mu"])
maha_raw = np.sqrt((D @ MAHA["precision"] * D).sum(1))
cced2_raw = (knn_raw - cfg["mk"])/cfg["sk"] + (maha_raw - cfg["mm"])/cfg["sm"]
S_q = per_file_norm(m, cced2_raw, SMOOTH_WIN)  # per-file MAD-normalised CCED2

# Fusion: α S_q + (1-α) z(cced2_raw)
# Standardize cced2_raw to z-score (unitless) so they're comparable
cced2_z_global = (cced2_raw - cced2_raw.mean()) / (cced2_raw.std() + 1e-9)

results = []
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    S_fused = alpha * S_q + (1.0 - alpha) * cced2_z_global
    cache = build_cache(m, S_fused)
    # Sweep theta to FP/h ≈ 10
    best = None
    for pctl in np.arange(70, 99.91, 0.1):
        th = float(np.percentile(S_fused, pctl))
        rec, fph, TP, FP = eval_at(cache, gt_by, th, hours)
        if best is None or abs(fph - TARGET_FP) < abs(best[1] - TARGET_FP):
            best = (rec, fph, th, pctl, TP, FP)
    results.append({"alpha": alpha, "Recall": best[0], "FP_h": best[1], "theta": best[2], "pctl": best[3], "TP": best[4], "FP": best[5]})

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"=== Fusion sweep over α ===")
print(df.to_string(index=False))
best_row = df.iloc[df["Recall"].argmax()]
print(f"\nBest Fusion: α={best_row['alpha']:.1f} Recall={best_row['Recall']:.4f} FP/h={best_row['FP_h']:.2f}")
print(f"\nBaselines for comparison (winaware paper protocol):")
print(f"  Quiet (α=1.0, S_q only): see α=1.0 row")
print(f"  Old paper Fusion: 0.0838 / FP/h=10.74")
