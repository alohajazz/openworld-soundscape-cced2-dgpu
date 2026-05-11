#!/usr/bin/env python3
"""Table 3 FRDR ablation: kNN_z / Mahalanobis_z / CCED2_z for each of 3 BEATs+DAPT encoders + Perch.
Paper §4.5 protocol: per-file MAD norm + smooth=3, binary search theta to FP/h=10."""
import os, json, glob, joblib
import numpy as np, pandas as pd
from pathlib import Path

# Paths
MAN_CSV = "/workspace/data/externaldata/frdr_upcall/manifests/frdr_B_continuous_hop2s.csv"
ANN_CSV = "/workspace/data/externaldata/frdr_upcall/raw/data/continuous/dataset_B/annotations_B_cont.csv"
OUT_CSV = Path("/workspace/release_repo/paper_artifacts/frdr_table3_winaware_2026-05-09.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# 3 BEATs+DAPT encoders + Perch
BEATs_dirs = [
    ("BEATs+DAPT_fulldata", "/workspace/embeddings/frdr_fulldata_winaware"),
    ("BEATs+DAPT_dapt_b_3ep", "/workspace/embeddings/frdr_dapt_b_3ep_winaware"),
    ("BEATs+DAPT_continual_palaoa", "/workspace/embeddings/frdr_continual_palaoa_winaware"),
]
PERCH_DIR = "/workspace/embeddings/perch_frdr_hop2s"

# InD references — encoder-paired
BEATs_KNN = "/workspace/embeddings/known56/models/knn_dapt.pkl"
BEATs_MAHA = "/workspace/embeddings/known56/models/maha_dapt.pkl"
BEATs_NORM = "/workspace/embeddings/cced2_norm.json"
PERCH_KNN = "/workspace/embeddings/perch_ind_models/knn_perch.pkl"
PERCH_MAHA = "/workspace/embeddings/perch_ind_models/maha_perch.pkl"
PERCH_NORM = "/workspace/embeddings/perch_ind_models/cced2_norm_perch.json"

K_CONSEC = 2; GAP_SEC = 3.0; TOL_SEC = 10.0; SMOOTH_WIN = 3; TARGET_FP = 10.0

def mad(x): return float(np.median(np.abs(x - np.median(x))) + 1e-6)
def smooth_same(x, w):
    if w <= 1: return x.astype(np.float32)
    k = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, k, mode="same").astype(np.float32)

def per_file_norm(m, S, sw):
    out = np.empty_like(S, dtype=np.float32)
    for b, idx in m.groupby("base").groups.items():
        idx = np.array(list(idx))
        cc = m.loc[idx, "center_sec"].to_numpy()
        o = np.argsort(cc); idx_s = idx[o]
        v = S[idx_s].astype(np.float32)
        v = (v - np.median(v)) / mad(v)
        v = smooth_same(v, sw)
        out[idx_s] = v
    return out

def extract_events(c, s, th, k=K_CONSEC, gp=GAP_SEC):
    hi = s >= th; n = len(s); o = []; i = 0
    while i < n:
        if not hi[i]: i += 1; continue
        j = i
        while j+1 < n and hi[j+1] and (c[j+1] - c[j]) <= gp: j += 1
        if j - i + 1 >= k:
            km = i + int(np.argmax(s[i:j+1]))
            o.append(float(c[km]))
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
        idx = np.array(list(idx)); cc = m.loc[idx, "center_sec"].to_numpy()
        o = np.argsort(cc)
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

def sweep_to_fp(c, gt, S_arr, hr, target=TARGET_FP):
    res = []
    for pctl in np.arange(50, 99.91, 0.5):
        th = float(np.percentile(S_arr, pctl))
        rec, fph, TP, FP = eval_at(c, gt, th, hr)
        res.append((pctl, th, rec, fph, TP, FP))
    df = pd.DataFrame(res, columns=["pctl","theta","Recall","FP_h","TP","FP"])
    return df.iloc[(df["FP_h"] - target).abs().argsort().iloc[0]]

def cced2_scores(E, knn_pkl, maha_pkl, norm_json):
    KNN = joblib.load(knn_pkl); MAHA = joblib.load(maha_pkl); cfg = json.load(open(norm_json))
    dists, _ = KNN["knn"].kneighbors(E)
    knn_raw = dists.mean(1)
    D = E - np.asarray(MAHA["mu"])
    maha_raw = np.sqrt((D @ MAHA["precision"] * D).sum(1))
    knn_z = (knn_raw - cfg["mk"]) / cfg["sk"]
    maha_z = (maha_raw - cfg["mm"]) / cfg["sm"]
    return knn_z, maha_z, knn_z + maha_z

# ---- Load manifest, ann ----
m = pd.read_csv(MAN_CSV); m["base"] = m["path"].apply(lambda p: Path(p).name); m["center_sec"] = m["center_sec"].astype(float)
ann = pd.read_csv(ANN_CSV, sep=";"); ann.columns = [c.strip().lower() for c in ann.columns]
fc = next(c for c in ann.columns if "file" in c); tc = next(c for c in ann.columns if "time" in c)
ann["base"] = ann[fc].apply(lambda p: Path(str(p)).name)
gt_by = {b: g[tc].astype(float).tolist() for b, g in ann.groupby("base")}
hours = sum((g["center_sec"].max()+5.0)/3600.0 for _, g in m.groupby("base"))
print(f"Hours: {hours:.2f}, GT events: {sum(len(v) for v in gt_by.values())}")

results = []
# 3 BEATs+DAPT encoders
for enc_name, emb_dir in BEATs_dirs:
    print(f"\n=== {enc_name} ===")
    E = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{emb_dir}/embeddings_*.npy"))]).astype("float32")
    knn_z, maha_z, cced2_z = cced2_scores(E, BEATs_KNN, BEATs_MAHA, BEATs_NORM)
    for sname, S in [("kNN_z", knn_z), ("Mahalanobis_z", maha_z), ("CCED2_z", cced2_z)]:
        S_pf = per_file_norm(m, S, SMOOTH_WIN)
        cache = build_cache(m, S_pf)
        pick = sweep_to_fp(cache, gt_by, S_pf, hours)
        print(f"  {sname:15} Recall={pick['Recall']:.4f} FP/h={pick['FP_h']:.2f} TP={int(pick['TP'])} FP={int(pick['FP'])}")
        results.append({"encoder": enc_name, "score": sname, "Recall": float(pick["Recall"]), "FP_h": float(pick["FP_h"]),
                        "TP": int(pick["TP"]), "FP": int(pick["FP"]), "theta": float(pick["theta"]), "pctl": float(pick["pctl"])})

# Perch
print(f"\n=== Perch 2.0 ===")
E_p = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{PERCH_DIR}/embeddings_*.npy"))]).astype("float32")
knn_zp, maha_zp, cced2_zp = cced2_scores(E_p, PERCH_KNN, PERCH_MAHA, PERCH_NORM)
for sname, S in [("kNN_z", knn_zp), ("Mahalanobis_z", maha_zp), ("CCED2_z", cced2_zp)]:
    S_pf = per_file_norm(m, S, SMOOTH_WIN)
    cache = build_cache(m, S_pf)
    pick = sweep_to_fp(cache, gt_by, S_pf, hours)
    print(f"  {sname:15} Recall={pick['Recall']:.4f} FP/h={pick['FP_h']:.2f} TP={int(pick['TP'])} FP={int(pick['FP'])}")
    results.append({"encoder": "Perch 2.0", "score": sname, "Recall": float(pick["Recall"]), "FP_h": float(pick["FP_h"]),
                    "TP": int(pick["TP"]), "FP": int(pick["FP"]), "theta": float(pick["theta"]), "pctl": float(pick["pctl"])})

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}")
print("\n=== Table 3 Summary ===")
piv = df.pivot(index="encoder", columns="score", values="Recall")
print(piv.round(4).to_string())
