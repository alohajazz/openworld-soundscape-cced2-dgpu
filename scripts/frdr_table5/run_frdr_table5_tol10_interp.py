import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

FRDR_DIR = Path("/workspace/data/externaldata/frdr_upcall")
MAN_CSV  = FRDR_DIR / "manifests/frdr_B_continuous_hop2s.csv"
EMB_DIR  = FRDR_DIR / "embeddings/op2s_dapt"
ANN_CSV  = FRDR_DIR / "raw/data/continuous/dataset_B/annotations_B_cont.csv"

KNN_PKL   = Path("/workspace/embeddings/known56/models/knn_dapt.pkl")
MAHA_PKL  = Path("/workspace/embeddings/known56/models/maha_dapt.pkl")
NORM_JSON = Path("/workspace/embeddings/cced2_norm.json")

# ---- Protocol ----
TOL_SEC = 10.0
TARGET_FP_H = 10.0
K_CONSEC = 2
GAP_SEC  = 3.0
SMOOTH_WIN = 3

# 閾値スイープ分位点（十分に細かく）
Q_GRID = np.unique(np.concatenate([
    np.linspace(0.001, 0.10, 300, endpoint=False),
    np.linspace(0.10, 0.90, 1200, endpoint=False),
    np.linspace(0.90, 0.9999, 1500)
]))

OUT_SUMMARY = FRDR_DIR / f"table5_frdr_tol{int(TOL_SEC)}_interp_at_fp{int(TARGET_FP_H)}.csv"

def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-6

def smooth_same(x, win):
    if win <= 1: return x.astype(np.float32)
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same").astype(np.float32)

def load_embeddings(emb_dir: Path) -> np.ndarray:
    paths = sorted(glob.glob(str(emb_dir / "embeddings_*.npy")))
    if not paths:
        raise FileNotFoundError(f"embeddings_*.npy not found under {emb_dir}")
    return np.concatenate([np.load(p) for p in paths], axis=0).astype("float32")

def infer_hop_seconds(df):
    diffs = (df.groupby("base")["center_sec"]
               .apply(lambda s: pd.Series(np.diff(np.sort(s.values))))
               .reset_index(drop=True))
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0: return 2.0
    hop = float(pd.Series(np.round(diffs, 3)).mode().iloc[0])
    return hop if hop > 0 else 2.0

def total_hours_from_manifest(df):
    total = 0.0
    for base, g in df.groupby("base"):
        mx = float(np.max(g["center_sec"].values))
        total += (mx + 5.0)
    return total / 3600.0

def extract_events(centers, scores, theta):
    hi = scores >= theta
    n = len(scores)
    out = []
    i = 0
    while i < n:
        if not hi[i]:
            i += 1
            continue
        j = i
        while (j + 1 < n) and hi[j + 1] and ((centers[j + 1] - centers[j]) <= GAP_SEC):
            j += 1
        if (j - i + 1) >= K_CONSEC:
            seg = slice(i, j + 1)
            kmax = int(np.argmax(scores[seg])) + i
            out.append(float(centers[kmax]))
        i = j + 1
    return out

def match_events_1to1(pred_times, gt_times, tol):
    pred = np.array(sorted(pred_times), dtype=float)
    gt   = np.array(sorted(gt_times), dtype=float)
    i = j = 0
    TP = 0
    while i < len(gt) and j < len(pred):
        if pred[j] < gt[i] - tol:
            j += 1
        elif pred[j] > gt[i] + tol:
            i += 1
        else:
            TP += 1
            i += 1
            j += 1
    FP = int(len(pred) - TP)
    FN = int(len(gt) - TP)
    return TP, FP, FN

def preprocess_per_file(m, score):
    out = np.empty_like(score, dtype=np.float32)
    for base, idx in m.groupby("base").groups.items():
        idx = np.array(list(idx), dtype=int)
        centers = m.loc[idx, "center_sec"].to_numpy()
        order = np.argsort(centers)
        idx_s = idx[order]
        v = score[idx_s].astype(np.float32)
        v = (v - np.median(v)) / mad(v)
        v = smooth_same(v, SMOOTH_WIN)
        out[idx_s] = v
    return out

def build_cache(m, score_pf):
    cache = {}
    for base, idx in m.groupby("base").groups.items():
        idx = np.array(list(idx), dtype=int)
        centers = m.loc[idx, "center_sec"].to_numpy()
        order = np.argsort(centers)
        cache[base] = (centers[order], score_pf[idx[order]])
    return cache

def eval_theta(cache, gt_by_file, theta, hours):
    TP=FP=FN=0
    pred_total=0
    for base, gt_times in gt_by_file.items():
        if base in cache:
            centers, ss = cache[base]
            pred_times = extract_events(centers, ss, theta)
        else:
            pred_times = []
        tp, fp, fn = match_events_1to1(pred_times, gt_times, TOL_SEC)
        TP += tp; FP += fp; FN += fn
        pred_total += len(pred_times)
    recall = TP / (TP + FN + 1e-12)
    fp_h = FP / max(hours, 1e-12)
    return recall, fp_h, pred_total, TP, FP, FN

def interpolate_at_target(curve_df, target_fp):
    # fp_hでソートして、targetを挟む2点を探す
    c = curve_df.sort_values("FP_per_hour").reset_index(drop=True)

    below = c[c["FP_per_hour"] <= target_fp]
    above = c[c["FP_per_hour"] >= target_fp]

    if len(below)==0 or len(above)==0:
        # 片側しかない → 近い点を採用
        c["diff"] = (c["FP_per_hour"] - target_fp).abs()
        row = c.sort_values("diff").iloc[0]
        return {
            "method": "nearest",
            "Recall_at_target": float(row["Recall"]),
            "FP_per_hour_nearest": float(row["FP_per_hour"]),
            "theta_nearest": float(row["theta"]),
            "quantile_nearest": float(row["quantile"]),
        }

    lo = below.iloc[-1]
    hi = above.iloc[0]

    if float(lo["FP_per_hour"]) == float(hi["FP_per_hour"]):
        return {
            "method": "exact",
            "Recall_at_target": float(lo["Recall"]),
            "FP_per_hour_nearest": float(lo["FP_per_hour"]),
            "theta_nearest": float(lo["theta"]),
            "quantile_nearest": float(lo["quantile"]),
        }

    # 線形補間
    fp0, r0 = float(lo["FP_per_hour"]), float(lo["Recall"])
    fp1, r1 = float(hi["FP_per_hour"]), float(hi["Recall"])
    alpha = (target_fp - fp0) / (fp1 - fp0)
    r = r0 + alpha * (r1 - r0)
    return {
        "method": "interp",
        "Recall_at_target": float(r),
        "FP0": fp0, "R0": r0, "theta0": float(lo["theta"]), "q0": float(lo["quantile"]),
        "FP1": fp1, "R1": r1, "theta1": float(hi["theta"]), "q1": float(hi["quantile"]),
    }

# ---- load manifest & embeddings ----
m = pd.read_csv(MAN_CSV)
m["center_sec"] = m["center_sec"].astype(float)
m["base"] = m["path"].apply(lambda p: Path(p).name)

E = np.concatenate([np.load(p) for p in sorted(glob.glob(str(EMB_DIR / "embeddings_*.npy")))], axis=0).astype("float32")
if len(E) != len(m):
    raise RuntimeError(f"Row mismatch: manifest={len(m)} vs embeddings={len(E)}")

hop = infer_hop_seconds(m)
windows_per_hour = int(round(3600.0 / hop))
hours = total_hours_from_manifest(m)

# ---- load models & scores ----
KNN  = joblib.load(KNN_PKL)
MAHA = joblib.load(MAHA_PKL)
cfg  = json.load(open(NORM_JSON))

dists,_ = KNN["knn"].kneighbors(E)
knn_raw = dists.mean(1)
D = E - np.asarray(MAHA["mu"])
maha_raw = np.sqrt((D @ MAHA["precision"] * D).sum(1))

knn_z  = (knn_raw  - cfg["mk"]) / cfg["sk"]
maha_z = (maha_raw - cfg["mm"]) / cfg["sm"]
cced2_z = knn_z + maha_z

# ---- annotations ----
ann = pd.read_csv(ANN_CSV, sep=";")
ann = ann.rename(columns=lambda c: c.strip().lower())
file_col = next((c for c in ann.columns if "file" in c), None)
time_col = next((c for c in ann.columns if "time" in c), None)
ann["base"] = ann[file_col].apply(lambda p: Path(str(p)).name)
ann["t"] = ann[time_col].astype(float)
gt_by_file = {b: g["t"].tolist() for b, g in ann.groupby("base")}
GT_events = int(sum(len(v) for v in gt_by_file.values()))

def run_curve(name, raw_score):
    score_pf = preprocess_per_file(m, raw_score)
    cache = build_cache(m, score_pf)
    vals = score_pf[np.isfinite(score_pf)]

    rows = []
    for q in Q_GRID:
        theta = float(np.quantile(vals, q))
        recall, fp_h, pred_total, TP, FP, FN = eval_theta(cache, gt_by_file, theta, hours)
        rows.append({
            "score": name, "quantile": float(q), "theta": theta,
            "FP_per_hour": fp_h, "Recall": recall,
            "Pred_events": int(pred_total), "TP": int(TP), "FP": int(FP), "FN": int(FN)
        })

    dfc = pd.DataFrame(rows)
    dfc.to_csv(FRDR_DIR / f"curve_{name}_tol{int(TOL_SEC)}.csv", index=False)

    summ = interpolate_at_target(dfc, TARGET_FP_H)
    return dfc, summ

summaries = []
for name, raw in [("kNN_z", knn_z), ("Mahalanobis_z", maha_z), ("CCED2_z", cced2_z)]:
    dfc, summ = run_curve(name, raw)
    row = {
        "score": name,
        "tol_sec": TOL_SEC,
        "hop_sec": hop,
        "k_consec": K_CONSEC,
        "gap_sec": GAP_SEC,
        "smooth_win": SMOOTH_WIN,
        "Hours(approx)": hours,
        "GT_events": GT_events,
        "target_FP_per_hour": TARGET_FP_H,
        **summ
    }
    summaries.append(row)

out = pd.DataFrame(summaries)
out.to_csv(OUT_SUMMARY, index=False)
print(out[["score","method","Recall_at_target"]].to_string(index=False))
print("\n[OK] wrote:", OUT_SUMMARY)
