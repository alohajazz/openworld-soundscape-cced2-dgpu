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

# ---- Table 5 protocol (FRDR) ----
TOL_SEC = 10.0
TARGET_EVENT_FP_PER_HOUR = 10.0   # ★ここを10に揃える
K_CONSEC = 2
GAP_SEC  = 3.0
SMOOTH_WIN = 3

OUT_CSV = FRDR_DIR / f"table5_frdr_tol{int(TOL_SEC)}_eventFP{int(TARGET_EVENT_FP_PER_HOUR)}.csv"

def mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) + 1e-6)

def smooth_same(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32)
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same").astype(np.float32)

def load_embeddings(emb_dir: Path) -> np.ndarray:
    paths = sorted(glob.glob(str(emb_dir / "embeddings_*.npy")))
    if not paths:
        raise FileNotFoundError(f"embeddings_*.npy not found under {emb_dir}")
    return np.concatenate([np.load(p) for p in paths], axis=0).astype("float32")

def infer_hop_seconds(df: pd.DataFrame) -> float:
    diffs = (df.groupby("base")["center_sec"]
               .apply(lambda s: pd.Series(np.diff(np.sort(s.values))))
               .reset_index(drop=True))
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return 2.0
    hop = float(pd.Series(np.round(diffs, 3)).mode().iloc[0])
    return hop if hop > 0 else 2.0

def total_hours_from_manifest(df: pd.DataFrame) -> float:
    total = 0.0
    for base, g in df.groupby("base"):
        mx = float(np.max(g["center_sec"].values))
        total += (mx + 5.0)  # center+5s で近似
    return total / 3600.0

def extract_events(centers: np.ndarray, scores: np.ndarray, theta: float,
                   k_consec: int, gap_sec: float) -> list[float]:
    hi = scores >= theta
    n = len(scores)
    out: list[float] = []
    i = 0
    while i < n:
        if not hi[i]:
            i += 1
            continue
        j = i
        while (j + 1 < n) and hi[j + 1] and ((centers[j + 1] - centers[j]) <= gap_sec):
            j += 1
        if (j - i + 1) >= k_consec:
            seg = slice(i, j + 1)
            kmax = int(np.argmax(scores[seg])) + i
            out.append(float(centers[kmax]))
        i = j + 1
    return out

def match_events_1to1(pred_times: list[float], gt_times: list[float], tol: float) -> tuple[int,int,int]:
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

def preprocess_per_file(m: pd.DataFrame, score: np.ndarray) -> np.ndarray:
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

def build_cache(m: pd.DataFrame, score_pf: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for base, idx in m.groupby("base").groups.items():
        idx = np.array(list(idx), dtype=int)
        centers = m.loc[idx, "center_sec"].to_numpy()
        order = np.argsort(centers)
        cache[base] = (centers[order], score_pf[idx[order]])
    return cache

def eval_at_theta(cache, gt_by_file, theta, hours):
    TP=FP=FN=0
    pred_total=0
    for base, gt_times in gt_by_file.items():
        if base in cache:
            centers, ss = cache[base]
            pred_times = extract_events(centers, ss, theta, K_CONSEC, GAP_SEC)
        else:
            pred_times = []
        tp, fp, fn = match_events_1to1(pred_times, gt_times, TOL_SEC)
        TP += tp; FP += fp; FN += fn
        pred_total += len(pred_times)
    recall = TP / (TP + FN + 1e-12)
    fp_h = FP / max(hours, 1e-12)
    return recall, fp_h, pred_total, TP, FP, FN

def pick_theta_by_binary_search(cache, gt_by_file, vals, hours, target_fp_h):
    # thetaを直接二分探索（単調性：theta↑ → pred↓ → FP/h↓ を仮定）
    lo = float(np.min(vals) - 10.0)   # ほぼ全て陽性
    hi = float(np.max(vals) + 10.0)   # ほぼ全て陰性

    # 範囲確認
    _, fp_lo, *_ = eval_at_theta(cache, gt_by_file, lo, hours)
    _, fp_hi, *_ = eval_at_theta(cache, gt_by_file, hi, hours)

    # 最大でもターゲットに届かないなら、最大側を返す
    if fp_lo < target_fp_h:
        rec, fp_h, pred_total, TP, FP, FN = eval_at_theta(cache, gt_by_file, lo, hours)
        return dict(theta=lo, fp_h=fp_h, recall=rec, pred_events=pred_total, TP=TP, FP=FP, FN=FN,
                    diff=abs(fp_h-target_fp_h), note="target unreachable (max FP/h below target)")

    best = None
    for _ in range(50):
        mid = (lo + hi) / 2.0
        rec, fp_h, pred_total, TP, FP, FN = eval_at_theta(cache, gt_by_file, mid, hours)
        diff = abs(fp_h - target_fp_h)
        cand = (diff, mid, fp_h, rec, pred_total, TP, FP, FN)
        if (best is None) or (cand < best):
            best = cand
        # fp_hが高すぎる→閾値を上げる（hi側へ）
        if fp_h > target_fp_h:
            lo = mid
        else:
            hi = mid

    diff, theta, fp_h, rec, pred_total, TP, FP, FN = best
    return dict(theta=theta, fp_h=fp_h, recall=rec, pred_events=pred_total, TP=TP, FP=FP, FN=FN,
                diff=diff, note="ok")

def run_one(name, score, m, gt_by_file, hours, hop, windows_per_hour):
    score_pf = preprocess_per_file(m, score)
    cache = build_cache(m, score_pf)
    vals = score_pf[np.isfinite(score_pf)]
    picked = pick_theta_by_binary_search(cache, gt_by_file, vals, hours, TARGET_EVENT_FP_PER_HOUR)
    row = {
        "score": name,
        "tol_sec": TOL_SEC,
        "target_event_fp_per_hour": TARGET_EVENT_FP_PER_HOUR,
        "hop_sec": hop,
        "windows_per_hour": windows_per_hour,
        "k_consec": K_CONSEC,
        "gap_sec": GAP_SEC,
        "smooth_win": SMOOTH_WIN,
        "Hours(approx)": hours,
        "GT_events": int(sum(len(v) for v in gt_by_file.values())),
        "theta": picked["theta"],
        "FP_per_hour": picked["fp_h"],
        "Recall": picked["recall"],
        "Pred_events": picked["pred_events"],
        "TP": picked["TP"],
        "FP": picked["FP"],
        "FN": picked["FN"],
        "abs_diff_to_target": picked["diff"],
        "note": picked.get("note",""),
    }
    return row

# ---- main ----
m = pd.read_csv(MAN_CSV)
m["center_sec"] = m["center_sec"].astype(float)
m["base"] = m["path"].apply(lambda p: Path(p).name)

E = load_embeddings(EMB_DIR)
if len(E) != len(m):
    raise RuntimeError(f"Row mismatch: manifest={len(m)} vs embeddings={len(E)}")

hop = infer_hop_seconds(m)
windows_per_hour = int(round(3600.0 / hop))
hours = total_hours_from_manifest(m)

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

ann = pd.read_csv(ANN_CSV, sep=";")
ann = ann.rename(columns=lambda c: c.strip().lower())
file_col = next((c for c in ann.columns if "file" in c), None)
time_col = next((c for c in ann.columns if "time" in c), None)
if file_col is None or time_col is None:
    raise RuntimeError(f"Unexpected annotation columns: {ann.columns.tolist()}")
ann["base"] = ann[file_col].apply(lambda p: Path(str(p)).name)
ann["t"] = ann[time_col].astype(float)
gt_by_file = {b: g["t"].tolist() for b, g in ann.groupby("base")}

rows = [
    run_one("kNN_z", knn_z, m, gt_by_file, hours, hop, windows_per_hour),
    run_one("Mahalanobis_z", maha_z, m, gt_by_file, hours, hop, windows_per_hour),
    run_one("CCED2_z", cced2_z, m, gt_by_file, hours, hop, windows_per_hour),
]
out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)

print(out[["score","theta","FP_per_hour","Recall","Pred_events","GT_events","TP","FP","FN","abs_diff_to_target","note"]].to_string(index=False))
print("\n[OK] wrote:", OUT_CSV)
