import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

FRDR_DIR = Path("/workspace/data/externaldata/frdr_upcall")
MAN_CSV  = FRDR_DIR / "manifests/frdr_B_continuous_hop2s.csv"
EMB_DIR  = FRDR_DIR / "embeddings/op2s_dapt"   # BEATs+DAPT hop2s embeddings
ANN_CSV  = FRDR_DIR / "raw/data/continuous/dataset_B/annotations_B_cont.csv"

KNN_PKL   = Path("/workspace/embeddings/known56/models/knn_dapt.pkl")
MAHA_PKL  = Path("/workspace/embeddings/known56/models/maha_dapt.pkl")
NORM_JSON = Path("/workspace/embeddings/cced2_norm.json")

TOL_SEC   = 10.0
TARGET_FP = 10.0
K_CONSEC  = 2
MIN_SEP   = 3.0
SMOOTH_W  = 3

# 既に十分細かいので、まずはこのgridでOK
Q_GRID = np.unique(np.concatenate([
    np.linspace(0.001, 0.10, 300, endpoint=False),
    np.linspace(0.10, 0.90, 1200, endpoint=False),
    np.linspace(0.90, 0.9999, 1500)
]))

OUT_CSV = FRDR_DIR / f"table5_beats_tol{int(TOL_SEC)}_interp_at_fp{int(TARGET_FP)}.csv"

def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-6

def smooth_same(x, win):
    if win <= 1: return x.astype(np.float32)
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same").astype(np.float32)

def load_embeddings(d: Path) -> np.ndarray:
    ps = sorted(glob.glob(str(d / "embeddings_*.npy")))
    assert ps, f"no embeddings_*.npy in {d}"
    return np.concatenate([np.load(p) for p in ps], axis=0).astype("float32")

def hop_and_wph(m: pd.DataFrame) -> tuple[float, int]:
    hop = (
        m.groupby("base")["center_sec"]
         .diff()
         .dropna()
         .round(3)
         .mode()
         .iloc[0]
    )
    wph = max(1, int(round(3600.0 / float(hop))))
    return float(hop), int(wph)

def per_file_norm_smooth(m, score):
    out = np.empty_like(score, dtype="float32")
    for base, idx in m.groupby("base").groups.items():
        idx = np.asarray(list(idx))
        order = np.argsort(m.loc[idx, "center_sec"].to_numpy())
        idx = idx[order]
        v = score[idx].astype("float32")
        v = (v - np.median(v)) / mad(v)
        v = smooth_same(v, SMOOTH_W)
        out[idx] = v
    return out

def build_events_one_file(times, scores, theta):
    pos = np.where(scores >= theta)[0]
    if pos.size == 0:
        return []
    groups = []
    cur = [pos[0]]
    for i in range(1, len(pos)):
        if (times[pos[i]] - times[pos[i-1]]) <= MIN_SEP:
            cur.append(pos[i])
        else:
            groups.append(cur)
            cur = [pos[i]]
    groups.append(cur)
    groups = [g for g in groups if len(g) >= K_CONSEC]
    if not groups:
        return []
    events = []
    for g in groups:
        g = np.asarray(g)
        j = g[np.argmax(scores[g])]
        events.append((float(times[j]), float(scores[j])))
    events.sort(key=lambda x: x[0])
    kept = []
    for t, s in events:
        if not kept:
            kept.append((t, s)); continue
        if (t - kept[-1][0]) >= MIN_SEP:
            kept.append((t, s))
        else:
            if s > kept[-1][1]:
                kept[-1] = (t, s)
    return [t for (t, _) in kept]

def eval_events(pred_by_base, gt_by_base, hours):
    # TP: for each GT, any pred within ±tol
    TP = 0
    gt_n = 0
    for b, gts in gt_by_base.items():
        gt_n += len(gts)
        preds = np.asarray(pred_by_base.get(b, []), dtype="float32")
        if preds.size == 0:
            continue
        for gt in gts:
            if np.any(np.abs(preds - gt) <= TOL_SEC):
                TP += 1
    FN = gt_n - TP

    # FP: pred not within tol of any GT
    FP = 0
    for b, preds in pred_by_base.items():
        gts = np.asarray(gt_by_base.get(b, []), dtype="float32")
        if gts.size == 0:
            FP += len(preds)
            continue
        for p in preds:
            if not np.any(np.abs(gts - p) <= TOL_SEC):
                FP += 1

    recall = TP / (TP + FN + 1e-9)
    fph = FP / max(hours, 1e-9)
    return float(recall), float(fph), int(gt_n), int(TP), int(FP), int(FN)

def interpolate(curve_df, target_fp):
    c = curve_df.sort_values("FP_per_hour").reset_index(drop=True)
    below = c[c["FP_per_hour"] <= target_fp]
    above = c[c["FP_per_hour"] >= target_fp]
    if len(below)==0 or len(above)==0:
        c["diff"] = (c["FP_per_hour"] - target_fp).abs()
        row = c.sort_values("diff").iloc[0]
        return ("nearest", float(row["Recall"]), dict(fp=float(row["FP_per_hour"]), theta=float(row["theta"]), q=float(row["quantile"])))
    lo = below.iloc[-1]; hi = above.iloc[0]
    fp0, r0 = float(lo["FP_per_hour"]), float(lo["Recall"])
    fp1, r1 = float(hi["FP_per_hour"]), float(hi["Recall"])
    if abs(fp1 - fp0) < 1e-12:
        return ("exact", r0, dict(fp=fp0, theta=float(lo["theta"]), q=float(lo["quantile"])))
    a = (target_fp - fp0) / (fp1 - fp0)
    r = r0 + a*(r1 - r0)
    dbg = dict(fp0=fp0, r0=r0, theta0=float(lo["theta"]), q0=float(lo["quantile"]),
               fp1=fp1, r1=r1, theta1=float(hi["theta"]), q1=float(hi["quantile"]))
    return ("interp", float(r), dbg)

def main():
    m = pd.read_csv(MAN_CSV)
    m["center_sec"] = m["center_sec"].astype("float32")
    m["base"] = m["path"].apply(lambda p: Path(p).name)

    ann = pd.read_csv(ANN_CSV, sep=";").rename(columns=lambda c: c.strip().lower())
    filename_col = next(k for k in ann.columns if "file" in k)
    time_col     = next(k for k in ann.columns if "time" in k)
    ann["base"]   = ann[filename_col].apply(lambda p: Path(str(p)).name)
    ann["center"] = ann[time_col].astype("float32")

    hop, wph = hop_and_wph(m)
    hours = len(m) / float(wph)

    E = load_embeddings(EMB_DIR)
    assert len(E) == len(m), (len(E), len(m))

    KNN  = joblib.load(KNN_PKL)
    MAHA = joblib.load(MAHA_PKL)
    cfg  = json.load(open(NORM_JSON))

    dists,_ = KNN["knn"].kneighbors(E)
    knn_raw = dists.mean(1).astype("float32")

    D = E - np.asarray(MAHA["mu"])
    maha_raw = np.sqrt((D @ MAHA["precision"] * D).sum(1)).astype("float32")

    knn_z  = (knn_raw  - cfg["mk"]) / (cfg["sk"] + 1e-8)
    maha_z = (maha_raw - cfg["mm"]) / (cfg["sm"] + 1e-8)
    cced2_z = (knn_z + maha_z).astype("float32")

    gt_by_base = {b: g["center"].tolist() for b, g in ann.groupby("base")}

    out_rows = []
    for name, raw in [("kNN_z", knn_z), ("Mahalanobis_z", maha_z), ("CCED2_z", cced2_z)]:
        s = per_file_norm_smooth(m, raw)

        curve = []
        for q in Q_GRID:
            theta = float(np.quantile(s, q))
            pred_by_base = {}
            for b, idx in m.groupby("base").groups.items():
                idx = np.asarray(list(idx))
                order = np.argsort(m.loc[idx, "center_sec"].to_numpy())
                idx = idx[order]
                times = m.loc[idx, "center_sec"].to_numpy(dtype="float32")
                sc = s[idx]
                pred_by_base[b] = build_events_one_file(times, sc, theta)
            recall, fph, gt_n, TP, FP, FN = eval_events(pred_by_base, gt_by_base, hours)
            curve.append({"score": name, "quantile": float(q), "theta": theta, "FP_per_hour": fph, "Recall": recall})

        dfc = pd.DataFrame(curve)
        method, r_at, dbg = interpolate(dfc, TARGET_FP)
        row = {"score": name, "tol_sec": TOL_SEC, "target_FP_per_hour": TARGET_FP,
               "method": method, "Recall_at_target": r_at, **dbg}
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_CSV, index=False)
    print(out[["score","method","Recall_at_target"]].to_string(index=False))
    print("\n[OK] wrote:", OUT_CSV)

if __name__ == "__main__":
    main()
