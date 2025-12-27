#!/usr/bin/env python
# run_frdr_table5_tol10_perch_interp.py
import glob, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# =========================
# Paths (from your logs)
# =========================
IND_MODELS = Path("/workspace/embeddings/perch_ind_models")
FRDR_EMB   = Path("/workspace/embeddings/perch_frdr_hop2s")
FRDR_MAN   = Path("/workspace/data/externaldata/frdr_upcall/manifests/frdr_B_continuous_hop2s.csv")
FRDR_ANN   = Path("/workspace/data/externaldata/frdr_upcall/raw/data/continuous/dataset_B/annotations_B_cont.csv")

# =========================
# Table-5 protocol knobs
# =========================
TOL_SEC   = 10.0
TARGET_FP = 10.0          # event FP per hour target
K_CONSEC  = 2             # at least k consecutive windows above theta (within gap)
MIN_SEP   = 3.0           # minimum separation between predicted events (NMS-style)
SMOOTH_W  = 3             # 3-point moving average (per file)

# quantile grid to bracket FP/h=10
Q_GRID = np.concatenate([
    np.linspace(0.90, 0.99, 60, endpoint=False),
    np.linspace(0.99, 0.9999, 120),
])

OUT_CSV_NEAR   = Path(f"/workspace/data/externaldata/frdr_upcall/table5_perch_tol{int(TOL_SEC)}_near_fp{int(TARGET_FP)}.csv")
OUT_CSV_INTERP = Path(f"/workspace/data/externaldata/frdr_upcall/table5_perch_tol{int(TOL_SEC)}_interp_at_fp{int(TARGET_FP)}.csv")

# -------------------------
# utils
# -------------------------
def load_embeddings(d: Path) -> np.ndarray:
    ps = sorted(glob.glob(str(d / "embeddings_*.npy")))
    assert ps, f"no embeddings_*.npy in {d}"
    return np.concatenate([np.load(p) for p in ps], axis=0).astype("float32")

def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-6)

def smooth_same(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype("float32")
    k = np.ones(win, dtype="float32") / float(win)
    return np.convolve(x, k, mode="same").astype("float32")

def hop_and_wph(m: pd.DataFrame) -> tuple[float, int]:
    # infer hop from mode of diff within files
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

def per_file_local_norm_and_smooth(m: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """per-file median/MAD -> z-like -> per-file 3pt smoothing"""
    out = np.empty_like(score, dtype="float32")
    for base, idx in m.groupby("base").groups.items():
        idx = np.asarray(list(idx))
        # keep time order within file
        order = np.argsort(m.loc[idx, "center_sec"].to_numpy())
        idx = idx[order]

        v = score[idx].astype("float32")
        v = (v - np.median(v)) / mad(v)
        v = smooth_same(v, SMOOTH_W)
        out[idx] = v
    return out

def build_events_for_one_file(times: np.ndarray, scores: np.ndarray, theta: float) -> list[tuple[float, float]]:
    """
    Convert window scores -> events for ONE file.
    Returns list of (event_center_sec, event_peak_score).
    """
    pos = np.where(scores >= theta)[0]
    if pos.size == 0:
        return []

    # group consecutive positives allowing small gaps up to MIN_SEP (we use time gap, not index gap)
    groups = []
    cur = [pos[0]]
    for i in range(1, len(pos)):
        if (times[pos[i]] - times[pos[i-1]]) <= MIN_SEP:  # treat as continuous cluster
            cur.append(pos[i])
        else:
            groups.append(cur)
            cur = [pos[i]]
    groups.append(cur)

    # require at least K_CONSEC windows in the group
    groups = [g for g in groups if len(g) >= K_CONSEC]
    if not groups:
        return []

    # represent each group by peak window
    events = []
    for g in groups:
        g = np.asarray(g)
        j = g[np.argmax(scores[g])]
        events.append((float(times[j]), float(scores[j])))

    # NMS-like min_sep between events (keep higher peak if too close)
    events.sort(key=lambda x: x[0])
    kept = []
    for t, s in events:
        if not kept:
            kept.append((t, s))
            continue
        if (t - kept[-1][0]) >= MIN_SEP:
            kept.append((t, s))
        else:
            # conflict: keep the higher score
            if s > kept[-1][1]:
                kept[-1] = (t, s)
    return kept

def build_events(m: pd.DataFrame, score: np.ndarray, theta: float) -> np.ndarray:
    """All files -> concatenated event centers (seconds), sorted."""
    all_events = []
    for base, idx in m.groupby("base").groups.items():
        idx = np.asarray(list(idx))
        order = np.argsort(m.loc[idx, "center_sec"].to_numpy())
        idx = idx[order]
        times = m.loc[idx, "center_sec"].to_numpy(dtype="float32")
        sc    = score[idx].astype("float32")
        ev = build_events_for_one_file(times, sc, theta)
        for (t, _) in ev:
            all_events.append((base, t))
    # store as structured array for matching by base
    return np.array(all_events, dtype=object)

def eval_events(pred_events: np.ndarray, ann: pd.DataFrame, hours: float) -> tuple[float, float, int, int, int, int]:
    """
    Matching rule:
      TP: each GT event is TP if any pred event within ±TOL_SEC (same base)
      FP: pred events not within ±TOL_SEC of any GT event (same base)
      FN: remaining GT
    """
    gt_n = len(ann)
    if gt_n == 0:
        return 0.0, 0.0, 0, 0, 0, 0

    # group preds by file base
    pred_by_base = {}
    for b, t in pred_events:
        pred_by_base.setdefault(b, []).append(float(t))
    for b in pred_by_base:
        pred_by_base[b].sort()

    # group GT by base
    gt_by_base = {}
    for b, t in zip(ann["base"].values, ann["center"].values):
        gt_by_base.setdefault(b, []).append(float(t))
    for b in gt_by_base:
        gt_by_base[b].sort()

    # TP/FN via "any pred in window" per GT
    TP = 0
    for b, gts in gt_by_base.items():
        preds = pred_by_base.get(b, [])
        if not preds:
            continue
        preds = np.asarray(preds, dtype="float32")
        for gt in gts:
            if np.any(np.abs(preds - gt) <= TOL_SEC):
                TP += 1
    FN = gt_n - TP

    # FP: pred not within tol of ANY gt
    FP = 0
    for b, preds in pred_by_base.items():
        gts = gt_by_base.get(b, [])
        if not gts:
            FP += len(preds)
            continue
        gts = np.asarray(gts, dtype="float32")
        for p in preds:
            if not np.any(np.abs(gts - p) <= TOL_SEC):
                FP += 1

    recall = TP / (TP + FN + 1e-9)
    fph    = FP / max(hours, 1e-9)
    return float(recall), float(fph), int(gt_n), int(len(pred_events)), int(TP), int(FP), int(FN)

def pick_two_points(score_norm: np.ndarray, m: pd.DataFrame, ann: pd.DataFrame, hours: float):
    """
    Scan quantiles -> get (FP/h, Recall, theta, q, n_pred_events).
    Return:
      best_near: closest to TARGET_FP
      bracket: (below, above) points that bracket TARGET_FP if possible
    """
    rows = []
    for q in Q_GRID:
        theta = float(np.quantile(score_norm, q))
        ev = build_events(m, score_norm, theta)
        rec, fph, gt_n, pred_n, TP, FP, FN = eval_events(ev, ann, hours)
        rows.append((fph, rec, theta, float(q), int(pred_n), TP, FP, FN))

    rows.sort(key=lambda x: x[0])  # sort by FP/h
    # closest
    best_near = min(rows, key=lambda x: abs(x[0] - TARGET_FP))

    # find bracket
    below = None
    above = None
    for r in rows:
        if r[0] <= TARGET_FP:
            below = r
        if r[0] >= TARGET_FP and above is None:
            above = r
    return best_near, below, above

def interp_recall_at_target(below, above):
    # linear interp in FP/h space
    fp0, r0, th0, q0, n0, TP0, FP0, FN0 = below
    fp1, r1, th1, q1, n1, TP1, FP1, FN1 = above
    if abs(fp1 - fp0) < 1e-9:
        return float(r0), dict(fp0=fp0, r0=r0, theta0=th0, q0=q0, fp1=fp1, r1=r1, theta1=th1, q1=q1)
    t = (TARGET_FP - fp0) / (fp1 - fp0)
    r = r0 + t * (r1 - r0)
    return float(r), dict(fp0=fp0, r0=r0, theta0=th0, q0=q0, fp1=fp1, r1=r1, theta1=th1, q1=q1)

# -------------------------
# main
# -------------------------
def main():
    # load manifest + annotation
    m = pd.read_csv(FRDR_MAN)
    m["center_sec"] = m["center_sec"].astype("float32")
    m["base"] = m["path"].apply(lambda p: Path(p).name)

    ann = pd.read_csv(FRDR_ANN, sep=";").rename(columns=lambda c: c.strip().lower())
    filename_col = next(k for k in ann.columns if "file" in k)
    time_col     = next(k for k in ann.columns if "time" in k)
    ann["base"]   = ann[filename_col].apply(lambda p: Path(str(p)).name)
    ann["center"] = ann[time_col].astype("float32")

    hop, wph = hop_and_wph(m)
    hours = len(m) / float(wph)
    print(f"[INFO] hop={hop}s  windows/hr≈{wph}  total_hours≈{hours:.3f}  GT_events={len(ann)}")

    # load embeddings (no re-extraction)
    E = load_embeddings(FRDR_EMB)
    assert len(E) == len(m), (len(E), len(m))

    # load Perch InD models
    KNN = joblib.load(IND_MODELS / "knn_perch.pkl")["knn"]
    MA  = joblib.load(IND_MODELS / "maha_perch.pkl")
    cfg = json.load(open(IND_MODELS / "cced2_norm_perch.json"))

    # raw distances -> z
    dists, _ = KNN.kneighbors(E)
    Kraw = dists.mean(1).astype("float32")
    D = E - np.asarray(MA["mu"])
    Mraw = np.sqrt((D @ MA["precision"] * D).sum(1)).astype("float32")

    Kz = (Kraw - cfg["mk"]) / (cfg["sk"] + 1e-8)
    Mz = (Mraw - cfg["mm"]) / (cfg["sm"] + 1e-8)
    Cz = (Kz + Mz).astype("float32")

    scores = {
        "kNN_z": Kz.astype("float32"),
        "Maha_z": Mz.astype("float32"),
        "CCED2_z": Cz.astype("float32"),
    }

    near_rows = []
    interp_rows = []

    for name, s in scores.items():
        s2 = per_file_local_norm_and_smooth(m, s)

        best_near, below, above = pick_two_points(s2, m, ann, hours)
        fp_n, r_n, th_n, q_n, pred_n, TP_n, FP_n, FN_n = best_near

        near_rows.append({
            "score": name,
            "tol_sec": TOL_SEC,
            "target_FP_per_hour": TARGET_FP,
            "method": "nearest",
            "theta": th_n,
            "quantile": q_n,
            "FP_per_hour": fp_n,
            "Recall": r_n,
            "Pred_events": pred_n,
            "GT_events": len(ann),
            "TP": TP_n,
            "FP": FP_n,
            "FN": FN_n,
            "Hours(approx)": hours,
        })

        if below is None or above is None:
            interp_rows.append({
                "score": name,
                "tol_sec": TOL_SEC,
                "target_FP_per_hour": TARGET_FP,
                "method": "interp",
                "Recall_at_target": np.nan,
                "note": "target not bracketed by grid (expand Q_GRID or check score range)",
                "Hours(approx)": hours,
            })
            continue

        r_at, dbg = interp_recall_at_target(below, above)
        interp_rows.append({
            "score": name,
            "tol_sec": TOL_SEC,
            "target_FP_per_hour": TARGET_FP,
            "method": "interp",
            "Recall_at_target": r_at,
            **dbg,
            "Hours(approx)": hours,
        })

    pd.DataFrame(near_rows).to_csv(OUT_CSV_NEAR, index=False)
    pd.DataFrame(interp_rows).to_csv(OUT_CSV_INTERP, index=False)

    print("\n[OK] wrote:", OUT_CSV_NEAR)
    print(pd.DataFrame(near_rows)[["score","theta","FP_per_hour","Recall","Pred_events","GT_events"]].to_string(index=False))
    print("\n[OK] wrote:", OUT_CSV_INTERP)
    print(pd.DataFrame(interp_rows)[["score","Recall_at_target","fp0","r0","fp1","r1"]].to_string(index=False))

if __name__ == "__main__":
    main()
