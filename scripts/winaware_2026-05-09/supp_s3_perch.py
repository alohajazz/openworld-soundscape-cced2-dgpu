#!/usr/bin/env python3
"""Supp Table S3 Perch baseline — 7 species, per-file Top-1, Perch InD ref, on 1705 OP subset only."""
import os, json, glob, joblib
import numpy as np, pandas as pd
from pathlib import Path

EMB_DIR = "/workspace/embeddings/perch_hiceas_op2s_10s_win10"
DET_XLSX = "/workspace/data/externaldata/bls-sound-eval/hiceas/metadata_DCLDE2020 DetectionData.xlsx"
KNN_PKL = "/workspace/embeddings/perch_ind_models/knn_perch.pkl"
MAHA_PKL = "/workspace/embeddings/perch_ind_models/maha_perch.pkl"
NORM_JSON = "/workspace/embeddings/perch_ind_models/cced2_norm_perch.json"
OUT_CSV = "/workspace/release_repo/paper_artifacts/supp_table_s3_perch_winaware_2026-05-09.csv"

SEVEN_SP = {
    71: "Minke whale",
    46: "Sperm whale",
    33: "False killer whale",
    36: "Short-finned pilot whale",
    15: "Rough-toothed dolphin",
    2:  "Offshore spotted dolphin",
    13: "Striped dolphin",
}
ODONTO_CODES = [46, 33, 36, 15, 2, 13]
Q_QUANTILE = 0.99

def cced2(E):
    KNN = joblib.load(KNN_PKL); MAHA = joblib.load(MAHA_PKL); cfg = json.load(open(NORM_JSON))
    dists, _ = KNN["knn"].kneighbors(E)
    knn_raw = dists.mean(1)
    D = E - np.asarray(MAHA["mu"])
    maha_raw = np.sqrt((D @ MAHA["precision"] * D).sum(1))
    knn_z = (knn_raw - cfg["mk"]) / cfg["sk"]
    maha_z = (maha_raw - cfg["mm"]) / cfg["sm"]
    return knn_z, maha_z, knn_z + maha_z

def parse_ts(fname):
    parts = Path(fname).stem.split("_")
    if len(parts) < 3: return None
    try: return pd.to_datetime(parts[1] + parts[2], format="%Y%m%d%H%M%S", utc=True)
    except Exception: return None

# Load Perch embeddings + index
E = np.load(f"{EMB_DIR}/embeddings_000.npy").astype("float32")
idx = pd.read_csv(f"{EMB_DIR}/index_000.csv")
idx["base"] = idx["path"].apply(lambda p: Path(p).name)
idx["center_sec"] = idx["center_sec"].astype(float)
print(f"Perch embeddings: {E.shape}, files: {idx['base'].nunique()}")

# Load detections (only 1705 since Perch is 1705 OP only)
xl = pd.ExcelFile(DET_XLSX)
det = xl.parse("1705_OdontoceteDetections")
det = det[det["Species1ID"].isin(ODONTO_CODES)].copy()
det["DetectionTimeStart_UTC"] = pd.to_datetime(det["DetectionTimeStart_UTC"], utc=True)
det["DetectionTimeEnd_UTC"] = pd.to_datetime(det["DetectionTimeEnd_UTC"], utc=True)
det = det.dropna(subset=["DetectionTimeStart_UTC"])
minke = xl.parse("1705_MinkeDetections")
minke["DetectionTimeStart_UTC"] = pd.to_datetime(minke["DetectionTimeStart_UTC"], utc=True)
minke["DetectionTimeEnd_UTC"] = pd.to_datetime(minke["DetectionTimeEnd_UTC"], utc=True)
minke["Species1ID"] = 71
minke = minke.dropna(subset=["DetectionTimeStart_UTC"])
all_det = pd.concat([det[["Species1ID", "DetectionTimeStart_UTC", "DetectionTimeEnd_UTC"]],
                    minke[["Species1ID", "DetectionTimeStart_UTC", "DetectionTimeEnd_UTC"]]], ignore_index=True)
print(f"Detections: 1705 odontocete={len(det)}, 1705 minke={len(minke)}")

# Match files to detections
unique_files = idx["base"].unique()
file_ts = {f: parse_ts(f) for f in unique_files}
file_ts = {f: t for f, t in file_ts.items() if t is not None}
gt_per_sp = {sp: {"files_with_event": set(), "events": {}} for sp in SEVEN_SP}
file_dur = pd.Timedelta(seconds=60)
for _, row in all_det.iterrows():
    sp_id = int(row["Species1ID"])
    if sp_id not in SEVEN_SP: continue
    t_start = row["DetectionTimeStart_UTC"]
    t_end = row["DetectionTimeEnd_UTC"] if pd.notna(row["DetectionTimeEnd_UTC"]) else t_start
    for f, fts in file_ts.items():
        if t_start <= fts + file_dur and t_end >= fts:
            gt_per_sp[sp_id]["files_with_event"].add(f)
            center_in_file = max(0, min(60, (t_start - fts).total_seconds()))
            gt_per_sp[sp_id]["events"].setdefault(f, []).append(center_in_file)

print("Per-species event coverage (1705 OP only):")
for sp_id, name in SEVEN_SP.items():
    nev = sum(len(v) for v in gt_per_sp[sp_id]["events"].values())
    nf = len(gt_per_sp[sp_id]["files_with_event"])
    print(f"  {sp_id:3d} {name:30s}: {nev:5d} events in {nf:4d} files")

# Compute Perch CCED2 + per-species evaluation
print("\nComputing Perch CCED2...")
knn_z, maha_z, cced2_z = cced2(E)
results = []
for score_name, S_full in [("-kNN_z", -knn_z), ("-Mahalanobis_z", -maha_z), ("-CCED2_z", -cced2_z)]:
    for tol_sec in [15.0, 20.0]:
        per_sp = []
        tot_TP = tot_FP = tot_FN = 0
        for sp_id, sp_data in gt_per_sp.items():
            dns_files = sp_data["files_with_event"]
            if not dns_files: continue
            mask = idx["base"].isin(dns_files)
            if mask.sum() == 0: continue
            m_dns = idx[mask].reset_index(drop=True)
            S_dns = S_full[mask.values]
            pred = {}
            for fbase, gidx in m_dns.groupby("base").groups.items():
                gidx = np.array(list(gidx))
                cc = m_dns.loc[gidx, "center_sec"].to_numpy()
                order = np.argsort(cc); cc = cc[order]; sc = S_dns[gidx[order]]
                if len(sc) < 5: continue
                theta = float(np.quantile(sc, Q_QUANTILE))
                sel = np.where(sc >= theta)[0]
                if len(sel) == 0: continue
                top1 = sel[np.argmax(sc[sel])]
                pred[fbase] = float(cc[top1])
            TP = FP = FN = 0
            for f in dns_files:
                gts = sp_data["events"].get(f, [])
                pt = pred.get(f)
                if pt is None: FN += len(gts); continue
                matched = False
                for gt in gts:
                    if abs(pt - gt) <= tol_sec: TP += 1; matched = True; break
                if not matched: FP += 1
                FN += max(0, len(gts) - (1 if matched else 0))
            P = TP/(TP+FP) if (TP+FP)>0 else 0.0
            R = TP/(TP+FN) if (TP+FN)>0 else 0.0
            F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
            hours = len(dns_files)*60/3600.0
            fph = FP/max(hours,1e-9)
            per_sp.append({"sp_id": sp_id, "P": P, "R": R, "F1": F1, "FP_h": fph, "TP": TP, "FP": FP, "FN": FN, "n_files": len(dns_files)})
            tot_TP += TP; tot_FP += FP; tot_FN += FN
        for avg in ["macro", "micro"]:
            if avg == "macro" and per_sp:
                P = np.mean([x["P"] for x in per_sp]); R = np.mean([x["R"] for x in per_sp])
                F1 = np.mean([x["F1"] for x in per_sp]); FPh = np.mean([x["FP_h"] for x in per_sp])
            else:
                P = tot_TP/(tot_TP+tot_FP) if (tot_TP+tot_FP)>0 else 0.0
                R = tot_TP/(tot_TP+tot_FN) if (tot_TP+tot_FN)>0 else 0.0
                F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
                th = sum(x["n_files"]*60/3600.0 for x in per_sp)
                FPh = tot_FP/max(th,1e-9)
            results.append({"encoder": "Perch_2.0_win10", "score": score_name, "tol": int(tol_sec), "avg": avg,
                            "P": P, "R": R, "F1": F1, "FP_h": FPh, "n_species": len(per_sp)})

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print("\n=== Perch 2.0 (win10) Supp S3 ===")
df["row_label"] = df["score"] + " " + df["avg"] + "@" + df["tol"].astype(str)
print(df[["row_label", "P", "R", "F1", "FP_h", "n_species"]].to_string(index=False))
print(f"\nWrote {OUT_CSV}")
