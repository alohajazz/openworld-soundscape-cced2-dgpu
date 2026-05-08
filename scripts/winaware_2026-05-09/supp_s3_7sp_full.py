#!/usr/bin/env python3
"""Supp Table S3 winaware — 7 species (PR set) + per-file Top-1 protocol (α 案)."""
import os, json, glob, joblib
import numpy as np, pandas as pd
from pathlib import Path

DET_XLSX = "/workspace/data/externaldata/bls-sound-eval/hiceas/metadata_DCLDE2020 DetectionData.xlsx"
OP_MAN = "/workspace/hiceas_op_manifest_winaware.csv"
SP_MAN = "/workspace/hiceas_1706_species_manifest_winsafe.csv"
KNN_PKL = "/workspace/embeddings/cced2_fulldata/knn_cced2.pkl"
MAHA_PKL = "/workspace/embeddings/cced2_fulldata/maha_cced2.pkl"
NORM_JSON = "/workspace/embeddings/cced2_fulldata/cced2_norm.json"
OUT_CSV = Path("/workspace/release_repo/paper_artifacts/supp_table_s3_7sp_winaware_2026-05-09.csv")

BEATs_dirs = [
    ("BEATs+DAPT_fulldata", "/workspace/embeddings/hiceas_op_fulldata_winaware", "/workspace/embeddings/hiceas_1706_fulldata_winaware"),
    ("BEATs+DAPT_dapt_b_3ep", "/workspace/embeddings/hiceas_op_dapt_b_3ep_winaware", "/workspace/embeddings/hiceas_1706_dapt_b_3ep_winaware"),
    ("BEATs+DAPT_continual_palaoa", "/workspace/embeddings/hiceas_op_continual_palaoa_winaware", "/workspace/embeddings/hiceas_1706_continual_palaoa_winaware"),
]

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

# Load detections
xl = pd.ExcelFile(DET_XLSX)
det = pd.concat([xl.parse("1705_OdontoceteDetections"), xl.parse("1706_OdontocetesDetections")], ignore_index=True)
det = det[det["Species1ID"].isin(ODONTO_CODES)].copy()
det["DetectionTimeStart_UTC"] = pd.to_datetime(det["DetectionTimeStart_UTC"], utc=True)
det["DetectionTimeEnd_UTC"] = pd.to_datetime(det["DetectionTimeEnd_UTC"], utc=True)
det = det.dropna(subset=["DetectionTimeStart_UTC"])
minke = xl.parse("1705_MinkeDetections")
minke["DetectionTimeStart_UTC"] = pd.to_datetime(minke["DetectionTimeStart_UTC"], utc=True)
minke["DetectionTimeEnd_UTC"] = pd.to_datetime(minke["DetectionTimeEnd_UTC"], utc=True)
minke["Species1ID"] = 71
minke = minke.dropna(subset=["DetectionTimeStart_UTC"])

print(f"Odontocete detections (6 species): {len(det)}")
print(f"Minke detections: {len(minke)}")

# Combine
all_det = pd.concat([det[["Species1ID", "DetectionTimeStart_UTC", "DetectionTimeEnd_UTC"]],
                     minke[["Species1ID", "DetectionTimeStart_UTC", "DetectionTimeEnd_UTC"]]], ignore_index=True)

# Manifests + file timestamps
m_op = pd.read_csv(OP_MAN); m_op["base"] = m_op["path"].apply(lambda p: Path(p).name)
m_sp = pd.read_csv(SP_MAN); m_sp["base"] = m_sp["path"].apply(lambda p: Path(p).name)
m_combined = pd.concat([m_op, m_sp], ignore_index=True)
m_combined["base"] = m_combined["path"].apply(lambda p: Path(p).name)
m_combined["center_sec"] = m_combined["center_sec"].astype(float)

unique_files = m_combined["base"].unique()
file_ts = {f: parse_ts(f) for f in unique_files}
file_ts = {f: t for f, t in file_ts.items() if t is not None}
print(f"Files with timestamps: {len(file_ts)}")

# Per-species event mapping (file → list of detection centers within file)
gt_per_sp = {sp_id: {"files_with_event": set(), "events": {}} for sp_id in SEVEN_SP}
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

print("\nPer-species (7) event counts:")
for sp_id, name in SEVEN_SP.items():
    nev = sum(len(v) for v in gt_per_sp[sp_id]["events"].values())
    nf = len(gt_per_sp[sp_id]["files_with_event"])
    print(f"  {sp_id:3d} {name:30s}: {nev:5d} events in {nf:4d} files")

# Run evaluation per encoder
results = []
for enc_name, op_dir, sp_dir in BEATs_dirs:
    print(f"\n=== {enc_name} ===")
    if not Path(op_dir + "/embeddings_000.npy").exists():
        print(f"  SKIP: {op_dir}"); continue
    E_op = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{op_dir}/embeddings_*.npy"))]).astype("float32")
    E_sp = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{sp_dir}/embeddings_*.npy"))]).astype("float32")
    E = np.concatenate([E_op, E_sp])
    n = min(len(E), len(m_combined))
    E = E[:n]; m_use = m_combined.iloc[:n].reset_index(drop=True)
    knn_z, maha_z, cced2_z = cced2(E)

    for score_name, S_full in [("-kNN_z", -knn_z), ("-Mahalanobis_z", -maha_z), ("-CCED2_z", -cced2_z)]:
        for tol_sec in [15.0, 20.0]:
            per_sp = []
            tot_TP = tot_FP = tot_FN = 0
            for sp_id, sp_data in gt_per_sp.items():
                dns_files = sp_data["files_with_event"]
                if not dns_files: continue
                mask = m_use["base"].isin(dns_files)
                if mask.sum() == 0: continue
                m_dns = m_use[mask].reset_index(drop=True)
                S_dns = S_full[mask.values]
                pred = {}
                for fbase, idx in m_dns.groupby("base").groups.items():
                    idx = np.array(list(idx))
                    cc = m_dns.loc[idx, "center_sec"].to_numpy()
                    order = np.argsort(cc); cc = cc[order]; sc = S_dns[idx[order]]
                    if len(sc) < 5: continue
                    theta = float(np.quantile(sc, Q_QUANTILE))
                    sel = np.where(sc >= theta)[0]
                    if len(sel) == 0: continue
                    top1 = sel[np.argmax(sc[sel])]
                    pred[fbase] = float(cc[top1])
                # Match per file
                TP = FP = FN = 0
                for f in dns_files:
                    gts = sp_data["events"].get(f, [])
                    pt = pred.get(f)
                    if pt is None:
                        FN += len(gts); continue
                    matched = False
                    for gt in gts:
                        if abs(pt - gt) <= tol_sec:
                            TP += 1; matched = True; break
                    if not matched: FP += 1
                    FN += max(0, len(gts) - (1 if matched else 0))
                P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
                hours = len(dns_files) * 60 / 3600.0
                fph = FP / max(hours, 1e-9)
                per_sp.append({"sp_id": sp_id, "P": P, "R": R, "F1": F1, "FP_h": fph, "TP": TP, "FP": FP, "FN": FN, "n_files": len(dns_files)})
                tot_TP += TP; tot_FP += FP; tot_FN += FN
            for avg in ["macro", "micro"]:
                if avg == "macro":
                    if not per_sp: continue
                    P = np.mean([x["P"] for x in per_sp]); R = np.mean([x["R"] for x in per_sp])
                    F1 = np.mean([x["F1"] for x in per_sp]); FPh = np.mean([x["FP_h"] for x in per_sp])
                else:
                    P = tot_TP / (tot_TP + tot_FP) if (tot_TP + tot_FP) > 0 else 0.0
                    R = tot_TP / (tot_TP + tot_FN) if (tot_TP + tot_FN) > 0 else 0.0
                    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
                    th = sum(x["n_files"] * 60 / 3600.0 for x in per_sp)
                    FPh = tot_FP / max(th, 1e-9)
                results.append({"encoder": enc_name, "score": score_name, "tol": int(tol_sec), "avg": avg,
                                "P": P, "R": R, "F1": F1, "FP_h": FPh, "n_species": len(per_sp)})

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print("\n=== Results (7 species, per-file Top-1) ===")
for enc in df["encoder"].unique():
    print(f"\n--- {enc} ---")
    sub = df[df["encoder"] == enc].copy()
    sub["row_label"] = sub["score"] + " " + sub["avg"] + "@" + sub["tol"].astype(str)
    print(sub[["row_label", "P", "R", "F1", "FP_h", "n_species"]].to_string(index=False))
print(f"\nWrote {OUT_CSV}")
