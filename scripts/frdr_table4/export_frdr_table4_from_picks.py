#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet_cmp", required=True)
    ap.add_argument("--union_pick", required=True)
    ap.add_argument("--fusion_pick", required=True)
    ap.add_argument("--out_csv", default="table4_frdr.csv")
    args = ap.parse_args()

    cmp_df = pd.read_csv(args.quiet_cmp)
    quiet_row = cmp_df[cmp_df["mode"].astype(str).str.contains("Quiet", case=False, na=False)].iloc[0]
    quiet_recall = float(quiet_row["Recall"])
    quiet_fph = float(quiet_row["FP_events_per_hr"])

    union_df = pd.read_csv(args.union_pick)
    fusion_df = pd.read_csv(args.fusion_pick)

    out = pd.DataFrame([
        {"Mode":"Quiet", "Recall":quiet_recall, "FP/h":quiet_fph, "theta":None},
        {"Mode":"Union", "Recall":float(union_df["Recall"].iloc[0]), "FP/h":float(union_df["FP_events_per_hr"].iloc[0]), "theta":float(union_df["theta"].iloc[0])},
        {"Mode":"Fusion","Recall":float(fusion_df["Recall"].iloc[0]), "FP/h":float(fusion_df["FP_events_per_hr"].iloc[0]), "theta":float(fusion_df["theta"].iloc[0])},
    ])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print("[OK] wrote:", out_path)

if __name__ == "__main__":
    main()
