#!/usr/bin/env python3
"""Table 3: recall at FP/h = 10 by linear interpolation of the FP/h-recall curve (the method stated in the Table 3 note).
Input: all_evaluations.csv written by run_frdr_table3_winaware_COPY.py (one row per evaluated threshold)."""
import json, sys
from pathlib import Path
import pandas as pd
DEFAULT = Path(__file__).resolve().parents[2] / "paper_artifacts/final_revision_2026-09-19/frdr/table3/all_evaluations.csv"
src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT; d = pd.read_csv(src); r = d["return"].apply(json.loads); d["recall"] = r.str[0]; d["fph"] = r.str[1]
out = {}
for k, g in d.groupby(["enc_name", "sname"]):
    g = g.drop_duplicates(["fph", "recall"]).sort_values("fph"); lo = g[g.fph <= 10].iloc[-1]; hi = g[g.fph > 10].iloc[0]
    out["|".join(k)] = round(float(lo.recall + (10 - lo.fph) * (hi.recall - lo.recall) / (hi.fph - lo.fph)), 6)   # six decimals: the manuscript rounds these values once, to three decimals
print(json.dumps(out, indent=1))
