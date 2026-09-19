#!/usr/bin/env python3
"""Table 1: per-encoder means and paired differences over the eight SED-head seeds.

Input : summary.csv written by sedgate.sh (seeds 42, 123, 777) and sedgate_more.sh (seeds 101-505).
Output: table1_seed_summary.json (mean, sample standard deviation, paired Stage 1 - PRETRAIN differences).
Rows for other checkpoints in summary.csv are not used."""
import csv, json, statistics as st, sys
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("summary.csv")
SEEDS = ["42", "123", "777", "101", "202", "303", "404", "505"]
ENC = {"PRETRAIN": "PRETRAIN", "DAPTfix": "BEATs+DAPT (Stage 1, step 127641)"}
MET = {"eventF1": "event", "clipF1": "clip", "seg2sF1": "segment_2s"}
rows = list(csv.DictReader(src.open()))
val = {m: {x["seed"]: x for x in rows if x["model"] == m} for m in ENC}
for m in ENC:
    assert sorted(val[m]) == sorted(SEEDS), (m, sorted(val[m]))
out = {"seeds": [int(s) for s in SEEDS], "encoders": {}, "paired_stage1_minus_pretrain": {}}
for m, label in ENC.items():
    out["encoders"][m] = {"label": label}
    for k, name in MET.items():
        v = [float(val[m][s][k]) for s in SEEDS]
        out["encoders"][m][name] = {"values": v, "mean": st.mean(v), "sample_std": st.stdev(v)}
for k, name in MET.items():
    d = [float(val["DAPTfix"][s][k]) - float(val["PRETRAIN"][s][k]) for s in SEEDS]
    out["paired_stage1_minus_pretrain"][name] = {"differences": [round(x, 4) for x in d], "mean": st.mean(d),
        "sample_std": st.stdev(d), "n_positive": sum(x > 0 for x in d), "n": len(d), "min": min(d), "max": max(d)}
dst = src.with_name("table1_seed_summary.json")
dst.write_text(json.dumps(out, indent=1) + "\n")
for m in ENC:
    print(m, {n: (round(out["encoders"][m][n]["mean"], 4), round(out["encoders"][m][n]["sample_std"], 4)) for n in MET.values()})
for n, r in out["paired_stage1_minus_pretrain"].items():
    print(n, round(r["mean"], 4), round(r["sample_std"], 4), f'{r["n_positive"]}/{r["n"]}', round(r["min"], 4), round(r["max"], 4))
