#!/usr/bin/env python3
"""Run Table 5 FRDR with winaware embeddings + ORIGINAL known56_sep25 InD ref (correct encoder pair)."""
import os, sys
from pathlib import Path
src_path = "/workspace/release_repo/scripts/frdr_table5/run_frdr_table5_tol10.py"
src = open(src_path).read()

# ONLY change EMB_DIR to winaware (keep as Path); keep KNN/Maha/norm as original
patched = src.replace(
    'EMB_DIR  = FRDR_DIR / "embeddings/op2s_dapt"',
    'EMB_DIR  = Path("/workspace/embeddings/frdr_fulldata_winaware")'
)
patched = patched.replace(
    'OUT_CSV = FRDR_DIR / f"table5_frdr_tol{int(TOL_SEC)}_eventFP{int(TARGET_EVENT_FP_PER_HOUR)}.csv"',
    'OUT_CSV = Path("/tmp/table5_frdr_winaware_v2_2026-05-09.csv")'
)

print(f"=== EMB_DIR = winaware, InD ref = known56_sep25 (paired) ===")
exec(compile(patched, "<patched>", "exec"))
