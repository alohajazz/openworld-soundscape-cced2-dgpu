"""Build winaware manifest for hiceas_1706_FLAC_part2 (mirrors existing manifest format)."""
import glob, os
from pathlib import Path
import pandas as pd

PART2_DIR = "/workspace/data/hiceas_1706_FLAC_part2"
OUT_MANIFEST = "/workspace/hiceas_1706_part2_manifest_winsafe.csv"

flac_files = sorted(glob.glob(f"{PART2_DIR}/*.flac"))
print(f"Found {len(flac_files)} FLAC files in part2")

# Window pattern (matches existing winsafe manifest):
#   center_sec from 5.0 to 55.0 in steps of 2.0 (= 26 windows)
#   start_sec = center_sec - 5.0
#   duration_sec = 10.0
rows = []
for path in flac_files:
    for k in range(26):
        start = k * 2.0
        center = start + 5.0
        rows.append((path, "eval", center, start, 10.0))

df = pd.DataFrame(rows, columns=["path", "label", "center_sec", "start_sec", "duration_sec"])
df.to_csv(OUT_MANIFEST, index=False)
print(f"Wrote {OUT_MANIFEST} ({len(df)} rows)")
print(f"  Sample first 3 rows:\n{df.head(3)}")
print(f"  Unique paths: {df['path'].nunique()}")
