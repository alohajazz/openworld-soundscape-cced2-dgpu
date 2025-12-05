# -*- coding: utf-8 -*-
"""
dapt_qc_manifest.py

DAPT Manifest Quality Control (Chunk-based / Large-scale).
- Checks path existence and readability.
- Verifies timestamp validity (start + duration <= actual length).
- Aggregates distribution stats (site, diel, month).

Usage:
  python3 dapt_qc_manifest.py /path/to/dapt_manifest.tsv
"""
import os, sys, json, time, datetime as dt
from pathlib import Path
import pandas as pd
import soundfile as sf

LOG_FILE = os.environ.get("LOG", "./logs/dapt_qc_manifest.log")
SAMPLE_N = int(os.environ.get("SAMPLE_N", "2000"))
CHUNKSIZE = int(os.environ.get("CHUNKSIZE", "500000"))

os.makedirs(Path(LOG_FILE).parent, exist_ok=True)

def log(msg: str):
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} UTC] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def get_audio_dur(path: str):
    try:
        info = sf.info(path)
        if info.samplerate and info.frames is not None:
            return info.frames / info.samplerate
    except Exception:
        return None
    return None

def main(tsv_path: str):
    start = time.time()
    log("==== dapt_qc_manifest: START ====")
    log(f"manifest = {tsv_path}")

    p = Path(tsv_path)
    if not p.exists():
        log(f"Error: Not found: {p}")
        return

    total_rows = 0
    missing_files = 0
    unreadable = 0
    out_of_range = 0

    site_counts = {}
    diel_counts = {}
    month_counts = {}

    validated_audio = 0
    already_checked = set()

    for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=CHUNKSIZE):
        total_rows += len(chunk)

        # Check existence
        exists = chunk["path"].apply(lambda s: Path(s).exists())
        missing_files += int((~exists).sum())

        # Stats
        for s, c in chunk["site_id"].value_counts().items():
            site_counts[s] = site_counts.get(s, 0) + int(c)
        for d, c in chunk["diel"].value_counts().items():
            diel_counts[d] = diel_counts.get(d, 0) + int(c)
        for m, c in chunk["month"].value_counts().items():
            month_counts[int(m)] = month_counts.get(int(m), 0) + int(c)

        # Sampling check for audio length
        if validated_audio < SAMPLE_N:
            for path in chunk["path"].unique():
                if validated_audio >= SAMPLE_N: break
                if path in already_checked: continue
                
                if not Path(path).exists():
                    already_checked.add(path)
                    continue
                    
                dur = get_audio_dur(path)
                if dur is None:
                    unreadable += 1
                    already_checked.add(path)
                    validated_audio += 1
                    continue
                
                sub = chunk[chunk["path"] == path][["start_sec", "duration_sec"]]
                bad = (sub["start_sec"] + sub["duration_sec"] > dur + 1e-6).sum()
                out_of_range += int(bad)
                validated_audio += 1
                already_checked.add(path)

        log(f"progress: rows={total_rows}, missing={missing_files}, unreadable={unreadable}")

    elapsed = int(time.time() - start)
    summary = {
        "total_rows": int(total_rows),
        "missing_files": int(missing_files),
        "unreadable_audio_headers": int(unreadable),
        "out_of_range_segments": int(out_of_range),
        "site_counts": dict(sorted(site_counts.items(), key=lambda x: -x[1])[:100]),
        "diel_counts": diel_counts,
        "month_counts": dict(sorted(month_counts.items())),
        "elapsed_sec": elapsed,
    }

    print(json.dumps(summary, indent=2))
    with open(p.with_suffix(".qc.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("==== dapt_qc_manifest: END ====")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 dapt_qc_manifest.py /path/to/dapt_manifest.tsv")
        sys.exit(1)
    main(sys.argv[1])