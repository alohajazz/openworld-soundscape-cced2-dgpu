# -*- coding: utf-8 -*-
"""
dapt_make_manifest_all.py

Large-scale DAPT manifest generator (Streaming version).
- Output TSV columns: path, start_sec, duration_sec, site_id, month, diel
- Scans audio files, infers timestamp/site from filenames/paths.
- Writes to TSV incrementally to handle massive datasets without memory issues.
"""

import os, sys, re, json, time, datetime as dt
from pathlib import Path
from typing import Optional, Tuple, List
import soundfile as sf

# ---------- Configuration (Environment Variables) ----------
# Default paths should be overridden by user
ROOTS = os.environ.get("ROOTS", "./data/raw_audio")
OUT = os.environ.get("OUT", "./data/dapt_manifest.tsv")
SEG_S = int(os.environ.get("SEG_S", "10"))
STRIDE_S = int(os.environ.get("STRIDE_S", str(SEG_S)))
MIN_AUDIO_S = int(os.environ.get("MIN_AUDIO_S", str(SEG_S)))
TZ_OFFSET_HOURS = int(os.environ.get("TZ_OFFSET_HOURS", "0"))
EXTS = [x.strip().lower() for x in os.environ.get("EXTS", ".flac,.wav").split(",") if x.strip()]
LOG_FILE = os.environ.get("LOG", "./logs/dapt_make_manifest.log")
MAX_FILES = int(os.environ["MAX_FILES"]) if os.environ.get("MAX_FILES") else None

os.makedirs(Path(LOG_FILE).parent, exist_ok=True)
os.makedirs(Path(OUT).parent, exist_ok=True)

def log(msg: str):
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} UTC] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ---------- Datetime Inference ----------
_PATTERNS = [
    # 20240131_235959
    re.compile(r"(20\d{2})(\d{2})(\d{2})[T_\- ]?(\d{2})(\d{2})(\d{2})"),
    # 2024-01-31T23-59-59
    re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})[T _\-]+(\d{2})[:_\-](\d{2})[:_\-](\d{2})"),
    # 20240131 (Date only)
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),
    # 2024-01-31 (Date only)
    re.compile(r"(20\d{2})-(\d{2})-(\d{2})"),
]

def parse_datetime_from_path(p: Path) -> Optional[dt.datetime]:
    s = p.as_posix()
    for pat in _PATTERNS:
        m = pat.search(s)
        if not m: continue
        gd = m.groups()
        try:
            if len(gd) >= 6:
                y, mo, d, H, M, S = map(int, gd[:6])
                return dt.datetime(y, mo, d, H, M, S)
            elif len(gd) == 3:
                y, mo, d = map(int, gd)
                return dt.datetime(y, mo, d, 0, 0, 0)
        except Exception:
            continue
    return None

def infer_datetime(p: Path) -> dt.datetime:
    t = parse_datetime_from_path(p)
    if t is None:
        try:
            t = dt.datetime.fromtimestamp(p.stat().st_mtime)
        except:
            t = dt.datetime.utcnow()
    if TZ_OFFSET_HOURS != 0:
        t = t + dt.timedelta(hours=TZ_OFFSET_HOURS)
    return t

# ---------- Site ID Inference ----------
def infer_site_id(p: Path) -> str:
    # Heuristic: Use the parent directory name as site ID
    # Users can customize this logic for specific dataset structures
    return p.parent.name or "unknown"

# ---------- Diel Cycle ----------
def infer_diel(t: dt.datetime) -> str:
    h = t.hour
    return "day" if 6 <= h < 18 else "night"

# ---------- Audio Info ----------
def safe_audio_info(path: Path) -> Optional[Tuple[int, int]]:
    try:
        info = sf.info(str(path))
        if info.samplerate and info.frames is not None:
            return int(info.samplerate), int(info.frames)
    except Exception as e:
        log(f"WARN: failed to read info: {path} ({e})")
    return None

def list_audio_files(root_dirs: List[Path], exts: List[str]) -> List[Path]:
    files = []
    for root in root_dirs:
        if not root.exists():
            log(f"WARN: root not found: {root}")
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
                if MAX_FILES and len(files) >= MAX_FILES:
                    return files
    return files

def main():
    start = time.time()
    log("==== dapt_make_manifest_all: START ====")
    log(f"ROOTS = {ROOTS}")
    log(f"OUT = {OUT}")
    
    root_dirs = [Path(x.strip()) for x in ROOTS.split(",") if x.strip()]
    files = list_audio_files(root_dirs, EXTS)
    log(f"Found files: {len(files)}")

    with open(OUT, "w") as fw:
        fw.write("path\tstart_sec\tduration_sec\tsite_id\tmonth\tdiel\n")

    stats = {
        "files_scanned": 0, "files_skipped_short": 0, "segments_written": 0,
        "by_site": {}, "by_diel": {"day": 0, "night": 0},
    }

    for i, path in enumerate(files, 1):
        ai = safe_audio_info(path)
        if ai is None: continue
        sr, frames = ai
        dur = frames / max(sr, 1)
        
        if dur < max(MIN_AUDIO_S, SEG_S):
            stats["files_skipped_short"] += 1
            continue

        t0 = infer_datetime(path)
        site = infer_site_id(path)
        diel = infer_diel(t0)
        month = t0.month

        nseg = 0
        start_sec = 0
        with open(OUT, "a") as fw:
            while start_sec + SEG_S <= dur + 1e-6:
                fw.write(f"{path.as_posix()}\t{int(start_sec)}\t{SEG_S}\t{site}\t{month}\t{diel}\n")
                nseg += 1
                start_sec += STRIDE_S

        stats["files_scanned"] += 1
        stats["segments_written"] += nseg
        stats["by_site"][site] = stats["by_site"].get(site, 0) + nseg
        stats["by_diel"][diel] = stats["by_diel"].get(diel, 0) + nseg

        if i % 1000 == 0:
            log(f"progress: {i}/{len(files)} files...")

    elapsed = time.time() - start
    log(f"DONE. scanned={stats['files_scanned']}, segments={stats['segments_written']}, time={int(elapsed)}s")

    stats_path = Path(OUT).with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()