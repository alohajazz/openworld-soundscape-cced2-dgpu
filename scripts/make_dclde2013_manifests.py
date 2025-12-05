#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dclde2013_manifests.py

Generates evaluation manifests for the DCLDE2013 dataset (NEFSC_SBNMS_200903_NOPP6_CH10).

Inputs:
  - input_root: Root directory of the DCLDE2013 dataset (containing 'source-audio' and 'detections').

Outputs:
  1) dclde2013_events_gt.csv              (Ground Truth events for PR evaluation)
  2) dclde2013_continuous_manifest.csv    (Continuous recording manifest for OP evaluation)

Design Logic:
- Extracts local time (US/Eastern) from WAV filenames and converts to UTC to match detection logs.
- Automatically infers detection log columns (supports both relative seconds and absolute timestamps).
- Converts 'start_utc' back to local time (Eastern) to assign 'diel' bins (00–06/06–12/12–18/18–24).
- Assigns a fixed site_id="SBNMS_CH10_200903".

Usage:
    python make_dclde2013_manifests.py \
      --input_root /path/to/DCLDE2013/nefsc_sbnms_200903_nopp6_ch10 \
      --out_dir    ./manifests
"""

import csv
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo  # Python >= 3.9
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore
    except ImportError:
        print("Error: 'zoneinfo' module not found. Please install 'backports.zoneinfo' for Python < 3.9.", file=sys.stderr)
        sys.exit(1)

import soundfile as sf

# Constants
SITE_ID = "SBNMS_CH10_200903"
EASTERN = ZoneInfo("America/New_York")

# ---- Helpers ----
# Matches filenames like: NOPP6_EST_20090328_000000_CH10.wav
WAV_RE = re.compile(r".*_(\d{8})_(\d{6})_CH(\d+)\.wav$", re.IGNORECASE)

def parse_wav_start_utc(wav_path: Path) -> datetime:
    m = WAV_RE.match(wav_path.name)
    if not m:
        raise ValueError(f"Unexpected WAV name format: {wav_path.name}")
    ymd, hms, ch = m.group(1), m.group(2), m.group(3)
    
    # Parse as naive local time -> Localize to Eastern (handling DST) -> Convert to UTC
    local_dt = datetime.strptime(f"{ymd}_{hms}", "%Y%m%d_%H%M%S").replace(tzinfo=EASTERN)
    return local_dt.astimezone(timezone.utc)

def wav_duration_sec(wav_path: Path) -> float:
    with sf.SoundFile(str(wav_path)) as f:
        return float(len(f) / f.samplerate)

def diel_bin(dt_utc: datetime) -> str:
    # Convert back to local (Eastern) time to determine 4-bin diel cycle
    local = dt_utc.astimezone(EASTERN)
    h = local.hour
    if 0 <= h < 6: return "00-06"
    if 6 <= h < 12: return "06-12"
    if 12 <= h < 18: return "12-18"
    return "18-24"

def month_of(dt_utc: datetime) -> int:
    return dt_utc.astimezone(EASTERN).month

def parse_abs_time_to_utc(s: str) -> datetime:
    """Parses various string formats to datetime (UTC)."""
    s = s.strip()
    # Handle ISO8601 variations
    # e.g., 2009-03-28T00:00:00Z, 2009-03-28T00:00:00.123Z, 2009-03-28 00:00:00-04:00
    try:
        # Handle Z suffix manually for older python versions if needed
        if s.endswith("Z"):
            s2 = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s2)
            return dt.astimezone(timezone.utc)
        
        # Parse as ISO; if no timezone, assume Eastern (dataset specific)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=EASTERN)
        return dt.astimezone(timezone.utc)
    except Exception:
        # Try legacy formats
        fmts = [
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for f in fmts:
            try:
                dt = datetime.strptime(s, f)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=EASTERN)
                return dt.astimezone(timezone.utc)
            except Exception:
                continue
        
        # Fallback: Numeric timestamp
        try:
            sec = float(s)
            return datetime.fromtimestamp(sec, tz=timezone.utc)
        except Exception:
            raise ValueError(f"Could not parse timestamp: {s}")

def parse_abs_time_to_utc_iso(s: str) -> str:
    return parse_abs_time_to_utc(s).strftime("%Y-%m-%dT%H:%M:%S%z")

def try_parse_detection_row(row: dict):
    """
    Parser for DCLDE2013/Raven logs, handling multiple dialects.
    
    Priority 1: Absolute Time (Start_DateTime_ISO8601 / End_DateTime_ISO8601)
    Priority 2: Raven Relative Seconds (Begin Time (s) / End Time (s))
    Priority 3: Generic Relative Seconds (start_sec / duration_sec)
    
    Returns:
      ("absolute", utc_start_iso, duration_sec or None)
      ("relative", start_sec, duration_sec or None)
    """
    # Normalize keys (strip whitespace/BOM)
    for k in list(row.keys()):
        v = row.pop(k)
        row[k.strip()] = v

    # ---- Priority 1: Absolute Time (Raven ISO8601)
    s_abs = None
    e_abs = None
    for key in ["Start_DateTime_ISO8601", "Start_ISO8601", "Start_UTC", "UTC_Start", "Date/Time", "utc_start", "start_time_utc"]:
        if key in row and str(row[key]).strip():
            s_abs = str(row[key]).strip(); break
    for key in ["End_DateTime_ISO8601", "End_ISO8601", "End_UTC", "UTC_End", "end_time_utc"]:
        if key in row and str(row[key]).strip():
            e_abs = str(row[key]).strip(); break
            
    if s_abs is not None:
        dur = None
        if e_abs is not None:
            try:
                t0 = parse_abs_time_to_utc(s_abs)
                t1 = parse_abs_time_to_utc(e_abs)
                dur = max(0.0, (t1 - t0).total_seconds())
            except Exception:
                dur = None
        return ("absolute", parse_abs_time_to_utc_iso(s_abs), dur)

    # ---- Priority 2: Raven Relative Seconds
    begin_keys = ["Begin Time (s)", "Begin Time", "Begin_Time_(s)", "Begin_Time"]
    end_keys   = ["End Time (s)", "End Time", "End_Time_(s)", "End_Time"]
    bkey = next((k for k in begin_keys if k in row and str(row[k]).strip() != ""), None)
    ekey = next((k for k in end_keys   if k in row and str(row[k]).strip() != ""), None)
    
    if bkey is not None and ekey is not None:
        try:
            b = float(row[bkey])
            e = float(row[ekey])
            return ("relative", b, max(0.0, e - b))
        except Exception:
            pass

    # ---- Priority 3: Generic Relative Seconds
    candidates_start = ["start_sec","start","begin_sec","begin","t_start","Start"]
    candidates_dur   = ["duration_sec","dur","Duration","Duration (s)","length","Length (s)"]
    key_s = next((c for c in candidates_start if c in row and str(row[c]).strip() != ""), None)
    key_d = next((c for c in candidates_dur   if c in row and str(row[c]).strip() != ""), None)
    
    if key_s is not None:
        try:
            s = float(row[key_s])
            d = float(row[key_d]) if key_d else None
            return ("relative", s, d)
        except Exception:
            pass

    raise ValueError(f"Unrecognized detection row format: keys={list(row.keys())[:10]}...")

def pick_label(row: dict, default_label: str) -> str:
    # Priority: Species > Call type > Label
    for key in ["Species", "Call type", "Call_type", "Label", "Call", "Type"]:
        if key in row and str(row[key]).strip():
            val = str(row[key]).strip().lower()
            val = val.replace(" ", "_").replace("-", "_")
            if "upcall" in val and "right" in val:
                return "right_whale_upcall"
            return val
    return default_label


# ---- Main Script ----

def main():
    parser = argparse.ArgumentParser(description="Generate DCLDE2013 evaluation manifests.")
    parser.add_argument("--input_root", required=True, type=Path,
                        help="Root directory of the DCLDE2013 dataset (containing 'source-audio' and 'detections').")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Output directory for the generated CSV manifests.")
    args = parser.parse_args()

    input_root = args.input_root
    out_dir = args.out_dir
    
    if not input_root.exists():
        print(f"Error: Input root not found: {input_root}", file=sys.stderr)
        sys.exit(1)
        
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Scan WAVs & Index by UTC start ----
    print(f"Scanning WAV files in {input_root / 'source-audio'}...")
    wav_list = sorted((input_root / "source-audio").glob("*.wav"))
    if not wav_list:
        print("Error: No .wav files found under source-audio directory.", file=sys.stderr)
        sys.exit(1)

    wav_meta = []
    for w in wav_list:
        try:
            utc0 = parse_wav_start_utc(w)
            dur = wav_duration_sec(w)
            utc1 = utc0 + timedelta(seconds=dur)
            wav_meta.append({"path": w, "utc0": utc0, "utc1": utc1, "dur": dur})
        except Exception as e:
            print(f"Warning: Skipping {w.name} due to error: {e}", file=sys.stderr)
            
    wav_meta.sort(key=lambda x: x["utc0"])
    
    def find_wav_for_utc(utc_dt: datetime):
        # Linear search is sufficient for this dataset size
        for m in wav_meta:
            if m["utc0"] <= utc_dt < m["utc1"]:
                return m
        return None

    # ---- Parse Detections ----
    det_dir = input_root / "detections"
    print(f"Parsing detections in {det_dir}...")
    
    csv_files = []
    csv_files += list(det_dir.glob("*upcall*-log*.csv"))
    csv_files += list(det_dir.glob("*allbaleen*updated*.csv"))
    if not csv_files:
        csv_files = list(det_dir.glob("*.csv"))  # Fallback

    events = []  # dict: wav_path, start_sec, duration_sec, label, utc_start_iso
    default_label_by_file = {
        "upcall": "right_whale_upcall",
        "allbaleen": "baleen_call",
    }

    for csv_path in sorted(csv_files):
        fname = csv_path.name.lower()
        default_label = "event"
        for key, lab in default_label_by_file.items():
            if key in fname:
                default_label = lab
                break
                
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    mode, v1, v2 = try_parse_detection_row(row)
                    
                    if mode == "relative":
                        start_rel = float(v1)
                        dur = float(v2) if v2 is not None else 1.0
                        
                        # Note: Mapping relative time requires knowing which WAV file the log corresponds to.
                        # This script assumes the relative time applies to the first WAV file that covers the range,
                        # or falls back to the first file. This path is rarely used for DCLDE2013 logs 
                        # as they typically contain absolute timestamps.
                        candidate = None
                        for m in wav_meta:
                            if 0 <= start_rel < m["dur"]:
                                candidate = m; break
                        if candidate is None and wav_meta:
                            candidate = wav_meta[0] # Fallback
                            
                        if candidate:
                            utc_start_iso = (candidate["utc0"] + timedelta(seconds=start_rel)).strftime("%Y-%m-%dT%H:%M:%S%z")
                            events.append({
                                "wav_path": str(candidate["path"]),
                                "start_sec": round(start_rel, 3),
                                "duration_sec": round(dur, 3),
                                "label": pick_label(row, default_label),
                                "utc_start_iso": utc_start_iso,
                            })
                            
                    else:
                        utc_iso = parse_abs_time_to_utc_iso(v1)
                        dt_utc = datetime.strptime(utc_iso, "%Y-%m-%dT%H:%M:%S%z")
                        
                        # Map UTC to WAV file
                        m = find_wav_for_utc(dt_utc)
                        if m is None:
                            continue # Skip events outside of audio coverage
                            
                        start_rel = (dt_utc - m["utc0"]).total_seconds()
                        dur = float(v2) if v2 is not None else 1.0
                        
                        events.append({
                            "wav_path": str(m["path"]),
                            "start_sec": round(start_rel, 3),
                            "duration_sec": round(dur, 3),
                            "label": pick_label(row, default_label),
                            "utc_start_iso": utc_iso,
                        })
                except Exception as e:
                    continue

    # ---- Write Events Manifest (for PR evaluation) ----
    ev_out = out_dir / "dclde2013_events_gt.csv"
    with open(ev_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "start_sec", "duration_sec", "site_id", "month", "diel", "label", "utc_start_iso"])
        for e in events:
            dt_utc = datetime.strptime(e["utc_start_iso"], "%Y-%m-%dT%H:%M:%S%z")
            w.writerow([
                e["wav_path"],
                f'{e["start_sec"]:.3f}',
                f'{e["duration_sec"]:.3f}',
                SITE_ID,
                month_of(dt_utc),
                diel_bin(dt_utc),
                e["label"],
                e["utc_start_iso"],
            ])

    # ---- Write Continuous Manifest (for OP evaluation) ----
    cont_out = out_dir / "dclde2013_continuous_manifest.csv"
    with open(cont_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "start_sec", "duration_sec", "site_id", "month", "diel"])
        for m in wav_meta:
            # Use the start time of the WAV file as the representative time for diel/month
            w.writerow([
                str(m["path"]),
                "0.0",
                f"{m['dur']:.3f}",
                SITE_ID,
                month_of(m["utc0"]),
                diel_bin(m["utc0"]),
            ])

    print(f"[OK] Events manifest saved to: {ev_out}")
    print(f"[OK] Continuous manifest saved to: {cont_out}")
    print(f"Stats: {len(events)} events, {len(wav_meta)} wav files processed.")

if __name__ == "__main__":
    main()