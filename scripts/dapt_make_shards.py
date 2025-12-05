# -*- coding: utf-8 -*-
"""
dapt_make_shards.py

Packs audio segments defined in a DAPT manifest into tar shards (WebDataset format).
This is an optional preprocessing step to improve I/O efficiency during large-scale training.

- Input: DAPT manifest TSV (path, start_sec, duration_sec, metadata...)
- Output: A directory containing numbered .tar files (shards).
          Each shard contains pairs of .wav (audio) and .json (metadata) files.

Usage:
    python dapt_make_shards.py \
      --tsv ./data/dapt_manifest.tsv \
      --out_dir ./data/shards \
      --shard_mb 800
"""

import os
import io
import tarfile
import json
import uuid
import argparse
import soundfile as sf
import pandas as pd
from pathlib import Path

def shard_path(out_dir, i):
    return os.path.join(out_dir, f"worlddapt-{i:06d}.tar")

def write_sample(tf, wav_bytes, meta):
    key = str(uuid.uuid4())
    info = tarfile.TarInfo(name=f"{key}.wav")
    info.size = len(wav_bytes)
    tf.addfile(info, io.BytesIO(wav_bytes))
    
    j = json.dumps(meta).encode("utf-8")
    info = tarfile.TarInfo(name=f"{key}.json")
    info.size = len(j)
    tf.addfile(info, io.BytesIO(j))

def load_segment(path, start_sec, dur_sec):
    with sf.SoundFile(path) as snd:
        sr = snd.samplerate
        snd.seek(int(start_sec * sr))
        x = snd.read(frames=int(dur_sec * sr), dtype='float32', always_2d=True).mean(axis=1)
    
    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, x, sr, format="WAV", subtype="PCM_16")
    return wav_bytes.getvalue(), sr

def main():
    parser = argparse.ArgumentParser(description="Pack DAPT manifest data into tar shards.")
    parser.add_argument("--tsv", required=True, help="Path to dapt_manifest.tsv")
    parser.add_argument("--out_dir", required=True, help="Output directory for shards")
    parser.add_argument("--shard_mb", type=int, default=800, help="Target shard size in MB")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.tsv, sep="\t")
    
    shard_i = 0
    tf = tarfile.open(shard_path(args.out_dir, shard_i), mode="w")
    bytes_in_shard = 0
    count = 0

    print(f"Starting sharding from {args.tsv} to {args.out_dir}...")

    for _, r in df.iterrows():
        try:
            wav_bytes, sr = load_segment(r.path, int(r.start_sec), int(r.duration_sec))
        except Exception:
            continue
            
        meta = {
            "site_id": r.site_id,
            "month": int(r.month),
            "diel": r.diel,
            "src_path": r.path,
            "start_sec": int(r.start_sec),
            "dur": int(r.duration_sec),
            "sr": int(sr)
        }
        write_sample(tf, wav_bytes, meta)
        bytes_in_shard += len(wav_bytes)
        count += 1
        
        if bytes_in_shard > args.shard_mb * 1024 * 1024:
            tf.close()
            print(f"Shard {shard_i} written.")
            shard_i += 1
            tf = tarfile.open(shard_path(args.out_dir, shard_i), mode="w")
            bytes_in_shard = 0
            
    tf.close()
    print(f"Done. {count} segments processed.")

if __name__ == "__main__":
    main()