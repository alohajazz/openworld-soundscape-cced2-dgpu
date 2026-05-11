#!/usr/bin/env python3
"""Pre-cache: decode each unique FLAC once, mean to mono, resample to 16 kHz, save as float32 .npy.
Parallelised across CPU workers. One-time cost; eliminates per-window decode + resample."""
import os, sys, time
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import numpy as np
import torch, torchaudio

CACHE_DIR = "/workspace/cache/16k_mono"
TARGET_SR = 16000
MANIFESTS = [
    "/workspace/hiceas_1706_species_manifest_winsafe.csv",
    "/workspace/hiceas_ood_manifest_winsafe.csv",
    "/workspace/hiceas_ood_manifest_full_winsafe.csv",
]

def cache_path_for(src: str) -> str:
    base = Path(src).stem
    return os.path.join(CACHE_DIR, base + ".npy")

def process_one(src: str) -> tuple:
    out = cache_path_for(src)
    if os.path.exists(out):
        try:
            arr = np.load(out, mmap_mode="r")
            return (src, "skip", arr.shape[0])
        except Exception:
            pass  # fall through to re-decode
    try:
        t0 = time.time()
        info = torchaudio.info(src)
        wav, sr = torchaudio.load(src)  # full file
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != TARGET_SR:
            wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
        arr = wav.squeeze(0).contiguous().numpy().astype(np.float32)
        np.save(out, arr)
        return (src, "ok", arr.shape[0], time.time() - t0)
    except Exception as e:
        return (src, "err", repr(e))

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = set()
    for mf in MANIFESTS:
        if os.path.exists(mf):
            df = pd.read_csv(mf, usecols=["path"])
            paths.update(df["path"].unique())
    paths = sorted(paths)
    print(f"Unique paths to cache: {len(paths)}", flush=True)
    n_workers = min(8, mp.cpu_count())
    print(f"Workers: {n_workers}", flush=True)
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    errors = []
    with mp.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, paths, chunksize=4)):
            status = res[1]
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            elif status == "err":
                n_err += 1
                errors.append(res)
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(paths) - (i + 1)) / max(rate, 1e-6)
                print(f"  {i+1}/{len(paths)} ok={n_ok} skip={n_skip} err={n_err} elapsed={el:.0f}s rate={rate:.2f}/s eta={eta:.0f}s", flush=True)
    print(f"DONE: ok={n_ok} skip={n_skip} err={n_err} elapsed={time.time()-t0:.0f}s")
    if errors:
        print("FIRST 10 ERRORS:")
        for e in errors[:10]:
            print(f"  {e}")

if __name__ == "__main__":
    main()
