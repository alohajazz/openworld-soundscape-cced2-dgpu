#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_winaware_cached.py — wrapper around dump_winaware_features.py that swaps
SegDataset for a version reading pre-decoded 16 kHz mono .npy cache files.
All other logic (BEATs load, head, dump format, CLI) inherited unchanged."""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CACHE_DIR = "/workspace/cache/16k_mono"

# Import original module (will not auto-run main due to __name__ guard)
sys.path.insert(0, "/workspace/scripts")
import dump_winaware_features as orig

class SegDatasetCached(Dataset):
    """Reads from CACHE_DIR/<basename>.npy (16 kHz mono float32 full-file array)."""
    def __init__(self, csv_path: str, target_seconds: float = 10.0, sr: int = 16000):
        df = pd.read_csv(csv_path)
        for k in ["path", "label"]:
            if k not in df.columns:
                raise ValueError(f"CSV missing column {k}: {csv_path}")
        self.items = df.to_dict("records")
        self.target_seconds = float(target_seconds)
        self.sr = int(sr)
        self.target_len = int(self.sr * self.target_seconds)

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _cache_path(src: str) -> str:
        return os.path.join(CACHE_DIR, Path(src).stem + ".npy")

    def __getitem__(self, i):
        it = self.items[i]
        path = it["path"]
        label = str(it["label"]).strip()
        center_sec = float(it.get("center_sec", self.target_seconds / 2.0))
        if "start_sec" in it and not pd.isna(it.get("start_sec")):
            start_sec_val = float(it["start_sec"])
        else:
            start_sec_val = max(0.0, center_sec - self.target_seconds / 2.0)
        cache = self._cache_path(path)
        # mmap read; copy out the slice.
        arr = np.load(cache, mmap_mode="r")
        start = max(0, int(round(start_sec_val * self.sr)))
        end = start + self.target_len
        seg = np.asarray(arr[start:end], dtype=np.float32)
        if seg.size < self.target_len:
            seg = np.concatenate([seg, np.zeros(self.target_len - seg.size, dtype=np.float32)])
        elif seg.size > self.target_len:
            seg = seg[:self.target_len]
        wav = torch.from_numpy(seg)  # [T]
        meta = {
            "path": path,
            "start_sec": float(start_sec_val),
            "duration_sec": float(self.target_seconds),
            "label": label,
            "center_sec": float(center_sec),
        }
        return wav, meta

# Hot-swap the dataset class so orig.main() picks it up
orig.SegDataset = SegDatasetCached

if __name__ == "__main__":
    orig.main()
