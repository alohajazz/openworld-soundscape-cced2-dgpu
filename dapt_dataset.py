# -*- coding: utf-8 -*-
"""
DAPT Dataset and Sampler reading directly from TSV.

- TSV columns: path, start_sec, duration_sec, site_id, month, diel
- Processing: On-the-fly cropping -> convert to mono -> resample to 16kHz
"""
import math, random, hashlib
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

TARGET_SR = 16000
SEG_S = 10
SEG_SAMPLES = TARGET_SR * SEG_S

class DAPTDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_path: str, split: str = "train", val_ratio: float = 0.1, seed: int = 42):
        self.df = pd.read_csv(tsv_path, sep="\t")
        
        # Simple split: deterministic split based on the hash of the file path
        def split_key(p):
            h = int(hashlib.md5(p.encode("utf-8")).hexdigest(), 16)
            return (h % 1000) / 1000.0
        
        self.df["__k__"] = self.df["path"].astype(str).map(split_key)
        
        if split == "train":
            self.df = self.df[self.df["__k__"] >= val_ratio].reset_index(drop=True)
        else:
            self.df = self.df[self.df["__k__"] <  val_ratio].reset_index(drop=True)
            
        # Ensure column types are safe/correct
        self.df["start_sec"] = self.df["start_sec"].astype(int)
        self.df["duration_sec"] = self.df["duration_sec"].astype(int)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def _load_segment(self, path: str, start_sec: int, duration_sec: int):
        """
        Crop 10s segment -> mono -> 16kHz. 
        Returns zeros for corrupted files to prevent training interruption.
        """
        try:
            with sf.SoundFile(path) as snd:
                sr = snd.samplerate
                snd.seek(int(start_sec * sr))
                x = snd.read(frames=int(duration_sec * sr), dtype="float32", always_2d=True)
                x = x.mean(axis=1)  # convert to mono
        except Exception:
            return np.zeros(SEG_SAMPLES, dtype=np.float32)
        
        if x.size == 0:
            return np.zeros(SEG_SAMPLES, dtype=np.float32)
            
        # Resample if needed
        if sr != TARGET_SR:
            x = torchaudio.functional.resample(torch.tensor(x), sr, TARGET_SR).numpy()
            
        # Adjust length to exactly 10s (trim or zero-pad at the end)
        if len(x) >= SEG_SAMPLES:
            x = x[:SEG_SAMPLES]
        else:
            pad = np.zeros(SEG_SAMPLES - len(x), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)
        return x.astype(np.float32)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav = self._load_segment(r["path"], r["start_sec"], r["duration_sec"])
        # Light preprocessing (e.g., whitening) can be added here if needed
        return torch.from_numpy(wav), r["site_id"], int(r["month"]), r["diel"]

def pad_collate(batch: List[Tuple[torch.Tensor, str, int, str]]):
    # Stack as is, since all segments are fixed to 10s length
    wavs = torch.stack([b[0] for b in batch], dim=0)  # (B, T)
    # Return additional metadata if needed
    return wavs, [b[1] for b in batch], torch.tensor([b[2] for b in batch]), [b[3] for b in batch]

class DielBalancedSampler(torch.utils.data.Sampler):
    """
    Samples 'diel' categories (00-06/06-12/12-18/18-24) roughly evenly.
    Passes through if no diel information is available.
    """
    def __init__(self, dataset: DAPTDataset, shuffle: bool = True, seed: int = 123):
        diel_to_idx = {}
        for i, d in enumerate(dataset.df["diel"].astype(str).tolist()):
            diel_to_idx.setdefault(d, []).append(i)
        self.groups = [v for _, v in diel_to_idx.items()]
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        # Round-robin based on the maximum length of groups
        self.max_len = max(len(g) for g in self.groups) if self.groups else 0

    def __len__(self):
        return sum(len(g) for g in self.groups)

    def __iter__(self):
        if not self.groups:
            return iter([])
        groups = [g[:] for g in self.groups]
        if self.shuffle:
            for g in groups:
                self.rng.shuffle(g)
        
        # Round-robin sampling
        out = []
        for j in range(self.max_len):
            for g in groups:
                if j < len(g):
                    out.append(g[j])
        return iter(out)