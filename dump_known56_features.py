#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dump_known56_features.py

A utility script to dump features (embeddings and/or logits) for a "known" labeled dataset
(e.g., the 56-class training set). This is distinct from the evaluation scripts.

Inputs:
    - CSV file containing at least "path" and "label" columns.
      (Optionally "center_sec" column; if missing, defaults to middle of the segment).

Outputs:
    - outdir/
        - embeddings_000.npy, embeddings_001.npy, ...  ([N_i, D] Time-pooled BEATs features)
        - logits_000.npy,     logits_001.npy, ...      ([N_i, 56] SED head logits, optional)
        - index_000.csv,      index_001.csv, ...       (Metadata: path, start_sec, label, etc.)
        - summary.json                                 (Metadata: total samples, dimensions, etc.)

Example Usage:
    python dump_known56_features.py \
      --csv /path/to/your_56label_train_manifest.csv \
      --ckpt_beats /path/to/checkpoints/BEATs_event_focus.pt \
      --head_ckpt  /path/to/checkpoints/SED_head_event_focus.pt \
      --outdir ./outputs/embeddings_known56 \
      --dump_logits --dump_embeddings --shard_size 200000
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Try to find bundled beats_core
BEATS_CORE = Path(__file__).parent / "beats_core"
if BEATS_CORE.exists():
    sys.path.append(str(BEATS_CORE))

try:
    from BEATs import BEATs, BEATsConfig
except ImportError:
    print("Error: Could not import 'BEATs'. Check if 'beats_core' folder exists or PYTHONPATH is set.", file=sys.stderr)
    sys.exit(1)

# -------------------------
# Dataset (Returns fixed-length segments)
# -------------------------
class SegDataset(Dataset):
    def __init__(self, csv_path: str, target_seconds: float = 10.0, sr: int = 16000):
        df = pd.read_csv(csv_path)
        req = ["path", "label"]
        for k in req:
            if k not in df.columns:
                raise ValueError(f"CSV must contain columns {req}: {csv_path}")
        self.items = df.to_dict("records")
        self.target_seconds = float(target_seconds)
        self.sr = int(sr)
        self.target_len = int(self.sr * self.target_seconds)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        path = it["path"]
        label = str(it["label"]).strip()
        # Default to center of the window if center_sec is not provided
        center_sec = float(it.get("center_sec", self.target_seconds / 2.0))
        
        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            # Fallback for missing/corrupt files: return zeros
            print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
            wav = torch.zeros(1, self.target_len)
            sr = self.sr

        # Mix down to mono
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        
        # Resample
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
            
        wav = wav.squeeze(0)  # (T,)
        T = wav.numel()
        
        # Pad or Crop to target length
        if T < self.target_len:
            wav = nn.functional.pad(wav, (0, self.target_len - T))
        else:
            wav = wav[:self.target_len]
            
        # Metadata for indexing
        meta = {
            "path": path,
            "start_sec": 0.0,
            "duration_sec": float(self.target_seconds),
            "label": label,
            "center_sec": float(center_sec)
        }
        return wav, meta


# -------------------------
# Helper: Load BEATs Model
# -------------------------
def load_beats(ckpt_path, device):
    print(f"Loading BEATs checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Handle potential differences in checkpoint keys
    cfg = BEATsConfig(ckpt["cfg"]) if "cfg" in ckpt else BEATsConfig(ckpt)
    model = BEATs(cfg)
    
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_feats(beats, wav_bT: torch.Tensor, target_seconds: float, device: torch.device):
    """
    Returns:
      feats: (B, T', C)  Time-series features from BEATs
      fps:   float       Effective frames per second
      Tp:    int         Number of time frames
    """
    feats, _ = beats.extract_features(wav_bT.to(device))  # (B, T', C)
    B, Tp, C = feats.shape
    fps = Tp / target_seconds
    return feats, fps, Tp


# -------------------------
# SED Head (Assumes 56 classes)
# -------------------------
class ConvSEDHead(nn.Module):
    def __init__(self, in_dim: int, n_class: int, p_drop=0.1):
        super().__init__()
        h = in_dim // 2
        
        def same_conv1d(cin, cout, k, d=1):
            assert (k % 2) == 1, "kernel_size should be odd for 'same' padding logic"
            pad = d * (k - 1) // 2
            return nn.Conv1d(cin, cout, kernel_size=k, padding=pad, dilation=d)
            
        self.b3  = same_conv1d(in_dim, h, k=3, d=1)
        self.b5  = same_conv1d(in_dim, h, k=5, d=1)
        self.b7d = same_conv1d(in_dim, h, k=7, d=3)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Conv1d(h * 3, n_class, kernel_size=1)

    def forward(self, feats_btC: torch.Tensor):
        x = feats_btC.transpose(1, 2)         # (B, C, T')
        b3 = self.b3(x)
        b5 = self.b5(x)
        b7 = self.b7d(x)
        h = torch.cat([b3, b5, b7], dim=1)    # (B, 3h, T')
        h = self.drop(self.act(h))
        return self.proj(h)                   # (B, K, T')


# -------------------------
# Helper: Save CSV Index
# -------------------------
def save_index_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not rows:
        return
    keys = ["path", "start_sec", "duration_sec", "label", "center_sec"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _collate_keep_meta(batch):
    # batch: List[(wav_1T, meta_dict)]
    wavs, metas = zip(*batch)                   # metas: tuple of dict
    wav_bT = torch.stack(wavs, dim=0)           # Stack fixed-length waveforms
    return wav_bT, list(metas)                  # Return metas as a list


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, 
                    help="Path to manifest CSV containing path, label, [,center_sec]")
    ap.add_argument("--ckpt_beats", required=True, 
                    help="Path to the BEATs checkpoint (e.g., BEATs_event_focus.pt)")
    ap.add_argument("--head_ckpt", default="", 
                    help="Path to the SED head checkpoint (optional, for dumping logits)")
    ap.add_argument("--outdir", required=True, 
                    help="Output directory for features and metadata")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--target_seconds", type=float, default=10.0)
    ap.add_argument("--dump_embeddings", action="store_true", 
                    help="If set, save time-pooled BEATs embeddings ([B, D])")
    ap.add_argument("--dump_logits", action="store_true", 
                    help="If set, save SED head logits ([B, 56])")
    ap.add_argument("--embedding_pool", type=str, default="mean", choices=["mean", "max"], 
                    help="Pooling method for time-series features (mean or max)")
    ap.add_argument("--shard_size", type=int, default=0, 
                    help="Split outputs into shards of this size. 0 means single file.")
    args = ap.parse_args()

    if not (args.dump_embeddings or args.dump_logits):
        ap.error("At least one of --dump_embeddings or --dump_logits must be specified.")

    os.makedirs(args.outdir, exist_ok=True)
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    
    device = torch.device(args.device)

    # Dataset/Loader
    ds = SegDataset(args.csv, target_seconds=args.target_seconds)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
        collate_fn=_collate_keep_meta,
    )

    # Load BEATs
    beats = load_beats(args.ckpt_beats, device)
    
    # Determine input dimension using dummy input
    dummy = torch.randn(1, int(16000 * args.target_seconds)).to(device)
    feats, fps, Tp = extract_feats(beats, dummy, args.target_seconds, device)
    in_dim = feats.shape[-1]
    print(f"BEATs loaded. Feature dim: {in_dim}, FPS: {fps:.2f}")

    # Load Head (Optional)
    head = None
    if args.dump_logits:
        if not args.head_ckpt:
            raise ValueError("--head_ckpt is required when --dump_logits is set.")
        print(f"Loading SED Head checkpoint: {args.head_ckpt}")
        head = ConvSEDHead(in_dim, n_class=56).to(device)
        ck = torch.load(args.head_ckpt, map_location="cpu")
        head.load_state_dict(ck["head"] if "head" in ck else ck)
        head.eval()

    # Buffers
    shard_id = 0
    emb_buf = []
    logit_buf = []
    idx_buf = []

    def flush():
        nonlocal shard_id, emb_buf, logit_buf, idx_buf
        if not idx_buf:
            return
        prefix = f"{shard_id:03d}"
        
        if args.dump_embeddings and emb_buf:
            np.save(os.path.join(args.outdir, f"embeddings_{prefix}.npy"),
                    np.concatenate(emb_buf, axis=0))  # [N_i, D]
            
        if args.dump_logits and logit_buf:
            np.save(os.path.join(args.outdir, f"logits_{prefix}.npy"),
                    np.concatenate(logit_buf, axis=0))  # [N_i, 56]
            
        save_index_csv(idx_buf, os.path.join(args.outdir, f"index_{prefix}.csv"))
        
        emb_buf.clear()
        logit_buf.clear()
        idx_buf.clear()
        shard_id += 1
        print(f"Saved shard {prefix}")

    # Main Loop
    total = 0
    print("Starting feature extraction...")
    with torch.no_grad():
        for i, (wav, meta) in enumerate(dl):
            wav = wav.to(device)  # (B, T)
            feats, fps, Tp = extract_feats(beats, wav, args.target_seconds, device)  # (B, T', C)
            
            # Embeddings (Time pooling)
            if args.dump_embeddings:
                if args.embedding_pool == "mean":
                    emb = feats.mean(dim=1)  # (B, C)
                else:
                    emb = feats.max(dim=1).values
                emb_np = emb.detach().float().cpu().numpy()  # [B, D]
                emb_buf.append(emb_np)

            # Logits (56 classes)
            if head is not None:
                logits = head(feats)              # (B, 56, T')
                agg = logits.mean(dim=2)          # Time average -> [B, 56]
                logit_np = agg.detach().float().cpu().numpy()
                logit_buf.append(logit_np)

            # Metadata index
            for m in meta:
                idx_buf.append({
                    "path": m["path"],
                    "start_sec": float(m.get("start_sec", 0.0)),
                    "duration_sec": float(m.get("duration_sec", args.target_seconds)),
                    "label": str(m.get("label", "")),
                    "center_sec": float(m.get("center_sec", args.target_seconds / 2.0))
                })

            total += len(meta)
            
            # Shard flush
            if args.shard_size > 0 and total % args.shard_size == 0:
                flush()
            
            if (i + 1) % 10 == 0:
                print(f"Processed {total} samples...", end="\r")

    # Final flush
    flush()

    # Summary
    summary = {
        "total_samples": total,
        "embedding": bool(args.dump_embeddings),
        "logits": bool(args.dump_logits),
        "fps": float(fps),
        "in_dim": int(in_dim),
        "target_seconds": float(args.target_seconds),
        "shards_written": shard_id
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Dump finished. Summary saved to {args.outdir}/summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # Optimize CUDA memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    main()