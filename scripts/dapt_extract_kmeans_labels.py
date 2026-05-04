#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dapt_extract_features_kmeans.py

Approach B: Extract BEATs patch-level features and run k-means to create
domain-specific discrete labels for MAM DAPT.

Step 1: Extract patch-level features (768-dim) for all segments
Step 2: Run k-means (faiss) to build codebook
Step 3: Assign labels and save

Usage:
  python dapt_extract_features_kmeans.py \
    --beats_ckpt /workspace/models/BEATs/BEATs_iter3_plus_AS2M.pt \
    --manifest /workspace/data/externaldata/dapt_manifest.tsv \
    --outdir /workspace/dapt_kmeans_labels \
    --k 1024
"""

import os
import sys
import time
import argparse
import logging
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import torchaudio

BEATS_PATH = Path("/workspace/third_party/beats")
sys.path.insert(0, str(BEATS_PATH))
from BEATs import BEATs, BEATsConfig

logger = logging.getLogger("kmeans_labels")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beats_ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", default="/workspace/dapt_kmeans_labels")
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_segments", type=int, default=0, help="0=all")
    parser.add_argument("--layer", type=int, default=-1, help="-1=last layer output")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load BEATs (frozen) ----
    logger.info(f"Loading BEATs from {args.beats_ckpt}")
    state = torch.load(args.beats_ckpt, map_location="cpu")
    cfg = BEATsConfig(state["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(state["model"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- Read manifest ----
    with open(args.manifest) as f:
        reader = csv.DictReader(f, delimiter="\t")
        items = list(reader)
    if args.max_segments > 0:
        items = items[:args.max_segments]
    logger.info(f"Segments: {len(items)}")

    # ---- Dataset for batched extraction ----
    from torch.utils.data import Dataset, DataLoader

    class ExtractDataset(Dataset):
        def __init__(self, items, sr=16000, target_len=160000):
            self.items = items
            self.sr = sr
            self.target_len = target_len
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            item = self.items[idx]
            try:
                wav, sr = torchaudio.load(
                    item["path"],
                    frame_offset=int(float(item["start_sec"]) * self.sr),
                    num_frames=self.target_len,
                )
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                wav = wav.squeeze(0)
                if wav.numel() < self.target_len:
                    wav = F.pad(wav, (0, self.target_len - wav.numel()))
                wav = wav[:self.target_len]
            except Exception:
                wav = torch.zeros(self.target_len)
            return wav

    # ---- Step 1: Extract features in chunks ----
    # Total: 196K segs * 496 patches * 768 dim * 4 bytes ≈ 282GB → must process in chunks
    num_patches = 496
    chunk_size = 2000  # segments per chunk (~3GB each)
    done_marker = outdir / "extraction_done.txt"
    chunk_files = sorted(outdir.glob("features_chunk_*.npy"))

    if done_marker.exists() and len(chunk_files) > 0:
        logger.info(f"Extraction already done ({len(chunk_files)} chunks)")
    else:
        logger.info(f"Extracting features (batch={args.batch_size}, workers={args.num_workers}, chunk={chunk_size} segs)...")
        ds = ExtractDataset(items)
        dl = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        chunk_feats = []
        chunk_count = len(chunk_files)  # resume from existing chunks
        seg_idx_start = chunk_count * chunk_size
        t0 = time.time()
        seg_idx = 0

        for batch_wav in dl:
            seg_idx += batch_wav.shape[0]
            if seg_idx <= seg_idx_start:
                continue  # skip already-processed chunks

            batch_wav = batch_wav.to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                feats, _ = model.extract_features(batch_wav)
            chunk_feats.append(feats.float().cpu().numpy().reshape(-1, 768))

            segs_in_chunk = sum(c.shape[0] for c in chunk_feats) // num_patches
            if segs_in_chunk >= chunk_size or seg_idx >= len(items):
                chunk = np.concatenate(chunk_feats, axis=0)
                chunk_path = outdir / f"features_chunk_{chunk_count:04d}.npy"
                np.save(chunk_path, chunk)
                chunk_count += 1
                chunk_feats = []

                elapsed = time.time() - t0
                segs_done = seg_idx_start + (seg_idx - seg_idx_start)
                remaining = len(items) - segs_done
                eta = elapsed / max(1, seg_idx - seg_idx_start) * remaining
                logger.info(f"  {segs_done}/{len(items)} ({elapsed:.0f}s, eta {eta:.0f}s), chunk {chunk_count}")

        with open(done_marker, "w") as f:
            f.write(f"chunks={chunk_count}\n")
        chunk_files = sorted(outdir.glob("features_chunk_*.npy"))
        logger.info(f"Extraction done: {len(chunk_files)} chunks")

    # ---- Step 2: k-means (subsample for training, chunk-wise for assignment) ----
    labels_path = outdir / f"labels_k{args.k}.npy"
    centroids_path = outdir / f"centroids_k{args.k}.npy"

    if labels_path.exists():
        logger.info(f"Labels already exist at {labels_path}, loading...")
        labels = np.load(labels_path)
    else:
        # 2a: Subsample for k-means training (max 2M vectors from ~97M total)
        max_train_vectors = 2_000_000
        total_vectors = len(items) * num_patches
        sample_ratio = min(1.0, max_train_vectors / total_vectors)
        logger.info(f"Subsampling {sample_ratio*100:.1f}% of vectors for k-means training...")

        rng = np.random.RandomState(42)
        train_samples = []
        for cp in chunk_files:
            chunk = np.load(cp)
            n = max(1, int(chunk.shape[0] * sample_ratio))
            idx = rng.choice(chunk.shape[0], n, replace=False)
            train_samples.append(chunk[idx])
        train_data = np.concatenate(train_samples, axis=0)

        # L2 normalize
        norms = np.linalg.norm(train_data, axis=1, keepdims=True)
        train_data = (train_data / np.clip(norms, 1e-8, None)).astype(np.float32)
        logger.info(f"k-means training data: {train_data.shape}")

        # 2b: Run k-means
        t0 = time.time()
        try:
            import faiss
            kmeans = faiss.Kmeans(768, args.k, niter=20, verbose=True,
                                 gpu=torch.cuda.is_available(), seed=42)
            kmeans.train(train_data)
            centroids = kmeans.centroids
            logger.info(f"faiss k-means done in {time.time()-t0:.0f}s")
        except ImportError:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=args.k, batch_size=10000,
                                n_init=3, max_iter=100, random_state=42, verbose=1)
            km.fit(train_data)
            centroids = km.cluster_centers_
            logger.info(f"sklearn k-means done in {time.time()-t0:.0f}s")

        np.save(centroids_path, centroids)

        # 2c: Assign labels chunk-by-chunk
        logger.info("Assigning labels chunk-by-chunk...")
        c_norm = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8, None)
        c_norm = c_norm.astype(np.float32)

        try:
            import faiss
            index = faiss.IndexFlatL2(768)
            index.add(c_norm)
        except ImportError:
            index = None

        all_labels = []
        for ci, cp in enumerate(chunk_files):
            chunk = np.load(cp)
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            chunk_norm = (chunk / np.clip(norms, 1e-8, None)).astype(np.float32)

            if index is not None:
                _, lab = index.search(chunk_norm, 1)
                lab = lab.squeeze(-1)
            else:
                from sklearn.metrics import pairwise_distances_argmin
                lab = pairwise_distances_argmin(chunk_norm, c_norm)

            all_labels.append(lab.astype(np.int64))
            if (ci + 1) % 10 == 0:
                logger.info(f"  assigned chunk {ci+1}/{len(chunk_files)}")

        labels = np.concatenate(all_labels, axis=0)
        np.save(labels_path, labels)
        logger.info(f"Labels saved: {labels.shape} to {labels_path}")

    # ---- Step 3: Quality report ----
    num_patches_per_seg = 496
    num_segs = len(items)

    unique_labels = np.unique(labels)
    counts = np.bincount(labels, minlength=args.k)
    probs = counts / counts.sum()
    entropy = -(probs[probs > 0] * np.log2(probs[probs > 0])).sum()
    max_entropy = np.log2(args.k)

    logger.info(f"\n=== k-means Label Quality Report ===")
    logger.info(f"Active codes: {len(unique_labels)}/{args.k} ({len(unique_labels)/args.k*100:.1f}%)")
    logger.info(f"Entropy: {entropy:.2f}/{max_entropy:.2f} = {entropy/max_entropy*100:.1f}%")

    sorted_counts = np.sort(counts[counts > 0])[::-1]
    cumsum = np.cumsum(sorted_counts) / sorted_counts.sum()
    for pct in [0.5, 0.8, 0.9]:
        idx = np.searchsorted(cumsum, pct)
        logger.info(f"  {pct*100:.0f}% coverage: top {idx+1} codes")

    top10_pct = sorted_counts[:10].sum() / sorted_counts.sum() * 100
    logger.info(f"Top 10 codes: {top10_pct:.1f}%")

    # Per-segment diversity
    if num_segs * num_patches_per_seg <= len(labels):
        seg_labels = labels[:num_segs * num_patches_per_seg].reshape(num_segs, num_patches_per_seg)
        per_seg_unique = np.array([len(np.unique(seg_labels[i])) for i in range(min(500, num_segs))])
        logger.info(f"Per-segment unique: mean={per_seg_unique.mean():.1f}, min={per_seg_unique.min()}, max={per_seg_unique.max()}")

    logger.info(f"\nOutput files:")
    logger.info(f"  Features: {features_path} ({os.path.getsize(features_path)/1e9:.1f}GB)")
    logger.info(f"  Labels:   {labels_path}")
    logger.info(f"  Centroids:{centroids_path}")


if __name__ == "__main__":
    main()
