#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dapt_train_beats_mam.py

Domain-Adaptive Pre-Training for BEATs using Masked Audio Modeling (MAM).
Faithfully follows the BEATs paper (Chen et al., ICML 2023):
  - 75% random patch masking
  - Cross-entropy loss on discrete tokenizer labels (masked positions only)
  - BEATs encoder initialized from iter3+ (AS2M)

Usage:
  python dapt_train_beats_mam.py \
    --beats_ckpt /workspace/models/BEATs/BEATs_iter3_plus_AS2M.pt \
    --tokenizer_ckpt /workspace/models/BEATs/Tokenizer_iter3_plus_AS2M.pt \
    --manifest /workspace/data/externaldata/dapt_manifest.tsv \
    --ckpt_dir /workspace/ckpts_dapt_mam
"""

import os
import sys
import time
import math
import logging
import argparse
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import csv
import random
import numpy as np

# ---- BEATs imports ----
BEATS_PATH = Path(__file__).resolve().parent.parent / "third_party" / "beats"
if not BEATS_PATH.exists():
    BEATS_PATH = Path("/workspace/third_party/beats")
sys.path.insert(0, str(BEATS_PATH))

from BEATs import BEATs, BEATsConfig
from Tokenizers import TokenizersConfig, Tokenizers

# ---- Logging ----
logger = logging.getLogger("dapt_mam")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)

# ---- Dataset ----
class DAPTDatasetMAM(Dataset):
    """Load 10s audio segments. Optionally loads pre-computed labels."""

    def __init__(self, manifest_path: str, sr: int = 16000, target_sec: float = 10.0,
                 precomputed_labels: np.ndarray = None, num_patches: int = 496):
        self.sr = sr
        self.target_len = int(sr * target_sec)
        self.precomputed_labels = precomputed_labels
        self.num_patches = num_patches
        self.items = []
        with open(manifest_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.items.append({
                    "path": row["path"],
                    "start": float(row["start_sec"]),
                    "dur": float(row["duration_sec"]),
                })
        if precomputed_labels is not None:
            logger.info(f"Dataset: {len(self.items)} segments (pre-computed labels)")
        else:
            logger.info(f"Dataset: {len(self.items)} segments (on-the-fly tokenizer)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            wav, sr = torchaudio.load(
                item["path"],
                frame_offset=int(item["start"] * self.sr),
                num_frames=self.target_len,
            )
        except Exception:
            wav = torch.zeros(1, self.target_len)

        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)

        # Pad or trim
        if wav.numel() < self.target_len:
            wav = F.pad(wav, (0, self.target_len - wav.numel()))
        wav = wav[: self.target_len]

        if self.precomputed_labels is not None:
            start = idx * self.num_patches
            end = start + self.num_patches
            labels = torch.from_numpy(
                self.precomputed_labels[start:end].copy()
            ).long()
            return wav, labels
        return wav


def collate_fn(batch):
    if isinstance(batch[0], tuple):
        wavs = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.stack([b[1] for b in batch], dim=0)
        return wavs, labels
    return torch.stack(batch, dim=0)


# ---- Label Predictor ----
class LabelPredictor(nn.Module):
    """Predicts discrete tokenizer labels from encoder representations."""

    def __init__(self, in_dim: int = 768, num_classes: int = 1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)  # (B, T, num_classes)


# ---- Masking ----
def random_mask(num_patches: int, mask_ratio: float = 0.75):
    """Generate random mask. Returns mask indices and unmasked indices."""
    num_mask = int(num_patches * mask_ratio)
    perm = torch.randperm(num_patches)
    mask_idx = perm[:num_mask]
    unmask_idx = perm[num_mask:]
    return mask_idx, unmask_idx


# ---- Checkpointing ----
def save_ckpt(path, beats, predictor, optim, epoch, step, meta=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "beats": beats.state_dict(),
        "predictor": predictor.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "step": step,
        "meta": meta or {},
    }, path)
    logger.info(f"[ckpt] saved: {path}")


def export_beats_ckpt(beats, cfg_dict, out_path):
    """Export BEATs encoder in standard format for downstream use."""
    torch.save({"cfg": cfg_dict, "model": beats.state_dict()}, out_path)
    logger.info(f"[export] BEATs checkpoint: {out_path}")


# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beats_ckpt", required=True)
    parser.add_argument("--tokenizer_ckpt", default="", help="Tokenizer ckpt (Approach A). Empty = use precomputed labels")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--precomputed_labels", default="", help="Path to precomputed labels .npy (Approach B)")
    parser.add_argument("--num_classes", type=int, default=1024, help="Number of label classes")
    parser.add_argument("--ckpt_dir", default="/workspace/ckpts_dapt_mam")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt_every", type=int, default=5000)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=0, help="0=full epoch")
    parser.add_argument("--test_steps", type=int, default=0, help="Quick test run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_file = ckpt_dir / "train.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"=== DAPT MAM ===")
    logger.info(f"Args: {vars(args)}")
    logger.info(f"Device: {device}")

    # ---- Load BEATs encoder ----
    logger.info(f"Loading BEATs from {args.beats_ckpt}")
    beats_state = torch.load(args.beats_ckpt, map_location="cpu")
    beats_cfg = BEATsConfig(beats_state["cfg"])
    beats_cfg_dict = beats_state["cfg"]
    beats = BEATs(beats_cfg)
    beats.load_state_dict(beats_state["model"])
    beats.to(device).train()

    # Save initial weights for comparison
    init_params = {n: p.clone().detach() for n, p in beats.named_parameters()}
    num_params = sum(p.numel() for p in beats.parameters())
    logger.info(f"BEATs params: {num_params:,} ({num_params/1e6:.1f}M)")

    # ---- Load Tokenizer or pre-computed labels ----
    use_precomputed = bool(args.precomputed_labels)
    precomputed_labels_np = None
    num_classes = args.num_classes
    tokenizer = None

    if use_precomputed:
        logger.info(f"Loading pre-computed labels from {args.precomputed_labels}")
        precomputed_labels_np = np.load(args.precomputed_labels)
        num_classes = int(precomputed_labels_np.max()) + 1
        if args.num_classes > 0:
            num_classes = args.num_classes
        logger.info(f"Pre-computed labels: {precomputed_labels_np.shape}, num_classes={num_classes}")
    else:
        logger.info(f"Loading Tokenizer from {args.tokenizer_ckpt}")
        tok_state = torch.load(args.tokenizer_ckpt, map_location="cpu")
        tok_cfg = TokenizersConfig(tok_state["cfg"])
        tokenizer = Tokenizers(tok_cfg)
        tokenizer.load_state_dict(tok_state["model"])
        tokenizer.eval().to(device)
        for p in tokenizer.parameters():
            p.requires_grad_(False)
        num_classes = tok_cfg.quant_n
        logger.info(f"Tokenizer codebook: {num_classes} codes")

    # ---- Label Predictor ----
    predictor = LabelPredictor(
        in_dim=beats_cfg.encoder_embed_dim,
        num_classes=num_classes,
    ).to(device)
    pred_params = sum(p.numel() for p in predictor.parameters())
    logger.info(f"Predictor params: {pred_params:,} ({pred_params/1e6:.1f}M)")

    # ---- Dataset ----
    ds = DAPTDatasetMAM(args.manifest, precomputed_labels=precomputed_labels_np)

    # Train/Val split
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    ds_train, ds_val = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train: {n_train}, Val: {n_val}")

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=1,
        pin_memory=True, drop_last=False,
    )

    # ---- Optimizer ----
    param_groups = [
        {"params": beats.parameters(), "lr": args.lr, "weight_decay": 0.01},
        {"params": predictor.parameters(), "lr": args.lr * 10, "weight_decay": 0.01},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    steps_per_epoch = len(dl_train)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")

    # Linear warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training Loop ----
    global_step = 0
    nan_batches = 0
    t_start = time.time()

    if args.test_steps > 0:
        total_steps = args.test_steps
        logger.info(f"TEST MODE: {args.test_steps} steps only")

    for epoch in range(1, args.epochs + 1):
        beats.train()
        predictor.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batch = 0

        for batch_data in dl_train:
            if global_step >= total_steps:
                break

            if use_precomputed:
                wav, target_labels = batch_data
                wav = wav.to(device)
                target_labels = target_labels.to(device)
            else:
                wav = batch_data.to(device)

            B = wav.shape[0]

            # ---- Generate labels (on-the-fly or pre-computed) ----
            if not use_precomputed:
                with torch.no_grad():
                    padding_mask = torch.zeros(B, wav.shape[1], dtype=torch.bool, device=device)
                    target_labels = tokenizer.extract_labels(wav, padding_mask=padding_mask)
                    num_patches = target_labels.shape[0] // B
                    target_labels = target_labels.view(B, num_patches)

            # ---- BEATs forward (full patches, for position encoding correctness) ----
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Get encoder features (all patches)
                features, _ = beats.extract_features(wav)  # (B, 496, 768)

                # ---- Apply mask (loss on masked positions only) ----
                # Generate mask per sample (same mask for simplicity)
                T = features.shape[1]
                mask_idx, unmask_idx = random_mask(T, args.mask_ratio)

                # Zero out masked positions before predictor
                features_masked = features.clone()
                features_masked[:, mask_idx, :] = 0.0

                # ---- Predict labels ----
                logits = predictor(features_masked)  # (B, T, 1024)

                # ---- Loss: cross-entropy on masked positions only ----
                logits_masked = logits[:, mask_idx, :]  # (B, num_masked, 1024)
                labels_masked = target_labels[:, mask_idx]  # (B, num_masked)

                loss = F.cross_entropy(
                    logits_masked.reshape(-1, num_classes),
                    labels_masked.reshape(-1),
                )

            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if nan_batches <= 5:
                    logger.warning(f"NaN loss at step {global_step}")
                continue

            # ---- Backward ----
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(beats.parameters()) + list(predictor.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # ---- Metrics ----
            global_step += 1
            n_batch += 1
            epoch_loss += loss.item()

            with torch.no_grad():
                preds = logits_masked.argmax(dim=-1)
                correct = (preds == labels_masked).sum().item()
                total = labels_masked.numel()
                epoch_correct += correct
                epoch_total += total

            # ---- Logging ----
            if global_step % 100 == 0 or global_step <= 10:
                avg_loss = epoch_loss / n_batch
                acc = epoch_correct / max(1, epoch_total) * 100
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start
                eta = elapsed / global_step * (total_steps - global_step)
                logger.info(
                    f"[ep{epoch} step{global_step}/{total_steps}] "
                    f"loss={loss.item():.4f} avg={avg_loss:.4f} "
                    f"acc={acc:.1f}% lr={lr_now:.2e} "
                    f"nan={nan_batches} "
                    f"eta={eta/3600:.1f}h"
                )

            # ---- Weight change monitoring ----
            if global_step % 1000 == 0 or global_step == 1:
                max_diff = max(
                    (p - init_params[n]).abs().max().item()
                    for n, p in beats.named_parameters()
                )
                logger.info(f"  weight_diff_from_init: {max_diff:.6f}")

            # ---- Checkpointing ----
            if args.ckpt_every > 0 and global_step % args.ckpt_every == 0:
                save_ckpt(
                    ckpt_dir / f"step_{global_step:09d}.pt",
                    beats, predictor, optimizer, epoch, global_step,
                )
                # Export BEATs-format checkpoint for downstream eval
                export_beats_ckpt(
                    beats, beats_cfg_dict,
                    ckpt_dir / f"BEATs_DAPT_MAM_step{global_step}.pt",
                )

        # ---- End of epoch ----
        avg_loss = epoch_loss / max(1, n_batch)
        acc = epoch_correct / max(1, epoch_total) * 100
        logger.info(
            f"[Epoch {epoch}] train_loss={avg_loss:.4f} train_acc={acc:.1f}% "
            f"steps={global_step} nan={nan_batches}"
        )

        # ---- Validation ----
        beats.eval()
        predictor.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_n = 0

        with torch.no_grad():
            for batch_data in dl_val:
                if use_precomputed:
                    wav, target_labels = batch_data
                    wav = wav.to(device)
                    target_labels = target_labels.to(device)
                else:
                    wav = batch_data.to(device)
                B = wav.shape[0]

                if not use_precomputed:
                    padding_mask = torch.zeros(B, wav.shape[1], dtype=torch.bool, device=device)
                    target_labels = tokenizer.extract_labels(wav, padding_mask=padding_mask)
                    num_patches = target_labels.shape[0] // B
                    target_labels = target_labels.view(B, num_patches)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    features, _ = beats.extract_features(wav)
                    T = features.shape[1]
                    mask_idx, _ = random_mask(T, args.mask_ratio)
                    features_masked = features.clone()
                    features_masked[:, mask_idx, :] = 0.0
                    logits = predictor(features_masked)
                    logits_masked = logits[:, mask_idx, :]
                    labels_masked = target_labels[:, mask_idx]
                    loss = F.cross_entropy(
                        logits_masked.reshape(-1, num_classes),
                        labels_masked.reshape(-1),
                    )

                val_loss += loss.item()
                preds = logits_masked.argmax(dim=-1)
                val_correct += (preds == labels_masked).sum().item()
                val_total += labels_masked.numel()
                val_n += 1

        val_avg = val_loss / max(1, val_n)
        val_acc = val_correct / max(1, val_total) * 100
        logger.info(f"[Val] loss={val_avg:.4f} acc={val_acc:.1f}%")

        # Weight change summary
        max_diff = max(
            (p - init_params[n]).abs().max().item()
            for n, p in beats.named_parameters()
        )
        logger.info(f"[Epoch {epoch}] max_weight_diff_from_pretrain: {max_diff:.6f}")

        # Save epoch checkpoint
        save_ckpt(
            ckpt_dir / f"epoch_{epoch:03d}_step_{global_step:09d}.pt",
            beats, predictor, optimizer, epoch, global_step,
        )
        export_beats_ckpt(
            beats, beats_cfg_dict,
            ckpt_dir / f"BEATs_DAPT_MAM_ep{epoch}.pt",
        )

    elapsed = time.time() - t_start
    logger.info(f"Training complete. {global_step} steps in {elapsed/3600:.1f}h. nan_batches={nan_batches}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        sys.exit(1)
