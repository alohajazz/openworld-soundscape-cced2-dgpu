#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dapt_train_beats_mam_fulldata.py

Full-scale DAPT with on-the-fly k-means labeling.
Handles mixed sample rates (5kHz-200kHz) via torchaudio resampling.
Uses k-means centroids for on-the-fly label assignment (no pre-computed labels needed).

Usage:
  python dapt_train_beats_mam_fulldata.py \
    --beats_ckpt /workspace/models/BEATs/BEATs_iter3_plus_AS2M.pt \
    --centroids /workspace/dapt_kmeans_labels/centroids_k1024.npy \
    --manifest /workspace/data/externaldata/dapt_manifest_all.tsv \
    --ckpt_dir /workspace/ckpts_dapt_mam_fulldata
"""

import os, sys, time, math, logging, argparse, traceback, csv
from pathlib import Path
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np

BEATS_PATH = Path("/workspace/third_party/beats")
sys.path.insert(0, str(BEATS_PATH))
from BEATs import BEATs, BEATsConfig

logger = logging.getLogger("dapt_mam_full")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)


# ---- Dataset with resampling support ----
class DAPTDatasetFull(Dataset):
    def __init__(self, manifest_path: str, target_sr: int = 16000, target_sec: float = 10.0):
        self.target_sr = target_sr
        self.target_len = int(target_sr * target_sec)
        self.target_sec = target_sec
        self.items = []
        with open(manifest_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.items.append({
                    "path": row["path"],
                    "start": float(row["start_sec"]),
                    "dur": float(row["duration_sec"]),
                })

    def __len__(self):
        return len(self.items)

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_resampler(orig_sr, target_sr):
        if orig_sr == target_sr:
            return None
        return torchaudio.transforms.Resample(orig_sr, target_sr)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            # All files are pre-resampled to 16kHz via dapt_manifest_all_16k.tsv
            frame_offset = int(item["start"] * self.target_sr)
            wav, sr = torchaudio.load(item["path"], frame_offset=frame_offset,
                                      num_frames=self.target_len)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = wav.squeeze(0)
            # Safety resample (should not be needed with 16k manifest)
            if sr != self.target_sr:
                resampler = self._get_resampler(sr, self.target_sr)
                wav = resampler(wav)
        except Exception:
            wav = torch.zeros(self.target_len)

        # Pad or trim to exact target length
        if wav.numel() < self.target_len:
            wav = F.pad(wav, (0, self.target_len - wav.numel()))
        wav = wav[:self.target_len]
        return wav


def collate_fn(batch):
    return torch.stack(batch, dim=0)


# ---- On-the-fly k-means labeling ----
class KMeansLabeler:
    """Assigns labels using pre-computed k-means centroids on GPU."""
    def __init__(self, centroids_path: str, device: torch.device):
        centroids = np.load(centroids_path).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.clip(norms, 1e-8, None)
        self.centroids = torch.from_numpy(centroids).to(device)  # (K, D)
        self.k = centroids.shape[0]
        logger.info(f"KMeansLabeler: {self.k} centroids, dim={centroids.shape[1]}")

    @torch.no_grad()
    def assign(self, features: torch.Tensor) -> torch.Tensor:
        """Assign labels to patch features.
        Args: features (B, T, D)
        Returns: labels (B, T) as long tensor
        """
        B, T, D = features.shape
        feat_flat = features.reshape(B * T, D)
        # L2 normalize
        feat_norm = F.normalize(feat_flat, dim=-1)
        # Compute distances (use negative cosine sim = find closest centroid)
        sims = feat_norm @ self.centroids.t()  # (B*T, K)
        labels = sims.argmax(dim=-1)  # (B*T,)
        return labels.reshape(B, T)


# ---- Label Predictor ----
class LabelPredictor(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes),
        )
    def forward(self, x):
        return self.head(x)


def random_mask(num_patches: int, mask_ratio: float = 0.75):
    num_mask = int(num_patches * mask_ratio)
    perm = torch.randperm(num_patches)
    return perm[:num_mask], perm[num_mask:]


def save_ckpt(path, beats, predictor, optim, epoch, step, meta=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "beats": beats.state_dict(), "predictor": predictor.state_dict(),
        "optim": optim.state_dict(), "epoch": epoch, "step": step,
        "meta": meta or {},
    }, path)
    logger.info(f"[ckpt] saved: {path}")


def export_beats_ckpt(beats, cfg_dict, out_path):
    torch.save({"cfg": cfg_dict, "model": beats.state_dict()}, out_path)
    logger.info(f"[export] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beats_ckpt", required=True)
    parser.add_argument("--centroids", required=True, help="k-means centroids .npy")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ckpt_dir", default="/workspace/ckpts_dapt_mam_fulldata")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1024)
    parser.add_argument("--ckpt_every", type=int, default=10000)
    parser.add_argument("--val_ratio", type=float, default=0.01)  # 1% for 2M segments
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--test_steps", type=int, default=0)
    parser.add_argument("--resume", default="", help="Resume from checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(ckpt_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"=== DAPT MAM Full-Data ===")
    logger.info(f"Args: {vars(args)}")

    # Load BEATs
    beats_state = torch.load(args.beats_ckpt, map_location="cpu")
    beats_cfg = BEATsConfig(beats_state["cfg"])
    beats_cfg_dict = beats_state["cfg"]
    beats = BEATs(beats_cfg)
    beats.load_state_dict(beats_state["model"])
    beats.to(device).train()
    init_params = {n: p.clone().detach() for n, p in beats.named_parameters()}
    logger.info(f"BEATs: {sum(p.numel() for p in beats.parameters())/1e6:.1f}M params")

    # K-means labeler (on-the-fly, no pre-computed labels needed)
    labeler = KMeansLabeler(args.centroids, device)

    # Label Predictor
    predictor = LabelPredictor(beats_cfg.encoder_embed_dim, args.num_classes).to(device)
    logger.info(f"Predictor: {sum(p.numel() for p in predictor.parameters())/1e6:.1f}M params")

    # Dataset
    ds = DAPTDatasetFull(args.manifest)
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    ds_train, ds_val = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    logger.info(f"Train: {n_train:,}, Val: {n_val:,}")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=args.num_workers,
                          pin_memory=True, drop_last=True,
                          persistent_workers=args.num_workers > 0, prefetch_factor=2)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=1, pin_memory=True)

    # Optimizer
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
    logger.info(f"Steps/epoch: {steps_per_epoch:,}, Total: {total_steps:,}, Warmup: {warmup_steps:,}")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    global_step = 0
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        beats.load_state_dict(ckpt["beats"])
        predictor.load_state_dict(ckpt["predictor"])
        optimizer.load_state_dict(ckpt["optim"])
        global_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        logger.info(f"Resumed from {args.resume} (step={global_step}, epoch={start_epoch})")

    # Training
    nan_batches = 0
    t_start = time.time()
    if args.test_steps > 0:
        total_steps = args.test_steps

    for epoch in range(start_epoch, args.epochs + 1):
        beats.train(); predictor.train()
        epoch_loss = 0.0; epoch_correct = 0; epoch_total = 0; n_batch = 0

        for wav in dl_train:
            if global_step >= total_steps: break
            wav = wav.to(device)
            B = wav.shape[0]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Single forward pass: use features for both labeling and training
                features, _ = beats.extract_features(wav)

                # Generate labels from detached features (no gradient through labeler)
                with torch.no_grad():
                    target_labels = labeler.assign(features.detach())  # (B, T)
                T = features.shape[1]
                mask_idx, unmask_idx = random_mask(T, args.mask_ratio)
                features_masked = features.clone()
                features_masked[:, mask_idx, :] = 0.0
                logits = predictor(features_masked)
                logits_masked = logits[:, mask_idx, :]
                labels_masked = target_labels[:, mask_idx]
                loss = F.cross_entropy(logits_masked.reshape(-1, args.num_classes),
                                       labels_masked.reshape(-1))

            if not torch.isfinite(loss):
                nan_batches += 1; optimizer.zero_grad(set_to_none=True)
                if nan_batches <= 5: logger.warning(f"NaN at step {global_step}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(beats.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)

            global_step += 1; n_batch += 1
            epoch_loss += loss.item()
            with torch.no_grad():
                epoch_correct += (logits_masked.argmax(-1) == labels_masked).sum().item()
                epoch_total += labels_masked.numel()

            if global_step % 500 == 0 or global_step <= 10:
                avg = epoch_loss / n_batch
                acc = epoch_correct / max(1, epoch_total) * 100
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start
                eta = elapsed / global_step * (total_steps - global_step)
                logger.info(f"[ep{epoch} step{global_step}/{total_steps}] "
                            f"loss={loss.item():.4f} avg={avg:.4f} acc={acc:.1f}% "
                            f"lr={lr_now:.2e} nan={nan_batches} eta={eta/3600:.1f}h")

            if global_step % 5000 == 0:
                max_diff = max((p - init_params[n]).abs().max().item()
                               for n, p in beats.named_parameters())
                logger.info(f"  weight_diff: {max_diff:.6f}")

            if args.ckpt_every > 0 and global_step % args.ckpt_every == 0:
                save_ckpt(ckpt_dir / f"step_{global_step:09d}.pt",
                          beats, predictor, optimizer, epoch, global_step)
                export_beats_ckpt(beats, beats_cfg_dict,
                                  ckpt_dir / f"BEATs_DAPT_MAM_step{global_step}.pt")

        avg = epoch_loss / max(1, n_batch)
        acc = epoch_correct / max(1, epoch_total) * 100
        logger.info(f"[Epoch {epoch}] loss={avg:.4f} acc={acc:.1f}% steps={global_step}")

        # Validation
        beats.eval(); predictor.eval()
        val_loss = 0.0; val_n = 0
        with torch.no_grad():
            for wav in dl_val:
                wav = wav.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    features, _ = beats.extract_features(wav)
                    target_labels = labeler.assign(features)
                    T = features.shape[1]
                    mask_idx, _ = random_mask(T, args.mask_ratio)
                    features_masked = features.clone()
                    features_masked[:, mask_idx, :] = 0.0
                    logits = predictor(features_masked)
                    loss = F.cross_entropy(logits[:, mask_idx, :].reshape(-1, args.num_classes),
                                           target_labels[:, mask_idx].reshape(-1))
                val_loss += loss.item(); val_n += 1
        logger.info(f"[Val] loss={val_loss/max(1,val_n):.4f}")

        max_diff = max((p - init_params[n]).abs().max().item()
                       for n, p in beats.named_parameters())
        logger.info(f"[Epoch {epoch}] weight_diff: {max_diff:.6f}")

        save_ckpt(ckpt_dir / f"epoch_{epoch:03d}_step_{global_step:09d}.pt",
                  beats, predictor, optimizer, epoch, global_step)
        export_beats_ckpt(beats, beats_cfg_dict,
                          ckpt_dir / f"BEATs_DAPT_MAM_ep{epoch}.pt")

    logger.info(f"Done. {global_step} steps in {(time.time()-t_start)/3600:.1f}h. nan={nan_batches}")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
