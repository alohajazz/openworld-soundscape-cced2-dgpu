# -*- coding: utf-8 -*-
"""
dapt_train.py

Self-Supervised DAPT (InfoNCE/SimCLR) for BEATs.

- Loads 10s/16kHz waveforms from DAPTDataset.
- Applies augmentations (Shift + Noise).
- Extracts BEATs features -> Projection Head -> InfoNCE Loss.
- Saves checkpoints for both BEATs backbone and Projector.
- Dependencies: Requires the bundled 'beats_core' module (BEATs, BEATsConfig).
"""
import os
import sys
import time
import math
import logging
import traceback
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- Imports ----
# 1. Add local dir for dapt_dataset
sys.path.append(str(Path(__file__).parent))
try:
    from dapt_dataset import DAPTDataset, pad_collate, DielBalancedSampler
except ImportError:
    pass

# 2. Add bundled beats_core for BEATs model
BEATS_CORE_PATH = Path(__file__).parent / "beats_core"
if BEATS_CORE_PATH.exists():
    sys.path.append(str(BEATS_CORE_PATH))
    try:
        from BEATs import BEATs, BEATsConfig
    except ImportError:
        print(f"Error: Could not import BEATs from {BEATS_CORE_PATH}. Check repo structure.")
        sys.exit(1)
else:
    # Fallback if user set PYTHONPATH manually
    try:
        from BEATs import BEATs, BEATsConfig
    except ImportError:
        print("Error: beats_core/ not found and BEATs module not in path.")
        sys.exit(1)

# --------- Configuration (Defaults) ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default paths use relative locations
TSV = os.environ.get("TSV", "./data/dapt_manifest.tsv")
BATCH = int(os.environ.get("BATCH", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
USE_DIEL = os.environ.get("USE_DIEL", "1") == "1"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.1"))
TEMP = float(os.environ.get("TEMP", "0.2"))  # InfoNCE temperature

# Default: Start from the provided top-up encoder
BEATS_CKPT = os.environ.get("BEATS_CKPT", "./weights/beats_dapt_topup_encoder.pt")

MAX_STEPS_PER_EPOCH = int(os.environ.get("MAX_STEPS_PER_EPOCH", "0"))

CKPT_DIR = Path(os.environ.get("CKPT_DIR", "./ckpts_dapt"))
CKPT_STEPS = int(os.environ.get("CKPT_STEPS", "10000"))
CKPT_MINS  = int(os.environ.get("CKPT_MINS", "15"))
RESUME     = os.environ.get("RESUME", "1") == "1"
RESUME_PATH = os.environ.get("RESUME_PATH", "")

LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "dapt_train.log"
CRASH_FILE = LOG_DIR / "dapt_train.crash.log"

# --------- Logging Setup ----------
logger = logging.getLogger("dapt_beats")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(fmt)
logger.addHandler(fh)

# --------- Model Loading Helper ----------
def load_beats(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    # Default config if starting from scratch (random init)
    default_cfg = {
        "encoder_layer": 12, "encoder_embed_dim": 768, "encoder_ffn_embed_dim": 3072,
        "encoder_attention_heads": 12, "input_patch_size": 16
    }
    
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found at {ckpt_path}. Using random initialization.")
        cfg = BEATsConfig(default_cfg)
        m = BEATs(cfg)
        m.to(device).train()
        return m, default_cfg

    logger.info(f"Loading BEATs from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    
    # Handle different checkpoint formats (dict vs state_dict)
    if "cfg" in state:
        cfg_dict = state["cfg"]
        model_state = state["model"]
    elif "model" in state:
        cfg_dict = default_cfg
        model_state = state["model"]
    else:
        cfg_dict = default_cfg
        model_state = state

    m = BEATs(BEATsConfig(cfg_dict))
    # Load weights (strict=False to allow missing keys e.g. predictor)
    msg = m.load_state_dict(model_state, strict=False)
    logger.info(f"Load status: {msg}")
    
    # Set to train mode for fine-tuning
    m.to(device).train()
    return m, cfg_dict

@torch.no_grad()
def feature_dim_from_beats(beats: nn.Module, seconds: float = 10.0) -> int:
    # Get device of BEATs parameters
    dev = next(beats.parameters()).device
    # Create dummy input on the same device
    dummy = torch.zeros(1, int(16000 * seconds), device=dev)
    feats, _ = beats.extract_features(dummy)
    return feats.shape[-1]  # (B, T', C) -> returns C

# --------- Projector & Loss ----------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.2, eps: float = 1e-6):
    # Safe handling for AMP: float32 conversion -> remove NaN/Inf -> normalize with epsilon
    z1 = torch.nan_to_num(z1.float(), nan=0.0, posinf=0.0, neginf=0.0)
    z2 = torch.nan_to_num(z2.float(), nan=0.0, posinf=0.0, neginf=0.0)
    z1 = F.normalize(z1, dim=-1, eps=eps)
    z2 = F.normalize(z2, dim=-1, eps=eps)
    
    logits = (z1 @ z2.t()) / max(1e-6, temp)
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

def augment_wav(wav: torch.Tensor, noise_std=0.005, shift_max=1600):
    # 16kHz 10s: Small time shift + small noise
    if shift_max > 0:
        shift = torch.randint(-shift_max, shift_max + 1, (wav.size(0),), device=wav.device)
        wav = torch.stack([torch.roll(w, int(s.item())) for w, s in zip(wav, shift)], dim=0)
    if noise_std > 0:
        wav = wav + noise_std * torch.randn_like(wav)
    return wav

# --------- Checkpointing ----------
def latest_ckpt_path(d: Path):
    if not d.exists():
        return None
    cs = sorted(d.glob("step_*.pt"))
    return cs[-1] if cs else None

def save_ckpt(path: Path, beats, projector, optim, scaler, epoch, step, meta: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "beats": beats.state_dict(),
        "projector": projector.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "meta": meta or {},
    }, path)
    logger.info(f"[ckpt] saved: {path}")

def load_ckpt(path: Path, beats, projector, optim, scaler) -> Tuple[int, int, dict]:
    obj = torch.load(path, map_location="cpu")
    beats.load_state_dict(obj["beats"], strict=True)
    projector.load_state_dict(obj["projector"], strict=True)
    if optim is not None and "optim" in obj:
        optim.load_state_dict(obj["optim"])
    if scaler is not None and obj.get("scaler"):
        scaler.load_state_dict(obj["scaler"])
    return int(obj.get("epoch", 0)), int(obj.get("step", 0)), obj.get("meta", {})

# --------- Main Loop ----------
def main():
    logger.info(f"BOOT device={DEVICE} tsv={TSV} batch={BATCH} epochs={EPOCHS}")
    
    # Dataset
    if not os.path.exists(TSV):
        logger.error(f"Manifest not found: {TSV}")
        return

    ds_tr = DAPTDataset(TSV, split="train", val_ratio=VAL_RATIO)
    ds_va = DAPTDataset(TSV, split="val",   val_ratio=VAL_RATIO)
    
    if len(ds_tr) == 0:
        logger.error("Train split is empty. Aborting.")
        return

    # Sampler
    if USE_DIEL:
        smp_tr = DielBalancedSampler(ds_tr, shuffle=True)
        dl_tr = DataLoader(ds_tr, batch_size=BATCH, sampler=smp_tr,
                           collate_fn=pad_collate, num_workers=NUM_WORKERS,
                           pin_memory=True, persistent_workers=NUM_WORKERS>0, prefetch_factor=4)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                           collate_fn=pad_collate, num_workers=NUM_WORKERS,
                           pin_memory=True, persistent_workers=NUM_WORKERS>0, prefetch_factor=4)
    
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False,
                       collate_fn=pad_collate, num_workers=max(1, NUM_WORKERS//2),
                       pin_memory=True, persistent_workers=NUM_WORKERS>0, prefetch_factor=2)

    # Model
    device = torch.device(DEVICE)
    beats, beats_cfg = load_beats(BEATS_CKPT, device)
    in_dim = feature_dim_from_beats(beats, seconds=10.0)
    projector = ProjectionHead(in_dim, out_dim=256).to(device)
    
    all_params = list(beats.parameters()) + list(projector.parameters())

    # Optimizer (Lower LR for BEATs backbone, Higher LR for Projector)
    params = [
        {"params": beats.parameters(), "lr": 5e-5, "weight_decay": 0.01},
        {"params": projector.parameters(), "lr": 1e-3, "weight_decay": 0.01},
    ]
    optim = torch.optim.AdamW(params)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # Resume
    start_epoch, global_step = 1, 0
    if RESUME:
        p = None
        if RESUME_PATH:
            p = Path(RESUME_PATH)
        else:
            p = latest_ckpt_path(CKPT_DIR)
            
        if p and p.exists():
            e, s, _ = load_ckpt(p, beats, projector, optim, scaler)
            # If starting from an epoch checkpoint, start from next epoch
            if str(p.name).startswith("epoch_"):
                start_epoch = e + 1
            else:
                start_epoch = max(1, e)
            global_step = s
            logger.info(f"[ckpt] Resumed from: {p} (epoch={e}, step={s})")

    # Throughput Estimation (Warm-up)
    logger.info("Warming up to estimate throughput...")
    beats.train()
    projector.train()
    
    warm_batches = min(5, len(dl_tr))
    t0 = time.time()
    cnt = 0
    
    for i, (wav, *_) in enumerate(dl_tr):
        wav = wav.to(device)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            w1 = augment_wav(wav)
            w2 = augment_wav(wav)
            h1, _ = beats.extract_features(w1)
            h2, _ = beats.extract_features(w2)
            z1 = projector(h1.mean(1))
            z2 = projector(h2.mean(1))
            loss = info_nce(z1, z2, TEMP)

        if not torch.isfinite(loss):
            logger.warning(f"[warmup] Non-finite loss at batch={i}; skipping.")
            optim.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

        cnt += wav.size(0)
        if i + 1 >= warm_batches:
            break
            
    warm_t = time.time() - t0
    spd = cnt / max(1e-6, warm_t)
    steps_per_epoch = max(1, math.ceil(len(ds_tr) / BATCH))
    limit_steps = min(steps_per_epoch, MAX_STEPS_PER_EPOCH) if MAX_STEPS_PER_EPOCH > 0 else steps_per_epoch
    
    est_epoch_sec = (limit_steps * BATCH) / max(1, spd)
    logger.info(f"Throughput ~ {spd:.1f} samples/s; Est. epoch time ~ {est_epoch_sec/60:.1f} min")

    last_ckpt_time = time.time()

    # Training Loop
    for ep in range(start_epoch, EPOCHS + 1):
        beats.train()
        projector.train()
        
        tot_loss = 0.0
        n_batch = 0
        pbar = tqdm(dl_tr, total=limit_steps, desc=f"Train Ep{ep}", leave=False)
        t_ep0 = time.time()
        
        for wav, *_ in pbar:
            wav = wav.to(device)
            
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                w1 = augment_wav(wav)
                w2 = augment_wav(wav)
                h1, _ = beats.extract_features(w1)
                h2, _ = beats.extract_features(w2)
                z1 = projector(h1.mean(1))
                z2 = projector(h2.mean(1))
                loss = info_nce(z1, z2, TEMP)

            # Detect non-finite loss and skip
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss at step {global_step}. Skipping batch.")
                optim.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            global_step += 1
            tot_loss += loss.item()
            n_batch += 1
            
            avg_loss = tot_loss / max(1, n_batch)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

            # Checkpointing
            if CKPT_STEPS > 0 and global_step % CKPT_STEPS == 0:
                save_ckpt(CKPT_DIR / f"step_{global_step:09d}.pt", beats, projector, optim, scaler, ep, global_step, {"tsv": str(TSV)})
            
            if CKPT_MINS > 0 and (time.time() - last_ckpt_time) >= CKPT_MINS * 60:
                save_ckpt(CKPT_DIR / f"step_{global_step:09d}.pt", beats, projector, optim, scaler, ep, global_step, {"tsv": str(TSV)})
                last_ckpt_time = time.time()

            if MAX_STEPS_PER_EPOCH > 0 and n_batch >= limit_steps:
                break

        ep_time = time.time() - t_ep0
        logger.info(f"[Ep {ep}] Train Avg Loss: {tot_loss/max(1, n_batch):.5f}, Time: {ep_time/60:.1f} min")

        # Validation (Monitor InfoNCE)
        beats.eval()
        projector.eval()
        val_loss_tot = 0.0
        val_n = 0
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            for wav, *_ in tqdm(dl_va, desc=f"Val Ep{ep}", leave=False):
                wav = wav.to(device)
                w1 = augment_wav(wav)
                w2 = augment_wav(wav)
                h1, _ = beats.extract_features(w1)
                h2, _ = beats.extract_features(w2)
                z1 = projector(h1.mean(1))
                z2 = projector(h2.mean(1))
                loss = info_nce(z1, z2, TEMP)
                val_loss_tot += float(loss.item())
                val_n += 1
        
        logger.info(f"[Ep {ep}] Val Avg Loss: {val_loss_tot/max(1, val_n):.5f}")
        
        save_ckpt(CKPT_DIR / f"epoch_{ep:03d}_step_{global_step:09d}.pt", beats, projector, optim, scaler, ep, global_step, {"tsv": str(TSV)})

    logger.info("Training finished.")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        # Log crash to file safely
        if 'CRASH_FILE' in globals():
            with open(CRASH_FILE, "a") as f:
                f.write(tb + "\n")
        sys.exit(1)