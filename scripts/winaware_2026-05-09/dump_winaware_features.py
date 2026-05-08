#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dump_known56_features.py

56ラベル学習データなど「既知集合」の特徴を保存する専用スクリプト（評価スクリプトとは別物）。
- 入力: CSV (path,label[,center_sec])  ※ center_sec は無くてもOK
- 出力: outdir/
    - embeddings_000.npy,  embeddings_001.npy, ...  （[N_i, D]  時間平均後のBEATs特徴）
    - logits_000.npy,      logits_001.npy, ...      （[N_i, 56]  SEDヘッドのロジット）
    - index_000.csv,       index_001.csv, ...       （path/start_sec/duration_sec など）
    - summary.json （次元や件数などのメタ）
- 使い方（例）：
    python dump_known56_features.py \
      --csv /path/to/your_56label_train_manifest.csv \
      --ckpt_beats /workspace/ckpts_dapt_beats/BEATs_event_focus.pt \
      --head_ckpt  /workspace/ckpts_dapt_beats/SED_head_event_focus.pt \
      --outdir /workspace/embeddings/known56 \
      --dump_logits --dump_embeddings --shard_size 200000
"""

import os, sys, json, argparse, csv
import numpy as np
import pandas as pd
import torch, torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

# === BEATs ===
sys.path.append("/workspace/third_party/beats")
from BEATs import BEATs, BEATsConfig

# -------------------------
# Dataset（path を返す）
# -------------------------
class SegDataset(Dataset):
    def __init__(self, csv_path: str, target_seconds: float = 10.0, sr: int = 16000):
        df = pd.read_csv(csv_path)
        req = ["path", "label"]
        for k in req:
            if k not in df.columns:
                raise ValueError(f"CSVに列 {req} が必要です: {csv_path}")
        self.items = df.to_dict("records")
        self.target_seconds = float(target_seconds)
        self.sr = int(sr)
        self.target_len = int(self.sr * self.target_seconds)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        path = it["path"]
        label = str(it["label"]).strip()
        center_sec = float(it.get("center_sec", self.target_seconds/2.0))
        # window-aware slice: center_sec ベース or manifest start_sec
        if "start_sec" in it and not pd.isna(it.get("start_sec")):
            start_sec_val = float(it["start_sec"])
        else:
            start_sec_val = max(0.0, center_sec - self.target_seconds / 2.0)
        # OPTIMIZATION: load only the required window using frame_offset/num_frames
        # — avoids decoding whole 60s file when we only need 10s slice
        info = torchaudio.info(path)
        sr_actual = info.sample_rate
        frame_offset = max(0, int(round(start_sec_val * sr_actual)))
        num_frames = int(round(self.target_seconds * sr_actual))
        # Right pad guard: if window extends beyond file, clip num_frames
        if frame_offset + num_frames > info.num_frames:
            num_frames = max(0, info.num_frames - frame_offset)
        wav, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        if wav.size(0)>1: wav = wav.mean(0, keepdim=True)
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = wav.squeeze(0)  # (T,)
        # 長さ統一 (right-pad if needed)
        if wav.numel() < self.target_len:
            wav = nn.functional.pad(wav, (0, self.target_len - wav.numel()))
        elif wav.numel() > self.target_len:
            wav = wav[:self.target_len]
        meta = {
            "path": path,
            "start_sec": float(start_sec_val),
            "duration_sec": float(self.target_seconds),
            "label": label,
            "center_sec": float(center_sec)
        }
        return wav, meta

# -------------------------
# BEATs ロード＆特徴抽出
# -------------------------
def load_beats(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    m = BEATs(BEATsConfig(ckpt["cfg"]))
    m.load_state_dict(ckpt["model"])
    m.eval().to(device)
    for p in m.parameters(): p.requires_grad_(False)
    return m

@torch.no_grad()
def extract_feats(beats, wav_bT: torch.Tensor, target_seconds: float, device: torch.device):
    """
    Returns:
      feats: (B, T', C)  ※BEATsの時系列特徴
      fps:   float       ※1秒あたりのフレーム数（目安）
      Tp:    int         ※フレーム長
    """
    feats,_ = beats.extract_features(wav_bT.to(device))  # (B,T',C)
    B,Tp,C = feats.shape
    fps = Tp/target_seconds
    return feats, fps, Tp

# -------------------------
# 学習済みSEDヘッド（56クラス想定）
# -------------------------
class ConvSEDHead(nn.Module):
    def __init__(self, in_dim: int, n_class: int, p_drop=0.1):
        super().__init__()
        h = in_dim // 2
        def same_conv1d(cin, cout, k, d=1):
            assert (k % 2) == 1, "kernel_sizeは奇数推奨"
            pad = d * (k - 1) // 2
            return nn.Conv1d(cin, cout, kernel_size=k, padding=pad, dilation=d)
        self.b3  = same_conv1d(in_dim, h, k=3, d=1)
        self.b5  = same_conv1d(in_dim, h, k=5, d=1)
        self.b7d = same_conv1d(in_dim, h, k=7, d=3)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Conv1d(h * 3, n_class, kernel_size=1)

    def forward(self, feats_btC: torch.Tensor):
        x = feats_btC.transpose(1, 2)         # (B,C,T')
        b3 = self.b3(x); b5 = self.b5(x); b7 = self.b7d(x)
        h = torch.cat([b3, b5, b7], dim=1)    # (B,3h,T')
        h = self.drop(self.act(h))
        return self.proj(h)                   # (B,K,T')

# -------------------------
def save_index_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not rows: return
    keys = ["path","start_sec","duration_sec","label","center_sec"]
    with open(out_csv, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,"") for k in keys})

def _collate_keep_meta(batch):
    # batch: List[(wav_1T, meta_dict)]
    wavs, metas = zip(*batch)                   # metas: tuple of dict
    wav_bT = torch.stack(wavs, dim=0)           # すでに固定長にしているのでstackでOK
    return wav_bT, list(metas)                  # ← metaをリストのまま返す


# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path,label[,center_sec] を含むCSV（56ラベル学習セットなど）")
    ap.add_argument("--ckpt_beats", required=True, help="BEATs 本体 ckpt（例：/workspace/ckpts_dapt_beats/BEATs_event_focus.pt）")
    ap.add_argument("--head_ckpt", default="", help="SEDヘッド ckpt（56クラス想定）。ロジット出力したい時に指定")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--target_seconds", type=float, default=10.0)
    ap.add_argument("--dump_embeddings", action="store_true", help="BEATs 時系列特徴の時間平均（[B,D]）を保存")
    ap.add_argument("--dump_logits", action="store_true", help="SEDヘッド（56次元）ロジットを保存")
    ap.add_argument("--embedding_pool", type=str, default="mean", choices=["mean","max"], help="時系列→1ベクトルへの集約方法")
    ap.add_argument("--shard_size", type=int, default=0, help="0で単一ファイル。>0でその件数ごとに分割保存")
    args = ap.parse_args()

    assert args.dump_embeddings or args.dump_logits, "少なくとも --dump_embeddings または --dump_logits を指定してください。"

    os.makedirs(args.outdir, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device=torch.device(args.device)

    # Dataset/Loader
    ds = SegDataset(args.csv, target_seconds=args.target_seconds)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(torch.cuda.is_available() and str(device) == "cuda"),  # GPU無ければFalse
        drop_last=False,
        collate_fn=_collate_keep_meta,   # ← これがポイント
    )


    # BEATs 構築
    beats = load_beats(args.ckpt_beats, device)
    # 入力次元 in_dim をダミー波形で取得
    dummy = torch.randn(1, int(16000*args.target_seconds)).to(device)
    feats, fps, Tp = extract_feats(beats, dummy, args.target_seconds, device)
    in_dim = feats.shape[-1]

    # ヘッド（任意）
    head = None
    if args.dump_logits:
        assert args.head_ckpt, "--dump_logits には --head_ckpt が必要です。"
        head = ConvSEDHead(in_dim, n_class=56).to(device)
        ck = torch.load(args.head_ckpt, map_location="cpu")
        head.load_state_dict(ck["head"])
        head.eval()

    # バッファと保存ヘルパ
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
                    np.concatenate(emb_buf, axis=0))  # [N_i,D]
        if args.dump_logits and logit_buf:
            np.save(os.path.join(args.outdir, f"logits_{prefix}.npy"),
                    np.concatenate(logit_buf, axis=0))  # [N_i,56]
        save_index_csv(idx_buf, os.path.join(args.outdir, f"index_{prefix}.csv"))
        emb_buf.clear(); logit_buf.clear(); idx_buf.clear()
        shard_id += 1

    # ループ
    import time as _time
    total = 0
    t0 = _time.time()
    n_batches = len(dl)
    print(f"[dump] Starting: {len(ds)} samples, {n_batches} batches, batch_size={args.batch_size}", flush=True)
    with torch.no_grad():
        for bi, (wav, meta) in enumerate(dl):
            wav = wav.to(device)  # (B,T)
            feats, fps, Tp = extract_feats(beats, wav, args.target_seconds, device)  # (B,T',C)
            B,Tp,C = feats.shape

            # 埋め込み（時間平均 or 最大）
            if args.dump_embeddings:
                if args.embedding_pool == "mean":
                    emb = feats.mean(dim=1)  # (B,C)
                else:
                    emb = feats.max(dim=1).values
                emb_np = emb.detach().float().cpu().numpy()  # [B,D]
                emb_buf.append(emb_np)

            # ロジット（56次元）
            if head is not None:
                logits = head(feats)              # (B,56,T')
                agg = logits.mean(dim=2)          # 時間平均で [B,56] に（maxでもOK）
                logit_np = agg.detach().float().cpu().numpy()
                logit_buf.append(logit_np)

            # index
            for m in meta:
                idx_buf.append({
                    "path": m["path"],
                    "start_sec": float(m.get("start_sec", 0.0)),
                    "duration_sec": float(m.get("duration_sec", args.target_seconds)),
                    "label": str(m.get("label","")),
                    "center_sec": float(m.get("center_sec", args.target_seconds/2.0))
                })

            total += len(meta)

            # Progress report
            if (bi+1) % 100 == 0 or (bi+1) == n_batches:
                elapsed = _time.time() - t0
                speed = total / elapsed
                eta = (len(ds) - total) / speed if speed > 0 else 0
                print(f"[dump] {total}/{len(ds)} ({total/len(ds)*100:.1f}%) "
                      f"{elapsed:.0f}s elapsed, {speed:.1f} samples/s, eta {eta:.0f}s",
                      flush=True)

            # shard flush
            if args.shard_size > 0 and len(idx_buf) >= args.shard_size:
                flush()

    # 残りを保存
    flush()

    # サマリー
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

    print("[OK] dump finished:", json.dumps(summary))

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]=os.environ.get("PYTORCH_CUDA_ALLOC_CONF","max_split_size_mb:128")
    main()
