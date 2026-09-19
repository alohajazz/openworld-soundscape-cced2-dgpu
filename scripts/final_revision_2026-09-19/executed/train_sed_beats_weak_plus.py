#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
弱教師SED（BEATs凍結）強化版
- 20ms刻みの時系列確率を出し、以下を統合:
  * スマートターゲット: σ自動・周期（comb）対応
  * ヘッド: 並列Conv + dilation（短音〜400msを一発で）
  * 損失: BCE or Focal Loss
  * 後処理: 平滑化 + ヒステリシス + 最小/最大長 + マージ
  * 評価: event-F1(クラス別tol_sec対応) + clip-F1 + segment-F1(1s/2s)
- CSV: path,label[,center_sec]（center_sec未指定は target_seconds/2）
"""

import os, sys, json, math, argparse, warnings
from typing import List, Tuple, Dict
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch, torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

# === BEATs ===
sys.path.append("/workspace/third_party/beats")
from BEATs import BEATs, BEATsConfig

# -------------------------
# Dataset
# -------------------------
class WeakSEDDataset(Dataset):
    def __init__(self, csv_path: str, label2id: Dict[str,int], target_seconds: float = 10.0, sr: int = 16000):
        self.df = pd.read_csv(csv_path)
        self.items = self.df.to_dict("records")
        self.label2id = label2id
        self.target_seconds = float(target_seconds)
        self.sr = int(sr)
        self.target_len = int(self.sr * self.target_seconds)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        path = it["path"]; label = str(it["label"]).strip()
        center_sec = float(it.get("center_sec", self.target_seconds/2.0))
        wav, sr = torchaudio.load(path)
        if wav.size(0)>1: wav = wav.mean(0, keepdim=True)
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = wav.squeeze(0)  # (T,)
        # pad/trim
        T = wav.numel()
        if T < self.target_len:
            wav = nn.functional.pad(wav, (0, self.target_len - T))
        else:
            wav = wav[:self.target_len]
        return wav, self.label2id[label], center_sec

# -------------------------
# BEATs frozen
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
    feats,_ = beats.extract_features(wav_bT.to(device))  # (B,T',C)
    B,Tp,C = feats.shape
    fps = Tp/target_seconds
    return feats, fps, Tp

# -------------------------
# Head: 並列Conv + dilation
# -------------------------
class ConvSEDHead(nn.Module):
    def __init__(self, in_dim: int, n_class: int, p_drop=0.1):
        super().__init__()
        h = in_dim // 2

        def same_conv1d(cin, cout, k, d=1):
            # 長さを保つ padding を自動設定
            assert (k % 2) == 1, "kernel_sizeは奇数推奨"
            pad = d * (k - 1) // 2
            return nn.Conv1d(cin, cout, kernel_size=k, padding=pad, dilation=d)

        # 3本の並列ブランチ（短〜中〜長の時間スケール）
        self.b3  = same_conv1d(in_dim, h, k=3, d=1)   # ≈60ms
        self.b5  = same_conv1d(in_dim, h, k=5, d=1)   # ≈100ms
        self.b7d = same_conv1d(in_dim, h, k=7, d=3)   # ≈420ms（padding=9）

        self.act  = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Conv1d(h * 3, n_class, kernel_size=1)

    def forward(self, feats_btC: torch.Tensor):
        x = feats_btC.transpose(1, 2)         # (B,C,T')
        b3 = self.b3(x)                       # (B,h,T')
        b5 = self.b5(x)                       # (B,h,T')
        b7 = self.b7d(x)                      # (B,h,T')
        # 念のため長さが一致しているか確認（開発時のみ）
        # assert b3.size(-1) == b5.size(-1) == b7.size(-1)
        h = torch.cat([b3, b5, b7], dim=1)    # (B,3h,T')
        h = self.drop(self.act(h))
        return self.proj(h)                   # (B,K,T')


# -------------------------
# Focal Loss（任意）
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, reduction="mean"):
        super().__init__(); self.g=gamma; self.red=reduction
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = p*targets + (1-p)*(1-targets)
        loss = (1-pt)**self.g * ce
        return loss.mean() if self.red=='mean' else loss.sum()

# -------------------------
# スマートターゲット: 長さ推定 + comb(周期)対応
# -------------------------
@torch.no_grad()
def estimate_len_and_period(wav_1T: torch.Tensor, sr: int, center_sec: float,
                            look_sec: float = 2.0) -> Tuple[float, float]:
    """中央±look_sec/2 の短窓から長さ(rough)と周期を推定"""
    T = wav_1T.numel()
    center = int(round(center_sec*sr))
    half = int(round(look_sec*sr/2.0))
    s = max(0, center-half); e = min(T, center+half)
    x = wav_1T[s:e].cpu().numpy()
    if x.size<8: return 0.20, None  # fallback

    # エネルギー包絡 → 2値化で長さ推定（ざっくり）
    frm = max(1, int(0.02*sr)); hop = max(1, int(0.01*sr))
    env=[]
    for i in range(0, len(x)-frm+1, hop):
        seg = x[i:i+frm]; env.append(np.sqrt(np.mean(seg**2)+1e-12))
    env = np.array(env); thr = np.percentile(env, 85)
    active = (env>=thr).astype(np.int32)
    # 最長連続区間
    L=0; cur=0
    for a in active:
        if a: cur+=1; L=max(L,cur)
        else: cur=0
    est_len_sec = max(0.10, min(0.80, L*hop/sr))

    # 自己相関で周期ピーク
    sig = x - x.mean()
    ac = np.correlate(sig, sig, mode='full')[len(sig)-1:]
    if ac.max()>0: ac = ac/(ac.max()+1e-12)
    lag_min=int(0.08*sr); lag_max=int(1.2*sr)
    if lag_max>=len(ac): lag_max=len(ac)-1
    if lag_min<lag_max and lag_max>0:
        lag = np.argmax(ac[lag_min:lag_max+1]) + lag_min
        ac_peak = float(ac[lag]); P = float(lag/sr) if ac_peak>0.25 else None
    else:
        P=None
    return float(est_len_sec), P

def smart_targets(wav_bT: torch.Tensor, centers_b: torch.Tensor,
                  fps: float, Tprime: int, n_class: int, class_ids: torch.Tensor,
                  device: torch.device, base_sigma: float = 0.20,
                  auto_period: bool = True, period_range=(0.25,1.2),
                  look_sec: float = 2.0) -> torch.Tensor:
    B = wav_bT.size(0); t = torch.arange(Tprime, device=device).float()
    targets = torch.zeros((B,n_class,Tprime), device=device)
    for b in range(B):
        k = int(class_ids[b].item()); c = float(centers_b[b].item())
        est_len, P = estimate_len_and_period(wav_bT[b], 16000, c, look_sec=look_sec)
        sigma = float(np.clip(est_len*0.4 if est_len else base_sigma, 0.10, 0.30))
        sigma_f = max(1e-6, sigma*fps)

        if auto_period and (P is not None) and (period_range[0] <= P <= period_range[1]):
            g = torch.zeros_like(t)
            # 5.0±mP ではなく center_sec=c を基準に左右に複数峰
            for m in range(-5,6):
                t0 = (c + m*P)*fps
                if 0<=t0<Tprime:
                    g += torch.exp(- (t - t0)**2 / (2.0*sigma_f**2))
            g = torch.clamp(g, 0.0, 1.0)
        else:
            t0 = c*fps
            g = torch.exp(- (t - t0)**2 / (2.0*sigma_f**2))
        targets[b,k,:] = g
    return targets

# -------------------------
# 平滑化 + ヒステリシス + 最小/最大長 + マージ
# -------------------------
def smooth_probs(probs_kt: np.ndarray, win: int = 3) -> np.ndarray:
    """
    各クラスについて 1 次元移動平均（'same'）で平滑化。
    入力: (K, T)
    出力: (K, T)（必ず同じ長さを保つ）
    """
    if win is None or win <= 1:
        return probs_kt
    K, T = probs_kt.shape
    out = np.empty_like(probs_kt, dtype=np.float32)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    for k in range(K):
        # np.convolve(..., mode='same') で長さ T を維持
        out[k, :] = np.convolve(probs_kt[k, :], kernel, mode="same")
    return out

def hysteresis_segments(prob_1t: np.ndarray, tau_high: float, tau_low: float,
                        min_len_f: int, max_len_f: int, merge_gap_f: int) -> List[Tuple[int,int]]:
    T = len(prob_1t); segs=[]
    aboveH = prob_1t >= tau_high
    aboveL = prob_1t >= tau_low
    s=None
    for i in range(T):
        if s is None:
            if aboveH[i]: s=i
        else:
            if not aboveL[i]:
                e=i-1
                if e-s+1 >= min_len_f and (max_len_f==0 or e-s+1<=max_len_f):
                    segs.append((s,e))
                s=None
    if s is not None:
        e=T-1
        if e-s+1 >= min_len_f and (max_len_f==0 or e-s+1<=max_len_f):
            segs.append((s,e))
    # マージ
    if merge_gap_f>0 and len(segs)>1:
        merged=[segs[0]]
        for s,e in segs[1:]:
            ps,pe=merged[-1]
            if s - pe - 1 <= merge_gap_f:
                merged[-1]=(ps,e)
            else:
                merged.append((s,e))
        segs=merged
    return segs

def center_seconds_of_segments(segs: List[Tuple[int,int]], fps: float) -> List[float]:
    return [ ((s+e)/2.0)/fps for s,e in segs ]

# -------------------------
# 評価: event-F1（クラス別tolに対応）
# -------------------------
def event_f1_with_tol(
    probs_bkt: np.ndarray,      # (B,K,T')
    labels_b: List[int],        # 正クラスID（弱教師）
    centers_b: List[float],     # 正クラス中心秒
    fps: float,
    tau: float,                 # 基本の二値化（ここからヒステリシスを派生）
    tol_by_class: Dict[int,float],
    smooth_win: int,
    min_len_sec: float, max_len_sec: float, merge_gap_sec: float,
    tau_low_delta: float = 0.15
) -> Tuple[float,float,float]:
    B,K,Tp = probs_bkt.shape
    TP=FP=FN=0
    seg_min = int(round(min_len_sec*fps))
    seg_max = int(round(max_len_sec*fps)) if max_len_sec>0 else 0
    seg_gap = int(round(merge_gap_sec*fps))
    for b in range(B):
        k = labels_b[b]
        tol = tol_by_class.get(k, 0.5)
        # 平滑
        probs_k = smooth_probs(probs_bkt[b], win=smooth_win)[k]
        segs = hysteresis_segments(
            probs_k, tau_high=tau, tau_low=max(0.05, tau - tau_low_delta),
            min_len_f=seg_min, max_len_f=seg_max, merge_gap_f=seg_gap
        )
        preds = center_seconds_of_segments(segs, fps)
        gt = centers_b[b]
        if len(preds)==0:
            FN += 1
        else:
            dists = [abs(p-gt) for p in preds]
            j = int(np.argmin(dists))
            if dists[j] <= tol:
                TP += 1
                FP += max(0, len(preds)-1)
            else:
                FN += 1
                FP += len(preds)
    P = TP/(TP+FP+1e-9); R = TP/(TP+FN+1e-9)
    F1 = 2*P*R/(P+R+1e-9)
    return P,R,F1

# -------------------------
# 評価: segment-F1（1s/2s） & クリップF1
# -------------------------
def segment_f1_from_probs(probs_bkt, labels_b, centers_b, fps, seg_len_sec=1.0, tau=0.5, use_max=False):
    B,K,Tp = probs_bkt.shape
    seg_frames = int(round(seg_len_sec*fps))
    n_seg = max(1, int(np.ceil(Tp/seg_frames)))
    pred_bkn = np.zeros((B,K,n_seg), dtype=np.float32)
    for n in range(n_seg):
        s=n*seg_frames; e=min(Tp,(n+1)*seg_frames)
        if s<e:
            if use_max:
                pred_bkn[:,:,n] = probs_bkt[:,:,s:e].max(axis=2)
            else:
                pred_bkn[:,:,n] = probs_bkt[:,:,s:e].mean(axis=2)
    pred_bin = (pred_bkn >= tau).astype(np.int32)
    gt = np.zeros_like(pred_bin)
    for b in range(B):
        k=labels_b[b]; idx=int(np.clip(np.floor(centers_b[b]/seg_len_sec), 0, n_seg-1))
        gt[b,k,idx]=1
    TP=int((pred_bin & gt).sum()); FP=int((pred_bin & (1-gt)).sum()); FN=int(((1-pred_bin) & gt).sum())
    P=TP/(TP+FP+1e-9); R=TP/(TP+FN+1e-9); F1=2*P*R/(P+R+1e-9)
    return P,R,F1

def clip_f1_from_probs(probs_bkt, labels_b, tau=0.5):
    B,K,Tp = probs_bkt.shape
    pred_bin = (probs_bkt.max(axis=2) >= tau).astype(np.int32)
    gt = np.zeros((B,K), dtype=np.int32)
    for b in range(B):
        gt[b, labels_b[b]] = 1
    TP=int((pred_bin & gt).sum()); FP=int((pred_bin & (1-gt)).sum()); FN=int(((1-pred_bin) & gt).sum())
    P=TP/(TP+FP+1e-9); R=TP/(TP+FN+1e-9); F1=2*P*R/(P+R+1e-9)
    return P,R,F1

# -------------------------
# Train/Eval
# -------------------------
def train_epoch(beats, head, loader, device, target_seconds, n_class,
                use_smart_targets, base_sigma, auto_period, period_range, look_sec,
                loss_fn, optimizer):
    head.train()
    loss_sum=n=0
    for wav, y, csec in loader:
        wav = wav.to(device); y = y.to(device); csec = csec.to(device)
        feats, fps, Tp = extract_feats(beats, wav, target_seconds, device)
        if use_smart_targets:
            targets = smart_targets(wav, csec, fps, Tp, n_class, y, device,
                                    base_sigma=base_sigma, auto_period=auto_period,
                                    period_range=period_range, look_sec=look_sec)
        else:
            # 従来ガウス
            t = torch.arange(Tp, device=device).float()
            targets = torch.zeros((wav.size(0), n_class, Tp), device=device)
            sigma_f = max(1e-6, base_sigma*fps)
            for b in range(wav.size(0)):
                k=int(y[b].item()); t0=csec[b].item()*fps
                g=torch.exp(- (t - t0)**2 / (2.0*sigma_f**2))
                targets[b,k,:]=g
        logits = head(feats)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        bs = wav.size(0); loss_sum += loss.item()*bs; n += bs
    return loss_sum/max(1,n)

@torch.no_grad()
def eval_epoch(beats, head, loader, device, target_seconds,
               base_tau, tol_by_class, smooth_win,
               min_len_sec, max_len_sec, merge_gap_sec,
               seg_tau, seg_len_sec_list):
    head.eval()
    probs_all=[]; labels_all=[]; centers_all=[]
    for wav, y, csec in loader:
        wav=wav.to(device); feats, fps, Tp = extract_feats(beats, wav, target_seconds, device)
        logits=head(feats); probs=torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
        labels_all.extend([int(i) for i in y.tolist()])
        centers_all.extend([float(i) for i in csec.tolist()])
    probs_all=np.concatenate(probs_all, axis=0)
    # event-F1（クラス別tol）
    evP,evR,evF1 = event_f1_with_tol(
        probs_all, labels_all, centers_all, fps, base_tau, tol_by_class,
        smooth_win, min_len_sec, max_len_sec, merge_gap_sec
    )
    # clip-F1
    cP,cR,cF1 = clip_f1_from_probs(probs_all, labels_all, tau=seg_tau)
    # segment-F1（1s/2s）
    segF1s={}
    for L in seg_len_sec_list:
        # sP,sR,sF1 = segment_f1_from_probs(probs_all, labels_all, centers_all, fps, seg_len_sec=L, tau=seg_tau)
        sP,sR,sF1 = segment_f1_from_probs(probs_all, labels_all, centers_all, fps, seg_len_sec=L, tau=seg_tau, use_max=True)
        segF1s[L]=(sP,sR,sF1)
    return evP,evR,evF1, cP,cR,cF1, segF1s

# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--ckpt", default="/workspace/models/BEATs/BEATs_iter3.pt")
    ap.add_argument("--outdir", default="/workspace/logs/sed_weak_plus")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs",     type=int, default=32)
    ap.add_argument("--lr",     type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_seconds", type=float, default=10.0)

    # Smart targets
    ap.add_argument("--use_smart_targets", action="store_true")
    ap.add_argument("--base_sigma", type=float, default=0.20)
    ap.add_argument("--auto_period", action="store_true")
    ap.add_argument("--period_min", type=float, default=0.25)
    ap.add_argument("--period_max", type=float, default=1.2)
    ap.add_argument("--look_sec",  type=float, default=2.0)

    # Loss
    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=1.5)

    # Postprocess / eval
    ap.add_argument("--base_tau", type=float, default=0.5)
    ap.add_argument("--tau_low_delta", type=float, default=0.15)
    ap.add_argument("--smooth_win", type=int, default=3)
    ap.add_argument("--min_len_sec", type=float, default=0.10)
    ap.add_argument("--max_len_sec", type=float, default=2.50)
    ap.add_argument("--merge_gap_sec", type=float, default=0.20)

    # Class-specific tolerance for event-F1（JSONで上書き可）
    ap.add_argument("--tol_sec_default", type=float, default=0.5)
    ap.add_argument("--tol_sec_json", type=str, default="")  # {"0":0.3,"1":0.5,...} もしくは {"ClassName":0.3,...}

    # segment/clip
    ap.add_argument("--seg_tau", type=float, default=0.5)
    ap.add_argument("--seg_lens", type=str, default="1.0,2.0")

    args = ap.parse_args()
    import random as _r; _r.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir,"hparams.json"),"w") as f: json.dump(vars(args), f, indent=2)

    device=torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    # Labels
    labels_all = sorted(pd.concat([pd.read_csv(args.train_csv)["label"], pd.read_csv(args.val_csv)["label"]]).unique())
    name2id = {n:i for i,n in enumerate(labels_all)}

    # Datasets
    tr_ds = WeakSEDDataset(args.train_csv, name2id, target_seconds=args.target_seconds)
    va_ds = WeakSEDDataset(args.val_csv,   name2id, target_seconds=args.target_seconds)
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,  num_workers=4, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    # BEATs & Head
    beats = load_beats(args.ckpt, device)
    dummy = torch.randn(1, int(16000*args.target_seconds)).to(device)
    feats,fps,Tp = extract_feats(beats, dummy, args.target_seconds, device)
    in_dim = feats.shape[-1]
    head = ConvSEDHead(in_dim, len(labels_all)).to(device)

    # Loss
    loss_fn = FocalLoss(gamma=args.focal_gamma) if args.use_focal else nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # tol_by_class（ID or 名前どちらでもOK）
    tol_by_class = {i: args.tol_sec_default for i in range(len(labels_all))}
    if args.tol_sec_json:
        with open(args.tol_sec_json) as f: td=json.load(f)
        # キーが名前ならIDに変換、数字文字列ならそのままID扱い
        for k,v in td.items():
            try:
                idx = int(k)
            except:
                idx = name2id.get(k, None)
            if idx is not None:
                tol_by_class[idx] = float(v)

    seg_len_list = [float(x) for x in args.seg_lens.split(",") if x.strip()]

    best_f1=-1.0; best_path=None
    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(
            beats, head, tr_dl, device, args.target_seconds, len(labels_all),
            use_smart_targets=args.use_smart_targets,
            base_sigma=args.base_sigma, auto_period=args.auto_period,
            period_range=(args.period_min, args.period_max), look_sec=args.look_sec,
            loss_fn=loss_fn, optimizer=optim
        )
        evP,evR,evF1, cP,cR,cF1, segF1s = eval_epoch(
            beats, head, va_dl, device, args.target_seconds,
            base_tau=args.base_tau,
            tol_by_class=tol_by_class,
            smooth_win=args.smooth_win,
            min_len_sec=args.min_len_sec,
            max_len_sec=args.max_len_sec,
            merge_gap_sec=args.merge_gap_sec,
            seg_tau=args.seg_tau,
            seg_len_sec_list=seg_len_list
        )
        seg_txt = " ".join([f"seg{L:.0f}sF1={segF1s[L][2]:.3f}" for L in seg_len_list])
        print(f"[ep{ep}] tr_loss={tr_loss:.4f}  eventF1={evF1:.3f} (P={evP:.3f} R={evR:.3f})  clipF1={cF1:.3f}  {seg_txt}")

        ck={"head":head.state_dict(),"labels":labels_all,"hparams":vars(args)}
        path=os.path.join(args.outdir,f"sed_head_ep{ep}.pt"); torch.save(ck, path)
        if evF1 > best_f1:
            best_f1, best_path = evF1, path

    print(f"BEST eventF1={best_f1:.3f} path={best_path}")
    with open(os.path.join(args.outdir,"best.json"),"w") as f:
        json.dump({"best_eventF1":best_f1, "best_path":best_path}, f, indent=2)

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]=os.environ.get("PYTORCH_CUDA_ALLOC_CONF","max_split_size_mb:128")
    main()


# python /workspace/scripts/train_sed_beats_weak_plus.py \
#   --train_csv /workspace/data/exp_beats_v1_ambient1/train.csv \
#   --val_csv   /workspace/data/exp_beats_v1_ambient1/val.csv \
#   --ckpt      /workspace/models/BEATs/BEATs_iter3.pt \
#   --outdir    /workspace/logs/sed_weak_plus_smart \
#   --epochs 8 --bs 32 --lr 2e-3 \
#   --use_smart_targets --auto_period --base_sigma 0.20 \
#   --period_min 0.25 --period_max 1.2 --look_sec 2.0
  
  
# 3) AS2M ckpt 比較
# --ckpt /workspace/models/BEATs/BEATs_iter3_plus_AS2M.pt
