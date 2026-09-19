#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_sed_beats_report.py
- 学習済み SED ヘッド（sed_head_epX.pt）を読み込み、val.csv で評価＆レポートを生成
- 出力：
    outdir/
      reports/
        class_report_clip.csv
        confusion_argmax.csv
        confusion_argmax_with_none.csv
        metrics_overview.json
- 既存の train_sed_beats_weak_plus.py と同じ評価ロジック（event/segment/clip）を実装
- tol_sec_json でクラス別 event-F1 の許容誤差（秒）を上書き可能
"""

import os, sys, json, argparse, warnings, csv, math
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
    def __init__(self, csv_path: str, label2id: dict, target_seconds: float = 10.0, sr: int = 16000):
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
# Head: 並列Conv + dilation（学習時と同構造）
# -------------------------
class ConvSEDHead(nn.Module):
    def __init__(self, in_dim: int, n_class: int, p_drop=0.1):
        super().__init__()
        h = in_dim // 2

        def same_conv1d(cin, cout, k, d=1):
            assert (k % 2) == 1, "kernel_sizeは奇数推奨"
            pad = d * (k - 1) // 2
            return nn.Conv1d(cin, cout, kernel_size=k, padding=pad, dilation=d)

        self.b3  = same_conv1d(in_dim, h, k=3, d=1)   # ≈60ms
        self.b5  = same_conv1d(in_dim, h, k=5, d=1)   # ≈100ms
        self.b7d = same_conv1d(in_dim, h, k=7, d=3)   # ≈420ms

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
# 後処理ユーティリティ
# -------------------------
def smooth_probs(probs_kt: np.ndarray, win: int = 3) -> np.ndarray:
    if win is None or win <= 1:
        return probs_kt
    K, T = probs_kt.shape
    out = np.empty_like(probs_kt, dtype=np.float32)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    for k in range(K):
        out[k, :] = np.convolve(probs_kt[k, :], kernel, mode="same")
    return out

def hysteresis_segments(prob_1t: np.ndarray, tau_high: float, tau_low: float,
                        min_len_f: int, max_len_f: int, merge_gap_f: int):
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

def center_seconds_of_segments(segs, fps: float):
    return [ ((s+e)/2.0)/fps for s,e in segs ]

# -------------------------
# 指標（event/segment/clip）
# -------------------------
def event_f1_with_tol(
    probs_bkt: np.ndarray, labels_b, centers_b, fps: float,
    tau: float, tol_by_class: dict, smooth_win: int,
    min_len_sec: float, max_len_sec: float, merge_gap_sec: float,
    tau_low_delta: float = 0.15
):
    B,K,Tp = probs_bkt.shape
    TP=FP=FN=0
    seg_min = int(round(min_len_sec*fps))
    seg_max = int(round(max_len_sec*fps)) if max_len_sec>0 else 0
    seg_gap = int(round(merge_gap_sec*fps))
    for b in range(B):
        k = labels_b[b]
        tol = tol_by_class.get(k, 0.5)
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
# レポート作成
# -------------------------
@torch.no_grad()
def gather_val_probs(beats, head, loader, device, target_seconds):
    head.eval()
    probs_all=[]; labels_all=[]; centers_all=[]
    fps_last=None
    for wav, y, csec in loader:
        wav=wav.to(device)
        feats, fps, Tp = extract_feats(beats, wav, target_seconds, device)
        logits=head(feats)
        probs=torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
        labels_all.extend([int(i) for i in y.tolist()])
        centers_all.extend([float(i) for i in csec.tolist()])
        fps_last = fps
    probs_all=np.concatenate(probs_all, axis=0)  # (B,K,T')
    return probs_all, labels_all, centers_all, fps_last

def per_class_clip_metrics(probs_bkt, labels_b, tau=0.5, label_names=None):
    B,K,T = probs_bkt.shape
    pred_bin = (probs_bkt.max(axis=2) >= tau).astype(np.int32)  # (B,K)
    gt = np.zeros((B,K), dtype=np.int32)
    for b in range(B):
        gt[b, labels_b[b]] = 1
    rows=[]
    for k in range(K):
        TP = int(((pred_bin[:,k]==1) & (gt[:,k]==1)).sum())
        FP = int(((pred_bin[:,k]==1) & (gt[:,k]==0)).sum())
        FN = int(((pred_bin[:,k]==0) & (gt[:,k]==1)).sum())
        P = TP/(TP+FP+1e-9); R = TP/(TP+FN+1e-9)
        F1 = 2*P*R/(P+R+1e-9)
        rows.append({
            "class_id": k,
            "class_name": (label_names[k] if label_names else str(k)),
            "support": int(gt[:,k].sum()),
            "precision": P, "recall": R, "f1": F1,
            "TP": TP, "FP": FP, "FN": FN
        })
    return rows

def confusion_matrix_argmax(probs_bkt, labels_b, tau_for_none=None, label_names=None):
    B,K,T = probs_bkt.shape
    agg = probs_bkt.max(axis=2)  # (B,K)
    pred = agg.argmax(axis=1)    # (B,)
    if tau_for_none is not None:
        none_mask = (agg.max(axis=1) < float(tau_for_none))
        pred = pred.copy()
        pred[none_mask] = K  # Noneクラス

    nK = K + (1 if tau_for_none is not None else 0)
    mat = np.zeros((K, nK), dtype=np.int32)  # 行=GT(0..K-1), 列=Pred(0..K-1[,K=None])
    for b in range(B):
        gt = labels_b[b]
        pd = int(pred[b])
        mat[gt, pd] += 1

    col_names = list(label_names) if label_names else [str(i) for i in range(K)]
    if tau_for_none is not None:
        col_names = col_names + ["None(<tau)"]
    row_names = list(label_names) if label_names else [str(i) for i in range(K)]
    return mat, row_names, col_names

def save_class_report_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if len(rows)==0: return
    fieldnames=list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k:(v if not isinstance(v,float) else f"{v:.6f}") for k,v in r.items()})

def save_confusion_csv(mat, row_names, col_names, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["GT\\Pred"] + col_names)
        for i, rn in enumerate(row_names):
            w.writerow([rn] + [int(x) for x in mat[i,:].tolist()])

def json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w") as f:
        json.dump(obj, f, indent=2)

def build_tol_by_class(args, labels_all, name2id):
    tol_by_class = {i: args.tol_sec_default for i in range(len(labels_all))}
    if args.tol_sec_json:
        with open(args.tol_sec_json) as f: td=json.load(f)
        for k,v in td.items():
            try:
                idx = int(k)
            except:
                idx = name2id.get(k, None)
            if idx is not None:
                tol_by_class[idx] = float(v)
    return tol_by_class

# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--head_ckpt", required=True, help="sed_head_epX.pt")
    ap.add_argument("--ckpt", default="/workspace/models/BEATs/BEATs_iter3.pt", help="BEATs本体ckpt")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target_seconds", type=float, default=10.0)

    # event 評価・後処理
    ap.add_argument("--base_tau", type=float, default=0.5)
    ap.add_argument("--tau_low_delta", type=float, default=0.15)
    ap.add_argument("--smooth_win", type=int, default=3)
    ap.add_argument("--min_len_sec", type=float, default=0.10)
    ap.add_argument("--max_len_sec", type=float, default=2.50)
    ap.add_argument("--merge_gap_sec", type=float, default=0.20)
    ap.add_argument("--tol_sec_default", type=float, default=0.5)
    ap.add_argument("--tol_sec_json", type=str, default="")

    # segment/clip
    ap.add_argument("--seg_tau", type=float, default=0.5)
    ap.add_argument("--seg_lens", type=str, default="1.0,2.0")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device=torch.device(args.device)

    # ラベル一覧（valから抽出）
    labels_all = sorted(pd.read_csv(args.val_csv)["label"].unique())
    name2id = {n:i for i,n in enumerate(labels_all)}

    # Dataset / Loader
    va_ds = WeakSEDDataset(args.val_csv, name2id, target_seconds=args.target_seconds)
    va_dl = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # BEATs と Head 構築
    beats = load_beats(args.ckpt, device)
    # in_dim をダミーで取得
    dummy = torch.randn(1, int(16000*args.target_seconds)).to(device)
    feats,fps,Tp = extract_feats(beats, dummy, args.target_seconds, device)
    in_dim = feats.shape[-1]
    head = ConvSEDHead(in_dim, len(labels_all)).to(device)

    # 学習済みヘッドをロード
    ck = torch.load(args.head_ckpt, map_location="cpu")
    head.load_state_dict(ck["head"])
    if "labels" in ck and list(ck["labels"]) != list(labels_all):
        print("[WARN] head_ckpt に保存されたクラス順と val のクラス順が異なる可能性があります。評価結果の解釈に注意してください。")

    # 全確率の収集
    probs_all, labels_all_ids, centers_all, fps = gather_val_probs(beats, head, va_dl, device, args.target_seconds)

    # 総合指標（event / clip / segment）
    tol_by_class = build_tol_by_class(args, labels_all, name2id)
    seg_len_list = [float(x) for x in args.seg_lens.split(",") if x.strip()]

    evP,evR,evF1 = event_f1_with_tol(
        probs_all, labels_all_ids, centers_all, fps,
        tau=args.base_tau,
        tol_by_class=tol_by_class,
        smooth_win=args.smooth_win,
        min_len_sec=args.min_len_sec,
        max_len_sec=args.max_len_sec,
        merge_gap_sec=args.merge_gap_sec,
        tau_low_delta=args.tau_low_delta
    )
    cP,cR,cF1 = clip_f1_from_probs(probs_all, labels_all_ids, tau=args.seg_tau)

    segF1s={}
    for L in seg_len_list:
        sP,sR,sF1 = segment_f1_from_probs(
            probs_all, labels_all_ids, centers_all, fps,
            seg_len_sec=L, tau=args.seg_tau, use_max=True
        )
        segF1s[L]=(sP,sR,sF1)

    # レポート保存
    rep_dir = os.path.join(args.outdir, "reports")
    os.makedirs(rep_dir, exist_ok=True)

    rows = per_class_clip_metrics(probs_all, labels_all_ids, tau=args.seg_tau, label_names=labels_all)
    save_class_report_csv(rows, os.path.join(rep_dir, "class_report_clip.csv"))

    mat1, rn, cn = confusion_matrix_argmax(probs_all, labels_all_ids, tau_for_none=None, label_names=labels_all)
    save_confusion_csv(mat1, rn, cn, os.path.join(rep_dir, "confusion_argmax.csv"))
    mat2, rn2, cn2 = confusion_matrix_argmax(probs_all, labels_all_ids, tau_for_none=args.seg_tau, label_names=labels_all)
    save_confusion_csv(mat2, rn2, cn2, os.path.join(rep_dir, "confusion_argmax_with_none.csv"))

    overview = {
        "eventF1": {"P":evP, "R":evR, "F1":evF1, "tau": args.base_tau,
                    "smooth_win": args.smooth_win,
                    "min_len_sec": args.min_len_sec,
                    "max_len_sec": args.max_len_sec,
                    "merge_gap_sec": args.merge_gap_sec,
                    "tau_low_delta": args.tau_low_delta},
        "clipF1": {"P":cP, "R":cR, "F1":cF1, "tau": args.seg_tau},
        "segmentF1": {f"{L}": {"P": segF1s[L][0], "R": segF1s[L][1], "F1": segF1s[L][2]} for L in segF1s},
        "fps": float(fps),
        "labels": labels_all,
        "tol_by_class": { (labels_all[k] if isinstance(k,int) else str(k)) : float(v) for k,v in tol_by_class.items() if isinstance(k,int) }
    }
    json_dump(overview, os.path.join(rep_dir, "metrics_overview.json"))

    print(f"[eval] eventF1={evF1:.3f}  clipF1={cF1:.3f}  " +
          " ".join([f"seg{float(L):.0f}sF1={segF1s[L][2]:.3f}" for L in segF1s]))
    print(f"[eval] reports saved under: {rep_dir}")

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]=os.environ.get("PYTORCH_CUDA_ALLOC_CONF","max_split_size_mb:128")
    main()

# ① ep10（BEST）でレポート一式を出力
# python /workspace/scripts/eval_sed_beats_report.py \
#   --val_csv /workspace/data/exp_beats_v1_ambient1/val.csv \
#   --head_ckpt /workspace/logs/sed_plus_balanced_v2/sed_head_ep10.pt \
#   --ckpt /workspace/models/BEATs/BEATs_iter3_plus_AS2M.pt \
#   --outdir /workspace/logs/sed_plus_balanced_v2 \
#   --base_tau 0.35 --smooth_win 2 --min_len_sec 0.06 --max_len_sec 2.5 --merge_gap_sec 0.25 \
#   --seg_tau 0.30 --seg_lens "1.0,2.0"

# python /workspace/scripts/eval_sed_beats_report.py \
#   --val_csv /workspace/data/exp_beats_v1_ambient1/val.csv \
#   --head_ckpt /workspace/logs/sed_plus_balanced_v2/sed_head_ep10.pt \
#   --ckpt /workspace/models/BEATs/BEATs_iter3_plus_AS2M.pt \
#   --outdir /workspace/logs/sed_plus_balanced_v2 \
#   --base_tau 0.35 --smooth_win 2 --min_len_sec 0.06 --max_len_sec 2.5 --merge_gap_sec 0.25 \
#   --seg_tau 0.30 --seg_lens "1.0,2.0" \
#   --tol_sec_json /workspace/configs/tol/click_tol.json