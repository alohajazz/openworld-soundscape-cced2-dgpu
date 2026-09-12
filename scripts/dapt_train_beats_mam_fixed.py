# -*- coding: utf-8 -*-
"""DAPT MAM 再学習(masking 位置バグ修正版) — SciRep 論文 結果更新用。
論文と同一: data=dapt_manifest_all_16k.tsv, init=BEATs_iter3+AS2M, centroids=PRETRAIN fit k1024,
  mask0.75/batch16/lr1e-4(head1e-3)/cosine+5%warmup/bf16/wd0.01/10s clip。
訂正1点: mask を encoder の「前」で input patch に mask_token 置換(論文本文 "masked input patches" 準拠)。
target: 凍結 teacher(PRETRAIN) の k-means label(固定, collapse 回避, 論文 "on PRETRAIN features" 準拠)。
"""
import os, sys, csv, math, time, random
from pathlib import Path
from functools import lru_cache
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
sys.path.append("/workspace/third_party/beats")
from BEATs import BEATs, BEATsConfig

SMOKE   = os.environ.get("SMOKE","0")=="1"
DEV     = "cuda"
MANIFEST= "/workspace/data/externaldata/dapt_manifest_all_16k.tsv"
CENTROIDS="/workspace/dapt_kmeans_labels/centroids_k1024.npy"
INIT    = "/workspace/models/BEATs/BEATs_iter3_plus_AS2M.pt"
CKPTDIR = Path("/workspace/ckpts_dapt_mam_fixed"); CKPTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR  = Path("/workspace/logs/dapt_mam_fixed"); LOGDIR.mkdir(parents=True, exist_ok=True)
BATCH=16; LR=1e-4; MASK_RATIO=0.75; NUM_CLASSES=1024; WD=0.01; WARMUP=0.05
TARGET_SR=16000; TARGET_SEC=10.0; TARGET_LEN=int(TARGET_SR*TARGET_SEC)
CKPT_EVERY= 200 if SMOKE else 5000
EPOCHS=1; NUM_WORKERS= 0 if SMOKE else 4

def log(m):
    s=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"; print(s,flush=True)
    with open(LOGDIR/"run.log","a") as f: f.write(s+"\n")

class DAPTData(torch.utils.data.Dataset):
    def __init__(s):
        s.items=[]
        with open(MANIFEST) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                s.items.append((r["path"], float(r["start_sec"])))
        if SMOKE: s.items=s.items[:2000]
    def __len__(s): return len(s.items)
    @staticmethod
    @lru_cache(maxsize=8)
    def _rs(o,t): return None if o==t else torchaudio.transforms.Resample(o,t)
    def __getitem__(s, i):
        p, st = s.items[i]
        try:
            import soundfile as sf
            info=sf.info(p); off=int(st*info.samplerate); need=int(TARGET_SEC*info.samplerate)
            x,_=sf.read(p, start=off, frames=need, dtype="float32", always_2d=True); x=x.mean(1)
            w=torch.from_numpy(x)
            if info.samplerate!=TARGET_SR:
                rs=s._rs(info.samplerate, TARGET_SR); w=rs(w.unsqueeze(0)).squeeze(0) if rs else w
            if w.numel()<TARGET_LEN: w=F.pad(w,(0,TARGET_LEN-w.numel()))
            return w[:TARGET_LEN]
        except Exception:
            return torch.zeros(TARGET_LEN)

def build_beats():
    ck=torch.load(INIT, map_location="cpu")
    m=BEATs(BEATsConfig(ck["cfg"])); m.load_state_dict(ck["model"], strict=False)
    return m, ck["cfg"]

def main():
    log(f"START dapt_mam_fixed SMOKE={SMOKE} batch={BATCH} mask={MASK_RATIO}")
    student, cfg = build_beats(); student.to(DEV).train()
    teacher, _   = build_beats(); teacher.to(DEV).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    init_sd={k:v.detach().clone() for k,v in student.state_dict().items()}
    cent=np.load(CENTROIDS).astype(np.float32)
    cent=torch.from_numpy(cent/np.clip(np.linalg.norm(cent,axis=1,keepdims=True),1e-8,None)).to(DEV)
    emb=cfg["encoder_embed_dim"]; pe=cfg["embed_dim"]
    mask_token=nn.Parameter(torch.zeros(pe, device=DEV))
    predictor=nn.Sequential(nn.Linear(emb,emb),nn.GELU(),nn.LayerNorm(emb),nn.Linear(emb,NUM_CLASSES)).to(DEV)
    opt=torch.optim.AdamW([{"params":student.parameters(),"lr":LR},
                           {"params":list(predictor.parameters())+[mask_token],"lr":LR*10}], weight_decay=WD)
    ds=DAPTData(); total=(len(ds)//BATCH)*EPOCHS
    warm=int(total*WARMUP)
    sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda st:(st/max(1,warm)) if st<warm else 0.5*(1+math.cos(math.pi*(st-warm)/max(1,total-warm))))
    dl=torch.utils.data.DataLoader(ds,batch_size=BATCH,shuffle=True,num_workers=NUM_WORKERS,drop_last=True,pin_memory=True)
    log(f"dataset={len(ds)} total_steps={total} warmup={warm}")

    @torch.no_grad()
    def labels_of(wav):
        tf,_=teacher.extract_features(wav)                 # 凍結 teacher encoder出力 (B,T,768)
        fn=F.normalize(tf.reshape(-1,tf.shape[-1]),dim=-1)
        return (fn@cent.t()).argmax(-1).reshape(tf.shape[0],tf.shape[1])

    def student_masked(wav):
        fb=student.preprocess(wav).unsqueeze(1)
        x=student.patch_embedding(fb); x=x.reshape(x.shape[0],pe,-1).transpose(1,2)   # (B,L,512)
        B,L,_=x.shape; nm=int(L*MASK_RATIO)
        ids=torch.rand(B,L,device=DEV).argsort(1); mb=torch.zeros(B,L,dtype=torch.bool,device=DEV)
        mb.scatter_(1,ids[:,:nm],True)
        x=torch.where(mb.unsqueeze(-1), mask_token.to(x.dtype), x)   # ← encoder 前で mask
        x=student.layer_norm(x)
        if student.post_extract_proj is not None: x=student.post_extract_proj(x)
        x=student.dropout_input(x); x,_=student.encoder(x,padding_mask=None)
        return predictor(x), mb

    step=0; t0=time.time()
    for ep in range(1,EPOCHS+1):
        for wav in dl:
            wav=wav.to(DEV)
            with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                lab=labels_of(wav)
                logits,mb=student_masked(wav)
                loss=F.cross_entropy(logits[mb].float(), lab[mb])
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(student.parameters())+list(predictor.parameters()),1.0)
                opt.step(); sched.step()
            step+=1
            if step%50==0 or step==1:
                with torch.no_grad():
                    acc=(logits[mb].argmax(-1)==lab[mb]).float().mean().item()*100
                    sd=student.state_dict()
                    wr=sum(((sd[k]-init_sd[k]).float().norm()**2).item() for k in init_sd)**0.5/ \
                       sum((init_sd[k].float().norm()**2).item() for k in init_sd)**0.5
                log(f"step {step}/{total} loss={loss.item():.3f} acc={acc:.1f}% enc_wrel={wr:.4f} ({step/(time.time()-t0):.2f} it/s)")
            if step%CKPT_EVERY==0:
                torch.save({"cfg":cfg,"model":student.state_dict()}, CKPTDIR/f"BEATs_DAPT_MAM_fixed_step{step}.pt")
                log(f"[ckpt] step{step}")
            if SMOKE and step>=400: break
        if SMOKE: break
    torch.save({"cfg":cfg,"model":student.state_dict()}, CKPTDIR/f"BEATs_DAPT_MAM_fixed_step{step}.pt")
    log(f"DONE step={step} {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
