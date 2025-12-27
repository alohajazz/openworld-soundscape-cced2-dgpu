#!/usr/bin/env bash
set -euo pipefail
BASE="${HICEAS_OP_EMB:?}"; ANN="${HICEAS_OP_ANN:?}"

python - <<'PY'
import os, json, numpy as np, pandas as pd
from pathlib import Path

BASE=os.environ["HICEAS_OP_EMB"]; ANN=os.environ["HICEAS_OP_ANN"]

# ========= 出力先/設定 =========
feat_p = Path(BASE)/"hp_feats_ext_minlen4.ckpt.parquet"
pred_p = Path(BASE)/"predictions_op_quiet_base.csv"
ops_p  = Path(BASE)/"precision80_rule_ext/ops_rule.json"
strict_p = Path(BASE)/"precision80_rule_ext/ops_strict.json"   # 任意
out_csv = Path(BASE)/"precision80_rule_ext/events_ops_final.csv"

cfg=json.loads(Path(ops_p).read_text())
use_off = bool(cfg["post_filter"].get("use_offset", False))
use_m45 = bool(cfg["post_filter"].get("minus4p5", False))
q_thr   = float(cfg["post_filter"].get("q_s_hat_near", 0.995))
K       = int(cfg["post_filter"].get("topK_per_hour", 1))

# strict モード（後段ルール）と burst 拡張スイッチ
STRICT_MODE=None
ENABLE_BURST=False
if strict_p.exists():
    try:
        s=json.loads(Path(strict_p).read_text())
        STRICT_MODE=int(s.get("mode", None)) if s.get("mode") is not None else None
        ENABLE_BURST=bool(s.get("enable_burst", False)) or ENABLE_BURST
    except Exception:
        pass

# ========= 入力読み込み =========
feat=pd.read_parquet(feat_p)
pred=pd.read_csv(pred_p)
ann =pd.read_csv(ANN); ann.columns=[c.strip().lower() for c in ann.columns]
if "base" not in ann.columns or "center" not in ann.columns:
    from pathlib import Path as _P
    fcol=next((c for c in ann.columns if "file" in c or "name" in c or c=="base"), None)
    tcol=next((c for c in ann.columns if "time" in c or c=="center" or "stamp" in c), None)
    ann=pd.DataFrame({"base":ann[fcol].astype(str).apply(lambda p:_P(p).name),
                      "center":pd.to_numeric(ann[tcol],errors="coerce")}).dropna()
ann_bases=set(ann["base"].unique())

CENTER_CANDS=["event_center_off","event_center","center_sec_off","center_sec","center"]
center_col=next((c for c in CENTER_CANDS if c in feat.columns), None)
if center_col is None: raise SystemExit(f"[FEAT] center column not found. candidates={CENTER_CANDS}")

if "base" not in feat.columns:
    if "path" in feat.columns:
        from pathlib import Path as _P
        feat["base"]=feat["path"].apply(lambda p:_P(p).name)
    else:
        raise SystemExit("[FEAT] base or path column is required")

# 候補（注釈 base 限定）
ev=feat[["base",center_col]].rename(columns={center_col:"event_center"})
ev=ev[ev["base"].isin(ann_bases)].copy()

# pred の base を合わせる
if "base" not in pred.columns:
    from pathlib import Path as _P
    pred["base"]=pred["path"].apply(lambda p:_P(p).name)
pred=pred[pred["base"].isin(ann_bases)].copy()

# Quiet 近傍スコア付与
def attach_s_hat_near(df):
    out=df.copy(); out["s_hat_near"]=np.nan
    for b,g in out.groupby("base"):
        P=pred[pred["base"]==b][["center_sec","s_hat"]].to_numpy()
        if P.size==0: continue
        t=g["event_center"].astype(float).to_numpy()
        idx=np.abs(t[:,None]-P[:,0][None,:]).argmin(axis=1)
        out.loc[g.index,"s_hat_near"]=P[idx,1]
    return out.dropna(subset=["s_hat_near"])
ev=attach_s_hat_near(ev)

# ========= per-base offset（列名ゆらぎ対応／符号は「引く」） =========
off=None
if use_off:
    for name in ["offset_sweep_per_base_wide30.csv","offset_sweep_per_base_wide.csv"]:
        p=Path(BASE)/name
        if p.exists():
            tmp=pd.read_csv(p); tmp.columns=[c.lower() for c in tmp.columns]
            col = "best_offset" if "best_offset" in tmp.columns else \
                  ("best_off_sec" if "best_off_sec" in tmp.columns else \
                   ("offset"      if "offset"      in tmp.columns else \
                    ("offset_sec" if "offset_sec" in tmp.columns else None)))
            if col:
                off=tmp[["base",col]].rename(columns={col:"off"})
                break
if off is not None:
    ev=ev.merge(off, on="base", how="left")
    ev["event_center"]=ev["event_center"].astype(float) - ev["off"].fillna(0.0).astype(float)
    ev=ev.drop(columns=["off"])

# ========= -4.5s 補正 =========
if use_m45:
    ev["event_center"]=ev["event_center"].astype(float)-4.5

# ========= q しきい =========
pool=ev["s_hat_near"].astype(float).to_numpy()
pool=pool[~np.isnan(pool)]
if pool.size==0:
    ev.iloc[0:0][["base","event_center"]].to_csv(out_csv, index=False)
    print("[OK] wrote:", out_csv, "rows= 0"); raise SystemExit(0)
thr=float(np.quantile(pool, q_thr))
ev=ev.loc[ev["s_hat_near"]>=thr].copy()

# ========= 後段ルール（任意）=========
def post_constraints(df, mode):
    if mode is None: return df
    x=df.copy()
    if mode>=1 and {"run_len","local_density"}.issubset(df.columns):
        x=x[(x["run_len"]>=6)&(x["local_density"]<=1)]
    if mode>=2 and "peak_count" in x.columns:
        x=x[x["peak_count"]<=2]
    if mode>=3 and "downsweep" in x.columns:
        x=x[x["downsweep"]<=-10]
    if mode>=4 and "p_band_80_800" in x.columns:
        v=pd.to_numeric(x["p_band_80_800"], errors="coerce").dropna().values
        if v.size: x=x[x["p_band_80_800"]>=float(np.quantile(v,0.60))]
    if mode>=5 and "local_density" in x.columns:
        x=x[x["local_density"]>=2]
    return x
ev=post_constraints(ev, STRICT_MODE)

# ========= TopK/時 =========
if not ev.empty:
    ev["hour_bucket"]=(ev["event_center"]//3600).astype(int)
    ev=ev.sort_values(["base","hour_bucket","s_hat_near"], ascending=[True,True,False])
    ev=ev.groupby(["base","hour_bucket"]).head(K).drop(columns=["hour_bucket"])

# ========= burst 拡張（density>=2 の時間だけ K+1） =========
if ENABLE_BURST and "local_density" in feat.columns:
    try:
        if "hour_bucket" not in ev.columns:
            ev["hour_bucket"]=(ev["event_center"]//3600).astype(int)
        base_pool = feat[["base", center_col, "local_density"]].rename(columns={center_col:"event_center"})
        pool = base_pool.merge(ev[["base","event_center","hour_bucket"]],
                               on=["base","event_center"], how="right")
        burst_hours = pool.groupby(["base","hour_bucket"])["local_density"].max().reset_index()
        burst_hours = burst_hours[burst_hours["local_density"]>=2][["base","hour_bucket"]]
        if not burst_hours.empty:
            add_src = attach_s_hat_near(feat[["base",center_col]].rename(columns={center_col:"event_center"}))
            add_src["hour_bucket"]=(add_src["event_center"]//3600).astype(int)
            add_src = add_src.merge(burst_hours, on=["base","hour_bucket"])
            add_src = add_src.sort_values(["base","hour_bucket","s_hat_near"],
                                          ascending=[True,True,False]).groupby(["base","hour_bucket"]).head(1)
            ev = pd.concat([ev.drop(columns=["hour_bucket"], errors="ignore"),
                            add_src[["base","event_center"]]], ignore_index=True).drop_duplicates()
    except Exception as e:
        print("[WARN] burst extension skipped:", e)

# ========= 出力 =========
ev[["base","event_center"]].to_csv(out_csv, index=False)
print("[OK] wrote:", out_csv, "rows=", len(ev))
PY
