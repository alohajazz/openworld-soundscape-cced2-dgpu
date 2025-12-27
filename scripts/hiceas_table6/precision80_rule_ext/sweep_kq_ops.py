import os, sys, json, argparse
import numpy as np, pandas as pd
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=15.0, help="評価の許容秒（±）")
    ap.add_argument("--minus4p5", action="store_true", help="-4.5秒補正を適用")
    ap.add_argument("--dens2", action="store_true", help="後段に local_density>=2 を適用（あれば）")
    ap.add_argument("--use_offset", action="store_true", help="per-base offset CSV があれば適用")
    ap.add_argument("--qlist", type=str, default="0.990,0.992,0.995,0.997,0.998,0.999")
    ap.add_argument("--klist", type=str, default="1,2,3,5,10")
    ap.add_argument("--fp_target", type=float, default=0.5, help="推奨を出す FP/h の上限（例 0.5 or 1.0）")
    ap.add_argument("--update_ops", action="store_true", help="推奨の q,K,minus4p5 で ops_rule.json を更新")
    ap.add_argument("--tag", type=str, default="", help="出力CSVの接尾辞（任意）")
    ap.add_argument("--dens_min", type=int, default=2, help="local_density の下限 (default=2)")

    args=ap.parse_args()

    BASE=os.environ["HICEAS_OP_EMB"]; ANN=os.environ.get("HICEAS_OP_ANN","")
    feat_p=Path(BASE)/"hp_feats_ext_minlen4.ckpt.parquet"
    pred_p=Path(BASE)/"predictions_op_quiet_base.csv"
    outdir=Path(BASE)/"precision80_rule_ext"; outdir.mkdir(parents=True,exist_ok=True)
    ops_p=Path(BASE)/"precision80_rule_ext/ops_rule.json"
    off_p1=Path(BASE)/"offset_sweep_per_base_wide30.csv"
    off_p2=Path(BASE)/"offset_sweep_per_base_wide.csv"

    qlist=[float(x) for x in args.qlist.split(",") if x.strip()]
    klist=[int(x) for x in args.klist.split(",") if x.strip()]

    feat=pd.read_parquet(feat_p)
    pred=pd.read_csv(pred_p)

    ann=None
    has_ann=False
    if ANN and Path(ANN).exists():
        ann=pd.read_csv(ANN); ann.columns=[c.strip().lower() for c in ann.columns]
        if "base" not in ann.columns or "center" not in ann.columns:
            from pathlib import Path as _P
            fcol=next((c for c in ann.columns if "file" in c or "name" in c or c=="base"), None)
            tcol=next((c for c in ann.columns if "time" in c or c=="center" or "stamp" in c), None)
            if fcol and tcol:
                ann=pd.DataFrame({"base":ann[fcol].astype(str).apply(lambda p:_P(p).name),
                                  "center":pd.to_numeric(ann[tcol],errors="coerce")}).dropna()
        has_ann=("base" in ann.columns) and ("center" in ann.columns)

    # 中心列自動検出
    CENTER=["event_center_off","event_center","center_sec_off","center_sec","center"]
    center_col=next((c for c in CENTER if c in feat.columns), None)
    if center_col is None:
        raise SystemExit(f"[FEAT] center column not found. candidates={CENTER}")

    # base 列
    if "base" not in feat.columns:
        if "path" in feat.columns:
            from pathlib import Path as _P
            feat["base"]=feat["path"].apply(lambda p:_P(p).name)
        else:
            raise SystemExit("[FEAT] base or path column is required")

    # 注釈base限定
    if has_ann:
        ann_bases=set(ann["base"].unique())
        feat=feat[feat["base"].isin(ann_bases)].copy()
    else:
        ann_bases=set(feat["base"].unique())

    # pred base列
    if "base" not in pred.columns:
        from pathlib import Path as _P
        pred["base"]=pred["path"].apply(lambda p:_P(p).name)
    pred=pred[pred["base"].isin(ann_bases)].copy()

    # s_hat_near を付与
    def attach_s_hat_near(df, center_col):
        out=df.copy()
        out["s_hat_near"]=np.nan
        for b,g in out.groupby("base"):
            P=pred[pred["base"]==b][["center_sec","s_hat"]].to_numpy()
            if P.size==0: continue
            t=g[center_col].astype(float).to_numpy()
            idx=np.abs(t[:,None]-P[:,0][None,:]).argmin(axis=1)
            out.loc[g.index,"s_hat_near"]=P[idx,1]
        return out.dropna(subset=["s_hat_near"])

    ev=attach_s_hat_near(feat[["base",center_col]+[c for c in ["run_len","local_density","peak_count","downsweep","p_band_80_800"] if c in feat.columns]].copy(),
                          center_col).rename(columns={center_col:"event_center"})

    # per-base offset（任意）
    if args.use_offset:
        off=None
        for p in [off_p1, off_p2]:
            if p.exists():
                tmp=pd.read_csv(p); tmp.columns=[c.lower() for c in tmp.columns]
                col = "best_offset" if "best_offset" in tmp.columns else \
                    ("best_off_sec" if "best_off_sec" in tmp.columns else \
                    ("offset" if "offset" in tmp.columns else \
                    ("offset_sec" if "offset_sec" in tmp.columns else None)))

                if col:
                    off=tmp[["base",col]].rename(columns={col:"off"})
                    break
        if off is not None:
            ev=ev.merge(off, on="base", how="left")
            ev["event_center"]=ev["event_center"].astype(float)+ev["off"].fillna(0.0).astype(float)
            ev=ev.drop(columns=["off"])

    # -4.5s（任意）
    if args.minus4p5:
        ev["event_center"]=ev["event_center"].astype(float)-4.5

    # density>=2（任意）
    # if args.dens2 and ("local_density" in ev.columns):
    #     ev=ev[ev["local_density"]>=2].copy()

    # HOURS
    hop=(pred.groupby("base")["center_sec"].diff().dropna().round(3)).mode().iloc[0]
    WPH=max(1,int(round(3600.0/float(hop))))
    HOURS=len(pred)/WPH

    # スコア関数
    gt_by_base={}
    if has_ann:
        gt_by_base={b:ann.loc[ann["base"]==b,"center"].to_numpy() for b in ann_bases}

    def score(df):
        if df.empty:
            if has_ann:
                return 0.0,0.0,0.0,0.0,0,0,len(ann)
            else:
                return np.nan,np.nan,np.nan,0.0,0,0,0
        if not has_ann:
            # 注釈が無い場合は FP/h と N のみ
            FPH=len(df)/HOURS
            return np.nan,np.nan,np.nan,FPH,0,0,0
        TP=FP=0
        for _,r in df.iterrows():
            gt=gt_by_base.get(r["base"], np.array([]))
            if gt.size==0: continue
            d=np.abs(gt - float(r["event_center"]))
            TP+=int(d.min()<=args.tol); FP+=int(d.min()>args.tol)
        FN=len(ann)-TP
        P=TP/(TP+FP+1e-9); R=TP/(TP+FN+1e-9); F=2*P*R/(P+R+1e-9); FPH=FP/HOURS
        return P,R,F,FPH,TP,FP,FN

    def topk_per_hour(df,K):
        if df.empty: return df
        x=df.copy()
        x["hour_bucket"]=(x["event_center"]//3600).astype(int)
        x=x.sort_values(["base","hour_bucket","s_hat_near"], ascending=[True,True,False])
        return x.groupby(["base","hour_bucket"]).head(K).drop(columns=["hour_bucket"])

    rows=[]
    for q in qlist:
        pool = ev["s_hat_near"].astype(float).to_numpy()
        pool = pool[~np.isnan(pool)]
        if pool.size == 0:
            print(f"[SKIP] no candidates to compute quantile at q={q}")
            continue

        thr = float(np.quantile(pool, q))
        base_q = ev.loc[ev["s_hat_near"] >= thr].copy()
        if base_q.empty:
            print(f"[SKIP] no events after q={q}")
            continue

        # ★ dens_min バックオフ：dens2 指定時のみ
        if args.dens2 and ("local_density" in base_q.columns):
            kept = None
            for dmin in [args.dens_min, 1, 0]:
                tmp = base_q[base_q["local_density"] >= dmin].copy()
                if not tmp.empty:
                    kept = tmp; break
            if kept is None:
                print(f"[SKIP] no events after dens filter (min tried: {args.dens_min}->1->0) at q={q}")
                continue
            base_q = kept

        for K in klist:
            sel = topk_per_hour(base_q, K)
            P,R,F1,FPH,TP,FP,FN = score(sel)
            rows.append({
                "q":q, "K":K, "P":P, "R":R, "F1":F1, "FP_per_h":FPH,
                "TP":TP, "FP":FP, "FN":FN, "N":len(sel),
                "tol_sec":args.tol, "minus4p5":args.minus4p5,
                "dens2":args.dens2, "use_offset":args.use_offset,
                "dens_min": (args.dens_min if args.dens2 else np.nan)
            })


    res = pd.DataFrame(rows)
    tag = args.tag.strip()
    out_csv = outdir / f"sweep_kq{('_'+tag if tag else '')}.csv"
    if res.empty:
        res.to_csv(out_csv, index=False)
        print(f"[WARN] no combos produced any rows. wrote empty CSV: {out_csv}")
        sys.exit(0)

    res = res.sort_values(["K","q"])
    res.to_csv(out_csv, index=False)
    print("[OK] wrote:", out_csv)


    # 推奨（注釈がある場合のみ算出）
    if has_ann:
        cand=res[res["FP_per_h"]<=args.fp_target]
        if cand.empty:
            print(f"[MISS] no combo under FP/h<={args.fp_target}")
            sys.exit(0)
        pick=cand.sort_values(["P","F1","R","N"], ascending=[False,False,False,False]).head(1).iloc[0]
        print(f"[BEST FP/h<={args.fp_target}] q={pick['q']:.3f}  K={int(pick['K'])}  "
              f"P={pick['P']:.3f} R={pick['R']:.3f} F1={pick['F1']:.3f} FP/h={pick['FP_per_h']:.3f} "
              f"N={int(pick['N'])} tol={pick['tol_sec']}  -4.5={bool(pick['minus4p5'])} dens2={bool(pick['dens2'])} off={bool(pick['use_offset'])}")
        if args.update_ops:
            ops={"post_filter":{
                "use_offset":bool(pick["use_offset"]),
                "minus4p5":bool(pick["minus4p5"]),
                "q_s_hat_near":float(pick["q"]),
                "topK_per_hour":int(pick["K"])
            }}
            Path(ops_p).write_text(json.dumps(ops, indent=2))
            print("[OK] updated ops_rule.json")

if __name__=="__main__":
    main()
