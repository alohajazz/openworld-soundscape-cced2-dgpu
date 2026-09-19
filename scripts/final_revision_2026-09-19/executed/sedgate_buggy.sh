#!/bin/bash
set -u
OUT=/workspace/logs/sedgate_fixed_20260702
VAL=/workspace/data/exp_beats_v1_ambient1/val.csv
TRAIN=/workspace/data/exp_beats_v1_ambient1/train.csv
TOL=/workspace/configs/tol/tol_min_overrides.json
EVAL_ARGS="--val_csv $VAL --base_tau 0.35 --smooth_win 2 --min_len_sec 0.06 --max_len_sec 2.5 --merge_gap_sec 0.35 --tau_low_delta 0.15 --tol_sec_json $TOL --seg_tau 0.30 --seg_lens 1.0,2.0"
SUM="$OUT/summary.csv"
CKPT=/workspace/ckpts_dapt_mam_b/BEATs_DAPT_MAM_step6000.pt
echo "BUGGY GATE START $(date)" >> "$OUT/progress.log"
for seed in 42 123 777; do
  SED="$OUT/head_DAPTbuggy_s${seed}"
  python3 /workspace/scripts/train_sed_beats_weak_plus.py --ckpt "$CKPT" --train_csv "$TRAIN" --val_csv "$VAL" --outdir "$SED" --epochs 8 --bs 32 --lr 0.002 --seed "$seed" > "${SED}.trainlog" 2>&1
  BEST=$(python3 -c "import json;print(json.load(open('$SED/best.json'))['best_path'])")
  EV="$OUT/eval_DAPTbuggy_s${seed}"
  python3 /workspace/scripts/eval_sed_beats_report.py --ckpt "$CKPT" --head_ckpt "$BEST" --outdir "$EV" $EVAL_ARGS > "${EV}.evallog" 2>&1
  python3 - "$EV" "$BEST" "DAPTbuggy" "$seed" >> "$SUM" << 'PY'
import json,re,sys
ev,best,model,seed=sys.argv[1:5]
m=json.load(open(ev+"/reports/metrics_overview.json"))
ep=re.search(r'ep(\d+)',best).group(1)
print(f'{model},{seed},{ep},{m["eventF1"]["F1"]:.4f},{m["clipF1"]["F1"]:.4f},{m["segmentF1"]["2.0"]["F1"]:.4f}')
PY
  echo "[done] DAPTbuggy seed$seed $(date)" >> "$OUT/progress.log"
done
# 三者集計
python3 - "$SUM" >> "$OUT/threeway_verdict.txt" << 'PY'
import csv,statistics as st,sys,math
rows=list(csv.DictReader(open(sys.argv[1])))
def stats(model,met):
    v=[float(x[met]) for x in rows if x["model"]==model]
    return v, st.mean(v), (st.stdev(v) if len(v)>1 else 0.0)
def ttest(a,b):
    ma,mb=st.mean(a),st.mean(b); sa,sb=st.stdev(a),st.stdev(b); na,nb=len(a),len(b)
    se=math.sqrt(sa*sa/na+sb*sb/nb)
    return (mb-ma), (0 if se==0 else (mb-ma)/se)
print("=== 三者比較 (n=3 seed, sample sd) ===")
for met in ["eventF1","clipF1","seg2sF1"]:
    line=[]
    for model in ["PRETRAIN","DAPTbuggy","DAPTfix"]:
        v,m,s=stats(model,met); line.append(f"{model}={m:.4f}±{s:.4f}")
    print(met+": "+" | ".join(line))
    pv=[float(x[met]) for x in rows if x["model"]=="PRETRAIN"]
    bv=[float(x[met]) for x in rows if x["model"]=="DAPTbuggy"]
    fv=[float(x[met]) for x in rows if x["model"]=="DAPTfix"]
    d1,t1=ttest(pv,bv); d2,t2=ttest(pv,fv); d3,t3=ttest(bv,fv)
    print(f"   buggy-PRE={d1:+.4f}(t={t1:.2f}) | fix-PRE={d2:+.4f}(t={t2:.2f}) | fix-buggy={d3:+.4f}(t={t3:.2f})")
print("(|t|>~2.8 で n=3 両側 p<0.05 目安)")
PY
echo "BUGGY GATE DONE $(date)" >> "$OUT/progress.log"
