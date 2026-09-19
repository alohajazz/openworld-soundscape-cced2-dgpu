#!/bin/bash
set -u
OUT=/workspace/logs/sedgate_fixed_20260702
VAL=/workspace/data/exp_beats_v1_ambient1/val.csv
TRAIN=/workspace/data/exp_beats_v1_ambient1/train.csv
TOL=/workspace/configs/tol/tol_min_overrides.json
EA="--val_csv $VAL --base_tau 0.35 --smooth_win 2 --min_len_sec 0.06 --max_len_sec 2.5 --merge_gap_sec 0.35 --tau_low_delta 0.15 --tol_sec_json $TOL --seg_tau 0.30 --seg_lens 1.0,2.0"
SUM="$OUT/summary.csv"
declare -A M
M[PRETRAIN]=/workspace/models/BEATs/BEATs_iter3_plus_AS2M.pt
M[DAPTfix]=/workspace/ckpts_dapt_mam_fixed/BEATs_DAPT_MAM_fixed_step127641.pt
echo "MORESEEDS START $(date)" >> "$OUT/progress.log"
for seed in 101 202 303 404 505; do
  for model in PRETRAIN DAPTfix; do
    CKPT=${M[$model]}; SED="$OUT/head_${model}_s${seed}"
    python3 /workspace/scripts/train_sed_beats_weak_plus.py --ckpt "$CKPT" --train_csv "$TRAIN" --val_csv "$VAL" --outdir "$SED" --epochs 8 --bs 32 --lr 0.002 --seed "$seed" > "${SED}.trainlog" 2>&1
    BEST=$(python3 -c "import json;print(json.load(open('$SED/best.json'))['best_path'])")
    EV="$OUT/eval_${model}_s${seed}"
    python3 /workspace/scripts/eval_sed_beats_report.py --ckpt "$CKPT" --head_ckpt "$BEST" --outdir "$EV" $EA > "${EV}.evallog" 2>&1
    python3 - "$EV" "$BEST" "$model" "$seed" >> "$SUM" << 'PY'
import json,re,sys
ev,best,model,seed=sys.argv[1:5]
m=json.load(open(ev+"/reports/metrics_overview.json"))
ep=re.search(r'ep(\d+)',best).group(1)
print(f'{model},{seed},{ep},{m["eventF1"]["F1"]:.4f},{m["clipF1"]["F1"]:.4f},{m["segmentF1"]["2.0"]["F1"]:.4f}')
PY
    echo "[done] $model s$seed $(date)" >> "$OUT/progress.log"
  done
done
python3 - "$SUM" > "$OUT/paired_verdict.txt" << 'PY'
import csv,statistics as st,math,sys
rows=list(csv.DictReader(open(sys.argv[1])))
def by(model,met):
    return {x["seed"]:float(x[met]) for x in rows if x["model"]==model}
def paired(a,b):  # b-a over shared seeds
    ks=sorted(set(a)&set(b)); d=[b[k]-a[k] for k in ks]
    n=len(d); md=st.mean(d); sd=st.stdev(d) if n>1 else 0.0
    se=sd/math.sqrt(n) if n>1 else 0.0; t=md/se if se>0 else float('nan')
    return n,md,sd,t
print("=== PAIRED (DAPTfix - PRETRAIN), matched seeds ===")
for met in ["eventF1","clipF1","seg2sF1"]:
    n,md,sd,t=paired(by("PRETRAIN",met),by("DAPTfix",met))
    print(f"{met}: n={n} mean_diff={md:+.4f} sd={sd:.4f} t={t:.2f}  (df={n-1}; |t|>2.36 => p<0.05 at df=7)")
print("\n=== group means (n) ===")
for model in ["PRETRAIN","DAPTfix","DAPTbuggy"]:
    for met in ["eventF1","seg2sF1"]:
        v=[float(x[met]) for x in rows if x["model"]==model]
        if v: print(f"{model} {met}: mean={st.mean(v):.4f} sd={(st.stdev(v) if len(v)>1 else 0):.4f} n={len(v)}")
PY
echo "MORESEEDS DONE $(date)" >> "$OUT/progress.log"
