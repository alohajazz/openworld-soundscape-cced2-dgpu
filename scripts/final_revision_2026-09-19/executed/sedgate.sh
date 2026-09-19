#!/bin/bash
set -u
OUT=/workspace/logs/sedgate_fixed_20260702
mkdir -p "$OUT"
VAL=/workspace/data/exp_beats_v1_ambient1/val.csv
TRAIN=/workspace/data/exp_beats_v1_ambient1/train.csv
TOL=/workspace/configs/tol/tol_min_overrides.json
EVAL_ARGS="--val_csv $VAL --base_tau 0.35 --smooth_win 2 --min_len_sec 0.06 --max_len_sec 2.5 --merge_gap_sec 0.35 --tau_low_delta 0.15 --tol_sec_json $TOL --seg_tau 0.30 --seg_lens 1.0,2.0"
SUM="$OUT/summary.csv"
echo "model,seed,best_ep,eventF1,clipF1,seg2sF1" > "$SUM"
echo "GATE START $(date)" > "$OUT/progress.log"

declare -A M
M[PRETRAIN]=/workspace/models/BEATs/BEATs_iter3_plus_AS2M.pt
M[DAPTfix]=/workspace/ckpts_dapt_mam_fixed/BEATs_DAPT_MAM_fixed_step127641.pt

for model in PRETRAIN DAPTfix; do
  CKPT=${M[$model]}
  for seed in 42 123 777; do
    SED="$OUT/head_${model}_s${seed}"
    python3 /workspace/scripts/train_sed_beats_weak_plus.py --ckpt "$CKPT" --train_csv "$TRAIN" --val_csv "$VAL" --outdir "$SED" --epochs 8 --bs 32 --lr 0.002 --seed "$seed" > "${SED}.trainlog" 2>&1
    BEST=$(python3 -c "import json;print(json.load(open('$SED/best.json'))['best_path'])")
    EV="$OUT/eval_${model}_s${seed}"
    python3 /workspace/scripts/eval_sed_beats_report.py --ckpt "$CKPT" --head_ckpt "$BEST" --outdir "$EV" $EVAL_ARGS > "${EV}.evallog" 2>&1
    python3 - "$EV" "$BEST" "$model" "$seed" >> "$SUM" << 'PY'
import json,re,sys
ev,best,model,seed=sys.argv[1:5]
m=json.load(open(ev+"/reports/metrics_overview.json"))
ep=re.search(r'ep(\d+)',best).group(1)
print(f'{model},{seed},{ep},{m["eventF1"]["F1"]:.4f},{m["clipF1"]["F1"]:.4f},{m["segmentF1"]["2.0"]["F1"]:.4f}')
PY
    echo "[done] $model seed$seed $(date)" >> "$OUT/progress.log"
  done
done

python3 - "$SUM" >> "$OUT/progress.log" << 'PY'
import csv,statistics as st,sys
rows=list(csv.DictReader(open(sys.argv[1])))
res={}
for model in ["PRETRAIN","DAPTfix"]:
    r=[x for x in rows if x["model"]==model]
    res[model]={}
    for met in ["eventF1","clipF1","seg2sF1"]:
        v=[float(x[met]) for x in r]
        res[model][met]=(st.mean(v), st.pstdev(v), v)
        print(f'{model} {met}: mean={st.mean(v):.4f} sd={st.pstdev(v):.4f} vals={v}')
print("---VERDICT---")
for met in ["eventF1","clipF1","seg2sF1"]:
    pm,ps,_=res["PRETRAIN"][met]; dm,ds,_=res["DAPTfix"][met]
    gap=dm-pm; band=2*((ps**2+ds**2)**0.5)
    verd="DAPT WINS (>2sd)" if gap>band else ("PRETRAIN>= " if gap<=0 else "within noise")
    print(f'{met}: DAPT-PRETRAIN={gap:+.4f}  noiseband(2*pooled_sd)={band:.4f}  => {verd}')
PY
echo "GATE DONE $(date)" >> "$OUT/progress.log"
