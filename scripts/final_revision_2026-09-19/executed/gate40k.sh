#!/bin/bash
set -u
OUT=/workspace/logs/sedgate_fixed_20260702
VAL=/workspace/data/exp_beats_v1_ambient1/val.csv
TRAIN=/workspace/data/exp_beats_v1_ambient1/train.csv
TOL=/workspace/configs/tol/tol_min_overrides.json
EA="--val_csv $VAL --base_tau 0.35 --smooth_win 2 --min_len_sec 0.06 --max_len_sec 2.5 --merge_gap_sec 0.35 --tau_low_delta 0.15 --tol_sec_json $TOL --seg_tau 0.30 --seg_lens 1.0,2.0"
SUM="$OUT/summary.csv"
CKPT=/workspace/ckpts_dapt_mam_fixed/BEATs_DAPT_MAM_fixed_step40000.pt
echo "GATE40K START $(date)" >> "$OUT/progress.log"
for seed in 42 123 777 101 202 303 404 505; do
  SED="$OUT/head_DAPTfix40k_s${seed}"
  python3 /workspace/scripts/train_sed_beats_weak_plus.py --ckpt "$CKPT" --train_csv "$TRAIN" --val_csv "$VAL" --outdir "$SED" --epochs 8 --bs 32 --lr 0.002 --seed "$seed" > "${SED}.log" 2>&1
  BEST=$(python3 -c "import json;print(json.load(open('$SED/best.json'))['best_path'])")
  EV="$OUT/eval_DAPTfix40k_s${seed}"
  python3 /workspace/scripts/eval_sed_beats_report.py --ckpt "$CKPT" --head_ckpt "$BEST" --outdir "$EV" $EA > "${EV}.log" 2>&1
  python3 - "$EV" "$BEST" "$seed" >> "$SUM" << 'PY'
import json,re,sys
ev,best,seed=sys.argv[1:4]; m=json.load(open(ev+"/reports/metrics_overview.json"))
ep=re.search(r'ep(\d+)',best).group(1)
print(f'DAPTfix40k,{seed},{ep},{m["eventF1"]["F1"]:.4f},{m["clipF1"]["F1"]:.4f},{m["segmentF1"]["2.0"]["F1"]:.4f}')
PY
  echo "[done] DAPTfix40k s$seed $(date)" >> "$OUT/progress.log"
done
echo "GATE40K DONE $(date)" >> "$OUT/progress.log"
