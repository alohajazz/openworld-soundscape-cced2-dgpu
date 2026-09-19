#!/bin/bash
set -uo pipefail
A=/workspace/scripts/audit_2026-07-30
CK=${STAGE2_CKPT:?STAGE2_CKPT を指定してください}
S1=/workspace/ckpts_dapt_mam_fixed/BEATs_DAPT_MAM_fixed_step127641.pt
TAG=${TAG:-stage2}
E=/workspace/embeddings
LOG=$A/stage2_downstream
mkdir -p "$LOG"
P="$LOG/progress.log"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$P"; }
die(){ say "!! ABORT: $*"; exit 1; }

say "=== START downstream for $CK (tag=$TAG) ==="

DK=/workspace/scripts/dump_known56_features.py
DW=/workspace/scripts/winaware_2026-05-09/dump_winaware_cached.py
VAL=/workspace/data/exp_beats_v1_ambient1/val.csv
TRAIN=/workspace/data/exp_beats_v1_ambient1/train.csv
FRDR_MAN=/workspace/data/externaldata/frdr_upcall/manifests/frdr_B_continuous_hop2s.csv

[ -f "$E/ind_val_$TAG/embeddings_000.npy" ] || {
  say "[1/9] ind_val 抽出"
  python3 "$DK" --csv "$VAL" --ckpt_beats "$CK" --outdir "$E/ind_val_$TAG" --dump_embeddings --batch_size 32 >"$LOG/ind_val.log" 2>&1 || die "ind_val 抽出失敗"; }
for pair in "frdr:$FRDR_MAN" \
            "op:/workspace/hiceas_op_manifest_winaware.csv" \
            "1706species:/workspace/hiceas_1706_species_manifest_winsafe.csv" \
            "1706part2:/workspace/hiceas_1706_part2_manifest_winsafe.csv"; do
  n=${pair%%:*}; m=${pair#*:}
  [ -f "$E/${n}_$TAG/embeddings_000.npy" ] || {
    say "[1/9] $n 抽出"
    python3 "$DW" --csv "$m" --ckpt_beats "$CK" --outdir "$E/${n}_$TAG" --dump_embeddings >"$LOG/${n}.log" 2>&1 || die "$n 抽出失敗"; }
done
python3 - "$TAG" <<'PY' | tee -a "$P" || die "件数 assert 失敗"
import glob,sys,numpy as np
tag=sys.argv[1]
want={"ind_val":1623,"frdr":44900,"op":54419,"1706species":87407,"1706part2":173160}
bad=[]
for k,v in want.items():
    fs=sorted(glob.glob(f"/workspace/embeddings/{k}_{tag}/embeddings_*.npy"))
    n=sum(np.load(f,mmap_mode='r').shape[0] for f in fs) if fs else 0
    ok = (n==v)
    print("   %-14s n=%-8d 期待 %-8d %s"%(k,n,v,"OK" if ok else "MISMATCH"))
    if not ok: bad.append(k)
if bad: raise SystemExit("件数不一致: "+",".join(bad))
PY

say "[2/9] cced2 参照を fit (val split n=1623)"
python3 -c "
import sys;sys.path.insert(0,'/workspace/scripts')
from cced2_utils import fit_cced2
fit_cced2('/workspace/embeddings/ind_val_$TAG','/workspace/embeddings/cced2_$TAG',k=50)
" >"$LOG/fit_cced2.log" 2>&1 || die "fit_cced2 失敗"
cat "$E/cced2_$TAG/cced2_norm.json" | tee -a "$P"
REF=$E/cced2_$TAG

say "[3/9] Table 2 (Quiet/Promoter/Union)"
FR_EMB=$E/frdr_$TAG FR_REF=$REF FR_OUT=$A/t2_$TAG python3 -u "$A/frdr_supervised_promoter_COPY.py" >"$LOG/t2.log" 2>&1 || die "Table2 失敗"
tail -5 "$LOG/t2.log" | tee -a "$P"

say "[4/9] Fusion"
FU_EMB=$E/frdr_$TAG FU_REF=$REF FU_OUT=$A/fusion_$TAG.csv python3 -u "$A/run_fusion_winaware_COPY.py" >"$LOG/fusion.log" 2>&1 || die "Fusion 失敗"

say "[5/9] Table 3"
T3_NAME=OceanBEATs_$TAG T3_EMB=$E/frdr_$TAG T3_REF=$REF T3_OUT=$A/t3_$TAG.csv python3 -u "$A/run_frdr_table3_winaware_COPY.py" >"$LOG/t3.log" 2>&1 || die "Table3 失敗"
tail -6 "$LOG/t3.log" | tee -a "$P"

say "[6/9] Supplementary Table S3"
S3_ARM=custom S3_NAME=OceanBEATs_$TAG S3_OP_DIR=$E/op_$TAG S3_SP_DIR=$E/1706species_$TAG \
  S3_CCED2_DIR=$REF S3_OUT=$A/s3_$TAG.csv python3 -u "$A/supp_s3_7sp_full_COPY.py" >"$LOG/s3.log" 2>&1 || die "S3 失敗"
tail -14 "$LOG/s3.log" | tee -a "$P"

say "[7/9] Table 4 / cross-day (9 条件)"
for src in check8_table4_rerun.py unified_promoter_audit_FIXED.py; do
  dst="$A/${src%.py}_$TAG.py"
  sed -e "s#$E/op_fixed_step127641#$E/op_$TAG#" \
      -e "s#$E/1706species_fixed_step127641#$E/1706species_$TAG#" \
      -e "s#$E/1706part2_fixed_step127641#$E/1706part2_$TAG#" \
      -e "s#check8_table4.json#check8_table4_$TAG.json#" \
      -e "s#unified_promoter_audit_FIXED_2026-07-30#unified_promoter_audit_${TAG}#" "$A/$src" > "$dst"
  python3 -u "$dst" >"$LOG/$(basename $dst .py).log" 2>&1 || say "  !! $src 失敗（継続）"
done
grep -E "FIXED|cross-day" "$LOG/check8_table4_rerun_$TAG.log" 2>/dev/null | tail -10 | tee -a "$P"
grep -A9 "Per-species best" "$LOG/unified_promoter_audit_FIXED_$TAG.log" 2>/dev/null | tee -a "$P"

say "[8/9] Supp Table S4 / Fig S2-S4 の CCED2 重複統計"
for src in analyze_168.py perch_overlap.py; do
  s=/workspace/scripts/audit_2026-07-30/overlap_src/$src
  [ -f "$s" ] || { say "  !! $src 未配置（スキップ）"; continue; }
  dst="$A/${src%.py}_$TAG.py"
  sed -e "s#$E/hiceas_op_fulldata_winaware#$E/op_$TAG#g" \
      -e "s#$E/cced2_fulldata#$REF#g" "$s" > "$dst"
  python3 -u "$dst" >"$LOG/${src%.py}_$TAG.log" 2>&1 || say "  !! $src 失敗（継続）"
  tail -12 "$LOG/${src%.py}_$TAG.log" | tee -a "$P"
done

TOL=/workspace/configs/tol/tol_min_overrides.json
EVAL_ARGS="--val_csv $VAL --base_tau 0.35 --smooth_win 2 --min_len_sec 0.06 --max_len_sec 2.5 --merge_gap_sec 0.35 --tau_low_delta 0.15 --tol_sec_json $TOL --seg_tau 0.30 --seg_lens 1.0,2.0"
run_sed(){ # $1=ckpt $2=tag $3=seed $4=summary
  local SED="$A/sedruns/$2_s$3"; mkdir -p "$(dirname $SED)"
  [ -f "$SED/done" ] || { python3 /workspace/scripts/train_sed_beats_weak_plus.py --ckpt "$1" --train_csv "$TRAIN" \
      --val_csv "$VAL" --outdir "$SED" --epochs 8 --bs 32 --lr 0.002 --seed $3 >"$LOG/sed_$2_$3.log" 2>&1 && touch "$SED/done"; }
  local B=$(ls "$SED"/sed_head_ep*.pt 2>/dev/null | sort -V | tail -1)
  [ -z "$B" ] && { say "   $2 seed$3 head 生成失敗"; return; }
  python3 /workspace/scripts/eval_sed_beats_report.py --ckpt "$1" --head_ckpt "$B" --outdir "$A/sedruns/ev_$2_s$3" $EVAL_ARGS >"$LOG/ev_$2_$3.log" 2>&1
  python3 - "$A/sedruns/ev_$2_s$3" "$2" "$3" "$4" <<'PY'
import json,sys,os
ev,tag,seed,summ=sys.argv[1:5]
d=json.load(open(os.path.join(ev,"reports","metrics_overview.json")))
r="%s,%s,%.4f,%.4f,%.4f"%(tag,seed,d["eventF1"]["F1"],d["clipF1"]["F1"],d["segmentF1"]["2.0"]["F1"])
open(summ,"a").write(r+"\n"); print("   ",r)
PY
}
say "[9/9]a Table 1: OceanBEATs(Stage2) 8 seed"
SUM1="$A/sedgate_${TAG}_table1.csv"; echo "model,seed,eventF1,clipF1,seg2sF1" > "$SUM1"
for s in 42 123 777 101 202 303 404 505; do run_sed "$CK" "$TAG" $s "$SUM1"; done
cat "$SUM1" | tee -a "$P"

say "[9/9]b §4.6: Stage1 vs Stage2 を論文と同じ 10 seed で比較"
SUM2="$A/sed_stage1_vs_stage2_10seed.csv"; echo "model,seed,eventF1,clipF1,seg2sF1" > "$SUM2"
for s in 42 123 456 789 2024 1234 5678 9012 3456 7890; do
  run_sed "$S1" "stage1" $s "$SUM2"
  run_sed "$CK" "stage2" $s "$SUM2"
done
cat "$SUM2" | tee -a "$P"
say "=== ALL DONE ==="
