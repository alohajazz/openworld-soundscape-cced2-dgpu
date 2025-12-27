#!/usr/bin/env bash
set -euo pipefail
BASE="${HICEAS_OP_EMB:?}"; ANN="${HICEAS_OP_ANN:?}"
EXT="$BASE/precision80_rule_ext"
OPS="$EXT/ops_rule.json"
RUN="$EXT/run_ops.sh"

stamp="$(date +%Y%m%d_%H%M)"

# 0) いまのPROD設定をバックアップ
cp "$OPS" "$EXT/ops_rule.prod.bak"

# 1) REVIEW設定に切替（K=5, q=0.990, minus4p5=True, use_offset=False）
python - <<'PY'
import os, json
from pathlib import Path
BASE=os.environ["HICEAS_OP_EMB"]
p=Path(BASE,"precision80_rule_ext","ops_rule.json")
ops=json.loads(p.read_text())
ops["post_filter"].update({"topK_per_hour":5,"q_s_hat_near":0.990,"minus4p5":True,"use_offset":False})
p.write_text(json.dumps(ops, indent=2))
print("[OK] ops -> REVIEW", ops["post_filter"])
PY

# 2) REVIEW出力を実行
"$RUN"

# 3) REVIEW結果を保存（上書きと世代管理の両方）
mv "$EXT/events_ops_final.csv" "$EXT/events_ops_review.csv"
cp "$EXT/events_ops_review.csv" "$EXT/events_ops_review_${stamp}.csv"

# 4) PROD設定に戻す（バックアップから復元）
mv "$EXT/ops_rule.prod.bak" "$OPS"
echo "[OK] ops -> PROD (restored)"

# 5) PRODを再生成（最新の最終CSV）
"$RUN"

# 6) review_todo を作成（本番に無いレビュー候補＝要確認）
python - <<'PY'
import os, pandas as pd
from pathlib import Path
BASE=os.environ["HICEAS_OP_EMB"]; P=Path(BASE)/"precision80_rule_ext"
prod=pd.read_csv(P/"events_ops_final.csv")
rev =pd.read_csv(P/"events_ops_review.csv")
todo=(rev.merge(prod, on=["base","event_center"], how="left", indicator=True)
         .query('_merge=="left_only"')[["base","event_center"]])
todo.to_csv(P/"events_ops_review_todo.csv", index=False)
print(f"[OK] review_todo -> {P/'events_ops_review_todo.csv'} rows={len(todo)}")
PY

echo "[DONE] review run complete at ${stamp}"
