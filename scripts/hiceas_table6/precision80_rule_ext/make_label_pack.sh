#!/usr/bin/env bash
set -euo pipefail
BASE="${HICEAS_OP_EMB:?}"
EXT="$BASE/precision80_rule_ext"
stamp="$(date +%Y%m%d_%H%M)"
SRC="$EXT/events_ops_review_todo.csv"
OUTD="$EXT/label_packs/$stamp"
mkdir -p "$OUTD"
if [[ -s "$SRC" ]]; then
  cp "$SRC" "$OUTD/review_todo_${stamp}.csv"
  echo "[OK] label pack -> $OUTD"
else
  echo "[SKIP] no review_todo.csv or empty"
fi
