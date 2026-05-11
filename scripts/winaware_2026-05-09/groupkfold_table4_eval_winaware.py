#!/usr/bin/env python3
"""Re-run Table 4 GroupKFold/StratifiedKFold Promoter eval using winaware (bug-fixed) embeddings."""
import os, sys, glob, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as orig

DAPT_OP_DIR = "/workspace/embeddings/hiceas_op_fulldata_winaware"
DAPT_1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata_winaware"
CANON_DIR = "/workspace/revision1_species_canons"
OUT_CSV = "/workspace/revision1_outputs_groupkfold/table4_groupkfold_compare_winaware_2026-05-09.csv"

print("=== Loading WINAWARE fulldata DAPT embeddings ===", flush=True)
print(f"  OP dir: {DAPT_OP_DIR}")
print(f"  1706 dir: {DAPT_1706_DIR}")
op_emb, op_idx = orig.load_sharded(DAPT_OP_DIR)
sp_emb, sp_idx = orig.load_sharded(DAPT_1706_DIR)
print(f"  OP shape: {op_emb.shape}  1706 shape: {sp_emb.shape}", flush=True)
emb = np.concatenate([op_emb, sp_emb])
idx = pd.concat([op_idx, sp_idx], ignore_index=True)
canon_embs = orig.build_canon_embeddings(emb, idx)
print(f"  Loaded {len(canon_embs)} canons (= 60-s recordings)", flush=True)

neg_csv = os.path.join(CANON_DIR, "neg_all.csv")
rows = []
print(f"\n{'Species':<28} {'Strat AUC':>15} {'Group AUC':>15} {'Delta':>8} {'Groups':>7} {'Folds':>5}", flush=True)
print("-" * 96, flush=True)
for sp_name, pos_file in orig.SPECIES:
    pos_csv = os.path.join(CANON_DIR, pos_file)
    sk = orig.evaluate(canon_embs, pos_csv, neg_csv, "stratified")
    gk = orig.evaluate(canon_embs, pos_csv, neg_csv, "group")
    if sk is None or gk is None:
        print(f"  {sp_name:<28} (skip)", flush=True)
        continue
    delta = gk["auc_mean"] - sk["auc_mean"]
    print(f"  {sp_name:<28} {sk['auc_mean']:.3f}+/-{sk['auc_std']:.3f}  {gk['auc_mean']:.3f}+/-{gk['auc_std']:.3f}  {delta:+.3f}  {gk['n_unique_groups']:>5}  {gk['n_folds_used']:>3}", flush=True)
    rows.append({"species": sp_name, **{f"strat_{k}": v for k, v in sk.items()}, **{f"group_{k}": v for k, v in gk.items()}})

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}", flush=True)
