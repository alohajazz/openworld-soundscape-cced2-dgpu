"""Re-evaluate Table 4 (HICEAS 7-species Promoter) with post-Part2 canon_embs.

Compares:
  (a) StratifiedKFold (paper Table 4 protocol) — recording-level split
  (b) GroupKFold by deployment-day — cross-encounter generalization

Uses identical Promoter (LogReg C=1.0) and 7-species PR set.
Outputs CSV + console table.
"""
import sys, glob, os, json
sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base
import numpy as np
import pandas as pd

OP_DIR = "/workspace/embeddings/hiceas_op_fulldata_winaware"
S1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata_winaware"
PART2_DIR = "/workspace/embeddings/hiceas_1706_part2_fulldata_winaware"
OUT_DIR = "/workspace/scripts/winaware_2026-05-09/table4_post_part2_2026-05-09"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = f"{OUT_DIR}/table4_post_part2_compare.csv"

def load_dir(emb_dir):
    paths = sorted(glob.glob(f"{emb_dir}/embeddings_*.npy"))
    idxs = sorted(glob.glob(f"{emb_dir}/index_*.csv"))
    embs = np.concatenate([np.load(p) for p in paths]).astype("float32")
    idx = pd.concat([pd.read_csv(p) for p in idxs], ignore_index=True)
    return embs, idx

print("Loading combined embeddings (OP + 1706 + Part2)...", flush=True)
op_e, op_i = load_dir(OP_DIR)
s1_e, s1_i = load_dir(S1706_DIR)
p2_e, p2_i = load_dir(PART2_DIR)
emb = np.concatenate([op_e, s1_e, p2_e])
idx = pd.concat([op_i, s1_i, p2_i], ignore_index=True)
print(f"Combined: {emb.shape}, {len(idx)} rows", flush=True)

canon_embs = base.build_canon_embeddings(emb, idx)
print(f"Canons in canon_embs: {len(canon_embs)}", flush=True)

neg_csv = os.path.join(base.CANON_DIR, "neg_all.csv")
rows = []
print(f"\n{'Species':<28} {'Strat AUC':>15} {'Group AUC':>15} {'Delta':>8} {'days':>5} {'folds':>5}", flush=True)
print("-" * 96, flush=True)
for sp_name, pos_file in base.SPECIES:
    pos_csv = os.path.join(base.CANON_DIR, pos_file)
    sk = base.evaluate(canon_embs, pos_csv, neg_csv, "stratified")
    gk = base.evaluate(canon_embs, pos_csv, neg_csv, "group")
    if sk is None or gk is None:
        print(f"  {sp_name:<28} (skip)", flush=True)
        continue
    delta = gk["auc_mean"] - sk["auc_mean"]
    print(
        f"  {sp_name:<28} {sk['auc_mean']:.3f}±{sk['auc_std']:.3f}  "
        f"{gk['auc_mean']:.3f}±{gk['auc_std']:.3f}  "
        f"{delta:+.3f}  {gk['n_unique_groups']:>3}  {gk['n_folds_used']:>3}",
        flush=True,
    )
    rows.append({
        "species": sp_name,
        **{f"strat_{k}": v for k, v in sk.items()},
        **{f"group_{k}": v for k, v in gk.items()},
        "auc_delta_strat_minus_group": -delta,
    })

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}", flush=True)
