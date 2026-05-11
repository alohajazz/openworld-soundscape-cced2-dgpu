"""Coverage check after Part2 extraction.

Combine OP + 1706 (existing) + Part2 (new) winaware embeddings,
build canon_embs, then per-species:
 - count canons in canon_embs (recordings)
 - count unique deployment-days with positives
Compare against pre-Part2 baseline."""
import sys, glob
sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base
import numpy as np
import pandas as pd

OP_DIR = "/workspace/embeddings/hiceas_op_fulldata_winaware"
S1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata_winaware"
PART2_DIR = "/workspace/embeddings/hiceas_1706_part2_fulldata_winaware"

def load_dir(emb_dir):
    paths = sorted(glob.glob(f"{emb_dir}/embeddings_*.npy"))
    idxs = sorted(glob.glob(f"{emb_dir}/index_*.csv"))
    if not paths or not idxs:
        return None, None
    embs = np.concatenate([np.load(p) for p in paths]).astype("float32")
    idx = pd.concat([pd.read_csv(p) for p in idxs], ignore_index=True)
    return embs, idx

print("[1/3] Loading combined embeddings (OP + 1706 + Part2)...", flush=True)
op_emb, op_idx = load_dir(OP_DIR)
print(f"  OP: {op_emb.shape}, idx={len(op_idx)}")
s1706_emb, s1706_idx = load_dir(S1706_DIR)
print(f"  1706: {s1706_emb.shape}, idx={len(s1706_idx)}")
p2_emb, p2_idx = load_dir(PART2_DIR)
print(f"  Part2: {p2_emb.shape}, idx={len(p2_idx)}")

emb = np.concatenate([op_emb, s1706_emb, p2_emb])
idx = pd.concat([op_idx, s1706_idx, p2_idx], ignore_index=True)
print(f"  Combined: {emb.shape}, total idx rows={len(idx)}")

print("\n[2/3] Building canon_embs (60-s recording mean embeddings)...", flush=True)
canon_embs = base.build_canon_embeddings(emb, idx)
print(f"  Total canons: {len(canon_embs)}")

print("\n[3/3] Per-species coverage (POST-Part2):", flush=True)
print(f"{'Species':<28} {'pos_csv':>9} {'in_embs':>9} {'days_pos':>10} {'rec/day':>9}")
print("-" * 70)
canon_set = set(canon_embs.keys())
for sp_name, pos_file in base.SPECIES:
    pos = pd.read_csv(f"{base.CANON_DIR}/{pos_file}")
    pos["in_embs"] = pos["canon"].isin(canon_set)
    n_total = len(pos)
    n_in = pos["in_embs"].sum()
    pos_in = pos[pos["in_embs"]]
    pos_in = pos_in.copy()
    pos_in["day"] = pos_in["canon"].apply(base.deployment_day_group)
    n_days = pos_in["day"].nunique() if len(pos_in) else 0
    rec_per_day = n_in / n_days if n_days > 0 else 0
    print(f"{sp_name:<28} {n_total:>9} {n_in:>9} {n_days:>10} {rec_per_day:>9.1f}")
