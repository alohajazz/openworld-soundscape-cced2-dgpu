"""Diagnose Pilot whale 1057 -> 234 embedding gap. Identify which canons in pos_*.csv are missing from winaware embeddings."""
import sys, os, glob
sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base
import numpy as np
import pandas as pd

DAPT_OP_DIR = "/workspace/embeddings/hiceas_op_fulldata_winaware"
DAPT_1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata_winaware"

# Load winaware embeddings
op_emb, op_idx = base.load_sharded(DAPT_OP_DIR)
sp_emb, sp_idx = base.load_sharded(DAPT_1706_DIR)
emb = np.concatenate([op_emb, sp_emb])
idx = pd.concat([op_idx, sp_idx], ignore_index=True)
canon_embs = base.build_canon_embeddings(emb, idx)
canon_set = set(canon_embs.keys())
print(f"Total canons in winaware embedding: {len(canon_set)}")

# Identify cruise prefixes available
cruise_prefixes = set()
for canon in canon_set:
    parts = canon.split("_")
    if parts:
        cruise_prefixes.add(parts[0])
print(f"Cruise prefixes in winaware: {sorted(cruise_prefixes)}")

# Per-species coverage analysis
print("\n=== Per-species coverage ===")
print(f"{'Species':<28} {'pos_csv':>9} {'in_embs':>9} {'missing':>9} {'missing %':>10}")
print("-" * 75)
for sp_name, pos_file in base.SPECIES:
    pos = pd.read_csv(f"{base.CANON_DIR}/{pos_file}")
    pos["in_embs"] = pos["canon"].isin(canon_set)
    n_total = len(pos)
    n_in = pos["in_embs"].sum()
    n_missing = n_total - n_in
    pct = 100.0 * n_missing / n_total if n_total else 0
    print(f"{sp_name:<28} {n_total:>9} {n_in:>9} {n_missing:>9} {pct:>9.1f}%")

# Detailed per-cruise breakdown for Pilot whale (most affected)
print("\n=== Pilot whale: per-cruise breakdown ===")
pilot = pd.read_csv(f"{base.CANON_DIR}/pos_Short-finned_pilot_whale.csv")
pilot["cruise"] = pilot["canon"].apply(lambda c: c.split("_")[0] if "_" in c else "?")
pilot["day"] = pilot["canon"].apply(base.deployment_day_group)
pilot["in_embs"] = pilot["canon"].isin(canon_set)
print(pilot.groupby("cruise").agg(n_total=("canon", "count"), n_in_embs=("in_embs", "sum"), n_unique_days=("day", "nunique")).reset_index())

# Day-level: which Pilot days are in canon_embs?
print("\n=== Pilot whale per-day coverage ===")
pilot_day_summary = pilot.groupby("day").agg(n_total=("canon", "count"), n_in_embs=("in_embs", "sum")).reset_index()
print(pilot_day_summary.head(20))
