import sys
sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base
import pandas as pd

print("=== Per-species deployment-day counts ===")
print(f"{'Species':<28} {'N_rec':>7} {'N_days':>7} {'rec/day':>9}")
print("-" * 55)
for sp_name, pos_file in base.SPECIES:
    pos = pd.read_csv(f"{base.CANON_DIR}/{pos_file}")
    pos["group"] = pos["canon"].apply(base.deployment_day_group)
    n_groups = pos["group"].nunique()
    n_pos = len(pos)
    print(f"{sp_name:<28} {n_pos:>7} {n_groups:>7} {n_pos/n_groups:>9.1f}")
