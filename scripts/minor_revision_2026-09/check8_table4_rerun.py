#!/usr/bin/env python3
"""Check 8: recompute the SUBMITTED Table 4 (recording-level Promoter, 5-fold
stratified CV) for the submitted-era backbone and for FIXED/OceanBEATs, using
the identical construction that reproduces the submitted 'N recordings' column
(verified in check7).

Also reports the GroupKFold-by-day (cross-day) baseline for the same setup.
"""
import sys
import os
import glob
import json

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base

CONFIGS = [
    ("submitted-era", [
        "/workspace/embeddings/hiceas_op_fulldata_winaware",
        "/workspace/embeddings/hiceas_1706_fulldata_winaware",
        "/workspace/embeddings/hiceas_1706_part2_fulldata_winaware",
    ]),
    ("FIXED", [
        "/workspace/embeddings/op_fixed_step127641",
        "/workspace/embeddings/1706species_fixed_step127641",
        "/workspace/embeddings/1706part2_fixed_step127641",
    ]),
]

SUBMITTED = {  # AUC (5-fold stratified) and F1 as printed in Main_Noda_etal_R1.docx
    "Short-finned pilot whale": (0.974, 0.79),
    "False killer whale": (0.990, 0.89),
    "Rough-toothed dolphin": (0.993, 0.72),
    "Minke whale": (0.993, 0.88),
    "Offshore spotted dolphin": (0.969, 0.57),
    "Striped dolphin": (0.974, 0.60),
    "Sperm whale": (0.943, 0.75),
}


def load_dir(d):
    p = sorted(glob.glob(os.path.join(d, "embeddings_*.npy")))
    i = sorted(glob.glob(os.path.join(d, "index_*.csv")))
    return (np.concatenate([np.load(x) for x in p]).astype("float32"),
            pd.concat([pd.read_csv(x) for x in i], ignore_index=True))


neg_csv = os.path.join(base.CANON_DIR, "neg_all.csv")
out = {}
for label, dirs in CONFIGS:
    E, I = [], []
    for d in dirs:
        e, i = load_dir(d)
        E.append(e)
        I.append(i)
    canon = base.build_canon_embeddings(np.concatenate(E),
                                        pd.concat(I, ignore_index=True))
    print("=" * 100)
    print("%s   canons=%d" % (label, len(canon)))
    print("   %-28s %6s %7s %-18s %-8s | %-18s %s"
          % ("species", "N", "n_neg", "AUC (5-fold strat)", "F1", "cross-day AUC (group)", "submitted"))
    rows = []
    for sp, f in base.SPECIES:
        pos_csv = os.path.join(base.CANON_DIR, f)
        sk = base.evaluate(canon, pos_csv, neg_csv, "stratified")
        gk = base.evaluate(canon, pos_csv, neg_csv, "group")
        s_auc, s_f1 = SUBMITTED[sp]
        print("   %-28s %6d %7d %-18s %-8.2f | %-18s %.3f / %.2f"
              % (sp, sk["n_pos"], sk["n_neg"],
                 "%.4f ± %.4f" % (sk["auc_mean"], sk["auc_std"]), sk["f1_mean"],
                 "%.4f ± %.4f" % (gk["auc_mean"], gk["auc_std"]) if gk else "NA",
                 s_auc, s_f1))
        rows.append(dict(species=sp, n_pos=sk["n_pos"], n_neg=sk["n_neg"],
                         strat_auc=sk["auc_mean"], strat_auc_std=sk["auc_std"],
                         strat_f1=sk["f1_mean"],
                         group_auc=gk["auc_mean"] if gk else None,
                         group_auc_std=gk["auc_std"] if gk else None,
                         group_folds=gk["n_folds_used"] if gk else None,
                         submitted_auc=s_auc, submitted_f1=s_f1))
    out[label] = rows
    m = np.mean([r["group_auc"] for r in rows if r["group_auc"] is not None])
    print("   -> cross-day 7-species mean AUC = %.4f\n" % m)

json.dump(out, open("/workspace/scripts/audit_2026-07-30/check8_table4.json", "w"), indent=2)
print("WROTE /workspace/scripts/audit_2026-07-30/check8_table4.json")
