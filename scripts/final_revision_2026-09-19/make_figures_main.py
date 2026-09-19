#!/usr/bin/env python3
"""Figures 3 and 4 of the final revision, drawn from the Stage 2 aggregate results in this directory tree.
The layout follows the figures of the previous revision."""
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
REPO = Path(__file__).resolve().parents[2]; ART = REPO / "paper_artifacts/final_revision_2026-09-19"; OUT = ART / "figures"

# ---- Fig. 4 (layout of the previous revision, Stage 2 aggregate) --------------------------------------------------------------------
agg = pd.read_csv(ART / "hiceas/fig4_label_efficiency_agg.csv", dtype={"N_days": str})
order_idx = ["1", "2", "4", "8", "16", "all"]
species_list = list(dict.fromkeys(agg["species"])); assert len(species_list) == 7
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
axes = axes.flatten()
for i, sp_name in enumerate(species_list):
    ax = axes[i]
    sub = (agg[agg["species"] == sp_name].set_index("N_days").reindex(order_idx).reset_index())
    xs = list(range(len(sub)))
    ax.errorbar(xs, sub["auc_mean"], yerr=sub["auc_std"].fillna(0), fmt="o-", capsize=3, color="tab:green")
    ax.set_xticks(xs); ax.set_xticklabels(sub["N_days"], rotation=45)
    n_days = sub["n_pos_days_total"].dropna().iloc[0]
    ax.set_title(f"{sp_name}\n(N_days_total = {int(n_days)})", fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3); ax.axhline(0.9, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)
    if i % 4 == 0: ax.set_ylabel("AUC (5-fold GroupKFold by day)")
axes[-1].set_visible(False)
fig.suptitle("HICEAS Promoter — deployment-day label efficiency\n(all positive recordings within selected days; cross-day GroupKFold)", y=1.02, fontsize=11)
fig.supxlabel("N_days = number of independent deployment-days labelled")
fig.tight_layout()
fig.savefig(OUT / "Fig4.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- Fig. 3 (layout of the previous revision, Stage 2 operating points) ---------------------------------------------------------
t2 = {r["mode"].split(" ")[0]: r for r in csv.DictReader(open(ART / "frdr/table2/frdr_quiet_promoter_union_supervised.csv"))}
fu = next(r for r in csv.DictReader(open(ART / "frdr/fusion/fusion.csv")) if float(r["alpha"]) == 0.0)
pts = {"Quiet": (float(t2["Quiet"]["FP_h"]), float(t2["Quiet"]["Recall"]), "o", "tab:blue"),
       "Promoter": (float(t2["Promoter"]["FP_h"]), float(t2["Promoter"]["Recall"]), "s", "tab:green"),
       "Union": (float(t2["Union"]["FP_h"]), float(t2["Union"]["Recall"]), "^", "tab:red"),
       "Fusion": (float(fu["FP_h"]), float(fu["Recall"]), "D", "tab:purple")}
fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=200)
ax.axvline(10, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
ax.text(10.02, 0.55, "FP/h = 10", color="gray", fontsize=8)
for name, (x, y, m, col) in pts.items():
    ax.scatter([x], [y], marker=m, s=170, color=col, edgecolor="black", linewidth=1.0, label=name, zorder=3)
box = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1.0)
lab = lambda n: f"{n}: recall={pts[n][1]:.3f}, FP/h={pts[n][0]:.2f}"
ax.annotate(lab("Promoter"), xy=pts["Promoter"][:2], xytext=(9.27, 0.46), fontsize=8, bbox=box)
ax.annotate(lab("Union"), xy=pts["Union"][:2], xytext=(10.28, 0.30), fontsize=8, bbox=box)
ax.annotate(lab("Quiet"), xy=pts["Quiet"][:2], xytext=(10.45, 0.16), fontsize=8, bbox=box, arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
ax.annotate(lab("Fusion"), xy=pts["Fusion"][:2], xytext=(9.27, 0.16), fontsize=8, bbox=box, arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
ax.set_xlim(9.2, 11.2); ax.set_ylim(-0.02, 0.58)
ax.set_xlabel("False positives per hour (FP/h)"); ax.set_ylabel("Recall")
ax.set_title("FP/h–recall operating points on FRDR right-whale upcalls", fontsize=10)
ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "Fig3.png"); plt.close(fig)
print({k: (round(v[0], 2), round(v[1], 3)) for k, v in pts.items()})
