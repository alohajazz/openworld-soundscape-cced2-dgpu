#!/usr/bin/env python3
"""Supplementary Figures S1-S4 of the final revision.
S1 and S3 need window-level score arrays that are not distributed (see the README of the artifact directory);
S2 is drawn from the aggregate summary. S4 (Perch only; counts unchanged) reuses the previous image with the
leading minus signs of the three set labels removed (the selection was always the upper tail of positive distances)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import brentq
from PIL import Image
REPO = Path(__file__).resolve().parents[2]; ART = REPO / "paper_artifacts/final_revision_2026-09-19"; OUT = ART / "figures"

# ---- S1 -------------------------------------------------------------------------------------------------------
s = np.load(ART / "ood/supp_fig_s1_scores_stage2.npz")
summ = json.loads((ART / "ood/supp_fig_s1_summary_stage2.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=100)
for ax, (ind, ood, title, au) in zip(axes, ((s["beats_ind"], s["beats_ood"], "BEATs+DAPT (this work)", summ["beats"]["auroc"]),
                                            (s["perch_ind"], s["perch_ood"], "Perch 2.0 (no underwater DAPT)", summ["perch"]["auroc"]))):
    bins = np.linspace(min(ind.min(), ood.min()), max(ind.max(), ood.max()), 49)
    ax.hist(ind, bins=bins, density=True, alpha=0.55, color="tab:blue", edgecolor="black", linewidth=0.5, label=f"InD (n={len(ind):,})")
    ax.hist(ood, bins=bins, density=True, alpha=0.55, color="tab:red", edgecolor="black", linewidth=0.5, label=f"OOD HICEAS (n={len(ood):,})")
    ax.set_title(f"{title}\nAUROC = {au:.3f}"); ax.set_xlabel("CCED2 score"); ax.set_ylabel("Density"); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "SuppFigS1.png"); plt.close(fig)

# ---- S2 (area-proportional two-set Venn) and S3 (scatter) ---------------------------------------------------------
d = np.load(ART / "hiceas/supp_fig_s2_s3_data_stage2.npz")
sm = json.loads((ART / "hiceas/supp_fig_s2_s3_summary_stage2.json").read_text())
shared, ko, co, n_sel = sm["shared"], sm["knn_only"], sm["cced2_only"], sm["n_select"]
assert shared + ko == n_sel == shared + co and len(d["shared"]) == shared
def lens(dist, r=1.0): return 2 * r * r * np.arccos(dist / (2 * r)) - 0.5 * dist * np.sqrt(4 * r * r - dist * dist)
dist = brentq(lambda x: lens(x) - np.pi * shared / n_sel, 1e-6, 2 - 1e-6)
fig, ax = plt.subplots(figsize=(6.3, 4.4), dpi=150)
ax.add_patch(Circle((-dist / 2, 0), 1, color="#3b82f6", alpha=0.45, lw=0)); ax.add_patch(Circle((dist / 2, 0), 1, color="#ef4444", alpha=0.45, lw=0))
ax.text(-1.0, 1.25, f"kNN_z\n(top 1%, n={n_sel})", ha="center", color="#1e40af", fontsize=12)
ax.text(1.0, 1.25, f"CCED2\n(top 1%, n={n_sel})", ha="center", color="#991b1b", fontsize=12)
ax.text(-1.55, 0, f"kNN_z only\n{ko}", ha="center", va="center", fontsize=13)
ax.text(1.55, 0, f"CCED2 only\n{co}", ha="center", va="center", fontsize=13)
ax.text(0, 0, f"shared\n{shared}", ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(0, -1.45, f"Jaccard = {shared} / {sm['union']} = {sm['jaccard']:.3f}     symmetric difference = {sm['symmetric_difference']} windows", ha="center", fontsize=11)
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.7, 1.9); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(OUT / "SuppFigS2.png", facecolor="white"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 4.5), dpi=200)
ax.scatter(d["dk_z"], d["dm_z"], s=1, color="#cbd5e1", alpha=0.35, linewidths=0, label=f"All windows (n={len(d['dk_z']):,})", rasterized=True)
for idx, col, lab in ((d["shared"], "#7c3aed", f"Shared (n={shared})"), (d["knn_only"], "#3b82f6", f"kNN_z only (n={ko})"), (d["cced2_only"], "#ef4444", f"CCED2 only (n={co})")):
    ax.scatter(d["dk_z"][idx], d["dm_z"][idx], s=14, color=col, alpha=0.9, edgecolor="white", linewidths=0.3, label=lab)
ax.set_xlabel("d_knn_z  (kNN distance, z-scored)"); ax.set_ylabel("d_maha_z  (Mahalanobis distance, z-scored)")
ax.set_title(f"Top-1% selections in feature-space coordinates\n(HICEAS OP, n = {len(d['dk_z']):,} windows; BEATs+DAPT embeddings)", fontsize=9)
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "SuppFigS3.png"); plt.close(fig)

# ---- S4: previous image, leading minus signs of the three set labels removed ----------------------------------------
im = Image.open(ART / "figures/SuppFigS4_previous.png").convert("RGB"); px = np.array(im)
removed = []
for (x0, y0, x1, y1) in ((200, 95, 420, 150), (900, 95, 1260, 150), (560, 885, 800, 940)):   # label bands; the minus is the first glyph
    band = (px[y0:y1, x0:x1].sum(axis=2) < 300); cols = np.where(band.any(axis=0))[0]
    runs = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1); first = runs[0]                 # first connected column run = leading glyph
    rows = np.where(band[:, first].any(axis=1))[0]
    assert len(first) >= 12 and rows.max() - rows.min() <= 5, ("leading glyph is not a minus bar", len(first), rows.min(), rows.max())
    px[y0 + rows.min() - 3:y0 + rows.max() + 4, x0 + first.min() - 3:x0 + first.max() + 4] = (255, 255, 255)   # include anti-aliased fringe
    removed.append((x0 + int(first.min()), y0 + int(rows.min()), x0 + int(first.max()), y0 + int(rows.max())))
Image.fromarray(px).save(OUT / "SuppFigS4.png"); print("minus strokes removed at", removed)
