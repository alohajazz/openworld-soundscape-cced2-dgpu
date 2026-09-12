#!/usr/bin/env python3
"""Plot the frozen step-127,641 HICEAS OP component selections.

The NPZ input contains positive higher-is-more-unknown distances and the exact
top-1% membership sets.  This plotting-only script does not refit CCED2 or
change any selected window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def save_both(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz)
    dk_z = data["dk_z"]
    dm_z = data["dm_z"]
    knn_only = set(data["knn_only"].astype(int).tolist())
    cced2_only = set(data["cced2_only"].astype(int).tolist())
    shared = set(data["shared"].astype(int).tolist())
    n_total = len(dk_z)
    n_select = len(knn_only) + len(shared)
    union = knn_only | cced2_only | shared
    jaccard = len(shared) / len(union)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.add_patch(Circle((0.43, 0.53), 0.25, color="#4C86E8", alpha=0.55, ec="#3566B8", lw=1.2))
    ax.add_patch(Circle((0.57, 0.53), 0.25, color="#F05B5B", alpha=0.55, ec="#B63A3A", lw=1.2))
    ax.text(0.28, 0.86, rf"$\mathrm{{kNN}}_z$" + f"\n(top 1%, n={n_select})", ha="center", va="center", fontsize=15, color="#1746B6")
    ax.text(0.72, 0.86, rf"$\mathrm{{CCED2}}$" + f"\n(top 1%, n={n_select})", ha="center", va="center", fontsize=15, color="#A20D0D")
    ax.text(0.31, 0.53, f"$\\mathrm{{kNN}}_z$ only\n{len(knn_only)}", ha="center", va="center", fontsize=16)
    ax.text(0.50, 0.53, f"shared\n{len(shared)}", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.69, 0.53, f"$\\mathrm{{CCED2}}$ only\n{len(cced2_only)}", ha="center", va="center", fontsize=16)
    ax.text(0.50, 0.12, f"Jaccard = {len(shared)} / {len(union)} = {jaccard:.3f}    symmetric difference = {len(knn_only) + len(cced2_only)} windows", ha="center", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    save_both(fig, args.output_dir / "SuppFigS2_venn_top1pct_step127641")

    fig, ax = plt.subplots(figsize=(9.2, 7.9))
    ax.scatter(dk_z, dm_z, s=2, color="#CBD5E1", alpha=0.38, linewidths=0, label=f"All windows (n={n_total:,})")
    categories = [
        (shared, "#7C3AED", f"Shared (n={len(shared)})"),
        (knn_only, "#2F80ED", rf"$\mathrm{{kNN}}_z$ only (n={len(knn_only)})"),
        (cced2_only, "#EF4444", rf"$\mathrm{{CCED2}}$ only (n={len(cced2_only)})"),
    ]
    for indices, colour, label in categories:
        selected = np.asarray(sorted(indices), dtype=int)
        ax.scatter(dk_z[selected], dm_z[selected], s=42, color=colour, alpha=0.88, edgecolor="white", linewidth=0.35, label=label)
    ax.set_title(f"Top-1% selections in score space\n(HICEAS OP, n={n_total:,} windows; final BEATs+DAPT encoder)", fontsize=16)
    ax.set_xlabel(r"$d_{\mathrm{kNN},z}$ (kNN distance, z-scored)", fontsize=14)
    ax.set_ylabel(r"$d_{\mathrm{Mahalanobis},z}$ (Mahalanobis distance, z-scored)", fontsize=14)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    save_both(fig, args.output_dir / "SuppFigS3_scatter_top1pct_step127641")


if __name__ == "__main__":
    main()
