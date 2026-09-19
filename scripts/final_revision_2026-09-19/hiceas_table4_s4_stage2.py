#!/usr/bin/env python3
"""Regenerate Supplementary Figures S2 and S3 from the final encoder.

The calculation follows ``check17_s4_beats_overlap.py``: all HICEAS OP
windows are scored against the fixed in-distribution kNN and Mahalanobis
models, and the top ceil(1%) sets for kNN_z and CCED2 are compared. Larger
scores indicate greater distance from the in-distribution reference.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def save_both(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    emb_files = sorted(glob.glob(str(Path(args.emb_dir) / "embeddings_*.npy")))
    if not emb_files:
        raise FileNotFoundError(f"No embedding shards found under {args.emb_dir}")
    embeddings = np.concatenate([np.load(path).astype(np.float32) for path in emb_files], axis=0)

    ref = Path(args.ref_dir)
    knn = joblib.load(ref / "knn_cced2.pkl")
    maha = joblib.load(ref / "maha_cced2.pkl")
    with (ref / "cced2_norm.json").open() as handle:
        norm = json.load(handle)

    distances, _ = knn["knn"].kneighbors(embeddings, n_neighbors=knn["k"])
    d_knn = distances.mean(axis=1)
    mu = maha["mu"].astype(embeddings.dtype)
    precision = maha["precision"].astype(embeddings.dtype)
    diff = embeddings - mu
    d_maha = np.sqrt(np.einsum("nd,nd->n", diff, diff @ precision) + 1e-12)
    dk_z = (d_knn - norm["mk"]) / norm["sk"]
    dm_z = (d_maha - norm["mm"]) / norm["sm"]
    cced2 = dk_z + dm_z

    n_total = len(embeddings)
    n_select = int(np.ceil(n_total * 0.01))
    top_knn = set(np.argsort(-dk_z)[:n_select].tolist())
    top_cced2 = set(np.argsort(-cced2)[:n_select].tolist())
    knn_only = top_knn - top_cced2
    cced2_only = top_cced2 - top_knn
    shared = top_knn & top_cced2
    union = top_knn | top_cced2
    jaccard = len(shared) / len(union)

    # S2: two-set overlap diagram. Circles are schematic, while all labels are exact.
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
    save_both(fig, out / "SuppFigS2_venn_top1pct_stage2")

    # S3: score-space view of the same exact top-1% sets.
    fig, ax = plt.subplots(figsize=(9.2, 7.9))
    ax.scatter(dk_z, dm_z, s=2, color="#CBD5E1", alpha=0.38, linewidths=0, label=f"All windows (n={n_total:,})")
    categories = [
        (shared, "#7C3AED", f"Shared (n={len(shared)})"),
        (knn_only, "#2F80ED", rf"$\mathrm{{kNN}}_z$ only (n={len(knn_only)})"),
        (cced2_only, "#EF4444", rf"$\mathrm{{CCED2}}$ only (n={len(cced2_only)})"),
    ]
    for indices, color, label in categories:
        idx = np.asarray(sorted(indices), dtype=int)
        ax.scatter(dk_z[idx], dm_z[idx], s=42, color=color, alpha=0.88, edgecolor="white", linewidth=0.35, label=label)
    ax.set_title(f"Top-1% selections in score space\n(HICEAS OP, n={n_total:,} windows; final BEATs+DAPT encoder)", fontsize=16)
    ax.set_xlabel(r"$d_{\mathrm{kNN},z}$ (kNN distance, z-scored)", fontsize=14)
    ax.set_ylabel(r"$d_{\mathrm{Mahalanobis},z}$ (Mahalanobis distance, z-scored)", fontsize=14)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    save_both(fig, out / "SuppFigS3_scatter_top1pct_stage2")

    np.savez_compressed(
        out / "supp_fig_s2_s3_data_stage2.npz",
        dk_z=dk_z,
        dm_z=dm_z,
        cced2=cced2,
        knn_only=np.asarray(sorted(knn_only), dtype=np.int64),
        cced2_only=np.asarray(sorted(cced2_only), dtype=np.int64),
        shared=np.asarray(sorted(shared), dtype=np.int64),
    )
    summary = {
        "n_total": n_total,
        "n_select": n_select,
        "shared": len(shared),
        "knn_only": len(knn_only),
        "cced2_only": len(cced2_only),
        "union": len(union),
        "jaccard": jaccard,
        "symmetric_difference": len(knn_only) + len(cced2_only),
        "mean_scores": {
            name: {"d_knn_z": float(dk_z[np.asarray(sorted(indices), dtype=int)].mean()), "d_maha_z": float(dm_z[np.asarray(sorted(indices), dtype=int)].mean())}
            for name, indices in [("knn_only", knn_only), ("cced2_only", cced2_only), ("shared", shared)]
        },
    }
    with (out / "supp_fig_s2_s3_summary_stage2.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
