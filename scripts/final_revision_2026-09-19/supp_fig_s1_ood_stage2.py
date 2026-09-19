#!/usr/bin/env python3
"""Regenerate Supplementary Figure S1 with the final BEATs+DAPT encoder."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def load_shards(directory: str) -> np.ndarray:
    paths = sorted(glob.glob(str(Path(directory) / "embeddings_*.npy")))
    if not paths:
        raise FileNotFoundError(directory)
    return np.concatenate([np.load(path).astype(np.float32) for path in paths], axis=0)


def score(embeddings: np.ndarray, model_dir: str, prefix: str) -> np.ndarray:
    root = Path(model_dir)
    knn = joblib.load(root / f"knn_{prefix}.pkl")
    maha = joblib.load(root / f"maha_{prefix}.pkl")
    with (root / ("cced2_norm.json" if prefix == "cced2" else "cced2_norm_perch.json")).open() as handle:
        norm = json.load(handle)
    distances, _ = knn["knn"].kneighbors(embeddings, n_neighbors=knn.get("k", norm.get("k", 50)))
    d_knn = distances.mean(axis=1)
    mu = np.asarray(maha["mu"], dtype=embeddings.dtype)
    precision = np.asarray(maha["precision"], dtype=embeddings.dtype)
    diff = embeddings - mu
    d_maha = np.sqrt(np.einsum("nd,nd->n", diff, diff @ precision) + 1e-12)
    return (d_knn - norm["mk"]) / norm["sk"] + (d_maha - norm["mm"]) / norm["sm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beats-ind", required=True)
    parser.add_argument("--beats-ood", required=True)
    parser.add_argument("--beats-models", required=True)
    parser.add_argument("--perch-ind-a", required=True)
    parser.add_argument("--perch-ind-b", required=True)
    parser.add_argument("--perch-ood", required=True)
    parser.add_argument("--perch-models", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    beats_ind = load_shards(args.beats_ind)
    beats_ood = load_shards(args.beats_ood)
    perch_ind = np.concatenate([load_shards(args.perch_ind_a), load_shards(args.perch_ind_b)], axis=0)
    perch_ood = load_shards(args.perch_ood)

    beats_ind_score = score(beats_ind, args.beats_models, "cced2")
    beats_ood_score = score(beats_ood, args.beats_models, "cced2")
    perch_ind_score = score(perch_ind, args.perch_models, "perch")
    perch_ood_score = score(perch_ood, args.perch_models, "perch")

    def auc(ind: np.ndarray, ood: np.ndarray) -> float:
        labels = np.concatenate([np.zeros(len(ind)), np.ones(len(ood))])
        values = np.concatenate([ind, ood])
        return float(roc_auc_score(labels, values))

    beats_auc = auc(beats_ind_score, beats_ood_score)
    perch_auc = auc(perch_ind_score, perch_ood_score)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [
        (axes[0], beats_ind_score, beats_ood_score, "BEATs+DAPT (Stage 2, step 6,385)", beats_auc),
        (axes[1], perch_ind_score, perch_ood_score, "Perch 2.0 (no underwater DAPT)", perch_auc),
    ]
    for ax, ind, ood, title, auroc in panels:
        lo = float(min(ind.min(), ood.min()))
        hi = float(max(ind.max(), ood.max()))
        bins = np.linspace(lo, hi, 36)
        ax.hist(ind, bins=bins, density=True, alpha=0.55, color="#3B82B8", edgecolor="#34515E", linewidth=0.45, label=f"InD (n={len(ind):,})")
        ax.hist(ood, bins=bins, density=True, alpha=0.55, color="#E45756", edgecolor="#7F3A39", linewidth=0.45, label=f"HICEAS dataset (n={len(ood):,})")
        ax.set_title(f"{title}\nDataset-membership AUROC = {auroc:.3f}", fontsize=13)
        ax.set_xlabel("CCED2 score")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.22)
        ax.legend(framealpha=0.95)
    fig.suptitle("Bandwidth-matched cross-dataset discrimination (0–8 kHz)", fontsize=14, y=1.02)
    fig.tight_layout()
    png = out / "SuppFigS1_dataset_ood_stage2.png"
    pdf = out / "SuppFigS1_dataset_ood_stage2.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    np.savez_compressed(
        out / "supp_fig_s1_scores_stage2.npz",
        beats_ind=beats_ind_score,
        beats_ood=beats_ood_score,
        perch_ind=perch_ind_score,
        perch_ood=perch_ood_score,
    )
    summary = {
        "beats": {"n_ind": len(beats_ind_score), "n_ood": len(beats_ood_score), "auroc": beats_auc},
        "perch": {"n_ind": len(perch_ind_score), "n_ood": len(perch_ood_score), "auroc": perch_auc},
    }
    with (out / "supp_fig_s1_summary_stage2.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
