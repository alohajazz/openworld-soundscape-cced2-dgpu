#!/usr/bin/env python3
"""Plot the frozen Figure 4 aggregate CSV without rerunning any model.

The figure describes exploratory partial-positive-label masking: at a finite
day budget, unselected positive records remain in training with label 0.  It
is not a standard record-subsampling label-efficiency experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "paper_artifacts/minor_revision_2026-09/fig4_label_efficiency_fixed_agg.csv"
DEFAULT_OUT = REPO_ROOT / "paper_artifacts/minor_revision_2026-09/fig4_partial_positive_labels.png"
REQUIRED_COLUMNS = {
    "species", "N_days", "auc_mean", "auc_std", "n_seeds",
    "n_pos_days_total", "n_days_used_median",
}
BUDGETS = ["1", "2", "4", "8", "16", "all"]
SPECIES_ORDER = [
    "Minke whale",
    "Sperm whale",
    "False killer whale",
    "Short-finned pilot whale",
    "Rough-toothed dolphin",
    "Offshore spotted dolphin",
    "Striped dolphin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="frozen aggregate CSV to plot")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output PNG path")
    return parser.parse_args()


def load_aggregate(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"N_days": "string"})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if set(frame["species"]) != set(SPECIES_ORDER):
        raise ValueError("aggregate CSV does not contain exactly the seven expected species")
    expected_rows = len(SPECIES_ORDER) * len(BUDGETS)
    if len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} aggregate rows, found {len(frame)}")
    return frame


def plot(frame: pd.DataFrame, out: Path) -> None:
    # At 300 dpi this is 1,950 px wide, suitable for a six-inch Word placement.
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 5.25), sharey=True)
    axes = axes.ravel()
    positions = np.arange(len(BUDGETS))

    for index, species in enumerate(SPECIES_ORDER):
        axis = axes[index]
        subset = frame.loc[frame["species"] == species].copy()
        subset["N_days"] = pd.Categorical(subset["N_days"], categories=BUDGETS, ordered=True)
        subset = subset.sort_values("N_days")
        if subset["N_days"].isna().any() or len(subset) != len(BUDGETS):
            raise ValueError(f"incomplete or unexpected day budgets for {species}")
        mean = subset["auc_mean"].to_numpy(dtype=float)
        error = subset["auc_std"].fillna(0.0).to_numpy(dtype=float)
        total_days = int(subset["n_pos_days_total"].iloc[0])
        axis.errorbar(
            positions, mean, yerr=error, fmt="o-", lw=1.35, ms=3.6,
            capsize=2.2, capthick=0.9, color="#1676b5", ecolor="#1676b5",
        )
        axis.axhline(0.5, color="#8a8a8a", linestyle="--", lw=0.65, zorder=0)
        axis.set_title(f"{species}\n({total_days} positive days available)", fontsize=6.5, pad=3)
        axis.set_xticks(positions, ["1", "2", "4", "8", "16", "all"], fontsize=7)
        axis.set_ylim(0.35, 1.02)
        axis.set_yticks([0.4, 0.6, 0.8, 1.0])
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(axis="y", alpha=0.25, lw=0.55)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    note_axis = axes[-1]
    note_axis.axis("off")
    note_axis.text(
        0.02, 0.95,
        "Exploratory masking\n\n"
        "Finite budgets:\n"
        "chosen days: label 1\n"
        "other positives kept:\n"
        "training label 0\n\n"
        "All-days: true labels\n"
        "4–5 usable folds",
        va="top", ha="left", fontsize=6.2, linespacing=1.28,
    )

    fig.suptitle("HICEAS Promoter: exploratory partial-positive-label masking",
                 fontsize=9.5, fontweight="bold", y=0.985)
    fig.supxlabel("Positive deployment days retaining positive labels", fontsize=8.3, y=0.062)
    fig.supylabel("Cross-day AUC", fontsize=8.3, x=0.012)
    fig.subplots_adjust(left=0.09, right=0.965, bottom=0.15, top=0.89, wspace=0.34, hspace=0.43)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(load_aggregate(args.csv), args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
