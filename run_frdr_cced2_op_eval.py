#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_frdr_cced2_op_eval.py

Evaluate the operational behaviour of the CCED2 unknownness score on the FRDR
North Atlantic right whale upcall dataset (dataset_B) in a continuous setting.

Assumptions:
    - A continuous manifest CSV for FRDR is available with at least:
        path, center_sec
      where each row corresponds to a 10 s segment and center_sec is the segment
      centre time in seconds. The hop length between consecutive segments
      (typically 2 s) will be estimated from the data.
    - BEATs+DAPT embeddings have been extracted and CCED2 scores have been
      computed for all segments using `cced2_utils.py score`, yielding
      a 1-D array `score_cced2.npy` aligned with the manifest rows.
    - An annotation CSV for upcall timestamps is available and can be parsed by
      `read_ann_safe` in `fp_recall_helpers.py`.

This script:
    1) Sorts segments by (path, center_sec) and reads the CCED2 scores in
       matching order.
    2) Estimates the hop length and thus the number of windows per hour.
    3) Performs per-file median/MAD normalisation of CCED2 scores, and optional
       3-point moving-average smoothing, as described in Methods Section 5.1.4.
    4) Chooses a global CCED2 threshold so that the expected number of positive
       windows per hour corresponds to a target FP/h value.
    5) Groups consecutive suprathreshold windows into events (allowing gaps
       up to `gap_sec` seconds and requiring at least `k` consecutive windows).
    6) Matches detected events to reference upcalls using a tolerance window
       ±tol_sec and reports recall and FP/h.

Example usage:

    python run_frdr_cced2_op_eval.py \
      --manifest-csv /path/to/frdr_manifest_hop2s.csv \
      --score-npy    /path/to/score_cced2.npy \
      --ann-csv      /path/to/frdr_annotations.csv \
      --target-fp-per-hour 8.0 \
      --k 2 \
      --gap-sec 3.0 \
      --tol-sec 8.0 \
      --smooth
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fp_recall_helpers import (
    read_ann_safe,
    build_events_from_scored,
    eval_events_vs_ann,
)


def median_abs_deviation(x: np.ndarray) -> float:
    """Compute the median absolute deviation (MAD) with a small epsilon."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-csv",
        required=True,
        help="FRDR continuous manifest CSV with at least columns: path, center_sec."
    )
    parser.add_argument(
        "--score-npy",
        required=True,
        help="Path to .npy file containing CCED2 scores aligned with the manifest rows."
    )
    parser.add_argument(
        "--ann-csv",
        required=True,
        help="Annotation CSV for FRDR upcalls; must be readable by read_ann_safe()."
    )
    parser.add_argument(
        "--target-fp-per-hour",
        type=float,
        default=8.0,
        help="Target FP/h used to determine the global CCED2 threshold."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Minimum number of consecutive suprathreshold windows per event."
    )
    parser.add_argument(
        "--gap-sec",
        type=float,
        default=3.0,
        help="Maximum gap (in seconds) between consecutive suprathreshold windows "
             "to be merged into a single event."
    )
    parser.add_argument(
        "--tol-sec",
        type=float,
        default=8.0,
        help="Temporal tolerance (in seconds) for matching detected and reference events."
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="If set, apply 3-point moving-average smoothing to locally normalised CCED2 scores."
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_csv)
    score_path = Path(args.score_npy)
    ann_path = Path(args.ann_csv)

    # 1) Load manifest and CCED2 scores
    manifest = pd.read_csv(manifest_path)
    if "center_sec" not in manifest.columns:
        raise ValueError(f"manifest CSV must contain center_sec: {manifest_path}")
    manifest = manifest.sort_values(["path", "center_sec"]).reset_index(drop=True)

    scores = np.load(score_path)
    if len(scores) != len(manifest):
        raise ValueError(
            f"Length mismatch between scores and manifest rows: {len(scores)} vs {len(manifest)}"
        )

    # 2) Estimate hop length and windows per hour
    diffs = manifest.groupby("path")["center_sec"].diff().dropna()
    if diffs.empty:
        raise ValueError("Unable to estimate hop length from center_sec (insufficient data).")
    hop_sec = float(diffs.round(3).mode().iloc[0])
    windows_per_hour = max(1, int(round(3600.0 / hop_sec)))
    print(f"[FRDR] estimated hop ≈ {hop_sec:.3f} s, windows_per_hour ≈ {windows_per_hour}")

    # 3) Per-file median/MAD normalisation
    manifest["score_raw"] = scores
    S_loc = np.empty_like(scores, dtype=float)

    for path_str, idx in manifest.groupby("path").groups.items():
        v = manifest.loc[idx, "score_raw"].to_numpy()
        med = np.median(v)
        mad = median_abs_deviation(v)
        S_loc[idx] = (v - med) / mad

    manifest["score_loc"] = S_loc

    # 4) Optional 3-point moving-average smoothing
    if args.smooth:
        kernel = np.ones(3, dtype=float) / 3.0
        S_smooth = np.convolve(S_loc, kernel, mode="same")
    else:
        S_smooth = S_loc.copy()

    manifest["score_hat"] = S_smooth

    # 5) Determine global threshold from target FP/h
    #    Approximate: expected positive windows per hour = target_fp_per_hour,
    #    so window-level quantile p = 1 - (FP/h) / (windows/hour).
    p = 1.0 - float(args.target_fp_per_hour) / float(windows_per_hour)
    p = max(0.0, min(1.0, p))
    theta = float(np.quantile(S_smooth, p))
    print(
        f"[FRDR] target_fp_per_hour={args.target_fp_per_hour:.2f} "
        f"→ quantile p={p:.4f}, threshold theta={theta:.3f}"
    )

    # 6) Build per-window DataFrame and construct events
    pw = pd.DataFrame({
        "base": manifest["path"].apply(lambda p: Path(p).name),
        "center_sec": manifest["center_sec"].astype(float),
        "s_hat": manifest["score_hat"].astype(float),
    })

    events_df = build_events_from_scored(
        pw,
        theta=theta,
        gap=args.gap_sec,
        min_len=args.k,
    )

    # 7) Load annotations and evaluate
    ann = read_ann_safe(str(ann_path))
    total_hours = len(manifest) / float(windows_per_hour)

    recall, fp_per_hr = eval_events_vs_ann(
        events_df,
        ann,
        hours=total_hours,
        tol_sec=args.tol_sec,
    )

    print("==== FRDR CCED2 operational evaluation ====")
    print(f"ground-truth events : {len(ann)}")
    print(f"predicted events    : {len(events_df)}")
    print(f"recall              : {recall:.3f}")
    print(f"FP/h                : {fp_per_hr:.2f}")
    print(f"theta               : {theta:.3f}")
    print(f"min_len (k)         : {args.k}")
    print(f"gap_sec             : {args.gap_sec}")
    print(f"tol_sec             : {args.tol_sec}")
    print(f"windows per hour    : {windows_per_hour}")


if __name__ == "__main__":
    main()
