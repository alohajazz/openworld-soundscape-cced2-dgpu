#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_hiceas_multi_species_eval.py

HICEAS multi-species evaluation script (Quiet-based CAP-style operating points).

This script reproduces the multi-species Quiet evaluation on HICEAS described
in the paper (e.g., CAP50 / CAPvar2), and can also be applied to a user's own
prediction files.

Inputs
------
- manifest CSV (per-event GT):
    Required columns:
        - base    : basename of the source file (e.g. "1705_YYYYMMDD_HHMMSS_....wav")
        - center  : event center time in seconds
        - species : species identifier (string or int)
    The script derives:
        - canon   : canonical recording ID, e.g. "1705_YYYYMMDD_HHMMSS"

- predictions CSV (per-window Quiet scores):
    Required columns (or equivalents):
        - base       : basename of the source file
                       (if missing, 'path' column is used and basename(path) is taken)
        - center_sec : center time of each window in seconds
                       (if missing, the first column containing "center" or "time"
                        in its name is used and renamed to 'center_sec')
        - s_hat      : detection score (if missing, one of
                       ['score', 'quiet_score', 'prob', 'p_hat'] is used)

Outputs
-------
1) Per-species summary CSV (out-summary):
    - One row per species
    - Columns:
        species, q, K,
        P_15, R_15, F1_15, FP_h_15, TP_15, FP_15, FN_15, N_15,
        P_20, R_20, F1_20, FP_h_20, TP_20, FP_20, FN_20, N_20

2) Macro/micro summary CSV (out-macro):
    - Four rows:
        macro@15, micro@15, macro@20, micro@20
    - Columns:
        split, P, R, F1, FP/h

Usage (example)
---------------
python run_hiceas_multi_species_eval.py \\
  --manifest    /path/to/manifest_from_DOWNLOADED_MATCH_CAP50.csv \\
  --predictions /path/to/predictions_op_quiet_matched.csv \\
  --out-summary ./results/op_multi_species_summary_tol15_20_CAP50.csv \\
  --out-macro   ./results/op_CAP50_macro_micro.csv \\
  --q 0.99 \\
  --K 2

Notes
-----
- FP/h is computed as FP / total_hours, where total_hours is estimated from the
  typical hop size between consecutive 'center_sec' values.
- By default, event centers are shifted by -4.5 s (use '--no-minus4p5' to disable).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------- helpers ---------


def to_canon(s: str) -> str:
    """
    Normalise basename to a canonical recording ID.
    Example: "1705_YYYYMMDD_HHMMSS_....wav" -> "1705_YYYYMMDD_HHMMSS"
    """
    if not isinstance(s, str):
        return None
    m = re.match(r'^(\d{4}_\d{8}_\d{6})', s)
    return m.group(1) if m else None


def topk_per_hour(events: pd.DataFrame, K: int) -> pd.DataFrame:
    """
    Keep at most K events per hour per canon, based on s_hat descending.

    Parameters
    ----------
    events : DataFrame
        Must contain columns ['canon', 'event_center', 's_hat'].
    K : int
        Maximum number of events per hour per canon.

    Returns
    -------
    DataFrame
        Filtered DataFrame with at most K events per hour per canon.
    """
    x = events.copy()
    x["hour_bucket"] = (x["event_center"] // 3600).astype(int)
    # Sort by score descending within each hour bucket
    x = x.sort_values(
        ["canon", "hour_bucket", "s_hat"],
        ascending=[True, True, False],
    )
    # Keep top K
    x = x.groupby(["canon", "hour_bucket"]).head(K).reset_index(drop=True)
    return x


def score_events_for_species(
    man: pd.DataFrame,
    pred: pd.DataFrame,
    species_id: str,
    q: float,
    K: int,
    hours_total: float,
    tol_list=(15.0, 20.0),
    use_m45: bool = True,
) -> dict:
    """
    Compute P/R/F1/FP/h for a single species.

    Parameters
    ----------
    man : DataFrame
        Ground-truth manifest with columns ['base', 'center', 'species', 'canon'].
    pred : DataFrame
        Quiet predictions with columns ['base', 'center_sec', 's_hat', 'canon'].
    species_id : str
        Target species identifier.
    q : float
        Quantile threshold on s_hat (e.g., 0.99).
    K : int
        Top-K events per hour per canon.
    hours_total : float
        Total duration (in hours) across all canons.
    tol_list : tuple of float
        Tolerance (seconds) for event matching.
    use_m45 : bool
        If True, subtract 4.5 s from center_sec to obtain event_center.

    Returns
    -------
    dict
        Dictionary with metrics for each tolerance in tol_list.
    """
    sp_str = str(species_id)
    man_sp = man[man["species"].astype(str) == sp_str].copy()
    
    # Initialize output dictionary
    out = {"species": sp_str, "q": q, "K": K}
    
    if man_sp.empty:
        # No ground truth for this species -> cannot compute Recall properly
        # (Assuming we only evaluate species present in the manifest)
        return {}

    # Restrict predictions to canons where this species appears in GT
    canons = man_sp["canon"].dropna().unique().tolist()
    pred_sp = pred[pred["canon"].isin(canons)].copy()
    
    def zero_metrics(out_dict):
        for tol in tol_list:
            ttag = int(tol)
            out_dict.update({
                f"P_{ttag}": 0.0, f"R_{ttag}": 0.0, f"F1_{ttag}": 0.0, f"FP_h_{ttag}": 0.0,
                f"TP_{ttag}": 0, f"FP_{ttag}": 0, f"FN_{ttag}": int(len(man_sp)), f"N_{ttag}": int(len(man_sp)),
            })
        return out_dict

    if pred_sp.empty:
        return zero_metrics(out)

    # Species-wise quantile threshold on s_hat
    thr = float(pred_sp["s_hat"].quantile(q))
    cand = pred_sp[pred_sp["s_hat"] >= thr].copy()
    
    if cand.empty:
        return zero_metrics(out)

    # Event center adjustment (optionally -4.5 s correction)
    if use_m45:
        cand["event_center"] = cand["center_sec"].astype(float) - 4.5
    else:
        cand["event_center"] = cand["center_sec"].astype(float)

    # Apply per-hour Top-K per canon
    events = topk_per_hour(cand[["canon", "event_center", "s_hat"]], K)

    # Build GT arrays per canon for fast matching
    gt_by_canon = (
        man_sp.groupby("canon")["center"]
        .apply(lambda s: np.asarray(s.values, dtype=float))
        .to_dict()
    )

    for tol in tol_list:
        TP = FP = 0
        FN = 0

        # One-to-one matching flags
        used = {c: np.zeros(len(gt), dtype=bool) for c, gt in gt_by_canon.items()}

        # Greedy nearest-neighbour matching
        for _, r in events.iterrows():
            c = r["canon"]
            t_pred = float(r["event_center"])
            gt = gt_by_canon.get(c)
            
            if gt is None or gt.size == 0:
                FP += 1
                continue
                
            d = np.abs(gt - t_pred)
            j = int(d.argmin())
            
            if d[j] <= tol and not used[c][j]:
                TP += 1
                used[c][j] = True
            else:
                FP += 1

        # Unmatched GT events count as FN
        for c, gt in gt_by_canon.items():
            FN += int((~used[c]).sum())

        N_gt = int(len(man_sp))
        P = TP / (TP + FP + 1e-9)
        R = TP / (TP + FN + 1e-9)
        F1 = 2 * P * R / (P + R + 1e-9)
        FP_h = FP / max(hours_total, 1e-9)

        ttag = int(tol)
        out.update({
            f"P_{ttag}": float(P),
            f"R_{ttag}": float(R),
            f"F1_{ttag}": float(F1),
            f"FP_h_{ttag}": float(FP_h),
            f"TP_{ttag}": int(TP),
            f"FP_{ttag}": int(FP),
            f"FN_{ttag}": int(FN),
            f"N_{ttag}": N_gt,
        })

    return out


def compute_summary(
    man_csv: str,
    pred_csv: str,
    out_summary: str,
    out_macro: str,
    q: float = 0.99,
    K: int = 2,
    tol15: float = 15.0,
    tol20: float = 20.0,
    use_m45: bool = True,
) -> None:
    """Compute per-species and macro/micro metrics from manifest and predictions."""
    man_path = Path(man_csv)
    pred_path = Path(pred_csv)

    print(f"Loading manifest: {man_path}")
    man = pd.read_csv(man_path)
    man.columns = [c.strip().lower() for c in man.columns]
    
    print(f"Loading predictions: {pred_path}")
    pred = pd.read_csv(pred_path)
    pred.columns = [c.strip().lower() for c in pred.columns]

    # Validate Manifest columns
    if "base" not in man.columns:
        raise ValueError("Manifest must contain 'base' column.")
    if "center" not in man.columns:
        raise ValueError("Manifest must contain 'center' column (GT event center in seconds).")
    if "species" not in man.columns:
        raise ValueError("Manifest must contain 'species' column.")

    man["canon"] = man["base"].map(to_canon)

    # Validate/Fix Predictions columns
    if "base" not in pred.columns and "path" in pred.columns:
        pred["base"] = pred["path"].astype(str).apply(lambda p: Path(p).name)

    if "center_sec" not in pred.columns:
        # Fallback: look for generic time columns
        cen_cands = [c for c in pred.columns if "center" in c or "time" in c]
        if not cen_cands:
            raise ValueError("Predictions must contain 'center_sec' or a '*center*'/'*time*' column.")
        pred.rename(columns={cen_cands[0]: "center_sec"}, inplace=True)

    if "s_hat" not in pred.columns:
        # Fallback: look for common score column names
        for c in ["score", "quiet_score", "prob", "p_hat"]:
            if c in pred.columns:
                pred["s_hat"] = pred[c]
                break
        if "s_hat" not in pred.columns:
            raise ValueError("Predictions must contain 's_hat' or one of ['score', 'quiet_score', 'prob', 'p_hat'].")

    pred["canon"] = pred["base"].map(to_canon)

    # Restrict to intersection of canons (fair comparison)
    man_canons = set(man["canon"].dropna())
    pred_canons = set(pred["canon"].dropna())
    inter = man_canons & pred_canons
    
    print(f"Canons in Manifest: {len(man_canons)}, Predictions: {len(pred_canons)}")
    print(f"Intersection: {len(inter)}")
    
    man = man[man["canon"].isin(inter)].reset_index(drop=True)
    pred = pred[pred["canon"].isin(inter)].reset_index(drop=True)

    # Estimate hop & hours_total (common across species)
    diffs = pred.groupby("canon")["center_sec"].diff().dropna().round(3)
    if diffs.empty:
        hop = 2.0
    else:
        hop = float(diffs.mode().iloc[0])

    windows_per_hour = max(1, int(round(3600.0 / max(hop, 1e-6))))
    hours_total = len(pred) / float(windows_per_hour)
    print(f"Estimated hop: {hop}s, Total hours: {hours_total:.2f}h")

    # Score each species
    rows = []
    unique_species = sorted(man["species"].astype(str).unique())
    print(f"Evaluating {len(unique_species)} species...")
    
    for sp in unique_species:
        row = score_events_for_species(
            man=man,
            pred=pred,
            species_id=sp,
            q=q,
            K=K,
            hours_total=hours_total,
            tol_list=(tol15, tol20),
            use_m45=use_m45,
        )
        if row:
            rows.append(row)

    # Save per-species summary
    summary = pd.DataFrame(rows).sort_values("species").reset_index(drop=True)
    Path(out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    print(f"Per-species summary saved to: {out_summary}")

    # ---- Macro / Micro Aggregation ----
    df = summary

    def agg(df_sub, p_col, r_col, f1_col, fph_col, tp_col, fp_col, fn_col):
        macro = {
            "P": df_sub[p_col].mean(),
            "R": df_sub[r_col].mean(),
            "F1": df_sub[f1_col].mean(),
            "FP/h": df_sub[fph_col].mean(),
        }
        TP = df_sub[tp_col].sum()
        FP_ = df_sub[fp_col].sum()
        FN = df_sub[fn_col].sum()
        microP = TP / (TP + FP_ + 1e-9)
        microR = TP / (TP + FN + 1e-9)
        microF1 = 2 * microP * microR / (microP + microR + 1e-9)
        micro = {
            "P": microP,
            "R": microR,
            "F1": microF1,
            "FP/h": df_sub[fph_col].mean(),
        }
        return macro, micro

    macro15, micro15 = agg(df, "P_15", "R_15", "F1_15", "FP_h_15", "TP_15", "FP_15", "FN_15")
    macro20, micro20 = agg(df, "P_20", "R_20", "F1_20", "FP_h_20", "TP_20", "FP_20", "FN_20")

    tidy = pd.DataFrame([
        {"split": "macro@15", **macro15},
        {"split": "micro@15", **micro15},
        {"split": "macro@20", **macro20},
        {"split": "micro@20", **micro20},
    ]).round(4)
    
    Path(out_macro).parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_macro, index=False)
    print(f"Macro/Micro summary saved to: {out_macro}")


def main():
    ap = argparse.ArgumentParser(
        description="HICEAS multi-species Quiet evaluation (CAP-like manifest)."
    )
    ap.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest CSV (e.g., manifest_from_DOWNLOADED_MATCH_CAP50.csv).",
    )
    ap.add_argument(
        "--predictions",
        required=True,
        help="Path to Quiet predictions CSV (e.g., predictions_op_quiet_matched.csv).",
    )
    ap.add_argument(
        "--out-summary",
        required=True,
        help="Output CSV path for per-species summary.",
    )
    ap.add_argument(
        "--out-macro",
        required=True,
        help="Output CSV path for macro/micro summary.",
    )
    ap.add_argument(
        "--q",
        type=float,
        default=0.99,
        help="Quantile threshold on s_hat (default: 0.99).",
    )
    ap.add_argument(
        "--K",
        type=int,
        default=2,
        help="Top-K events per hour per canon (default: 2).",
    )
    ap.add_argument(
        "--tol15",
        type=float,
        default=15.0,
        help="Tolerance (seconds) for first evaluation (default: 15).",
    )
    ap.add_argument(
        "--tol20",
        type=float,
        default=20.0,
        help="Tolerance (seconds) for second evaluation (default: 20).",
    )
    ap.add_argument(
        "--no-minus4p5",
        action="store_true",
        help="If set, do NOT subtract 4.5 s from center_sec when computing event_center.",
    )
    args = ap.parse_args()

    use_m45 = not args.no_minus4p5
    
    compute_summary(
        man_csv=args.manifest,
        pred_csv=args.predictions,
        out_summary=args.out_summary,
        out_macro=args.out_macro,
        q=args.q,
        K=args.K,
        tol15=args.tol15,
        tol20=args.tol20,
        use_m45=use_m45,
    )


if __name__ == "__main__":
    main()