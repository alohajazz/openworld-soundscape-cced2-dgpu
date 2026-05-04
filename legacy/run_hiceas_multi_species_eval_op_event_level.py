#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_hiceas_multi_species_eval.py

HICEAS multi-species evaluation script (Quiet-based CAP-style operating points).

Log/Methods-aligned matching (Methods 4.5.1 style):
- TP is reference-based: each GT (reference) event is a TP if there exists >=1 predicted event within ±tol.
- FP is prediction-based: a predicted event is an FP only if it falls outside ±tol of ALL GT events.
- Duplicate predictions near the same GT are NOT counted as extra TP, and are NOT counted as FP.

DNS (canon-level):
- For each species, we only score canons that contain at least one GT annotation for that species.
- Within scored canons, we do not mask unlabelled time spans.

CAP-style selection:
- Species-wise quantile threshold on window-level s_hat (q)
- Then keep K most extreme events per hour per canon.
  * tail=high: higher s_hat is more "unknown" -> keep top (1-q) tail, then top-K by s_hat desc
  * tail=low : lower  s_hat is more "unknown" -> keep bottom (1-q) tail, then top-K by s_hat asc

FP/h denominator (hours_total):
- Computed per species over the scored canons (based on window-level predictions), using hop-derived windows_per_hour.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# --------- helpers ---------

def to_canon(s: str) -> str:
    """ "1705_YYYYMMDD_HHMMSS_....wav" -> "1705_YYYYMMDD_HHMMSS" """
    if not isinstance(s, str):
        return None
    m = re.match(r'^(\d{4}_\d{8}_\d{6})', s)
    return m.group(1) if m else None


def topk_per_hour(events: pd.DataFrame, K: int, tail: str = "high") -> pd.DataFrame:
    """
    Keep at most K events per hour per canon.
    tail=high: larger s_hat is more extreme -> sort desc
    tail=low : smaller s_hat is more extreme -> sort asc
    """
    x = events.copy()
    x["hour_bucket"] = (x["event_center"] // 3600).astype(int)

    # high tail -> descending; low tail -> ascending
    asc = True if tail == "low" else False
    x = x.sort_values(["canon", "hour_bucket", "s_hat"], ascending=[True, True, asc])
    x = x.groupby(["canon", "hour_bucket"]).head(K).reset_index(drop=True)
    return x


def _counts_anymatch_ref_and_fp_pred(
    events: pd.DataFrame,
    gt_by_canon: dict,
    tol: float,
) -> tuple[int, int, int]:
    """
    Returns (TP, FP, FN) using:
    - TP: per-reference any-match within tol
    - FP: per-prediction outside-all-GT within tol
    - FN: remaining references not matched
    """
    TP_total = 0
    FP_total = 0
    FN_total = 0

    # group predictions per canon for efficiency
    pred_by_canon = (
        events.groupby("canon")["event_center"]
        .apply(lambda s: np.asarray(s.values, dtype=float))
        .to_dict()
    )

    for canon, gt in gt_by_canon.items():
        gt = np.asarray(gt, dtype=float)
        pp = pred_by_canon.get(canon, np.asarray([], dtype=float))

        if gt.size == 0:
            # Should not happen in DNS-scored canons, but keep safe:
            FP_total += int(pp.size)
            continue

        if pp.size == 0:
            # No predictions -> all GT are FN
            FN_total += int(gt.size)
            continue

        # distance matrix (P, G)
        d = np.abs(pp[:, None] - gt[None, :])

        # TP: for each GT (column), is there any pred within tol?
        tp_ref = np.min(d, axis=0) <= tol
        TP = int(tp_ref.sum())

        # FP: for each pred (row), is it outside all GT windows?
        fp_pred = np.min(d, axis=1) > tol
        FP = int(fp_pred.sum())

        FN = int(gt.size) - TP

        TP_total += TP
        FP_total += FP
        FN_total += FN

    return TP_total, FP_total, FN_total


def score_events_for_species(
    man: pd.DataFrame,
    pred: pd.DataFrame,
    species_id: str,
    q: float,
    K: int,
    windows_per_hour: int,
    tol_list=(15.0, 20.0),
    use_m45: bool = True,
    tail: str = "high",
) -> dict:
    """Compute P/R/F1/FP/h for a single species."""
    sp_str = str(species_id)
    man_sp = man[man["species"].astype(str) == sp_str].copy()

    out = {"species": sp_str, "q": q, "K": K, "tail": tail}
    if man_sp.empty:
        return {}

    # DNS (canon-level): restrict to canons where this species appears in GT
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

    # FP/h denominator: scored hours for THIS species (DNS canons)
    hours_total = len(pred_sp) / float(max(1, windows_per_hour))

    # Species-wise quantile threshold on s_hat, with tail direction
    s = pred_sp["s_hat"].astype(float)

    if tail == "high":
        # keep top (1-q) tail
        thr = float(s.quantile(q))
        cand = pred_sp[s >= thr].copy()
    else:
        # keep bottom (1-q) tail
        thr = float(s.quantile(1.0 - q))
        cand = pred_sp[s <= thr].copy()

    if cand.empty:
        return zero_metrics(out)

    # Event center adjustment
    if use_m45:
        cand["event_center"] = cand["center_sec"].astype(float) - 4.5
    else:
        cand["event_center"] = cand["center_sec"].astype(float)

    # Apply per-hour Top-K per canon (direction-aware)
    events = topk_per_hour(cand[["canon", "event_center", "s_hat"]], K, tail=tail)

    # Build GT arrays per canon
    gt_by_canon = (
        man_sp.groupby("canon")["center"]
        .apply(lambda s: np.asarray(s.values, dtype=float))
        .to_dict()
    )

    for tol in tol_list:
        TP, FP, FN = _counts_anymatch_ref_and_fp_pred(events, gt_by_canon, float(tol))

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
    tail: str = "high",
) -> None:
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
        cen_cands = [c for c in pred.columns if "center" in c or "time" in c]
        if not cen_cands:
            raise ValueError("Predictions must contain 'center_sec' or a '*center*'/'*time*' column.")
        pred.rename(columns={cen_cands[0]: "center_sec"}, inplace=True)

    if "s_hat" not in pred.columns:
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

    # Estimate hop & windows_per_hour (used for FP/h denominator; per species we compute hours on DNS canons)
    diffs = pred.groupby("canon")["center_sec"].diff().dropna().round(3)
    hop = float(diffs.mode().iloc[0]) if not diffs.empty else 2.0
    windows_per_hour = max(1, int(round(3600.0 / max(hop, 1e-6))))
    print(f"Estimated hop: {hop}s, windows_per_hour: {windows_per_hour}")
    print(f"Selection tail: {tail} (q={q}, K={K})")

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
            windows_per_hour=windows_per_hour,
            tol_list=(tol15, tol20),
            use_m45=use_m45,
            tail=tail,
        )
        if row:
            rows.append(row)

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
    ap = argparse.ArgumentParser(description="HICEAS multi-species Quiet evaluation (CAP-like manifest).")
    ap.add_argument("--manifest", required=True, help="Path to manifest CSV.")
    ap.add_argument("--predictions", required=True, help="Path to predictions CSV (Quiet or alternative s_hat).")
    ap.add_argument("--out-summary", required=True, help="Output CSV path for per-species summary.")
    ap.add_argument("--out-macro", required=True, help="Output CSV path for macro/micro summary.")
    ap.add_argument("--q", type=float, default=0.99, help="Quantile threshold parameter q (default: 0.99).")
    ap.add_argument("--K", type=int, default=2, help="Top-K events per hour per canon (default: 2).")
    ap.add_argument("--tol15", type=float, default=15.0, help="Tolerance (seconds) for first evaluation.")
    ap.add_argument("--tol20", type=float, default=20.0, help="Tolerance (seconds) for second evaluation.")
    ap.add_argument("--tail", choices=["high", "low"], default="high",
                    help="Which tail is considered 'more unknown/extreme'. "
                         "high: keep top (1-q) tail; low: keep bottom (1-q) tail.")
    ap.add_argument("--no-minus4p5", action="store_true",
                    help="If set, do NOT subtract 4.5 s from center_sec when computing event_center.")
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
        tail=args.tail,
    )

if __name__ == "__main__":
    main()
