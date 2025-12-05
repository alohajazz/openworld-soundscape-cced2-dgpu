# -*- coding: utf-8 -*-
"""
fp_recall_helpers.py

Helper functions for parsing annotations and calculating False Positives per Hour (FP/h)
and Recall for continuous operational evaluation.

These utilities are primarily used by `run_frdr_cced2_op_eval.py` to:
1. Load ground-truth annotations safely.
2. Group continuous window-level scores into discrete events.
3. Match predicted events with ground-truth events to compute performance metrics.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

def read_ann_safe(ann_path: str) -> pd.DataFrame:
    """
    Reads an annotation CSV file robustly, handling different delimiters (',' or ';')
    and common column name variations.

    Args:
        ann_path (str): Path to the annotation CSV.

    Returns:
        pd.DataFrame: DataFrame with columns ['base', 'center'], where 'base' is the
                      filename and 'center' is the event timestamp in seconds.
    """
    if not os.path.exists(ann_path):
        raise FileNotFoundError(ann_path)
    
    # Try reading with semicolon delimiter first, then comma (default)
    try:
        ann = pd.read_csv(ann_path, sep=";", engine="python")
    except Exception:
        ann = pd.read_csv(ann_path, engine="python")
    
    # Handle single-column CSVs that might be malformed
    if ann.shape[1] == 1:
        ann = ann.iloc[:, 0].str.split(";", expand=True)
    
    # Normalize column names
    ann.columns = [str(c).strip().lower() for c in ann.columns]
    
    # Identify file and time columns
    fcol = next((c for c in ann.columns if "file" in c or "name" in c), None)
    tcol = next((c for c in ann.columns if "time" in c or "stamp" in c), None)
    
    # Fallback for simple CSVs without headers or standard names
    if fcol is None or tcol is None:
        if len(ann.columns) >= 2:
            fcol, tcol = ann.columns[0], ann.columns[1]
        else:
            raise KeyError(f"Annotation columns (filename/time) not found: {ann.columns.tolist()}")
            
    out = pd.DataFrame({
        "base":   ann[fcol].astype(str).apply(lambda p: Path(p).name),
        "center": pd.to_numeric(ann[tcol], errors="coerce"),
    }).dropna(subset=["base", "center"]).reset_index(drop=True)
    
    return out

def _unify_two(df: pd.DataFrame, a: str, b: str, out: str) -> pd.DataFrame:
    """Helper to merge two columns (a, b) into one (out)."""
    if a in df.columns and b in df.columns:
        df[out] = df[a].where(df[a].notna(), df[b])
        df.drop(columns=[a, b], inplace=True)
    elif a in df.columns:
        df.rename(columns={a: out}, inplace=True)
    elif b in df.columns:
        df.rename(columns={b: out}, inplace=True)
    elif out not in df.columns:
        df[out] = np.nan
    return df

def build_ev_safe(ev2d: pd.DataFrame, evlist: pd.DataFrame,
                  mem: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """
    Safely merges event clustering results with original metadata.
    (Used mainly for cluster analysis/promotion steps).
    """
    if "event_id" not in evlist.columns:
        evlist = evlist.copy()
        evlist["event_id"] = np.arange(len(evlist))
        
    ev = ev2d.merge(evlist, on="event_id", how="inner")
    ev = _unify_two(ev, "base_x", "base_y", "base")
    ev = _unify_two(ev, "event_center_x", "event_center_y", "event_center")
    
    # Fallback to recover base filenames if missing
    if ev["base"].isna().any():
        idx2base = pred["base"].to_numpy()
        eid2mem  = {int(r["event_id"]): r.get("members", []) for r in mem.to_dict("records")}
        for i in ev.index[ev["base"].isna()]:
            eid = int(ev.at[i, "event_id"])
            bases = [idx2base[j] for j in eid2mem.get(eid, []) if 0 <= j < len(idx2base)]
            if bases:
                ev.at[i, "base"] = pd.Series(bases).mode().iloc[0]
                
    ev = ev.dropna(subset=["base"]).reset_index(drop=True)
    return ev

def build_events_from_scored(pw: pd.DataFrame, theta: float, gap: float=3.0, min_len: int=1) -> pd.DataFrame:
    """
    Groups consecutive windows above a threshold into discrete events.

    Args:
        pw (pd.DataFrame): Per-window DataFrame with columns ['base', 'center_sec', 's_hat'].
        theta (float): Detection threshold.
        gap (float): Max gap (in seconds) allowed to merge consecutive segments.
        min_len (int): Minimum number of consecutive windows required to form an event.

    Returns:
        pd.DataFrame: DataFrame of detected events with columns ['base', 'event_center'].
    """
    sel = pw[pw["s_hat"] >= theta].copy().sort_values(["base", "center_sec"])
    rows = []
    
    for b, g in sel.groupby("base"):
        if g.empty:
            continue
            
        cs = g["center_sec"].to_numpy()
        st = cs[0]
        ed = cs[0]
        run = 1
        
        for t in cs[1:]:
            if t - ed <= gap:
                ed = t
                run += 1
            else:
                if run >= min_len:
                    rows.append([b, (st + ed) / 2.0])
                st = ed = t
                run = 1
                
        if run >= min_len:
            rows.append([b, (st + ed) / 2.0])
            
    return pd.DataFrame(rows, columns=["base", "event_center"])

def eval_events_vs_ann(ev_df: pd.DataFrame, ann: pd.DataFrame, hours: float, tol_sec: float=10.0):
    """
    Evaluates detected events against ground-truth annotations.

    Args:
        ev_df (pd.DataFrame): Detected events (['base', 'event_center']).
        ann (pd.DataFrame): Ground truth events (['base', 'center']).
        hours (float): Total duration of the processed audio in hours.
        tol_sec (float): Temporal tolerance in seconds for a correct match.

    Returns:
        tuple: (recall, fp_per_hr)
    """
    TP = 0
    for b, grp in ann.groupby("base", sort=False):
        pp = ev_df[ev_df["base"] == b]["event_center"].to_numpy()
        if pp.size == 0:
            continue
            
        # Check for matches within tolerance
        for c in grp["center"].to_numpy():
            if np.min(np.abs(pp - c)) <= tol_sec:
                TP += 1
                
    # Calculate metrics
    FP = len(ev_df) - TP
    recall = TP / (len(ann) + 1e-9)
    fp_per_hr = FP / hours
    
    return recall, fp_per_hr