# -*- coding: utf-8 -*-
"""
fp_recall_helpers.py

Helper functions for parsing annotations and calculating False Positives per Hour (FP/h)
and Recall for continuous operational evaluation.

Log/Methods-aligned matching (Methods 4.5.1 style):
- TP is reference-based: each GT event is a TP if there exists >=1 predicted event within ±tol.
- FP is prediction-based: a predicted event is an FP only if it falls outside ±tol of ALL GT events.
- Duplicate predictions near the same GT are NOT counted as extra TP, and are NOT counted as FP.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def read_ann_safe(ann_path: str) -> pd.DataFrame:
    if not os.path.exists(ann_path):
        raise FileNotFoundError(ann_path)

    try:
        ann = pd.read_csv(ann_path, sep=";", engine="python")
    except Exception:
        ann = pd.read_csv(ann_path, engine="python")

    if ann.shape[1] == 1:
        ann = ann.iloc[:, 0].str.split(";", expand=True)

    ann.columns = [str(c).strip().lower() for c in ann.columns]

    fcol = next((c for c in ann.columns if "file" in c or "name" in c), None)
    tcol = next((c for c in ann.columns if "time" in c or "stamp" in c), None)

    if fcol is None or tcol is None:
        if len(ann.columns) >= 2:
            fcol, tcol = ann.columns[0], ann.columns[1]
        else:
            raise KeyError(f"Annotation columns (filename/time) not found: {ann.columns.tolist()}")

    out = pd.DataFrame({
        "base": ann[fcol].astype(str).apply(lambda p: Path(p).name),
        "center": pd.to_numeric(ann[tcol], errors="coerce"),
    }).dropna(subset=["base", "center"]).reset_index(drop=True)

    return out


def _unify_two(df: pd.DataFrame, a: str, b: str, out: str) -> pd.DataFrame:
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
    if "event_id" not in evlist.columns:
        evlist = evlist.copy()
        evlist["event_id"] = np.arange(len(evlist))

    ev = ev2d.merge(evlist, on="event_id", how="inner")
    ev = _unify_two(ev, "base_x", "base_y", "base")
    ev = _unify_two(ev, "event_center_x", "event_center_y", "event_center")

    if ev["base"].isna().any():
        idx2base = pred["base"].to_numpy()
        eid2mem = {int(r["event_id"]): r.get("members", []) for r in mem.to_dict("records")}
        for i in ev.index[ev["base"].isna()]:
            eid = int(ev.at[i, "event_id"])
            bases = [idx2base[j] for j in eid2mem.get(eid, []) if 0 <= j < len(idx2base)]
            if bases:
                ev.at[i, "base"] = pd.Series(bases).mode().iloc[0]

    ev = ev.dropna(subset=["base"]).reset_index(drop=True)
    return ev


def build_events_from_scored(pw: pd.DataFrame, theta: float, gap: float = 3.0, min_len: int = 1) -> pd.DataFrame:
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


def eval_events_vs_ann(ev_df: pd.DataFrame, ann: pd.DataFrame, hours: float, tol_sec: float = 10.0):
    """
    Evaluates detected events against ground-truth annotations.

    Returns:
        (recall, fp_per_hr)

    Matching definition (Methods 4.5.1 style):
    - TP counts per GT reference event: a GT is a TP if any predicted event falls within ±tol.
    - FP counts per predicted event: a pred is an FP if it is outside ±tol of all GT events.
    """
    if ev_df is None or len(ev_df) == 0:
        recall = 0.0
        fp_per_hr = 0.0
        return recall, fp_per_hr

    # Ensure expected columns exist
    if "base" not in ev_df.columns or "event_center" not in ev_df.columns:
        raise ValueError("ev_df must contain columns ['base', 'event_center']")
    if "base" not in ann.columns or "center" not in ann.columns:
        raise ValueError("ann must contain columns ['base', 'center']")

    TP = 0
    FP = 0

    # Group predictions by base for efficient matching
    pred_by_base = (
        ev_df.groupby("base")["event_center"]
        .apply(lambda s: np.asarray(s.values, dtype=float))
        .to_dict()
    )

    for b, grp in ann.groupby("base", sort=False):
        gt = grp["center"].to_numpy(dtype=float)
        pp = pred_by_base.get(b, np.asarray([], dtype=float))

        if pp.size == 0:
            continue

        # distance matrix (P, G)
        d = np.abs(pp[:, None] - gt[None, :])

        # TP (per GT): any pred within tol?
        TP += int((np.min(d, axis=0) <= tol_sec).sum())

        # FP (per pred): outside all GT windows?
        FP += int((np.min(d, axis=1) > tol_sec).sum())

    recall = TP / (len(ann) + 1e-9)
    fp_per_hr = FP / max(hours, 1e-9)

    return float(recall), float(fp_per_hr)
