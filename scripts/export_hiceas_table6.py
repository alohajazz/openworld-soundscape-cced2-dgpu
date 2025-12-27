#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_hiceas_table6.py

Generate FINAL_summary.md and FINAL_table_15s.tex from _FINAL_SPECIES_TABLE.csv.

Selection rule (default):
- Filter tol == 15 (or user-specified)
- For each species, pick the row that maximizes Recall subject to:
    Precision >= 0.9 and FP/h <= 0.5
  Ties are broken by F1.
- If no row satisfies constraints, pick the row with max F1.

This matches the "Table 6 (main)" selection logic described in the paper.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def pick_rows(df: pd.DataFrame, tol: float, p_min: float, fph_max: float) -> pd.DataFrame:
    df = df.copy()
    df = df[df["tol"].astype(float) == float(tol)].copy()

    picks = []
    for sp, g in df.groupby("species_name", sort=False):
        g = g.copy()
        # ensure numeric
        for c in ["P", "R", "F1", "FP/h"]:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        ok = g[(g["P"] >= p_min) & (g["FP/h"] <= fph_max)]
        if len(ok) > 0:
            best = ok.sort_values(["R", "F1"], ascending=[False, False]).iloc[0]
        else:
            best = g.sort_values(["F1", "R"], ascending=[False, False]).iloc[0]
        picks.append(best)

    out = pd.DataFrame(picks)
    # stable order: by the order in the file for tol==15
    # (groupby(sort=False) already preserves first occurrence order)
    return out


def write_md(df_pick: pd.DataFrame, tol: float, out_md: Path) -> None:
    lines = []
    lines.append(f"# HICEAS Final (DNS, tol=±{int(tol)}s)")
    lines.append("")
    lines.append("| Species | Mode | P | R | F1 | FP/h |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in df_pick.iterrows():
        sp = r["species_name"]
        mode = r["method"]
        P = float(r["P"])
        R = float(r["R"])
        F1 = float(r["F1"])
        fph = float(r["FP/h"])
        lines.append(f"| {sp} | {mode} | {P:.4f} | {R:.4f} | {F1:.4f} | {fph:.2f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(df_pick: pd.DataFrame, tol: float, out_tex: Path, label: str = "tab:hiceas15") -> None:
    # LaTeX booktabs table
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Species & Mode & P & R & F1 & FP/h \\")
    lines.append(r"\midrule")
    for _, r in df_pick.iterrows():
        sp = str(r["species_name"]).replace("&", r"\&")
        mode = str(r["method"]).replace("&", r"\&")
        P = float(r["P"]); R = float(r["R"]); F1 = float(r["F1"]); fph = float(r["FP/h"])
        lines.append(f"{sp} & {mode} & {P:.4f} & {R:.4f} & {F1:.4f} & {fph:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{Per-species performance on HICEAS under DNS and tol=$\pm${int(tol)}\,s.}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True, help="Path to _FINAL_SPECIES_TABLE.csv")
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--p_min", type=float, default=0.9)
    ap.add_argument("--fph_max", type=float, default=0.5)
    ap.add_argument("--out_md", type=str, required=True, help="Output FINAL_summary.md")
    ap.add_argument("--out_tex", type=str, required=True, help="Output FINAL_table_15s.tex")
    ap.add_argument("--out_pick_csv", type=str, default="", help="Optional: save picked rows as csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    need_cols = {"species_name", "method", "tol", "P", "R", "F1", "FP/h"}
    missing = need_cols - set(df.columns)
    if missing:
        raise SystemExit(f"[ERR] missing columns in {args.in_csv}: {sorted(missing)}")

    picked = pick_rows(df, tol=args.tol, p_min=args.p_min, fph_max=args.fph_max)

    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    out_tex = Path(args.out_tex); out_tex.parent.mkdir(parents=True, exist_ok=True)
    write_md(picked, tol=args.tol, out_md=out_md)
    write_tex(picked, tol=args.tol, out_tex=out_tex, label="tab:hiceas15")

    if args.out_pick_csv:
        Path(args.out_pick_csv).parent.mkdir(parents=True, exist_ok=True)
        picked.to_csv(args.out_pick_csv, index=False)

    print("[OK] wrote:", out_md)
    print("[OK] wrote:", out_tex)
    if args.out_pick_csv:
        print("[OK] wrote:", args.out_pick_csv)


if __name__ == "__main__":
    main()
