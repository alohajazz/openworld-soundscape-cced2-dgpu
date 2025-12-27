#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_frdr_ablate_scores.py

FRDR North Atlantic right whale upcall dataset (dataset_B) 上で、
未知度スコアのアブレーション (kNN_z, Mahalanobis_z, CCED2_z) を
既存の CCED2 パイプラインと同じ評価ロジックで比較する。

やっていることは:
  1) manifest と注釈CSVを読み込み、(path, center_sec) 順にソート。
  2) BEATs+DAPT 埋め込み E_all を embeddings_*.npy から読み込み。
  3) InD モデル (kNN, Mahalanobis) と InD 正規化係数 (mk,sk,mm,sm) を読み込み。
  4) 全セグメントについて kNN, Mahalanobis 距離を計算し、InDベース z正規化し、
     kNN_z, Maha_z, CCED2_z = kNN_z + Maha_z を作る。
  5) 各スコアについて:
     - ファイルごとに median/MAD でローカル正規化
     - 必要なら 3点移動平均で smoothing (--smooth オプション)
     - CCED2本編と同じ近似式で閾値θを決定 (p = 1 - FP/h / windows_per_hour)
     - build_events_from_scored + eval_events_vs_ann で FP/h と Recall を計算
  6) 各スコアのベストOP (Recall, FP/h, θ) を表示し、JSONに保存する。
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fp_recall_helpers import (
    read_ann_safe,
    build_events_from_scored,
    eval_events_vs_ann,
)


# 既存のCCED2パイプラインで使っているInDモデル・正規化係数のパス
KNN_MODEL = Path("/workspace/embeddings/known56/models/knn_dapt.pkl")
MAHA_MODEL = Path("/workspace/embeddings/known56/models/maha_dapt.pkl")
NORM_JSON  = Path("/workspace/embeddings/cced2_norm.json")  # {mk,sk,mm,sm}


def median_abs_deviation(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-6)


def load_embeddings_op2s(emb_dir: Path) -> np.ndarray:
    emb_files = sorted(emb_dir.glob("embeddings_*.npy"))
    if not emb_files:
        raise RuntimeError(f"No embeddings_*.npy under {emb_dir}")
    xs = [np.load(p) for p in emb_files]
    E = np.concatenate(xs, axis=0).astype("float32")
    return E


def compute_base_scores(E: np.ndarray) -> dict:
    """
    BEATs+DAPT 埋め込み E_all (N, D) から、
    kNN / Mahalanobis の raw 距離 → InDベース z正規化 → CCED2_z を作る。
    """
    knn_obj = joblib.load(KNN_MODEL)
    maha_obj = joblib.load(MAHA_MODEL)
    cfg = json.loads(NORM_JSON.read_text())
    mk, sk = cfg["mk"], cfg["sk"]
    mm, sm = cfg["mm"], cfg["sm"]

    # kNN distance: mean distance to k neighbours
    d_knn, _ = knn_obj["knn"].kneighbors(E)
    knn_raw = d_knn.mean(axis=1).astype("float32")

    # Mahalanobis distance
    mu = np.asarray(maha_obj["mu"])         # shape (1, D)
    P  = np.asarray(maha_obj["precision"])  # shape (D, D)
    D  = E - mu
    maha_raw = np.sqrt((D @ P * D).sum(axis=1)).astype("float32")

    knn_z  = (knn_raw  - mk) / (sk + 1e-8)
    maha_z = (maha_raw - mm) / (sm + 1e-8)
    cced2_z = knn_z + maha_z

    return {
        "knn_z":   knn_z,
        "maha_z":  maha_z,
        "cced2_z": cced2_z,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-csv",
        required=True,
        help="FRDR continuous manifest CSV with at least columns: path, center_sec.",
    )
    parser.add_argument(
        "--emb-dir",
        required=True,
        help="Directory containing embeddings_*.npy for FRDR (op2s_dapt の出力).",
    )
    parser.add_argument(
        "--ann-csv",
        required=True,
        help="Annotation CSV for FRDR upcalls; must be readable by read_ann_safe().",
    )
    parser.add_argument(
        "--target-fp-per-hour",
        type=float,
        default=8.0,
        help="Target FP/h used to determine the global threshold.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Minimum number of consecutive suprathreshold windows per event.",
    )
    parser.add_argument(
        "--gap-sec",
        type=float,
        default=3.0,
        help="Maximum gap (in seconds) between consecutive suprathreshold windows "
             "to be merged into a single event.",
    )
    parser.add_argument(
        "--tol-sec",
        type=float,
        default=8.0,
        help="Temporal tolerance (in seconds) for matching detected and reference events.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="If set, apply 3-point moving-average smoothing to locally normalised scores.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_csv)
    emb_dir       = Path(args.emb_dir)
    ann_path      = Path(args.ann_csv)

    # 1) manifest を読み込み、(path, center_sec) でソート
    manifest = pd.read_csv(manifest_path)
    if "center_sec" not in manifest.columns:
        raise ValueError(f"manifest CSV must contain center_sec: {manifest_path}")
    manifest = manifest.sort_values(["path", "center_sec"]).reset_index(drop=True)

    # 2) hop と windows_per_hour を推定
    diffs = manifest.groupby("path")["center_sec"].diff().dropna()
    if diffs.empty:
        raise ValueError("Unable to estimate hop length from center_sec (insufficient data).")
    hop_sec = float(diffs.round(3).mode().iloc[0])
    windows_per_hour = max(1, int(round(3600.0 / hop_sec)))
    print(f"[FRDR] estimated hop ≈ {hop_sec:.3f} s, windows_per_hour ≈ {windows_per_hour}")

    # 3) 埋め込みを読み込み、未知度スコア (knn_z / maha_z / cced2_z) を計算
    print("== Loading embeddings and computing base scores ==")
    E_all = load_embeddings_op2s(emb_dir)
    if len(E_all) != len(manifest):
        raise ValueError(
            f"Embeddings length {len(E_all)} does not match manifest rows {len(manifest)}"
        )
    base_scores = compute_base_scores(E_all)

    # 4) 注釈を読み込み
    ann = read_ann_safe(str(ann_path))
    total_hours = len(manifest) / float(windows_per_hour)

    # 5) 各スコアについて、既存CCED2ロジックと同じ方法でローカル正規化 → smoothing → 閾値 → event生成 → eval
    all_results = {}

    for name, scores_z in base_scores.items():
        print(f"\n==== Evaluating FRDR with score: {name} ====")

        # per-file median/MAD でローカル正規化
        S_loc = np.empty_like(scores_z, dtype=float)
        manifest["score_raw"] = scores_z
        for path_str, idx in manifest.groupby("path").groups.items():
            v = manifest.loc[idx, "score_raw"].to_numpy()
            med = np.median(v)
            mad = median_abs_deviation(v)
            S_loc[idx] = (v - med) / mad

        manifest["score_loc"] = S_loc

        # optional smoothing（CCED2本編と同じ）
        if args.smooth:
            kernel = np.ones(3, dtype=float) / 3.0
            S_smooth = np.convolve(S_loc, kernel, mode="same")
        else:
            S_smooth = S_loc.copy()

        manifest["score_hat"] = S_smooth

        # 閾値決定（CCED2本編と同じ近似式）
        p = 1.0 - float(args.target_fp_per_hour) / float(windows_per_hour)
        p = max(0.0, min(1.0, p))
        theta = float(np.quantile(S_smooth, p))
        print(
            f"[FRDR:{name}] target_fp_per_hour={args.target_fp_per_hour:.2f} "
            f"→ quantile p={p:.4f}, threshold theta={theta:.3f}"
        )

        # per-window DataFrame を作って event 生成
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

        # eval
        recall, fp_per_hr = eval_events_vs_ann(
            events_df,
            ann,
            hours=total_hours,
            tol_sec=args.tol_sec,
        )

        print(f"[FRDR:{name}] ground-truth events : {len(ann)}")
        print(f"[FRDR:{name}] predicted events    : {len(events_df)}")
        print(f"[FRDR:{name}] recall              : {recall:.3f}")
        print(f"[FRDR:{name}] FP/h                : {fp_per_hr:.2f}")
        print(f"[FRDR:{name}] theta               : {theta:.3f}")
        print(f"[FRDR:{name}] min_len (k)         : {args.k}")
        print(f"[FRDR:{name}] gap_sec             : {args.gap_sec}")
        print(f"[FRDR:{name}] tol_sec             : {args.tol_sec}")

        all_results[name] = {
            "recall": float(recall),
            "fp_per_hour": float(fp_per_hr),
            "theta": float(theta),
            "windows_per_hour": float(windows_per_hour),
            "target_fp_per_hour": float(args.target_fp_per_hour),
            "k": int(args.k),
            "gap_sec": float(args.gap_sec),
            "tol_sec": float(args.tol_sec),
        }

    # 6) JSONにまとめて保存
    out_json = emb_dir / "ablation_frdr_scores.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print("[OK] wrote", out_json)


if __name__ == "__main__":
    main()
