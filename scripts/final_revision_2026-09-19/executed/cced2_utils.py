#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cced2_utils.py

CCED2 (Cross-Class Embedding Distance 2) の最小実装モジュール。

- fit:
    In-distribution (InD) 埋め込み集合から kNN 参照器・Mahalanobis 参照器・
    z 正規化パラメータを学習し、指定ディレクトリに保存する。

- score:
    学習済み参照器と z 正規化パラメータを読み込み、任意の埋め込み集合に対して
    CCED2 スコアを計算し、.npy 形式で保存する。

基本的な使い方の例:

1) InD 埋め込みから CCED2 モデルを学習する:

    python cced2_utils.py fit \
      --embeddings_dir /path/to/ind_embeddings \
      --out_dir        /path/to/cced2_model \
      --k              50

2) 学習済みモデルを使って別データセットの CCED2 を計算する:

    python cced2_utils.py score \
      --model_dir      /path/to/cced2_model \
      --embeddings_dir /path/to/target_embeddings \
      --out_path       /path/to/target_embeddings/score_cced2.npy

埋め込みディレクトリには、dump_known56_features.py などで出力した
embeddings_*.npy ファイル（[N_i, D]）が複数存在することを前提とする。
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
import joblib


def load_embeddings_from_dir(emb_dir: str,
                             pattern: str = "embeddings_*.npy",
                             dtype=np.float32) -> np.ndarray:
    """
    embeddings_*.npy をすべて読み込み、[N, D] に結合して返す。
    """
    emb_files = sorted(glob.glob(os.path.join(emb_dir, pattern)))
    if not emb_files:
        raise FileNotFoundError(f"No embeddings matched: {emb_dir}/{pattern}")
    arrs = []
    for p in emb_files:
        e = np.load(p)
        if e.ndim != 2:
            raise ValueError(f"Expected 2D array in {p}, got shape {e.shape}")
        arrs.append(e.astype(dtype, copy=False))
    emb = np.concatenate(arrs, axis=0)
    return emb


def fit_cced2(emb_dir: str,
              out_dir: str,
              k: int = 50,
              knn_fname: str = "knn_cced2.pkl",
              maha_fname: str = "maha_cced2.pkl",
              norm_fname: str = "cced2_norm.json") -> None:
    """
    InD 埋め込み集合から CCED2 の参照器と z 正規化パラメータを学習し out_dir に保存する。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # InD 埋め込み読み込み
    print(f"[fit_cced2] loading embeddings from: {emb_dir}")
    E = load_embeddings_from_dir(emb_dir)  # [N, D]
    N, D = E.shape
    print(f"[fit_cced2] embeddings shape: {E.shape}")

    # kNN 参照器学習
    k_eff = min(k, max(1, N))
    print(f"[fit_cced2] training kNN (k={k_eff}) ...")
    knn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    knn.fit(E)

    # Mahalanobis 用の平均＆共分散 (Ledoit–Wolf)
    print(f"[fit_cced2] fitting LedoitWolf covariance ...")
    # LedoitWolf は float64 を前提とするので変換
    lw = LedoitWolf()
    lw.fit(E.astype(np.float64))
    mu = E.mean(axis=0, keepdims=True)  # [1, D]
    precision = lw.precision_           # [D, D]

    # InD 上で距離を計算して z 正規化パラメータを推定
    print(f"[fit_cced2] computing kNN & Mahalanobis distances on InD ...")
    dists, _ = knn.kneighbors(E, n_neighbors=k_eff)
    d_knn = dists.mean(axis=1)                     # [N]
    diff = (E - mu.astype(E.dtype))               # [N, D]
    # Mahalanobis: sqrt( (x-μ)^T Σ^{-1} (x-μ) )
    maha_raw = diff @ precision.astype(E.dtype)   # [N, D]
    d_maha = np.sqrt(np.einsum("nd,nd->n", diff, maha_raw) + 1e-12)  # [N]

    mk = float(d_knn.mean())
    sk = float(d_knn.std() + 1e-8)
    mm = float(d_maha.mean())
    sm = float(d_maha.std() + 1e-8)

    print(f"[fit_cced2] z-normalisation stats:")
    print(f"  kNN:   mean={mk:.6f}, std={sk:.6f}")
    print(f"  Maha:  mean={mm:.6f}, std={sm:.6f}")

    # モデルと正規化パラメータを保存
    print(f"[fit_cced2] saving models to: {out_dir}")
    joblib.dump({"knn": knn, "k": k_eff}, out_path / knn_fname)
    joblib.dump({"mu": mu, "precision": precision}, out_path / maha_fname)
    cfg = {"mk": mk, "sk": sk, "mm": mm, "sm": sm}
    with open(out_path / norm_fname, "w") as f:
        json.dump(cfg, f, indent=2)

    print("[fit_cced2] done.")


def _load_cced2_models(model_dir: str,
                       knn_fname: str = "knn_cced2.pkl",
                       maha_fname: str = "maha_cced2.pkl",
                       norm_fname: str = "cced2_norm.json"):
    """
    保存済み kNN / Mahalanobis モデルと z 正規化パラメータを読み込む。
    """
    model_path = Path(model_dir)
    knn = joblib.load(model_path / knn_fname)
    maha = joblib.load(model_path / maha_fname)
    with open(model_path / norm_fname, "r") as f:
        cfg = json.load(f)
    return knn, maha, cfg


def score_cced2(model_dir: str,
                emb_dir: str,
                out_path: str,
                pattern: str = "embeddings_*.npy") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    学習済み CCED2 モデルを使って emb_dir 内の埋め込みに対して
    d_knn^z, d_maha^z, CCED2 を計算し、out_path に CCED2 を保存する。

    戻り値:
        d_knn_z, d_maha_z, cced2_score (いずれも shape [N])
    """
    knn, maha, cfg = _load_cced2_models(model_dir)
    print(f"[score_cced2] loading embeddings from: {emb_dir}")
    E = load_embeddings_from_dir(emb_dir, pattern=pattern)  # [N, D]
    print(f"[score_cced2] embeddings shape: {E.shape}")

    k_eff = knn["k"]
    dists, _ = knn["knn"].kneighbors(E, n_neighbors=k_eff)
    d_knn = dists.mean(axis=1)  # [N]

    mu = maha["mu"].astype(E.dtype)           # [1, D]
    precision = maha["precision"].astype(E.dtype)  # [D, D]
    diff = (E - mu)
    maha_raw = diff @ precision               # [N, D]
    d_maha = np.sqrt(np.einsum("nd,nd->n", diff, maha_raw) + 1e-12)

    mk, sk = cfg.get("mk", 0.0), cfg.get("sk", 1.0)
    mm, sm = cfg.get("mm", 0.0), cfg.get("sm", 1.0)
    sk = sk if sk > 0 else 1e-8
    sm = sm if sm > 0 else 1e-8

    d_knn_z = (d_knn - mk) / sk
    d_maha_z = (d_maha - mm) / sm
    cced2_score = d_knn_z + d_maha_z

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, cced2_score.astype(np.float32))
    print(f"[score_cced2] saved CCED2 scores to: {out_path}")

    return d_knn_z, d_maha_z, cced2_score


def main():
    ap = argparse.ArgumentParser()
    subparsers = ap.add_subparsers(dest="command", required=True)

    # fit サブコマンド
    ap_fit = subparsers.add_parser("fit", help="fit CCED2 models on InD embeddings")
    ap_fit.add_argument("--embeddings_dir", required=True,
                        help="directory containing embeddings_*.npy for InD")
    ap_fit.add_argument("--out_dir", required=True,
                        help="directory to save CCED2 models and stats")
    ap_fit.add_argument("--k", type=int, default=50,
                        help="number of neighbours for kNN")

    # score サブコマンド
    ap_score = subparsers.add_parser("score", help="score CCED2 for a target embeddings directory")
    ap_score.add_argument("--model_dir", required=True,
                          help="directory containing knn_cced2.pkl, maha_cced2.pkl and cced2_norm.json")
    ap_score.add_argument("--embeddings_dir", required=True,
                          help="directory containing embeddings_*.npy to be scored")
    ap_score.add_argument("--out_path", required=True,
                          help="output .npy file path for CCED2 scores")
    ap_score.add_argument("--pattern", default="embeddings_*.npy",
                          help="glob pattern for embeddings files (default: embeddings_*.npy)")

    args = ap.parse_args()

    if args.command == "fit":
        fit_cced2(args.embeddings_dir, args.out_dir, k=args.k)
    elif args.command == "score":
        score_cced2(args.model_dir, args.embeddings_dir, args.out_path, pattern=args.pattern)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
