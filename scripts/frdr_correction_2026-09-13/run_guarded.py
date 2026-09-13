#!/usr/bin/env python3
"""Fail-closed launcher for authorised, isolated FRDR correction reruns."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SOURCE = PACKAGE / "source"
WORKSPACE = Path("/workspace")
MANIFEST = WORKSPACE / "data/externaldata/frdr_upcall/manifests/frdr_B_continuous_hop2s.csv"
ANNOTATIONS = WORKSPACE / "data/externaldata/frdr_upcall/raw/data/continuous/dataset_B/annotations_B_cont.csv"
PERCH_DIR = WORKSPACE / "embeddings/perch_frdr_hop2s"
PERCH_REF = WORKSPACE / "embeddings/perch_ind_models"
EXPECTED_ROWS, EXPECTED_FILES, EXPECTED_ANNOTATIONS, EXPECTED_POSITIVE_FILES = 44900, 50, 1157, 44
FROZEN_PARTS = ("paper_artifacts/minor_revision_2026-09", "scripts/minor_revision_2026-09",
                "paper_artifacts/frdr_correction_2026-09-13")
SOURCE_BY_MODE = {"table2": "frdr_supervised_promoter_COPY.py", "table3": "run_frdr_table3_winaware_COPY.py",
                  "fusion": "run_fusion_winaware_COPY.py", "labels": "frdr_label_efficiency_param.py"}

def die(message):
    raise SystemExit("REFUSING TO RUN: " + message)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def receipt_hashes(path):
    hashes = json.loads(path.read_text(encoding="utf-8")).get("inputs")
    if not isinstance(hashes, dict) or not hashes:
        die("--execution-receipt must contain non-empty execution_receipt.json inputs map")
    return hashes

def checked_paths(mode, beats, reference):
    paths = [MANIFEST, ANNOTATIONS, *sorted(beats.glob("embeddings_*.npy")), *sorted(beats.glob("index_*.csv"))]
    if not any(p.name.startswith("embeddings_") for p in paths) or not any(p.name.startswith("index_") for p in paths):
        die("BEATs embedding directory requires embeddings_*.npy and index_*.csv")
    if mode != "labels":
        if reference is None: die(f"{mode} requires --beats-ref-dir")
        paths += [reference / n for n in ("knn_cced2.pkl", "maha_cced2.pkl", "cced2_norm.json")]
    if mode == "table3":
        paths += [*sorted(PERCH_DIR.glob("embeddings_*.npy")), *sorted(PERCH_DIR.glob("index_*.csv")),
                  *[PERCH_REF / n for n in ("knn_perch.pkl", "maha_perch.pkl", "cced2_norm_perch.json")]]
    return paths

def verify_receipt(paths, hashes):
    for path in dict.fromkeys(paths):
        if not path.is_file(): die(f"required runtime input missing: {path}")
        expected = hashes.get(str(path))
        if not isinstance(expected, str) or len(expected) != 64: die(f"execution receipt lacks exact hash coverage: {path}")
        if sha256(path) != expected.lower(): die(f"execution receipt hash mismatch: {path}")

def path_center_columns(frame, label):
    path = next((c for c in frame if str(c).lower() == "path"), None)
    center = next((c for c in frame if "center" in str(c).lower()), None)
    if path is None or center is None: die(f"{label} index lacks path/center columns")
    return path, center

def validate_set(directory, dimension, label, manifest, np, pd):
    arrays, indexes = sorted(directory.glob("embeddings_*.npy")), sorted(directory.glob("index_*.csv"))
    if not arrays or len(arrays) != len(indexes): die(f"{label} embedding/index shard count mismatch")
    frames = []
    for array_path, index_path in zip(arrays, indexes, strict=True):
        array = np.load(array_path, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != dimension or not np.isfinite(array).all():
            die(f"{label} requires finite {dimension}-D embeddings: {array_path}")
        index = pd.read_csv(index_path); path, center = path_center_columns(index, label)
        index = index[[path, center]].rename(columns={path: "path", center: "center_sec"})
        index["center_sec"] = pd.to_numeric(index["center_sec"], errors="coerce")
        if len(index) != len(array) or not np.isfinite(index["center_sec"]).all(): die(f"{label} index/array mismatch")
        frames.append(index)
    index = pd.concat(frames, ignore_index=True)
    if len(index) != len(manifest): die(f"{label} index row count differs from manifest")
    if index["path"].astype(str).tolist() != manifest["path"].astype(str).tolist(): die(f"{label} index path order differs from manifest")
    if not np.array_equal(index["center_sec"].to_numpy(), manifest["center_sec"].to_numpy()): die(f"{label} index center order differs from manifest")

def validate_universe(mode, beats):
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc: die(f"validation requires NumPy/pandas: {exc}")
    manifest = pd.read_csv(MANIFEST)
    if not {"path", "center_sec"}.issubset(manifest): die("manifest lacks path/center_sec")
    manifest = manifest[["path", "center_sec"]].copy()
    manifest["path"] = manifest["path"].astype(str); manifest["center_sec"] = pd.to_numeric(manifest["center_sec"], errors="coerce")
    manifest["base"] = manifest["path"].map(lambda x: Path(x).name)
    if len(manifest) != EXPECTED_ROWS or manifest["base"].nunique() != EXPECTED_FILES or not np.isfinite(manifest["center_sec"]).all():
        die("manifest row count, all-file universe, or finite centers differ")
    for _, group in manifest.groupby("base", sort=False):
        c = group["center_sec"].to_numpy()
        if len(c) > 1 and not np.all(c[1:] > c[:-1]): die("manifest centers are not strictly monotonic within a file")
    ann = pd.read_csv(ANNOTATIONS, sep=";"); ann.columns = [str(c).strip().lower() for c in ann]
    file = next((c for c in ann if "file" in c), None); time = next((c for c in ann if "time" in c), None)
    bases = ann[file].map(lambda x: Path(str(x)).name) if file else []
    times = pd.to_numeric(ann[time], errors="coerce").to_numpy() if time else []
    if (file is None or time is None or len(ann) != EXPECTED_ANNOTATIONS or bases.nunique() != EXPECTED_POSITIVE_FILES
        or not np.isfinite(times).all() or not set(bases).issubset(set(manifest["base"]))): die("annotation validation failed")
    hours = sum((g["center_sec"].max() + 5.0) / 3600.0 for _, g in manifest.groupby("base", sort=False))
    if abs(hours - 451 / 18) > 1e-12: die(f"nominal duration changed: {hours!r}")
    validate_set(beats, 768, "BEATs", manifest, np, pd)
    if mode == "table3": validate_set(PERCH_DIR, 1536, "Perch", manifest, np, pd)

def validate_output(out):
    if not out.is_absolute(): die("--output must be absolute")
    if out.exists(): die("--output must not already exist, including for --dry-run")
    if any(part in out.as_posix() for part in FROZEN_PARTS): die("--output must not target frozen artifacts")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=sorted(SOURCE_BY_MODE)); parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--beats-emb-dir", required=True, type=Path); parser.add_argument("--beats-ref-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--execution-receipt", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    if args.workspace.resolve() != WORKSPACE: die("sources retain fixed /workspace manifest/annotation/Perch paths")
    out = args.output.resolve()
    validate_output(out)
    verify_receipt(checked_paths(args.mode, args.beats_emb_dir, args.beats_ref_dir), receipt_hashes(args.execution_receipt))
    validate_universe(args.mode, args.beats_emb_dir)  # after hashes, before source np.load/joblib.
    env = os.environ.copy(); env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    if args.mode == "table2": env.update(FR_EMB=str(args.beats_emb_dir), FR_REF=str(args.beats_ref_dir), FR_OUT=str(out))
    elif args.mode == "table3": env.update(T3_NAME="OceanBEATs_fixed_valref", T3_EMB=str(args.beats_emb_dir), T3_REF=str(args.beats_ref_dir), T3_OUT=str(out))
    elif args.mode == "fusion": env.update(FU_EMB=str(args.beats_emb_dir), FU_REF=str(args.beats_ref_dir), FU_OUT=str(out))
    else: env.update(LE_EMB_DIR=str(args.beats_emb_dir), LE_OUT=str(out))
    command = [sys.executable, str(SOURCE / SOURCE_BY_MODE[args.mode])]
    if args.dry_run: print("PASS guarded inputs; would run:", " ".join(command)); return
    out.parent.mkdir(parents=True, exist_ok=True); subprocess.run(command, cwd=SOURCE, env=env, check=True)
if __name__ == "__main__": main()
