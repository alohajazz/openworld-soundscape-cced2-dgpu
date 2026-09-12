#!/usr/bin/env python3
"""Verify the frozen September 2026 minor-revision artifacts.

This checks public byte hashes, final S3 CSV schemas, aggregate metrics and
HICEAS provenance. Optional authorised inputs enable the original S2/S3 NPZ
summary and fitted-model byte checks. It does not rerun data-dependent analyses
and never deserializes pickle files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


ARTIFACT_RELATIVE = Path("paper_artifacts/minor_revision_2026-09")
EXPECTED_HASHES = {
    "table2_fixed_step127641.csv": "7e4c9041b975b3e40ca0824456ab67207e94c454ab86c3706aa806f6e2b50609",
    "table2_fusion_fixed_step127641.csv": "2abd81431e6f09a4c393e5d85c43d6a61967caa14d746d36f773260dfe9772b6",
    "table3_fixed_step127641.csv": "8ccb32e9014be780353c416f182d09e0e13cc8c75010606b3b8912de40680ce0",
    "supp_table_s3_fixed_step127641_unknown_high.csv": "4483c7b0af2bb208bf6e217855ec08adc229e7b692c369f7e0a3ff03c1586139",
    "supp_table_s3_perch_unknown_high.csv": "727c22c42e5c6c2ca9e24c25904fb682f6dc4fba5c38e7b7f1e954925e48bfc0",
    "fig4_label_efficiency_fixed.csv": "57c4256272f2faf3e21f4fd819647a078aadc441e5f28a31a3e8027f1ab2a5c0",
    "fig4_label_efficiency_fixed_agg.csv": "585286b8285e83ba3c7403c058ab8258c8dc1378ddeb177f3db6588c670ec86d",
    "supp_fig_s1_summary_step127641.json": "ba45360039edab0facd80fafa988e67c554b05596440d229700d8b06d48468a6",
    "supp_fig_s2_s3_summary_step127641.json": "a42dbd3621858c6f737159c7c2f62b5d751ad067d6f396a871bbd7272bb0e142",
    "table4_fixed_and_submitted_audit.json": "00b3170c5a89d6930d40ade55f174f332a2329a604e36ef5f5baf15212890c72",
    "table4_fold_auc_audit.csv": "18a98a9d61330863997f589714989b0301df203b169909b0e461f9c617a0dea9",
    "table1_beats_as2m_s42_metrics.json": "f41d7408fdd08fade158bcb09245f302af3c0b183544315e3885fb82cc75606e",
    "table1_dapt_fixed_s42_metrics.json": "366ee5059c4f8f9b610c92f8656cc9292ae1f9aaed14368cdca6c26dfb21e892",
    "manifests/input_provenance.json": "4dadde4dc649704d02f7d5c98301787316fee293f9d4b825456eb27f6ade5299",
    "manifests/hiceas_op_manifest_winaware.csv.gz": "d2078c97b1d872eb0c94d0caa1163bbeb96dd17f2a76851ec1eaf57129542761",
    "manifests/hiceas_1706_species_manifest_winsafe.csv.gz": "dc86f56f0ba84896f19872f28d471b9c1dcfd2297642239238b12b5c1a36febc",
    "manifests/hiceas_1706_part2_manifest_winsafe.csv.gz": "520caf1cac18df42b8987d52c80e7ba04aadb8d37717ec54edf2661918074912",
    "manifests/hiceas_recording_labels.zip": "378eb1820f12fe9ee1f17852af213be8b35819e06462f5d5734518a7da3a524d",
}
EXPECTED_CCED2_HASHES = {
    "cced2_norm.json": "4e7ae1a1348b4c227b3cdce421bfa11abe5cd86768b34383fb13cc9e949548a3",
    "theta_cced2.json": "6b42ebb4c810049eac3310bf45efd79f9f63b9173075fcf624916fc5206e7957",
}
PRIVATE_REFERENCE_HASHES = {
    "knn_cced2.pkl": "ca1fa4d90c2d6c45aa0b2f7b0bb617e9d71718a7f524bd86839cf52f57d2535c",
    "maha_cced2.pkl": "bdbe7866d1f65f5830566b7ebc36404158855b71421543a56764c2f8988151d1",
}
PRIVATE_NPZ_NAME = "supp_fig_s2_s3_data_step127641.npz"
PRIVATE_NPZ_HASH = "2cb139ede7353531188c95aa4653251d64aa35a9e3326dcfa2194d2448c41f28"
S3_COLUMNS = ("encoder", "score", "tol", "avg", "P", "R", "F1", "FP_h", "n_species")
TABLE4_EXPECTED = {
    "Minke whale": (1094, 0.991, 0.001, 0.87),
    "Sperm whale": (1968, 0.922, 0.005, 0.68),
    "False killer whale": (1442, 0.986, 0.002, 0.87),
    "Short-finned pilot whale": (1054, 0.959, 0.002, 0.71),
    "Rough-toothed dolphin": (246, 0.984, 0.008, 0.55),
    "Offshore spotted dolphin": (266, 0.949, 0.009, 0.43),
    "Striped dolphin": (371, 0.956, 0.005, 0.36),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing required artifact: {path}")
        return
    actual = sha256(path)
    if actual != expected:
        failures.append(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def validate_s3_csv(path: Path, expected_encoder: str, failures: list[str]) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error) as error:
        failures.append(f"cannot read {path}: {error}")
        return
    if fieldnames != S3_COLUMNS:
        failures.append(f"unexpected S3 schema in {path}: expected {S3_COLUMNS}, got {fieldnames}")
    if len(rows) != 12:
        failures.append(f"unexpected S3 row count in {path}: expected 12, got {len(rows)}")
        return
    expected_pairs = {(score, tol, avg) for score in ("kNN_z", "Mahalanobis_z", "CCED2")
                      for tol in ("15", "20") for avg in ("macro", "micro")}
    observed_pairs = {(row.get("score"), row.get("tol"), row.get("avg")) for row in rows}
    if observed_pairs != expected_pairs:
        failures.append(f"unexpected score/tolerance/averaging cells in {path}")
    if {row.get("encoder") for row in rows} != {expected_encoder}:
        failures.append(f"unexpected encoder label in {path}")
    for row in rows:
        try:
            values = [float(row[column]) for column in ("P", "R", "F1", "FP_h")]
            species = int(row["n_species"])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"non-numeric S3 value in {path}: {error}")
            continue
        if not all(np.isfinite(value) for value in values) or species != 7:
            failures.append(f"invalid numeric S3 values in {path}: {row}")


def validate_npz(artifact_dir: Path, private_artifact_dir: Path, failures: list[str]) -> None:
    path = private_artifact_dir / PRIVATE_NPZ_NAME
    summary_path = artifact_dir / "supp_fig_s2_s3_summary_step127641.json"
    try:
        with np.load(path, allow_pickle=False) as data:
            expected = {"dk_z", "dm_z", "cced2", "knn_only", "cced2_only", "shared"}
            if set(data.files) != expected:
                failures.append(f"unexpected NPZ keys in {path}: {data.files}")
                return
            dk_z, dm_z, cced2 = (data[name] for name in ("dk_z", "dm_z", "cced2"))
            knn_only, cced2_only, shared = (data[name] for name in ("knn_only", "cced2_only", "shared"))
            if any(array.ndim != 1 for array in (dk_z, dm_z, cced2, knn_only, cced2_only, shared)):
                failures.append("NPZ arrays must all be one-dimensional")
                return
            if len(dk_z) != 54419 or len(dm_z) != 54419 or len(cced2) != 54419:
                failures.append("NPZ score arrays must each contain 54,419 windows")
            if not all(np.issubdtype(array.dtype, np.number) for array in (dk_z, dm_z, cced2)):
                failures.append("NPZ score arrays must be numeric")
            if not np.all(np.isfinite(dk_z)) or not np.all(np.isfinite(dm_z)) or not np.all(np.isfinite(cced2)):
                failures.append("NPZ score arrays contain non-finite values")
            if not np.allclose(cced2, dk_z + dm_z, rtol=1e-11, atol=1e-11):
                failures.append("NPZ CCED2 scores do not equal dk_z + dm_z")
            sets = [set(array.astype(int).tolist()) for array in (knn_only, cced2_only, shared)]
            if any(len(values) != len(array) for values, array in zip(sets, (knn_only, cced2_only, shared))):
                failures.append("NPZ membership arrays contain duplicate indices")
            if any(values & other for i, values in enumerate(sets) for other in sets[i + 1:]):
                failures.append("NPZ membership sets are not disjoint")
            if any(index < 0 or index >= 54419 for values in sets for index in values):
                failures.append("NPZ membership index outside [0, 54419)")
            knn_set, cced2_set, shared_set = sets
            result = {
                "n_total": 54419,
                "n_select": len(knn_set) + len(shared_set),
                "shared": len(shared_set),
                "knn_only": len(knn_set),
                "cced2_only": len(cced2_set),
                "union": len(knn_set | cced2_set | shared_set),
                "jaccard": len(shared_set) / len(knn_set | cced2_set | shared_set),
                "symmetric_difference": len(knn_set) + len(cced2_set),
            }
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:  # type: ignore[name-defined]
        failures.append(f"cannot safely load NPZ {path}: {error}")
        return
    try:
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"cannot read NPZ summary {summary_path}: {error}")
        return
    for key, value in result.items():
        if key == "jaccard":
            matches = np.isclose(recorded.get(key), value, rtol=0, atol=1e-15)
        else:
            matches = recorded.get(key) == value
        if not matches:
            failures.append(f"NPZ summary mismatch for {key}: expected {value!r}, got {recorded.get(key)!r}")
    if result != {"n_total": 54419, "n_select": 545, "shared": 501, "knn_only": 44,
                  "cced2_only": 44, "union": 589, "jaccard": 501 / 589,
                  "symmetric_difference": 88}:
        failures.append(f"unexpected final S2/S3 membership summary: {result}")


def validate_table4(artifact_dir: Path, failures: list[str]) -> None:
    """Check the FIXED stratified arm against its fold-level audit evidence."""
    json_path = artifact_dir / "table4_fixed_and_submitted_audit.json"
    fold_path = artifact_dir / "table4_fold_auc_audit.csv"
    try:
        audit = json.loads(json_path.read_text(encoding="utf-8"))
        fixed_rows = audit["FIXED"]
        with fold_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != ("config", "species", "mode", "fold", "auc"):
                failures.append("unexpected Table 4 fold-audit CSV schema")
                return
            folds = list(reader)
    except (OSError, KeyError, json.JSONDecodeError, csv.Error) as error:
        failures.append(f"cannot read Table 4 audit artifacts: {error}")
        return
    fixed_by_species = {row.get("species"): row for row in fixed_rows}
    if set(fixed_by_species) != set(TABLE4_EXPECTED):
        failures.append("Table 4 FIXED arm does not contain exactly the seven final species")
        return
    for species, (n_pos, expected_auc, expected_sd, expected_f1) in TABLE4_EXPECTED.items():
        row = fixed_by_species[species]
        matching = [float(value["auc"]) for value in folds
                    if value.get("config") == "FIXED" and value.get("mode") == "stratified"
                    and value.get("species") == species]
        try:
            auc = float(row["strat_auc"])
            sd = float(row["strat_auc_std"])
            f1 = float(row["strat_f1"])
            reported_n = int(row["n_pos"])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"invalid FIXED Table 4 row for {species}: {error}")
            continue
        if len(matching) != 5:
            failures.append(f"Table 4 FIXED/stratified fold count for {species}: expected 5, got {len(matching)}")
            continue
        if not np.isclose(auc, np.mean(matching), rtol=0, atol=1e-12):
            failures.append(f"Table 4 mean AUC mismatch for {species}")
        if not np.isclose(sd, np.std(matching, ddof=0), rtol=0, atol=1e-12):
            failures.append(f"Table 4 population SD mismatch for {species}")
        if reported_n != n_pos or round(auc, 3) != expected_auc or round(sd, 3) != expected_sd or round(f1, 2) != expected_f1:
            failures.append(f"Table 4 final rounded value/count mismatch for {species}")


def canon_from_path(path: str) -> str:
    parts = Path(path).stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else Path(path).stem


def validate_table1(artifact_dir: Path, failures: list[str]) -> None:
    expected = {
        "table1_beats_as2m_s42_metrics.json": (0.483, 0.784, 0.506),
        "table1_dapt_fixed_s42_metrics.json": (0.493, 0.749, 0.518),
    }
    for name, rounded in expected.items():
        try:
            metrics = json.loads((artifact_dir / name).read_text(encoding="utf-8"))
            values = (float(metrics["eventF1"]["F1"]), float(metrics["clipF1"]["F1"]),
                      float(metrics["segmentF1"]["2.0"]["F1"]))
            labels = metrics["labels"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"cannot read Table 1 aggregate {name}: {error}")
            continue
        if tuple(round(value, 3) for value in values) != rounded or len(labels) != 56:
            failures.append(f"Table 1 aggregate mismatch in {name}: values={values}, labels={len(labels)}")


def validate_input_provenance(artifact_dir: Path, failures: list[str]) -> None:
    manifest_dir = artifact_dir / "manifests"
    try:
        provenance = json.loads((manifest_dir / "input_provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"cannot read input provenance: {error}")
        return
    all_canons: set[str] = set()
    for filename, details in provenance.get("manifests", {}).items():
        gz_path = manifest_dir / f"{filename}.gz"
        try:
            raw = gzip.decompress(gz_path.read_bytes())
            rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
        except (OSError, UnicodeDecodeError, gzip.BadGzipFile, csv.Error) as error:
            failures.append(f"cannot read manifest {gz_path}: {error}")
            continue
        if sha256(gz_path) != details.get("sha256_gzip") or hashlib.sha256(raw).hexdigest() != details.get("sha256_uncompressed"):
            failures.append(f"manifest hash mismatch for {filename}")
        if len(rows) != details.get("windows") or len({row.get("path") for row in rows}) != details.get("unique_files"):
            failures.append(f"manifest count mismatch for {filename}")
        all_canons.update(canon_from_path(row["path"]) for row in rows if row.get("path"))
    archive_path = manifest_dir / "hiceas_recording_labels.zip"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            expected_labels = provenance["labels"]
            if set(archive.namelist()) != set(expected_labels):
                failures.append("unexpected label-archive member set")
                return
            labels: dict[str, set[str]] = {}
            for name, details in expected_labels.items():
                raw = archive.read(name)
                rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
                canons = {row.get("canon", "") for row in rows}
                if len(rows) != details.get("source_rows") or len(canons) != len(rows):
                    failures.append(f"label row/duplicate mismatch for {name}")
                if hashlib.sha256(raw).hexdigest() != details.get("sha256"):
                    failures.append(f"label hash mismatch for {name}")
                if len(canons & all_canons) != details.get("available_in_three_manifests"):
                    failures.append(f"label availability mismatch for {name}")
                labels[name] = canons
            negatives = labels["neg_all.csv"]
            if any(negatives & canons for name, canons in labels.items() if name != "neg_all.csv"):
                failures.append("negative label set overlaps a positive label set")
            if len(negatives & all_canons) != 6135:
                failures.append("available common-negative count is not 6,135")
    except (OSError, KeyError, UnicodeDecodeError, zipfile.BadZipFile, csv.Error) as error:
        failures.append(f"cannot safely read label archive: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                        help="repository root (defaults to this script's parent)")
    parser.add_argument("--private-artifact-dir", type=Path,
                        help="optional authorised directory containing the original window-score NPZ")
    parser.add_argument("--private-reference-dir", type=Path,
                        help="optional authorised directory containing original fitted reference pickle files")
    args = parser.parse_args()
    repo = args.repo.resolve()
    artifact_dir = repo / ARTIFACT_RELATIVE
    failures: list[str] = []
    for name, expected in EXPECTED_HASHES.items():
        require_hash(artifact_dir / name, expected, failures)
    for name, expected in EXPECTED_CCED2_HASHES.items():
        require_hash(repo / "weights/cced2_step127641" / name, expected, failures)
    validate_s3_csv(artifact_dir / "supp_table_s3_fixed_step127641_unknown_high.csv", "BEATs+DAPT", failures)
    validate_s3_csv(artifact_dir / "supp_table_s3_perch_unknown_high.csv", "Perch_2.0_win10", failures)
    if args.private_artifact_dir is not None:
        require_hash(args.private_artifact_dir / PRIVATE_NPZ_NAME, PRIVATE_NPZ_HASH, failures)
        validate_npz(artifact_dir, args.private_artifact_dir, failures)
    else:
        print("NOT CHECKED: window-level NPZ and derived membership summary (not publicly supplied).")
    if args.private_reference_dir is not None:
        for name, expected in PRIVATE_REFERENCE_HASHES.items():
            require_hash(args.private_reference_dir / name, expected, failures)
    else:
        print("NOT CHECKED: fitted reference pickle hashes (not publicly supplied).")
    validate_table4(artifact_dir, failures)
    validate_table1(artifact_dir, failures)
    validate_input_provenance(artifact_dir, failures)
    if failures:
        print("Minor-revision artifact verification FAILED:", file=sys.stderr)
        print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
        return 1
    print("Public minor-revision artifact verification passed; no model rerun or publication status is verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
