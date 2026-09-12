#!/usr/bin/env python3
"""Run the final Table 4 FIXED protocol using the archived audited evaluator.

This wrapper only supplies paths, enforces embedding/index alignment, and
serializes results.  The classifier, splits, species order, and scoring remain
in ``scripts/minor_revision_2026-09/groupkfold_table4_eval.py``.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_BASE = REPO_ROOT / "scripts/minor_revision_2026-09/groupkfold_table4_eval.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-dir", type=Path,
                        default=Path(os.environ.get("HICEAS_OP_DIR", "/workspace/embeddings/op_fixed_step127641")))
    parser.add_argument("--species-dir", type=Path,
                        default=Path(os.environ.get("HICEAS_1706_DIR", "/workspace/embeddings/1706species_fixed_step127641")))
    parser.add_argument("--part2-dir", type=Path,
                        default=Path(os.environ.get("HICEAS_PART2_DIR", "/workspace/embeddings/1706part2_fixed_step127641")))
    parser.add_argument("--canon-dir", type=Path,
                        default=Path(os.environ.get("HICEAS_CANON_DIR", "/workspace/revision1_species_canons")))
    parser.add_argument("--out-json", type=Path,
                        default=Path(os.environ.get("DGPU_TABLE4_OUT", "/workspace/outputs/table4_fixed_step127641.json")))
    return parser.parse_args()


def load_base():
    spec = importlib.util.spec_from_file_location("minor_revision_table4_base", ARCHIVE_BASE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load archived Table 4 evaluator: {ARCHIVE_BASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sharded_strict(directory: Path) -> tuple[np.ndarray, pd.DataFrame]:
    embedding_paths = sorted(Path(path) for path in glob.glob(str(directory / "embeddings_*.npy")))
    index_paths = sorted(Path(path) for path in glob.glob(str(directory / "index_*.csv")))
    if not embedding_paths or len(embedding_paths) != len(index_paths):
        raise FileNotFoundError(f"require matching non-empty embeddings_*.npy and index_*.csv shards in {directory}")
    embeddings = np.concatenate([np.load(path) for path in embedding_paths]).astype("float32")
    index = pd.concat([pd.read_csv(path) for path in index_paths], ignore_index=True)
    if len(embeddings) != len(index):
        raise ValueError(f"embedding/index row mismatch in {directory}: {len(embeddings)} != {len(index)}")
    if not {"path", "center_sec"}.issubset(index.columns):
        raise ValueError(f"index shards in {directory} require path and center_sec columns")
    return embeddings, index


def require_label_files(canon_dir: Path, species: list[tuple[str, str]]) -> Path:
    neg_csv = canon_dir / "neg_all.csv"
    required = [neg_csv, *(canon_dir / filename for _, filename in species)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Table 4 label files: " + ", ".join(missing))
    return neg_csv


def main() -> None:
    args = parse_args()
    base = load_base()
    op_embeddings, op_index = load_sharded_strict(args.op_dir)
    species_embeddings, species_index = load_sharded_strict(args.species_dir)
    part2_embeddings, part2_index = load_sharded_strict(args.part2_dir)
    embeddings = np.concatenate([op_embeddings, species_embeddings, part2_embeddings])
    index = pd.concat([op_index, species_index, part2_index], ignore_index=True)
    if len(embeddings) != len(index):
        raise ValueError(f"combined embedding/index row mismatch: {len(embeddings)} != {len(index)}")
    canon_embeddings = base.build_canon_embeddings(embeddings, index)
    neg_csv = require_label_files(args.canon_dir, base.SPECIES)

    rows = []
    for species, filename in base.SPECIES:
        pos_csv = args.canon_dir / filename
        stratified = base.evaluate(canon_embeddings, pos_csv, neg_csv, "stratified")
        group = base.evaluate(canon_embeddings, pos_csv, neg_csv, "group")
        if stratified is None or group is None:
            raise RuntimeError(f"no usable Table 4 evaluation folds for {species}")
        rows.append({
            "species": species,
            "n_pos": stratified["n_pos"],
            "n_neg": stratified["n_neg"],
            "strat_auc": stratified["auc_mean"],
            "strat_auc_std": stratified["auc_std"],
            "strat_f1": stratified["f1_mean"],
            "group_auc": group["auc_mean"],
            "group_auc_std": group["auc_std"],
            "group_folds": group["n_folds_used"],
        })
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"FIXED": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
