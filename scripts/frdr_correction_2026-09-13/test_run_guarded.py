#!/usr/bin/env python3
"""Focused isolated tests for run_guarded.py; no model execution."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("guarded", HERE / "run_guarded.py")
guarded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guarded)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class GuardedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.manifest, self.annotations = root / "manifest.csv", root / "annotations.csv"
        self.beats, self.perch, self.ref, self.pref = root / "beats", root / "perch", root / "ref", root / "pref"
        for directory in (self.beats, self.perch, self.ref, self.pref): directory.mkdir()
        rows = [{"path": f"/x/{base}.wav", "center_sec": center}
                for base in ("a", "b") for center in (0.0, 45095.0)]
        pd.DataFrame(rows).to_csv(self.manifest, index=False)
        pd.DataFrame({"file": ["a.wav"] * 3, "time": [1.0, 2.0, 3.0]}).to_csv(self.annotations, sep=";", index=False)
        self._set_constants()
        self._write_set(self.beats, 768, rows)
        self._write_set(self.perch, 1536, rows)
        for directory, names in ((self.ref, ("knn_cced2.pkl", "maha_cced2.pkl", "cced2_norm.json")),
                                 (self.pref, ("knn_perch.pkl", "maha_perch.pkl", "cced2_norm_perch.json"))):
            for name in names: (directory / name).write_bytes(name.encode())

    def tearDown(self): self.temp.cleanup()

    def _set_constants(self):
        guarded.MANIFEST, guarded.ANNOTATIONS = self.manifest, self.annotations
        guarded.PERCH_DIR, guarded.PERCH_REF = self.perch, self.pref
        guarded.EXPECTED_ROWS, guarded.EXPECTED_FILES = 4, 2
        guarded.EXPECTED_ANNOTATIONS, guarded.EXPECTED_POSITIVE_FILES = 3, 1

    def _write_set(self, directory, dimension, rows):
        np.save(directory / "embeddings_000.npy", np.ones((4, dimension), dtype=np.float32))
        pd.DataFrame(rows).to_csv(directory / "index_000.csv", index=False)

    def _receipt(self, paths):
        receipt = Path(self.temp.name) / "receipt.json"
        receipt.write_text(json.dumps({"inputs": {str(path): digest(path) for path in paths}}))
        return receipt

    def test_correct_shape_and_coverage_accepted(self):
        paths = guarded.checked_paths("table3", self.beats, self.ref)
        receipt = self._receipt(paths)
        guarded.verify_receipt(paths, guarded.receipt_hashes(receipt))
        guarded.validate_universe("table3", self.beats)

    def test_missing_receipt_coverage_refused(self):
        paths = guarded.checked_paths("table2", self.beats, self.ref)
        with self.assertRaisesRegex(SystemExit, "lacks exact hash coverage"):
            guarded.verify_receipt(paths, {})

    def test_wrong_index_order_refused(self):
        index = pd.read_csv(self.beats / "index_000.csv")
        index.iloc[[0, 1]] = index.iloc[[1, 0]].to_numpy()
        index.to_csv(self.beats / "index_000.csv", index=False)
        with self.assertRaisesRegex(SystemExit, "order differs"):
            guarded.validate_universe("labels", self.beats)

    def test_existing_output_refused(self):
        output = Path(self.temp.name) / "already.csv"
        output.write_text("x")
        with self.assertRaisesRegex(SystemExit, "must not already exist"):
            guarded.validate_output(output)

if __name__ == "__main__":
    unittest.main()
