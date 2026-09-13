# FRDR 50-file annotation-completeness correction (2026-09-13)

**Status: frozen public correction release
`v3.0.5-sr-minor-2026-09-13`.** This package is the accepted correction
record for the revised FRDR results. It supersedes the old FRDR results while
preserving their historical files and release manifest unchanged.

## What changed

The original FRDR annotation dictionary contained only files with one or more
annotations. The corrected sources construct the dictionary over the complete
manifest file universe: all 50 manifest files are present and files without an
annotation receive an empty list. Annotation files absent from the manifest
fail closed.

Only this correction is made:

- Each of the four executable evaluators imports and calls
  `complete_annotations` immediately after constructing the annotation
  dictionary.
- `frdr_label_efficiency_param.py` also accepts `LE_OUT`, so a rerun can
  write to an isolated directory instead of a frozen artifact directory.

The helper/method logic, event processing, reference models, embeddings,
encoders, folds, labels, threshold grids and operating-point selection rules
are otherwise preserved. The nominal duration remains
`sum(max(center_sec) + 5) / 3600 = 451/18 = 25.055555555555554` hours.

`original/` holds the four exact pre-correction archived sources.
`source/` holds the exact executed correction sources. The
`source_changes.diff` is a review aid; `regression_tests.py` checks that
reversing the allowed edits recovers each original and exercises the relevant
boundary behavior.

## Layout

- `source/` — executed corrected sources plus the small annotation-universe helper
- `original/` — exact pre-correction sources, retained for provenance
- `regression_tests.py` — non-output regression checks; it uses NumPy and pandas
- `artifact_manifest.json` — package identity, source hashes, constraints, and
  approved aggregate-artifact hashes
- `verify_frdr_correction_package.py` — standard-library-only package verifier
- `prepare_frdr_manuscript_values.py` — retained review source for the
  CSV-to-Figure-3 bridge. It preserves its original sibling-directory
  `ROOT` assumption and is not a portable public CLI.
- `plot_frdr_correction.py` — portable aggregate-only Figure 3 regeneration
  CLI; requires Matplotlib and refuses to overwrite frozen outputs.

No embeddings, NPZ files, fitted model pickles, raw audio, row-level restricted
metadata are distributed in this source directory. Aggregate results are
in `paper_artifacts/frdr_correction_2026-09-13/`. A full public-data rerun is
therefore not claimed. The FRDR public recordings alone are insufficient:
the exact corrected embedding arrays, aligned manifests, event annotations,
and fitted reference inputs remain required.

## Local verification

From the repository root:

```bash
python3 scripts/frdr_correction_2026-09-13/verify_frdr_correction_package.py --artifact-root paper_artifacts/frdr_correction_2026-09-13
python3 scripts/frdr_correction_2026-09-13/regression_tests.py
```

Regenerate Figure 3 without private inputs or model fitting:

```bash
python3 scripts/frdr_correction_2026-09-13/plot_frdr_correction.py --output-dir /absolute/new/figure3_output
```

The original manuscript figure was rendered with Matplotlib 3.10.8. The
portable plot was checked for pixel identity in that same plotting environment;
other Matplotlib/font versions may change layout without changing the values.

The second command was tested in the execution-compatible environment:

```text
Python 3.12.10
numpy 1.26.4
pandas 2.3.3
scikit-learn 1.7.0
scipy 1.13.1
CPU threads: 1
```

A controlled rerun must use `run_guarded.py`; do not invoke a source module
directly because its archived fallback defaults point to the old
`frdr_fulldata_winaware` layout. The launcher requires the unchanged
`/workspace` FRDR manifest/annotation and (for Table 3) Perch locations,
explicit BEATs embedding/reference paths, a fresh output path, and the
approved `execution_receipt.json` from the controlled execution. Its
`inputs` map must cover the exact selected manifest, annotations, every
BEATs embedding/index shard, each BEATs reference file, and—only for Table
3—every Perch embedding/index shard and all three Perch reference files.
The matching hashes are checked before an array is loaded or a pickle is
opened. It then checks 44,900 finite aligned embedding/manifest rows, exact
index path and centre order, 768-D BEATs and 1536-D Perch dimensions, 50
unique manifest files, 1,157 annotations across 44 positive files, monotonic
centres, and the 451/18-h nominal duration.

```bash
python3 scripts/frdr_correction_2026-09-13/run_guarded.py table2 \
  --workspace /workspace \
  --beats-emb-dir /workspace/authorised/final_beats_embeddings \
  --beats-ref-dir /workspace/authorised/final_beats_reference \
  --execution-receipt /workspace/authorised/execution_receipt.json \
  --output /workspace/authorised/new_outputs/table2.csv --dry-run
```

The shown input paths are placeholders, not public path conventions. The
launcher supports `table2`, `table3`, `fusion`, and `labels`; it sets
the relevant source environment variables (including one CPU thread and the
final `OceanBEATs_fixed_valref` Table 3 label) and never rewrites source
code. `test_run_guarded.py` covers receipt coverage, wrong index order,
correct mock shapes, and output refusal in an isolated temporary directory.
Do not write to `paper_artifacts/minor_revision_2026-09/` or modify
`scripts/minor_revision_2026-09/`.

Approved aggregate-output hashes are recorded in `artifact_manifest.json`.
Verify them with:

```bash
python3 scripts/frdr_correction_2026-09-13/verify_frdr_correction_package.py \
  --artifact-root paper_artifacts/frdr_correction_2026-09-13
```

The verifier rejects private embedding/model formats in either the package or
the supplied artifact root.

## Scientific scope

This correction changes FRDR file accounting from the prior 44
annotation-bearing files to all 50 manifest files. The six newly represented
files have no annotations and can contribute false positives; they do not add
true positives or false negatives. It does not change:

- Table 2 selection: Quiet and Promoter nearest 10 FP/h; Union maximum recall
  subject to FP/h ≤ 10.5; Fusion selects the highest-recall result over its
  alpha grid after each alpha's 0.1-grid FP/h selector. The observed selected
  alpha is 1.0; it is not a complementary-signal fusion claim.
- Table 3 selection: nearest 10 FP/h on the discrete 0.5-percentile grid,
  without interpolation.
- Figure 3: FRDR results only. Figure 4 is the unchanged HICEAS
  partial-positive-label analysis.
- Table 1: descriptive validation reporting uses 6,489 training and 1,623
  validation examples for the same head selection; it is not an independent
  test set.
