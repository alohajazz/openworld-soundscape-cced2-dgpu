# Archived minor-revision execution evidence

Files in this directory are verbatim archived evidence from the minor-revision
execution and audit record. They preserve their original paths, environment
assumptions, and calculation behavior. Do not edit them to create a new run;
their role is provenance, comparison, and recovery of the audited procedure.

The current guarded entrypoints are outside this directory:

- `scripts/winaware_2026-05-09/supp_s3_7sp_full.py` and
  `supp_s3_perch.py` default to final corrected S3 inputs, write outside frozen
  artifacts, and reject embedding/index/manifest alignment mismatches.
- `scripts/winaware_2026-05-09/hiceas_label_efficiency_postpart2.py` defaults
  to final three-part HICEAS embeddings and retains unselected positives with
  training label 0 at finite budgets.
- `scripts/run_table4_fixed.py` validates three embedding/index sets before
  invoking this directory's archived Table 4 evaluator.

The archived and current scripts require the original public-data inputs,
embedding arrays and indexes, label CSVs, and compatible Python packages
(including NumPy, pandas, scikit-learn, and, for S3, joblib). They do not make
the restricted reference embeddings, event mappings, or corrected audio
embeddings publicly available. Frozen manuscript outputs should be verified
with `scripts/verify_minor_revision_artifacts.py`, not overwritten by a rerun.
