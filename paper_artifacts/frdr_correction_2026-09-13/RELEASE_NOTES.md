# FRDR all-50-file correction release notes

Tag: `v3.0.5-sr-minor-2026-09-13`

This frozen correction replaces the FRDR results used for Table 2, Table 3,
and Figure 3. It completes the ground-truth annotation dictionary over all 50
manifest files: the six files without annotations now contribute empty
ground-truth lists and can therefore contribute false positives. The nominal
duration remains 25.055555555555554 h and the annotation-event count remains
1,157.

The correction preserves the original event processing, references, encoders,
folds, labels, threshold grids, and discrete selectors. Table 2 Quiet and
Promoter select the nearest 10-FP/h point; Union selects maximum recall at
FP/h ≤ 10.5; Fusion selects the highest recall across its alpha grid after the
0.1-percentile-grid FP/h selection at each alpha (the selected alpha is 1.0).
Table 3 uses the nearest 10-FP/h point on its 0.5-percentile grid, without
interpolation.

No audio re-encoding, encoder retraining, or reference-model refitting was
performed. The existing small supervised Promoter heads were refit only under
the unchanged folds, labels, and selection protocol. Figure 4/HICEAS and
Table 1 are unchanged.

The release contains aggregate CSVs, Figure 3, executable-source provenance,
and receipts only. It does not include raw audio, embeddings, NPZ/model arrays,
or fitted reference pickles. Therefore it does not claim full raw-input
reproducibility; see `MANUSCRIPT_ARTIFACT_MAP.md`.

