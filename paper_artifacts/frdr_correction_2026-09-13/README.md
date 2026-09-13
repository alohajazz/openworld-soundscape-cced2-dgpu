# FRDR all-50-file correction artifacts

**Frozen public correction release
`v3.0.5-sr-minor-2026-09-13`.** These are the approved aggregate outputs
for the 2026-09-13 FRDR annotation-completeness correction. They supersede
the old FRDR results,
while the historical `paper_artifacts/minor_revision_2026-09/` files and
release manifest remain unchanged as a record of the preceding release.

The correction completes the annotation dictionary over 50 manifest files:
the six files with no annotations contribute empty ground-truth lists. The
nominal duration is 25.055555555555554 hours and the annotation-event count is
1,157. The numerical acceptance receipt records selector/grid and input-hash
checks; the execution receipt records the controlled environment and hashes.

`results/*/*.csv` contains aggregate operating-point/sweep outputs only.
No raw audio, embeddings, NPZ/model arrays, fitted reference pickles, or logs
are included. `manuscript/Fig3_FRDR_all50.{png,pdf}` is generated from the
approved aggregate CSVs. `before_after.json` and the two receipts are
provenance records; their workspace paths do not redistribute the referenced
private inputs.

Run the new verifier, not the historical frozen-artifact verifier:

```bash
python3 scripts/frdr_correction_2026-09-13/verify_frdr_correction_package.py \
  --artifact-root paper_artifacts/frdr_correction_2026-09-13
```
