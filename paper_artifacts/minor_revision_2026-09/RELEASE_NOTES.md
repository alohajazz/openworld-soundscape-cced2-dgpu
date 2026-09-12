# September 2026 corrected code and aggregate-results release

Tag: `v3.0.3-sr-minor-2026-09-13`

This release supersedes the public May 2026 analysis route with the corrected
step-127,641 input-mask MAM implementation, window-aware analysis entrypoints,
positive higher-is-more-unknown S3 selection, frozen aggregate results, and
an item-by-item manuscript artifact map. It includes the HICEAS recording
identifiers, time windows, species-label archives and their provenance hashes.

No raw audio, restricted 56-class individual embeddings, corrected fitted
reference pickle files, or window-level score/membership NPZ is added. The
original private analysis history is not an ancestor of this release. Aggregate
CCED2 normalisation and threshold JSON files are included; they do not replace
the fitted reference models needed for exact CCED2 scoring.

Corrected encoder and SED-head hashes are recorded in `release_manifest.json`,
but the files are not attached to this GitHub release. The public OceanBEATs
Hugging Face revision `dbb29a3dfc4fe1605c9fdd87079723db12903849`, checked on
2026-09-13, contains legacy weights, not the corrected encoder/head. A later
model publication must identify its own immutable revision and match the
recorded SHA-256 hashes before being used for the corrected manuscript results.

Run `python scripts/verify_minor_revision_artifacts.py` for the public frozen
artifact checks. Optional private-input checks are separately labelled;
passing public checks does not establish full raw-data reproduction or model
availability. See `MANUSCRIPT_ARTIFACT_MAP.md` for limitations, including the
restricted 56-class corpus, diagnostic S3 FP/h denominator, distinct BEATs/Perch
reference pools, and exploratory Figure 4 partial-positive-label masking.
