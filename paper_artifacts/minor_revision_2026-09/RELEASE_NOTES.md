# September 2026 corrected code and aggregate-results release

Tag: `v3.0.4-sr-minor-2026-09-13`

> **Historical release note.** The frozen FRDR correction under
> `scripts/frdr_correction_2026-09-13/` and
> `paper_artifacts/frdr_correction_2026-09-13/` is release
> `v3.0.5-sr-minor-2026-09-13`. It corrects FRDR annotation completion from
> 44 annotation-bearing files to all 50 manifest files while retaining the
> nominal 25.055555555555554-h duration and the established evaluators.
> It does not alter this `v3.0.4` historical artifact
> set, HICEAS Figure 4, or Table 1's descriptive 6,489-train/1,623-validation
> head-selection reporting.

This follow-up changes model-availability documentation only: scientific code,
frozen numerical outputs and the data-publication boundary are unchanged from
`v3.0.3-sr-minor-2026-09-13`. The earlier release and tag are preserved.

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

Corrected encoder and SED-head hashes are recorded in `release_manifest.json`.
The files are publicly available on Hugging Face, not attached to this GitHub
release. `MODEL_AVAILABILITY.json` records the immutable model revision, tag,
download URLs, sizes and SHA-256 values. Both files were downloaded without
authentication from that revision and their full byte hashes verified.
The older OceanBEATs revision `dbb29a3dfc4fe1605c9fdd87079723db12903849` and
its two legacy weight files remain unchanged; they are not the corrected pair.

Run `python scripts/verify_minor_revision_artifacts.py` for the public frozen
artifact checks. Optional private-input checks are separately labelled;
passing public checks does not establish full raw-data reproduction or model
availability. See `MANUSCRIPT_ARTIFACT_MAP.md` for limitations, including the
restricted 56-class corpus, diagnostic S3 FP/h denominator, distinct BEATs/Perch
reference pools, and exploratory Figure 4 partial-positive-label masking.
