# Table 4 — HICEAS canon-level Promoter discrimination (7 species)

This folder reproduces manuscript Table 4 (Noda et al., revision 2.2):
canon-level Promoter discrimination of seven cetacean species in HICEAS.

## File

- `table4_canon_promoter_7species.csv` — per-species results (7 species,
  AUC mean ± std across 5-fold StratifiedKFold cross-validation, plus F1,
  Precision, Recall).

## Source script

`run_hiceas_canon_promoter.py` (top-level) — fully parameterised CLI tool.
The original analysis snapshot is preserved as
`scripts/eval_b2_canon_promoter_compare.py`.

## Method (matches manuscript Methods §4.1.4 + §4.4.4)

- Positives: 60-s FLAC recordings overlapping any annotated
  DetectionTimeStart-DetectionTimeEnd interval for the species.
- Negatives: FLACs without any annotation for the species, sampled at 2x
  the positive count.
- Promoter: scikit-learn LogisticRegression(C=1.0, max_iter=1000,
  solver='lbfgs') on canon-mean BEATs+DAPT embeddings (768-dim).
- Cross-validation: StratifiedKFold(n_splits=5, shuffle=True,
  random_state=42).
- Total canons across all 7 species: 5,316 (HICEAS OP + 1706 manifest
  combined).

## In-band signal column — primary literature anchors

For each species, the "in-band signal (0-8 kHz)" column lists the vocal
repertoire component within the 0-8 kHz analysis band of the 16-kHz
resampled HICEAS recordings, with the primary literature citation that
quantifies the relevant frequency:

- *Globicephala macrorhynchus* — Jensen et al. 2011: deep-dive tonal
  calls with median peak frequency 3.9 kHz (5-95th percentile 1.8-12.3 kHz).
- *Pseudorca crassidens* — Sanino & Fowle 2006: south-eastern Pacific
  whistles, mean 6.9 kHz, range 3.8-13.6 kHz.
- *Steno bredanensis* — Lima et al. 2012: whistles with fundamental
  frequencies 2.24-13.94 kHz (Brazilian coast).
- *Balaenoptera acutorostrata* — Martin et al. 2013: Hawaiian minke
  whale "boing" call, dominant spectral component 1,384 Hz.
- *Stenella attenuata* — Silva et al. 2016: Hawaiian DTAG whistles
  ranged in frequency from 9.7 ± 2.8 to 19.8 ± 4.2 kHz.
- *Stenella coeruleoalba* — Papale et al. 2013: Atlantic stock whistles,
  minimum freq 7,882 ± 1,723 Hz, maximum 17,171 ± 3,500 Hz.
- *Physeter macrocephalus* — Møhl et al. 2003: multi-pulsed click p0
  pulse centroid 7.2 kHz (cBWrms 5.0 kHz) plus inter-pulse interval
  ≈ 5 ms (~200 Hz periodicity).

## Anchor: Hawaiian odontocete clicks lie above 0-8 kHz

Ziegenhorn et al. 2022 classified ten click types in HARP recordings off
Kauaʻi, Kona and Pearl & Hermes Reef: all peak frequencies were ≥ 12.5
kHz. Hence the discrimination achieved within the retained 0-8 kHz band
is anchored on whistles, tonal calls, the minke boing, and the sperm
whale low-frequency p0 pulse / IPI structure — not on full-bandwidth
echolocation clicks.
