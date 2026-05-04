# Legacy artifacts (pre-revision-2.2)

This folder preserves scripts, weights, and CCED2 fits from the original
December 2026 paper submission, **before** the major revision (revision 2.2,
2026-05).

## Why kept

- **Reproducibility / archaeology**: allows comparison of original-vs-revised
  results when reviewers or future readers ask.
- **Documents the bugs that were corrected**: the original DAPT weights were
  affected by an AMP fp16 numerical instability that prevented weight updates
  (DAPT encoder ≈ PRETRAIN); see `revision1_DAPT_weight_issue.md` in the
  manuscript repository for details.

## What is here (do NOT use for new work)

| Path | Description | Reason for legacy |
|------|-------------|-------------------|
| `dapt_train_simclr_buggy.py` | Original SimCLR/InfoNCE-based DAPT training script (AMP fp16) | Numerical instability prevented BEATs encoder weight updates; superseded by Masked Audio Modeling (MAM) DAPT in `dapt_train.py` (top-level) |
| `run_dclde2013_cced2_eval.py` | DCLDE 2013 CCED2 evaluation | Bandwidth mismatch (DCLDE 2013 = 0–1 kHz silence above 1 kHz vs the 56-class InD reference); removed entirely from manuscript revision 2.2 |
| `run_hiceas_multi_species_eval_op_event_level.py` | OP event-level grid-search HICEAS evaluation | Replaced by canon-level Promoter discrimination (`run_hiceas_canon_promoter.py` at top-level) for the manuscript Table 4 |
| `scripts/dclde_table3_ablation.py` | DCLDE Table 3 ablation | Same DCLDE removal reason |
| `scripts/make_dclde2013_manifests.py` | DCLDE manifest construction | Same |
| `weights/beats_dapt_topup_encoder_buggy.pt` *(local only — gitignored)* | Original buggy DAPT weights (= PRETRAIN since updates failed) | Functionally equivalent to PRETRAIN; do not use for any DAPT comparison |
| `weights/sed_head_56_topup_ep8_buggy.pt` *(local only — gitignored)* | SED head trained on top of buggy encoder | Same |
| `weights/cced2/{cced2_norm.json,knn_dapt.pkl,maha_dapt.pkl,theta_cced.json,theta_cced2.json,theta_cced_legacy_pretrain.json}` | CCED2 fit computed from buggy encoder embeddings | Old normalization values (mk=2.14 vs new 2.54) and theta thresholds; new fit at `weights/cced2/` |

## When this folder will be removed

After paper acceptance and confirmation of revision 2.2 reproducibility, this
folder may be deleted. Until then it serves as ground truth for the original
submission's behaviour.
