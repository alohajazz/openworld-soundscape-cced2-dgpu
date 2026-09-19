# Executed versions of shared scripts

The files in this directory are the bytes that were executed on the analysis server. They are kept separately
because files with the same names elsewhere in this repository are documented, generic versions that differ from
the executed ones. Paths inside the scripts refer to the analysis server and must be adapted before reuse.

| File | SHA-256 (first 16) | Used for |
|---|---|---|
| `sedgate.sh` | `b277ed1d2799dac8` | Launcher of the Table 1 runs, seeds 42, 123 and 777: trains the 56-class SED head on BEATs AS-2M and on the Stage 1 encoder and evaluates each head, with the exact arguments |
| `sedgate_more.sh` | `e12dfced0e8f331f` | Same runs with the same arguments for seeds 101, 202, 303, 404 and 505; appends to the same `summary.csv` |
| `sedgate_buggy.sh`, `gate40k.sh` | `eeedd470cd166249`, `db189fb6d7e478fd` | Same protocol for two checkpoints that are not used in the manuscript (rows `DAPTbuggy`: checkpoint `BEATs_DAPT_MAM_step6000.pt` written by the superseded training code, three seeds; rows `DAPTfix40k`: intermediate step-40,000 checkpoint of the corrected Stage 1 run). Included because they wrote rows to the same `summary.csv` |
| `train_sed_beats_weak_plus.py` | `4887bdcc0d22294b` | SED-head training called by `sedgate.sh` (applies `--seed`) |
| `eval_sed_beats_report.py` | `fd2f3bb9dff5151a` | SED evaluation called by `sedgate.sh`; writes the metrics reported in Table 1 |
| `dump_known56_features.py` | `acd383e2682e673a` | Extraction of the 1,623 validation-clip embeddings that define the CCED2 reference |
| `cced2_utils.py` | `de20b5d5213ca221` | `fit_cced2` used to fit the CCED2 reference (k = 50) |
| `run_stage2_downstream_commands.sh` | — | Commands of the Stage 2 driver: embedding extraction with row-count checks, CCED2 reference fit, and the calls of the FRDR, HICEAS and overlap scripts. Full-line comments of the original file were removed; every command line is unchanged |

Table 1 of the manuscript reports the mean of the eight seeds for the rows `PRETRAIN` and `DAPTfix` of
`paper_artifacts/final_revision_2026-09-19/table1/summary.csv`; `../table1_seed_summary.py` computes the means, sample
standard deviations and paired differences. The 56-class audio, labels and splits
referenced by these scripts are not public (see `MANUSCRIPT_ARTIFACT_MAP.md`).
