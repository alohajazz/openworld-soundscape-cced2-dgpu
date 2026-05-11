# Table 2 — FRDR Detection performance (Phase 2 v5)

`frdr_quiet_promoter_union_supervised.csv` — Phase 2 v5 (window-aware) primary source for Table 2.

| Mode | Recall | FP/h | TP | FP |
|---|---|---|---|---|
| Quiet (CCED2) | 0.0743 | 9.7783 | 86 | 245 |
| Promoter (supervised LogReg, 5-fold OOF) | 0.4339 | 9.7783 | 502 | 245 |
| Union (Quiet ∪ Promoter) | 0.4270 | 10.3370 | 494 | 259 |

`frdr_fusion_winaware_2026-05-09.csv` — Fusion alpha sweep. alpha=0.9 row (Recall=0.0700, FP/h=9.978) used in Table 2.

`*_sweep_phase2v5.csv` — full threshold sweep tables for each mode (Phase 2 v5 source data).

## Legacy (Phase 2 v4)
The following pre-fix CSVs are retained for transparency:
- `table4_frdr.csv` — Pre-fix Quiet/Union/Fusion picks
- `comparison_quiet_vs_promoted.csv`, `fusion_pick.csv`, `union_quiet_promoted_pick.csv`
