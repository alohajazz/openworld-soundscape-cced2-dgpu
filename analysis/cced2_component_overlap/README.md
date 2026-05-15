# CCED2 Component Overlap Analysis (2026-05-15)

Analysis script archive supporting Discussion §3.2 + §3.4 + Supplementary
Table S5 + Supplementary Fig S4 added in revision 2.14 (main) / 2.16 (supp).

## Question addressed

How much does the CCED2 score (= z(kNN) + z(Maha)) overlap with each single-distance
score in selecting the top-1% most-extreme windows? Is the relative balance between
the kNN and Mahalanobis components encoder-dependent?

## Files

- `analyze_168.py`        : First overlap analysis (BEATs+DAPT, kNN vs CCED2 only)
                            generated Fig S2 Venn data (461 shared / 84 each unique)
- `perch_overlap.py`      : Pairwise overlap stats (kNN vs Maha vs CCED2) for both
                            BEATs+DAPT and Perch on HICEAS OP (n = 54,419 windows)
- `make_perch_venn3.py`   : Generated Supplementary Fig S4 (Perch 3-way Venn)

## Key numerical findings

| Encoder    | kNN-Maha | kNN-CCED2 | Maha-CCED2 |
|------------|----------|-----------|------------|
| BEATs+DAPT |   81.8%  |   84.6%   |   97.1%    |
| Perch      |   60.0%  |   74.5%   |   84.4%    |

Pearson r over all 54,419 windows:
- BEATs+DAPT: r(dk, dm) = 0.698, r(dm, cced2) = 0.995
- Perch:      r(dk, dm) = 0.942, r(dm, cced2) = 0.991

## How to re-run

```bash
ssh adelie-linux
docker exec -it d58770e91330 bash
cd /tmp
python3 perch_overlap.py     # numerical comparison
python3 make_perch_venn3.py  # regenerate Fig S4 PNG
```

Required inputs (already on Adelie):
- `/workspace/embeddings/hiceas_op_fulldata_winaware/` (BEATs+DAPT embeddings)
- `/workspace/embeddings/perch_hiceas_op2s_10s_win10/` (Perch embeddings)
- `/workspace/embeddings/cced2_fulldata/{knn_cced2.pkl,maha_cced2.pkl,cced2_norm.json}`
- `/workspace/embeddings/perch_ind_models/{knn_perch.pkl,maha_perch.pkl,cced2_norm_perch.json}`
