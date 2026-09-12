# CCED2 fit for the corrected step-127,641 encoder

These parameters were fitted on the held-out 56-class validation reference
embeddings (`n = 1,623`) produced by
`BEATs_DAPT_MAM_fixed_step127641.pt`. Only the aggregate normalisation and
threshold JSON files are supplied in this release. The fitted pickle files
are withheld: the kNN file retains all 1,623 individual reference embeddings,
and this release does not distribute fitted model files. The hashes below
identify the original fits for authorised verification, not public downloads.

| File | SHA-256 |
|---|---|
| `cced2_norm.json` | `4e7ae1a1348b4c227b3cdce421bfa11abe5cd86768b34383fb13cc9e949548a3` |
| `knn_cced2.pkl` | `ca1fa4d90c2d6c45aa0b2f7b0bb617e9d71718a7f524bd86839cf52f57d2535c` |
| `maha_cced2.pkl` | `bdbe7866d1f65f5830566b7ebc36404158855b71421543a56764c2f8988151d1` |
| `theta_cced2.json` | `6b42ebb4c810049eac3310bf45efd79f9f63b9173075fcf624916fc5206e7957` |

Exact CCED2 scoring cannot be reproduced from these two JSON files alone.
Authorised holders of the original fitted files can hash-check them with the
verifier's `--private-reference-dir` option. Never deserialize untrusted pickle
files; the verifier checks their bytes without loading them.
