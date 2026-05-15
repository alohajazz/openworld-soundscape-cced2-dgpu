"""Analyze the 168 windows that differ between top-1% selections of -kNN_z and -CCED2.
Compute qualitative characterization (mean z-distances, species distribution if available)
and generate Venn diagram + scatter plot."""
import sys, glob, json
sys.path.insert(0, "/workspace/scripts")
import numpy as np
import joblib
from pathlib import Path

emb_files = sorted(glob.glob("/workspace/embeddings/hiceas_op_fulldata_winaware/embeddings_*.npy"))
E = np.concatenate([np.load(f).astype(np.float32) for f in emb_files], axis=0)

# Also load the per-window index (file + center_sec + species label) for species characterization
import csv
idx_files = sorted(glob.glob("/workspace/embeddings/hiceas_op_fulldata_winaware/index_*.csv"))
rows = []
for f in idx_files:
    with open(f) as fh:
        r = csv.reader(fh)
        header = next(r)
        for row in r:
            rows.append(row)
print(f"index header: {header}")
print(f"n_index_rows: {len(rows)}, n_emb: {len(E)}")

model_dir = Path("/workspace/embeddings/cced2_fulldata")
knn = joblib.load(model_dir / "knn_cced2.pkl")
maha = joblib.load(model_dir / "maha_cced2.pkl")
cfg = json.load(open(model_dir / "cced2_norm.json"))
k_eff = knn["k"]
dists, _ = knn["knn"].kneighbors(E, n_neighbors=k_eff)
d_knn = dists.mean(axis=1)
mu = maha["mu"].astype(E.dtype); P = maha["precision"].astype(E.dtype)
diff = E - mu
d_maha = np.sqrt(np.einsum("nd,nd->n", diff, diff @ P) + 1e-12)
dk = (d_knn - cfg["mk"]) / cfg["sk"]
dm = (d_maha - cfg["mm"]) / cfg["sm"]
cced2 = dk + dm

n_total = len(E); n_select = int(np.ceil(n_total * 0.01))
top_knn = np.argsort(-dk)[:n_select]
top_cced = np.argsort(-cced2)[:n_select]
set_knn = set(top_knn.tolist()); set_cced = set(top_cced.tolist())
shared = set_knn & set_cced
knn_only = set_knn - set_cced
cced_only = set_cced - set_knn
print(f"\nn_total={n_total} n_select={n_select}")
print(f"|shared|={len(shared)} |knn_only|={len(knn_only)} |cced_only|={len(cced_only)}")

# Qualitative metrics
def stats(idxs, name):
    idxs = np.array(list(idxs))
    print(f"\n--- {name} (n={len(idxs)}) ---")
    print(f"  mean dk_z = {dk[idxs].mean():.3f} (std {dk[idxs].std():.3f})")
    print(f"  mean dm_z = {dm[idxs].mean():.3f} (std {dm[idxs].std():.3f})")
    print(f"  mean cced2 = {cced2[idxs].mean():.3f}")
    return idxs

idx_knn_only = stats(knn_only, "kNN-only")
idx_cced_only = stats(cced_only, "CCED2-only")
idx_shared = stats(shared, "shared (top-1% both)")

# Species distribution (best effort from index)
def get_field(i, name):
    if name in header:
        return rows[i][header.index(name)]
    return None
species_field = None
for cand in ('species','label','class','species_label'):
    if cand in header:
        species_field = cand; break
print(f"\nspecies field: {species_field}")
if species_field is not None:
    from collections import Counter
    for name, idxs in [("kNN-only", idx_knn_only), ("CCED2-only", idx_cced_only), ("shared", idx_shared)]:
        species = [get_field(i, species_field) for i in idxs]
        c = Counter(species).most_common(10)
        print(f"{name}: top species = {c}")

# Save figure data
np.savez('/tmp/fig_data.npz',
    dk_z=dk, dm_z=dm, cced2=cced2,
    top_knn_idx=top_knn, top_cced_idx=top_cced,
    knn_only=np.array(sorted(knn_only)),
    cced_only=np.array(sorted(cced_only)),
    shared=np.array(sorted(shared)))
print("\nSaved /tmp/fig_data.npz")
