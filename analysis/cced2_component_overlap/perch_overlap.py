import glob, json
import numpy as np, joblib
from pathlib import Path

emb_files = sorted(glob.glob("/workspace/embeddings/perch_hiceas_op2s_10s_win10/embeddings_*.npy"))
E = np.concatenate([np.load(f).astype(np.float32) for f in emb_files], axis=0)
print(f"Perch HICEAS OP win10: n_windows = {len(E)}, dim = {E.shape[1]}")

m = Path("/workspace/embeddings/perch_ind_models")
knn = joblib.load(m / "knn_perch.pkl")
maha = joblib.load(m / "maha_perch.pkl")
cfg = json.load(open(m / "cced2_norm_perch.json"))
k_eff = knn.get("k", cfg["k"])
dists, _ = knn["knn"].kneighbors(E, n_neighbors=k_eff)
d_knn = dists.mean(axis=1)
mu = maha["mu"].astype(E.dtype); P = maha["precision"].astype(E.dtype)
diff = E - mu
d_maha = np.sqrt(np.einsum("nd,nd->n", diff, diff @ P) + 1e-12)
dk = (d_knn - cfg["mk"]) / cfg["sk"]
dm = (d_maha - cfg["mm"]) / cfg["sm"]
cced2 = dk + dm

n = len(E); n_select = int(np.ceil(n * 0.01))
top_knn = set(np.argsort(-dk)[:n_select].tolist())
top_maha = set(np.argsort(-dm)[:n_select].tolist())
top_cced = set(np.argsort(-cced2)[:n_select].tolist())

print(f"\nn_total = {n}, n_select (top 1%) = {n_select}")
print(f"--- Perch HICEAS OP pairwise overlaps ---")
for n1, s1, n2, s2 in [
    ("kNN", top_knn, "Maha", top_maha),
    ("kNN", top_knn, "CCED2", top_cced),
    ("Maha", top_maha, "CCED2", top_cced),
]:
    inter = len(s1 & s2); uni = len(s1 | s2)
    jacc = inter / uni
    print(f"  {n1} vs {n2}: shared={inter}, union={uni}, Jaccard={jacc:.3f}, overlap_pct={inter/n_select:.1%}")
print(f"\n3-way intersection = {len(top_knn & top_maha & top_cced)}")
print(f"\nPearson r(dk, dm) = {np.corrcoef(dk, dm)[0,1]:.3f}")
print(f"Pearson r(dk, cced2) = {np.corrcoef(dk, cced2)[0,1]:.3f}")
print(f"Pearson r(dm, cced2) = {np.corrcoef(dm, cced2)[0,1]:.3f}")
