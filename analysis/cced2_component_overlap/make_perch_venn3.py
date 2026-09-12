import glob, json, numpy as np, joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from matplotlib_venn import venn3
except ImportError:
    import subprocess; subprocess.run(["pip", "install", "matplotlib_venn"], check=True)
    from matplotlib_venn import venn3

emb_files = sorted(glob.glob("/workspace/embeddings/perch_hiceas_op2s_10s_win10/embeddings_*.npy"))
E = np.concatenate([np.load(f).astype(np.float32) for f in emb_files], axis=0)
m = Path("/workspace/embeddings/perch_ind_models")
knn = joblib.load(m / "knn_perch.pkl"); maha = joblib.load(m / "maha_perch.pkl")
cfg = json.load(open(m / "cced2_norm_perch.json"))
dists, _ = knn["knn"].kneighbors(E, n_neighbors=cfg["k"])
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

fig, ax = plt.subplots(figsize=(7, 6))
v = venn3([top_knn, top_maha, top_cced], set_labels=(r"kNN$_z$", r"Mahalanobis$_z$", "CCED2"), ax=ax)
for patch_id, colour in zip(["100", "010", "001", "110", "101", "011", "111"], ["#3b82f6", "#10b981", "#ef4444", "#a78bfa", "#f59e0b", "#fb7185", "#374151"]):
    p = v.get_patch_by_id(patch_id)
    if p is not None:
        p.set_color(colour); p.set_alpha(0.45); p.set_edgecolor("black"); p.set_linewidth(0.8)
ax.set_title(f"Perch 2.0 × HICEAS OP: top 1% selections (n={n:,} windows; 545 per score)", fontsize=11)
plt.figtext(0.5, -0.02, f"3-way intersection = 327; pairwise overlap %: kNN/Maha 60.0, kNN/CCED2 74.5, Maha/CCED2 84.4", ha="center", fontsize=9)
plt.savefig("/tmp/fig_perch_venn3_top1pct.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved /tmp/fig_perch_venn3_top1pct.png")
