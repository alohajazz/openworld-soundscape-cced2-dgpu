"""Unified Promoter audit: 3 predictors × 3 calibrations + ensemble + wd_AUC.

Conditions evaluated per species under 5-fold GroupKFold by deployment-day:

  Predictors:
    LR    = LogReg C=1.0 on raw embedding (paper Table 4 protocol)
    MLP   = 768->128 LayerNorm ReLU Dropout -> Linear (z-norm embedding)
    DANN  = MLP + day classifier head + GRL (z-norm embedding)

  Calibrations (applied to predictor's test-fold scores):
    raw   = no transformation
    z     = per-day z-score within test fold
    rank  = per-day rank-normalize to [0,1] within test fold

  Auxiliary:
    wd_AUC = mean of per-day within-day AUC (encoder ceiling)
    ensemble_rank = mean of per-day-rank-normalized (LR, MLP, DANN) scores

Reference:
  TENT (per-day rank/z calibration motivation): Wang et al. 2021 ICLR (arXiv:2006.10726)
  DANN: Ganin et al. 2016 JMLR (arXiv:1505.07818)

Output:
  unified_promoter_compare.csv  — wide table with all 9 conditions + wd_AUC + ensemble
  per_species_best.csv          — best (predictor, calibration) per species
"""
import sys, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base

OP_DIR = "/workspace/embeddings/op_stage2"
S1706_DIR = "/workspace/embeddings/1706species_stage2"
PART2_DIR = "/workspace/embeddings/1706part2_stage2"
OUTDIR = Path("/workspace/scripts/winaware_2026-05-09/unified_promoter_audit_stage2")
OUTDIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024]
N_FOLDS = 5
N_EPOCHS = 100
HIDDEN = 128
LR_INIT = 1e-3
WD = 1e-3
DROPOUT = 0.3
BATCH = 256


# ------------- Models -------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lam, None

class Trunk(nn.Module):
    def __init__(self, in_dim=768, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MLPPromoter(nn.Module):
    def __init__(self, in_dim=768, hidden=128, dropout=0.3):
        super().__init__()
        self.trunk = Trunk(in_dim, hidden, dropout)
        self.species_head = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.species_head(self.trunk(x)).squeeze(-1)

class DANNPromoter(nn.Module):
    def __init__(self, in_dim=768, hidden=128, n_days=8, dropout=0.3):
        super().__init__()
        self.trunk = Trunk(in_dim, hidden, dropout)
        self.species_head = nn.Linear(hidden, 1)
        self.day_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_days),
        )
    def forward(self, x, lam=1.0):
        h = self.trunk(x)
        sp_logit = self.species_head(h).squeeze(-1)
        h_rev = GradReverse.apply(h, lam)
        day_logit = self.day_head(h_rev)
        return sp_logit, day_logit

def lambda_schedule(epoch, n_epochs):
    p = epoch / max(n_epochs - 1, 1)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ------------- Trainers -------------
def fit_mlp(X_tr, y_tr, X_te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLPPromoter(in_dim=X_tr.shape[1], hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True)
    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = bce(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Xe = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        return torch.sigmoid(model(Xe)).cpu().numpy()

def fit_dann(X_tr, y_tr, day_tr, X_te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    day_uniq = np.unique(day_tr)
    day_to_idx = {d: i for i, d in enumerate(day_uniq)}
    day_tr_idx = np.array([day_to_idx[d] for d in day_tr], dtype=np.int64)
    model = DANNPromoter(in_dim=X_tr.shape[1], hidden=HIDDEN,
                         n_days=len(day_uniq), dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    dt = torch.tensor(day_tr_idx, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt, dt), batch_size=BATCH, shuffle=True)
    model.train()
    for epoch in range(N_EPOCHS):
        lam = lambda_schedule(epoch, N_EPOCHS)
        for xb, yb, db in loader:
            xb, yb, db = xb.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
            sp, day = model(xb, lam=lam)
            loss = bce(sp, yb) + ce(day, db)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Xe = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        sp, _ = model(Xe, lam=0.0)
        return torch.sigmoid(sp).cpu().numpy()


# ------------- Calibrations -------------
def per_day_zscore(scores, days):
    out = np.array(scores, dtype=np.float64).copy()
    for d in np.unique(days):
        m = days == d
        if m.sum() < 2: continue
        sd = out[m].std()
        if sd < 1e-9: continue
        out[m] = (out[m] - out[m].mean()) / sd
    return out

def per_day_rank(scores, days):
    out = np.array(scores, dtype=np.float64).copy()
    for d in np.unique(days):
        m = days == d
        n = m.sum()
        if n < 2: continue
        out[m] = (rankdata(out[m], method="average") - 1.0) / (n - 1.0)
    return out

def within_day_auc(scores, y, days):
    aucs = []
    for d in np.unique(days):
        m = days == d
        if len(np.unique(y[m])) < 2: continue
        aucs.append(roc_auc_score(y[m], scores[m]))
    return float(np.mean(aucs)) if aucs else float("nan")


# ------------- Load -------------
def load_dir(emb_dir):
    paths = sorted(glob.glob(f"{emb_dir}/embeddings_*.npy"))
    idxs = sorted(glob.glob(f"{emb_dir}/index_*.csv"))
    embs = np.concatenate([np.load(p) for p in paths]).astype("float32")
    idx = pd.concat([pd.read_csv(p) for p in idxs], ignore_index=True)
    return embs, idx

print(f"Device: {DEVICE}", flush=True)
print("Loading combined embeddings (OP + 1706 + Part2)...", flush=True)
op_e, op_i = load_dir(OP_DIR)
s1_e, s1_i = load_dir(S1706_DIR)
p2_e, p2_i = load_dir(PART2_DIR)
emb = np.concatenate([op_e, s1_e, p2_e])
idx = pd.concat([op_i, s1_i, p2_i], ignore_index=True)
canon_embs = base.build_canon_embeddings(emb, idx)
print(f"Total canons: {len(canon_embs)}", flush=True)

# Pre-compute z-normalized embeddings for MLP/DANN training stability
all_embs_arr = np.array(list(canon_embs.values()), dtype=np.float32)
emb_mean = all_embs_arr.mean(axis=0)
emb_std = all_embs_arr.std(axis=0) + 1e-8
canon_embs_norm = {k: ((v - emb_mean) / emb_std).astype(np.float32)
                   for k, v in canon_embs.items()}

neg_csv = os.path.join(base.CANON_DIR, "neg_all.csv")
all_results = []

print(f"\nProcessing {len(base.SPECIES)} species, 5-fold GroupKFold by day, "
      f"3 seeds for MLP/DANN...", flush=True)

for sp_name, pos_file in base.SPECIES:
    pos_csv = os.path.join(base.CANON_DIR, pos_file)
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    Xraw, Xnorm, y, g = [], [], [], []
    for _, row in pos_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs:
            Xraw.append(canon_embs[c])
            Xnorm.append(canon_embs_norm[c])
            y.append(1)
            g.append(base.deployment_day_group(c))
    for _, row in neg_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs:
            Xraw.append(canon_embs[c])
            Xnorm.append(canon_embs_norm[c])
            y.append(0)
            g.append(base.deployment_day_group(c))
    Xraw = np.array(Xraw, dtype=np.float32)
    Xnorm = np.array(Xnorm, dtype=np.float32)
    y = np.array(y); g = np.array(g)

    # Conditions: predictor in {LR, MLP, DANN} × calib in {raw, z, rank}
    # plus wd_AUC per predictor and ensemble
    fold_results = {
        f"{p}_{c}": [] for p in ("lr", "mlp", "dann") for c in ("raw", "z", "rank")
    }
    fold_results.update({f"{p}_wd": [] for p in ("lr", "mlp", "dann")})
    fold_results["ensemble_rank"] = []

    splitter = GroupKFold(n_splits=N_FOLDS)
    for train_idx, test_idx in splitter.split(Xraw, y, groups=g):
        if len(np.unique(y[test_idx])) < 2 or sum(y[train_idx] == 1) < 2:
            continue
        days_test = g[test_idx]; y_test = y[test_idx]

        # LR on raw embedding (paper Table 4 protocol)
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(Xraw[train_idx], y[train_idx])
        s_lr = clf.predict_proba(Xraw[test_idx])[:, 1]

        # MLP and DANN on z-normalized embedding (training stability), average over seeds
        s_mlp_seeds, s_dann_seeds = [], []
        for seed in SEEDS:
            s_mlp_seeds.append(fit_mlp(Xnorm[train_idx], y[train_idx],
                                       Xnorm[test_idx], seed))
            s_dann_seeds.append(fit_dann(Xnorm[train_idx], y[train_idx],
                                         g[train_idx], Xnorm[test_idx], seed))
        s_mlp = np.mean(s_mlp_seeds, axis=0)
        s_dann = np.mean(s_dann_seeds, axis=0)

        for name, s in [("lr", s_lr), ("mlp", s_mlp), ("dann", s_dann)]:
            fold_results[f"{name}_raw"].append(roc_auc_score(y_test, s))
            fold_results[f"{name}_z"].append(roc_auc_score(y_test,
                                                            per_day_zscore(s, days_test)))
            fold_results[f"{name}_rank"].append(roc_auc_score(y_test,
                                                              per_day_rank(s, days_test)))
            fold_results[f"{name}_wd"].append(within_day_auc(s, y_test, days_test))

        # Ensemble: mean of per-day-rank-normalized scores
        ens = (per_day_rank(s_lr, days_test)
               + per_day_rank(s_mlp, days_test)
               + per_day_rank(s_dann, days_test)) / 3.0
        fold_results["ensemble_rank"].append(roc_auc_score(y_test, ens))

    if not fold_results["lr_raw"]:
        continue

    res = {"species": sp_name, "n_folds": len(fold_results["lr_raw"])}
    for k, vs in fold_results.items():
        valid = [v for v in vs if not np.isnan(v)]
        res[k] = float(np.mean(valid)) if valid else float("nan")
    all_results.append(res)

    # Print compact line
    print(f"\n{sp_name} (folds={res['n_folds']}):", flush=True)
    print(f"           raw    z      rank   wd_AUC", flush=True)
    for p in ("lr", "mlp", "dann"):
        print(f"  {p:<6}: {res[f'{p}_raw']:.3f}  {res[f'{p}_z']:.3f}  "
              f"{res[f'{p}_rank']:.3f}  {res[f'{p}_wd']:.3f}", flush=True)
    print(f"  ensemble_rank: {res['ensemble_rank']:.3f}", flush=True)

# Save wide
df = pd.DataFrame(all_results)
df.to_csv(OUTDIR / "unified_promoter_compare.csv", index=False)
print(f"\nSaved {OUTDIR}/unified_promoter_compare.csv", flush=True)

# Best per species
best_rows = []
for r in all_results:
    cells = {f"{p}_{c}": r[f"{p}_{c}"]
             for p in ("lr", "mlp", "dann") for c in ("raw", "z", "rank")}
    cells["ensemble_rank"] = r["ensemble_rank"]
    best_key = max(cells, key=cells.get)
    best_rows.append({
        "species": r["species"],
        "best_condition": best_key,
        "best_auc": cells[best_key],
        "lr_raw_baseline": r["lr_raw"],
        "delta_vs_paper": cells[best_key] - r["lr_raw"],
        "wd_AUC_ceiling": max(r["lr_wd"], r["mlp_wd"], r["dann_wd"]),
    })
pd.DataFrame(best_rows).to_csv(OUTDIR / "per_species_best.csv", index=False)
print(f"\n=== Per-species best ===", flush=True)
for r in best_rows:
    print(f"  {r['species']:<28} best={r['best_condition']:<14} "
          f"AUC={r['best_auc']:.3f}  Δ={r['delta_vs_paper']:+.3f}  "
          f"(wd_AUC={r['wd_AUC_ceiling']:.3f})", flush=True)
