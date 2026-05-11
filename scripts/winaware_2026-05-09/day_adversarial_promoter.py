"""Day-adversarial Promoter — Tier 1 (a) cross-day fix.

Compares 3 promoter architectures on 7 HICEAS species under 5-fold GroupKFold
by deployment-day. Frozen BEATs+DAPT encoder (no fine-tune).

  (1) LR    : LogReg C=1.0 (paper Table 4 baseline)
  (2) MLP   : MLP species classifier alone (controls for capacity vs LogReg)
  (3) DANN  : Day-adversarial MLP (Ganin et al. 2016 JMLR, arXiv:1505.07818)

DANN architecture:
  emb (768) -> Linear -> BN -> ReLU -> Dropout -> [species head, day head via GRL]
  Loss: BCE_species + lambda(t) * CE_day  (gradient reversed for trunk)
  lambda schedule: 2/(1+exp(-10*p)) - 1   (Ganin §6.2 protocol)

For each fold:
  - day classifier learns to predict deployment-day from train embeddings
  - gradient reversal pushes trunk to MAX day CE = day-invariant features
  - species head learns to classify species from day-invariant features
  - At test: day head discarded, species head used on held-out days

Run: 3 seeds; report mean ± std across seeds (within-seed = 5-fold mean).
"""
import sys, glob, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/scripts/revision1")
import groupkfold_table4_eval as base

OP_DIR = "/workspace/embeddings/hiceas_op_fulldata_winaware"
S1706_DIR = "/workspace/embeddings/hiceas_1706_fulldata_winaware"
PART2_DIR = "/workspace/embeddings/hiceas_1706_part2_fulldata_winaware"
OUTDIR = Path("/workspace/scripts/winaware_2026-05-09/day_adversarial_promoter_2026-05-09")
OUTDIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024]
N_FOLDS = 5
N_EPOCHS = 100
HIDDEN = 128
LR = 1e-3
WD = 1e-3
DROPOUT = 0.3
BATCH = 256

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
        # LayerNorm (per-sample) instead of BatchNorm to avoid batch-size=1 crash
        # at the tail of the DataLoader. Ganin's original DANN used neither;
        # for our small-data regime LN is more stable than BN.
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
        h = self.trunk(x)
        return self.species_head(h).squeeze(-1)

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

def fit_mlp(X_tr, y_tr, X_te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLPPromoter(in_dim=X_tr.shape[1], hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = bce(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Xe = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        prob = torch.sigmoid(model(Xe)).cpu().numpy()
    return prob

def fit_dann(X_tr, y_tr, day_tr, X_te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    day_uniq = np.unique(day_tr)
    day_to_idx = {d: i for i, d in enumerate(day_uniq)}
    day_tr_idx = np.array([day_to_idx[d] for d in day_tr], dtype=np.int64)
    model = DANNPromoter(in_dim=X_tr.shape[1], hidden=HIDDEN,
                         n_days=len(day_uniq), dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    dt = torch.tensor(day_tr_idx, dtype=torch.long)
    ds = TensorDataset(Xt, yt, dt)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    model.train()
    for epoch in range(N_EPOCHS):
        lam = lambda_schedule(epoch, N_EPOCHS)
        for xb, yb, db in loader:
            xb, yb, db = xb.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
            sp_logit, day_logit = model(xb, lam=lam)
            loss = bce(sp_logit, yb) + ce(day_logit, db)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Xe = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        sp_logit, _ = model(Xe, lam=0.0)
        prob = torch.sigmoid(sp_logit).cpu().numpy()
    return prob

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

# Normalize embeddings (z-score per dim) for MLP training stability
all_embs = np.array(list(canon_embs.values()))
emb_mean = all_embs.mean(axis=0)
emb_std = all_embs.std(axis=0) + 1e-8
canon_embs_norm = {k: ((v - emb_mean) / emb_std).astype(np.float32) for k, v in canon_embs.items()}

neg_csv = os.path.join(base.CANON_DIR, "neg_all.csv")
results = []
print(f"\n{'Species':<28} {'LR (paper)':>13} {'MLP':>13} {'DANN':>13} {'Δ(DANN-LR)':>11} folds", flush=True)
print("-" * 100, flush=True)

for sp_name, pos_file in base.SPECIES:
    pos_csv = os.path.join(base.CANON_DIR, pos_file)
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    X, y, g = [], [], []
    for _, row in pos_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs_norm:
            X.append(canon_embs_norm[c]); y.append(1); g.append(base.deployment_day_group(c))
    for _, row in neg_df.iterrows():
        c = row.get("canon", "")
        if c in canon_embs_norm:
            X.append(canon_embs_norm[c]); y.append(0); g.append(base.deployment_day_group(c))
    X = np.array(X, dtype=np.float32); y = np.array(y); g = np.array(g)

    splitter = GroupKFold(n_splits=N_FOLDS)
    aucs_lr = []
    aucs_mlp_per_seed = {s: [] for s in SEEDS}
    aucs_dann_per_seed = {s: [] for s in SEEDS}

    for train_idx, test_idx in splitter.split(X, y, groups=g):
        if len(np.unique(y[test_idx])) < 2 or sum(y[train_idx] == 1) < 2:
            continue
        # (1) Baseline LogReg (deterministic, no seed needed beyond solver)
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        prob_lr = clf.predict_proba(X[test_idx])[:, 1]
        aucs_lr.append(roc_auc_score(y[test_idx], prob_lr))
        # (2) MLP and (3) DANN — multi-seed
        for seed in SEEDS:
            prob_mlp = fit_mlp(X[train_idx], y[train_idx], X[test_idx], seed=seed)
            aucs_mlp_per_seed[seed].append(roc_auc_score(y[test_idx], prob_mlp))
            prob_dann = fit_dann(X[train_idx], y[train_idx], g[train_idx],
                                  X[test_idx], seed=seed)
            aucs_dann_per_seed[seed].append(roc_auc_score(y[test_idx], prob_dann))

    if not aucs_lr:
        continue
    lr_mean = float(np.mean(aucs_lr)); lr_std = float(np.std(aucs_lr))
    mlp_means_per_seed = [float(np.mean(aucs_mlp_per_seed[s])) for s in SEEDS]
    dann_means_per_seed = [float(np.mean(aucs_dann_per_seed[s])) for s in SEEDS]
    mlp_m = float(np.mean(mlp_means_per_seed)); mlp_s = float(np.std(mlp_means_per_seed))
    dann_m = float(np.mean(dann_means_per_seed)); dann_s = float(np.std(dann_means_per_seed))

    res = {
        "species": sp_name,
        "lr_auc_mean": lr_mean, "lr_auc_std_folds": lr_std,
        "mlp_auc_mean": mlp_m, "mlp_auc_std_seeds": mlp_s,
        "dann_auc_mean": dann_m, "dann_auc_std_seeds": dann_s,
        "delta_dann_minus_lr": dann_m - lr_mean,
        "delta_dann_minus_mlp": dann_m - mlp_m,
        "n_folds": len(aucs_lr),
    }
    results.append(res)
    print(f"{sp_name:<28} {lr_mean:.3f}±{lr_std:.3f}  "
          f"{mlp_m:.3f}±{mlp_s:.3f}  {dann_m:.3f}±{dann_s:.3f}  "
          f"{res['delta_dann_minus_lr']:+.3f}     {res['n_folds']:>2}", flush=True)

df = pd.DataFrame(results)
df.to_csv(OUTDIR/"day_adversarial_compare.csv", index=False)
print(f"\nSaved {OUTDIR}/day_adversarial_compare.csv", flush=True)
