"""Count annotated upcall events in each training fold of the FRDR label-budget analysis.
Same inputs and the same GroupKFold(n_splits=5) by file as frdr_label_efficiency_param.py; read-only."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
MAN = "/workspace/data/externaldata/frdr_upcall/manifests/frdr_B_continuous_hop2s.csv"
ANN = "/workspace/data/externaldata/frdr_upcall/raw/data/continuous/dataset_B/annotations_B_cont.csv"
m = pd.read_csv(MAN); m["base"] = m["path"].apply(lambda p: Path(p).name)
ann = pd.read_csv(ANN, sep=";"); ann.columns = [c.strip().lower() for c in ann.columns]
fc = next(c for c in ann.columns if "file" in c); ann["base"] = ann[fc].apply(lambda p: Path(str(p)).name)
groups = m["base"].values
folds = list(GroupKFold(n_splits=5).split(np.zeros(len(m)), np.zeros(len(m)), groups=groups))
out = {"manifest_rows": int(len(m)), "files": int(m["base"].nunique()), "annotated_files": int(ann["base"].nunique()), "events_total": int(len(ann)), "folds": []}
for k, (tr, te) in enumerate(folds):
    trf, tef = set(np.unique(groups[tr])), set(np.unique(groups[te])); assert not (trf & tef)
    ntr, nte = int(ann["base"].isin(trf).sum()), int(ann["base"].isin(tef).sum()); assert ntr + nte == len(ann)
    out["folds"].append({"fold": k, "train_files": len(trf), "test_files": len(tef), "train_events": ntr, "test_events": nte, "share_300": round(300 / ntr, 4)})
tr = [f["train_events"] for f in out["folds"]]
out["train_events_mean"] = round(float(np.mean(tr)), 1); out["share_300_of_mean"] = round(300 / float(np.mean(tr)), 4)
out["share_300_min_max"] = [round(300 / max(tr), 4), round(300 / min(tr), 4)]; out["share_300_of_all_1157"] = round(300 / len(ann), 4)
print("RESULT_JSON " + json.dumps(out))
