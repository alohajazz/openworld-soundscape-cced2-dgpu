# HICEAS OP: Ops Point (FP/h ≤ 0.5)

## Post-filter config
```json
{
  "post_filter": {
    "use_offset": false,
    "minus4p5": false,
    "q_s_hat_near": 0.995,
    "topK_per_hour": 1
  }
}
```
## Reproduce
```bash
python - <<'PY'
import os, numpy as np, pandas as pd
from pathlib import Path
BASE=os.environ['HICEAS_OP_EMB']; ANN=os.environ['HICEAS_OP_ANN']
ev_p=Path(BASE)/'precision80_rule_ext/events_ops_final.csv'
pred_p=Path(BASE)/'predictions_op_quiet_base.csv'
ev=pd.read_csv(ev_p); pred=pd.read_csv(pred_p); ann=pd.read_csv(ANN)
ann.columns=[c.strip().lower() for c in ann.columns]
if 'base' not in ann.columns:
  from pathlib import Path as _P
  fcol=next((c for c in ann.columns if 'file'in c or 'name'in c or c=='base'), None)
  tcol=next((c for c in ann.columns if 'time'in c or c=='center' or 'stamp'in c), None)
  ann=pd.DataFrame({'base':ann[fcol].astype(str).apply(lambda p:_P(p).name), 'center':pd.to_numeric(ann[tcol],errors='coerce')}).dropna()
ann_bases=set(ann['base'].unique())
ev=ev[ev['base'].isin(ann_bases)].copy()
if 'base' not in pred.columns:
  from pathlib import Path as _P
  pred['base']=pred['path'].apply(lambda p:_P(p).name)
pred=pred[pred['base'].isin(ann_bases)].copy()
hop=(pred.groupby('base')['center_sec'].diff().dropna().round(3)).mode().iloc[0]
WPH=max(1,int(round(3600.0/float(hop)))); HOURS=len(pred)/WPH
gt={b:ann.loc[ann['base']==b,'center'].to_numpy() for b in ann_bases}
TP=FP=0
for _,r in ev.iterrows():
  g=gt.get(r['base'], [])
  import numpy as np
  if len(g)==0: continue
  d=np.abs(np.array(g)-float(r['event_center']))
  TP+=int(d.min()<=15.0); FP+=int(d.min()>15.0)
FN=len(ann)-TP
P=TP/(TP+FP+1e-9); R=TP/(TP+FN+1e-9); F=2*P*R/(P+R+1e-9); FPH=FP/HOURS
print(f'P={P:.3f} R={R:.3f} F1={F:.3f} FP/h={FPH:.3f} N={len(ev)} (tol=±15s)')
PY
```