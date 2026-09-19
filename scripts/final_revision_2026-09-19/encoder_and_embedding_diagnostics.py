"""Read-only recomputation of R1 diagnostics that earlier revisions deleted instead of updating.
Writes nothing under /workspace; prints one JSON object."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
import glob, json, hashlib, numpy as np, torch, joblib
from sklearn.metrics import roc_auc_score
E='/workspace/embeddings/'
OOD2='/tmp/dgpu-stage2-ood-20260919-2spn1uac/embeddings'
def load(d):
    ps=sorted(glob.glob(d+'/embeddings_*.npy')); assert ps,d
    return np.concatenate([np.load(p).astype(np.float64) for p in ps],0)
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
def diag(ind,ood):
    ci,co=ind.mean(0),ood.mean(0)
    lo=np.maximum(ind.min(0),ood.min(0)); hi=np.minimum(ind.max(0),ood.max(0))
    return {'n_ind':int(len(ind)),'n_ood':int(len(ood)),'dim':int(ind.shape[1]),
            'centroid_cosine':float(ci@co/np.linalg.norm(ci)/np.linalg.norm(co)),
            'l2norm_mean_ind':float(np.linalg.norm(ind,axis=1).mean()),'l2norm_mean_ood':float(np.linalg.norm(ood,axis=1).mean()),
            'dims_with_range_overlap':int((hi>=lo).sum())}
def cced2(x,mdir):
    knn=joblib.load(mdir+'/knn_cced2.pkl'); maha=joblib.load(mdir+'/maha_cced2.pkl'); norm=json.load(open(mdir+'/cced2_norm.json'))
    x=x.astype(np.float32); dist,_=knn['knn'].kneighbors(x,n_neighbors=knn.get('k',norm.get('k',50))); dk=dist.mean(1)
    mu=np.asarray(maha['mu'],dtype=x.dtype); P=np.asarray(maha['precision'],dtype=x.dtype); df=x-mu
    dm=np.sqrt(np.einsum('nd,nd->n',df,df@P)+1e-12)
    return (dk-norm['mk'])/norm['sk']+(dm-norm['mm'])/norm['sm']
def auroc(ind,ood,mdir):
    s=np.concatenate([cced2(ind,mdir),cced2(ood,mdir)]); y=np.r_[np.zeros(len(ind)),np.ones(len(ood))]
    return float(roc_auc_score(y,s))
def load_sd(p):
    o=torch.load(p,map_location='cpu',weights_only=False)
    if isinstance(o,dict):
        for k in ('model','beats','state_dict'):
            if k in o and isinstance(o[k],dict): return o[k]
    return o
def wdelta(pa,pb):
    a,b=load_sd(pa),load_sd(pb); mx=0.0;num=den=0.0;n=0
    for k,vb in b.items():
        if k not in a: continue
        va=a[k]
        if not torch.is_floating_point(vb) or va.shape!=vb.shape: continue
        d=va.float()-vb.float(); mx=max(mx,float(d.abs().max())); num+=float((d*d).sum()); den+=float((vb.float()**2).sum()); n+=1
    return {'tensors':n,'max_abs':mx,'rel_fro':(num/den)**0.5}
out={}
ind2=load(E+'ind_val_stage2'); ood2=load(OOD2)
out['stage2_ind_vs_ood']=diag(ind2,ood2)
out['stage2_auroc']=auroc(ind2,ood2,E+'cced2_stage2')
indp=load(E+'ind_val_pretrain')
out['pretrain_sets']={}
for name in ('hiceas_ood_pretrain','ood_pretrain'):
    try:
        o=load(E+name); s=json.load(open(E+name+'/summary.json')) if os.path.exists(E+name+'/summary.json') else {}
        out['pretrain_sets'][name]={'diag':diag(indp,o),'auroc':auroc(indp,o,E+'cced2_pretrain'),'summary':{k:s[k] for k in list(s)[:12]}}
    except Exception as ex: out['pretrain_sets'][name]={'error':repr(ex)}
PRE='/workspace/models/BEATs/BEATs_iter3_plus_AS2M.pt'; S1='/workspace/ckpts_dapt_mam_fixed/BEATs_DAPT_MAM_fixed_step127641.pt'; S2='/workspace/ckpts_dapt_mam_fixed_palaoa/BEATs_DAPT_MAM_fixed_palaoa_step6385.pt'
out['weights']={'stage1_vs_pretrain':wdelta(S1,PRE),'stage2_vs_stage1':wdelta(S2,S1),'stage2_vs_pretrain':wdelta(S2,PRE)}
out['sha256']={'S1':sha(S1),'S2':sha(S2),'PRE':sha(PRE),'ood2_emb':sha(OOD2+'/embeddings_000.npy'),'ind2_emb':sha(E+'ind_val_stage2/embeddings_000.npy')}
print('RESULT_JSON '+json.dumps(out))
