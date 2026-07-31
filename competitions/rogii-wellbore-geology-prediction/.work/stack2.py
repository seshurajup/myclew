import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
feat=pd.read_parquet('results/honest_feat.parquet')
folds=pd.read_csv('config/well_field_folds.csv')[['well','field_fold']]
feat=feat.merge(folds,on='well',how='left')
FEATS=['pf_dtvt','conf','md_since','z_since','z_rate','rowidx_since','beta','icpt',
       'zsig','n_known','gr','gr_res','a','tvt_ps','twspan','proj_dtvt']
y=feat['dtvt_true'].values
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
print('BASE pf_dtvt pooled %.3f'%rmse(feat.pf_dtvt,y),flush=True)
params=dict(objective='regression',learning_rate=0.05,num_leaves=63,min_child_samples=200,
            subsample=0.7,colsample_bytree=0.8,reg_lambda=10.,n_estimators=350,verbose=-1,n_jobs=6)
rng=np.random.RandomState(0); Xall=feat[FEATS].values
def run(axis):
    pred=np.full(len(feat),np.nan)
    if axis=='well':
        splits=list(GroupKFold(5).split(feat,y,feat.well.values))
    else:
        g=feat.field_fold.values
        splits=[(np.where(g!=f)[0],np.where(g==f)[0]) for f in sorted(feat.field_fold.unique())]
    for tr,va in splits:
        if len(tr)>600000: tr=rng.choice(tr,600000,replace=False)
        m=lgb.LGBMRegressor(**params); m.fit(Xall[tr],y[tr]); pred[va]=m.predict(Xall[va])
    r=rmse(pred,y); print(f'GBM[{axis}] pooled %.3f'%r,flush=True)
    for w in [0,.3,.5,.7,1.]:
        print(f'   blend w={w} %.3f'%rmse(w*pred+(1-w)*feat.pf_dtvt.values,y),flush=True)
    return pred
pw=run('well'); pf=run('field_fold')
m=lgb.LGBMRegressor(**params); tr=rng.choice(len(feat),600000,replace=False)
m.fit(Xall[tr],y[tr])
print('top feats:',sorted(zip(FEATS,m.feature_importances_),key=lambda x:-x[1])[:8],flush=True)
np.save('results/honest_gbm_oof_well.npy',pw); np.save('results/honest_gbm_oof_field.npy',pf)
print('SAVED',flush=True)
