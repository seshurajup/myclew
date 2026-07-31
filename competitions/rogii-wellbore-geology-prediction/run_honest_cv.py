import sys, importlib.util, warnings, time; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0,'fleet_agents')
spec=importlib.util.spec_from_file_location("activity","fleet_agents/activity.py")
act=importlib.util.module_from_spec(spec); spec.loader.exec_module(act)
S="rogii-wellbore-geology-prediction"
def beat(a,s,d=None,k="running"):
    try: act.beat(S,a,s,detail=d,kind=k,run="honest-cv")
    except Exception: pass
    print(f"[{a}] {s}",flush=True)
import geology_honest as H
beat("engine","honest engine run_oof START — affine-cal + multi-scale NCC + PF + per-well selector","full 773 wells, well+field CV",k="running")
t0=time.time()
oof=H.run_oof("input/train", training=True, limit=None, use_selector=True)
oof.to_csv("results/honest_engine_oof.csv",index=False)
dt=time.time()-t0
# CV
def pooled(df,col='pred_dtvt',tru='dtvt_true'):
    return float(np.sqrt(((df[col]-df[tru])**2).mean()))
cols=oof.columns.tolist()
beat("engine",f"run_oof done in {dt/60:.1f}min — cols {cols[:6]}","computing CV",k="running")
# find prediction & truth columns robustly
pcol=[c for c in cols if 'pred' in c and 'dtvt' in c] or [c for c in cols if c.endswith('_dtvt') and c!='dtvt_true']
tcol=[c for c in cols if c in ('dtvt_true','true_dtvt')]
pc=pcol[0] if pcol else cols[2]; tc=tcol[0] if tcol else 'dtvt_true'
cv=float(np.sqrt(((oof[pc]-oof[tc])**2).mean()))
# field-grouped: pooled is per-row so same number; report per-well and field breakdown
folds=pd.read_csv("config/well_field_folds.csv")[["well","field_fold"]]
m=oof.merge(folds,on="well",how="left") if "well" in cols else oof
beat("engine",f"HONEST ENGINE CV (pooled) = {cv:.3f}  [PF baseline 11.13, const 15.91, target <6]",
     f"pred={pc} truth={tc}, {len(oof)} rows, {dt/60:.1f}min",k="result")
# journal
sp=importlib.util.spec_from_file_location("db","fleet_agents/db.py"); db=importlib.util.module_from_spec(sp); sp.loader.exec_module(db)
db.upsert_journal(S,[dict(exp="honest_engine_full",cv=round(cv,4),
  desc=f"Full honest engine: affine-cal + multi-scale NCC + PF + selector (pooled CV, {dt/60:.0f}min)",
  change="honest",trn_set="full",stage="engine",kept=cv<11.13,ts=None)])
print(f"=== HONEST ENGINE CV {cv:.4f} ({dt/60:.1f}min) ===",flush=True)
