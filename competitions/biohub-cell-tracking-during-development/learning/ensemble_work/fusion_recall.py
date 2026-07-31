import numpy as np, pandas as pd, zarr
from pathlib import Path
from scipy.spatial import cKDTree
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT/"input/biohub-cell-tracking-during-development/train"
EW = ROOT/"learning/ensemble_work"
VOX = np.array([1.625,0.40625,0.40625]); GATE=4.5; MATCH=7.0
stems=sorted(p.stem for p in (EW/"pilkwang_nodes").glob("*.csv"))
def gt(stem):
    g=TRAIN/f"{stem}.geff/nodes/props"
    return pd.DataFrame({a:np.asarray(zarr.open(str(g/f"{a}/values"),mode="r")[:]) for a in "tzyx"})
rows=[]
for s in stems:
    P=pd.read_csv(EW/"pilkwang_nodes"/f"{s}.csv"); C=pd.read_csv(EW/"canqiang_nodes"/f"{s}.csv"); G=gt(s)
    brec=frec=ngt=added=0
    for t,Gf in G.groupby("t"):
        Pf=P[P.t==t][["z","y","x"]].values; Cf=C[C.t==t][["z","y","x"]].values; Gv=Gf[["z","y","x"]].values
        if len(Gv)==0: continue
        # conservative fusion: canqiang centers > GATE µm from any backbone node
        keep=Cf
        if len(Pf) and len(Cf):
            d,_=cKDTree(Pf*VOX).query(Cf*VOX,k=1); keep=Cf[d>GATE]
        cap=min(70,int(round(0.035*max(len(Pf),1))))
        keep=keep[:cap]
        F=np.vstack([Pf,keep]) if len(Pf) else keep
        # GT recall within MATCH µm
        def rec(N): 
            if len(N)==0: return 0
            d,_=cKDTree(N*VOX).query(Gv*VOX,k=1); return int((d<=MATCH).sum())
        brec+=rec(Pf); frec+=rec(F); ngt+=len(Gv); added+=len(keep)
    rows.append({"stem":s,"emb":s[:4],"nGT":ngt,"backbone_rec":round(brec/ngt,4),
                 "fused_rec":round(frec/ngt,4),"gain":round((frec-brec)/ngt,4),"added":added})
df=pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\nMEAN backbone recall={df.backbone_rec.mean():.4f}  fused recall={df.fused_rec.mean():.4f}  "
      f"GAIN=+{(df.fused_rec.mean()-df.backbone_rec.mean()):.4f}")
print(f"GT cells recovered by fusion: {int((df.fused_rec*df.nGT).sum()-(df.backbone_rec*df.nGT).sum())}  "
      f"| total centers added: {df.added.sum()}  ({df.added.sum()/(df.added.sum()+1):.1%} kept sparse)")
