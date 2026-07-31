import numpy as np, pandas as pd, zarr
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter
ROOT=Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN=ROOT/"input/biohub-cell-tracking-during-development/train"; EW=ROOT/"learning/ensemble_work"
VOX=np.array([1.625,0.40625,0.40625]); GATE=4.5; MATCH=7.0
stems=sorted(p.stem for p in (EW/"pilkwang_nodes").glob("*.csv"))
def gt(s):
    g=TRAIN/f"{s}.geff/nodes/props"
    return pd.DataFrame({a:np.asarray(zarr.open(str(g/f"{a}/values"),mode="r")[:]) for a in "tzyx"})
def dog_peaks(vol):  # simple multi-scale DoG center detector (classical, no training)
    v=vol.astype(np.float32); lo,hi=np.quantile(v,0.5),np.quantile(v,0.999)
    v=np.clip((v-lo)/(hi-lo+1e-6),0,3)
    dog=gaussian_filter(v,1.0)-gaussian_filter(v,2.2)          # difference of gaussians
    mx=maximum_filter(dog,size=3,mode="nearest")
    pk=np.argwhere((dog==mx)&(dog>0.03))                       # local maxima above thresh
    return pk.astype(np.float32)
rows=[]
for s in stems:
    P=pd.read_csv(EW/"pilkwang_nodes"/f"{s}.csv"); C=pd.read_csv(EW/"canqiang_nodes"/f"{s}.csv"); G=gt(s)
    z=zarr.open(str(TRAIN/f"{s}.zarr/0"),mode="r")
    br=cr=dr=ngt=0
    for t,Gf in G.groupby("t"):
        Gv=Gf[["z","y","x"]].values
        if len(Gv)==0: continue
        Pf=P[P.t==t][["z","y","x"]].values; Cf=C[C.t==t][["z","y","x"]].values
        # DoG on the real frame at (1,4,4) then rescale to full-res coords
        vol=np.asarray(z[int(t),:,::4,::4]); dpk=dog_peaks(vol)
        if len(dpk): dpk=dpk*np.array([1,4,4])+np.array([0,1.5,1.5])
        def gate(N,ref):
            if len(N)==0: return N
            if len(ref)==0: return N
            d,_=cKDTree(ref*VOX).query(N*VOX,k=1); return N[d>GATE]
        def rec(N): 
            if len(N)==0: return 0
            d,_=cKDTree(N*VOX).query(Gv*VOX,k=1); return int((d<=MATCH).sum())
        Cg=gate(Cf,Pf)[:min(70,int(0.035*max(len(Pf),1)))]     # canqiang gated+capped
        F2=np.vstack([Pf,Cg]) if len(Pf) else Cg
        Dg=gate(dpk,F2)[:min(70,int(0.035*max(len(Pf),1)))]     # DoG gated vs (backbone+canqiang)
        F3=np.vstack([F2,Dg]) if len(F2) else Dg
        br+=rec(Pf); cr+=rec(F2); dr+=rec(F3); ngt+=len(Gv)
    rows.append({"stem":s,"nGT":ngt,"backbone":round(br/ngt,4),"+canq":round(cr/ngt,4),"+canq+DoG":round(dr/ngt,4)})
df=pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\nMEAN  backbone={df.backbone.mean():.4f}  +canqiang={df['+canq'].mean():.4f}  +canqiang+DoG={df['+canq+DoG'].mean():.4f}")
print(f"canqiang fusion gain=+{df['+canq'].mean()-df.backbone.mean():.4f} | DoG ADDS +{df['+canq+DoG'].mean()-df['+canq'].mean():.4f} | ceiling(1.0-fused)={1-df['+canq+DoG'].mean():.4f}")
