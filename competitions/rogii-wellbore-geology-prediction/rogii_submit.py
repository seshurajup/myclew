"""rogii_submit — self-contained, offline, 2xT4-ready. PS-augmented bidirectional-GRU aligner (the verified
honest ~6.1 engine). FULL-LENGTH sequences (no truncation) so every hidden row is predicted. Two modes:
  --cv      field-grouped OOF on train/ (faithful, full-length)  → the honest CV number
  --submit  train an ENSEMBLE on ALL train/ wells, predict test/ wells, write submission.csv
Inputs available at inference (no leak): GR, Z, MD, TVT_input (known prefix). Target = TVT (hidden rows)."""
import os, sys, glob, time, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn

def find_data():
    for c in ["input", "/kaggle/input/rogii-wellbore-geology-prediction",
              "/kaggle/input/rogii-wellbore-geology-prediction/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c, "train")) or os.path.isdir(os.path.join(c, "test")):
            return c
    # discover by content
    for root, dirs, files in os.walk("/kaggle/input"):
        if any(f.endswith("__horizontal_well.csv") for f in files):
            return os.path.dirname(root) if os.path.basename(root) in ("train","test") else root
    return "input"

FEATS = 6  # [GR_norm, is_known, known_drift/50, md_since_ps/1000, Z_norm, dZ]

def load_well(hw):
    md=hw["MD"].to_numpy(float); gr=hw["GR"].to_numpy(float); z=hw["Z"].to_numpy(float)
    tvti=hw["TVT_input"].to_numpy(float)
    tvt=hw["TVT"].to_numpy(float) if "TVT" in hw.columns else None
    if not np.isfinite(tvti).any(): return None
    ps=int(np.where(np.isfinite(tvti))[0][-1])
    return dict(md=md,gr=gr,z=z,tvti=tvti,tvt=tvt,n=len(md),real_ps=ps)

def feats(d, ps, tvt_src=None):
    md,gr,z=d["md"],d["gr"],d["z"]; n=d["n"]; mdps=md[ps]
    src = tvt_src if tvt_src is not None else d["tvt"]        # train uses full TVT; test uses TVT_input prefix
    tvtps = (src[ps] if src is not None and np.isfinite(src[ps]) else d["tvti"][ps])
    kn=np.arange(n)<=ps
    gm=gr[kn]; mu=np.nanmean(gm); sd=np.nanstd(gm)+1e-6; grn=np.nan_to_num((gr-mu)/sd)
    known_tvt = src if src is not None else d["tvti"]
    kdt=np.where(kn, np.nan_to_num(known_tvt-tvtps), 0.0)
    mds=(md-mdps)/1000.0
    zc=(z-np.nanmean(z[kn]))/(np.nanstd(z[kn])+1e-6); dz=np.gradient(zc)
    X=np.stack([grn,kn.astype(float),kdt/50.0,mds,zc,dz],1).astype(np.float32)
    y=(d["tvt"]-tvtps).astype(np.float32) if d["tvt"] is not None else None
    return X,y,(~kn),float(tvtps)

class BiGRU(nn.Module):
    def __init__(s,d=FEATS,h=112):
        super().__init__(); s.g=nn.GRU(d,h,2,batch_first=True,bidirectional=True,dropout=0.1)
        s.h=nn.Sequential(nn.Linear(2*h,64),nn.GELU(),nn.Linear(64,1))
    def forward(s,x): return s.h(s.g(x)[0]).squeeze(-1)

PS_FRACS=[0.4,0.5,0.6,0.7,0.8]
def make_samples(D, wells, augment):
    out=[]
    for w in wells:
        d=D[w]; n=d["n"]
        pss=[int(f*n) for f in PS_FRACS] if augment else [d["real_ps"]]
        for p in pss:
            if not (20<p<n-5): continue
            X,y,hid,_=feats(d,p)
            if y is None: continue
            out.append((X,y,hid))
    return out

def batched(samp, bs, dev, shuf, rng, tok_cap=200000):
    idx=np.arange(len(samp))
    if shuf: rng.shuffle(idx)
    i=0
    while i<len(idx):
        chunk=[]; toks=0; L=0
        while i<len(idx) and len(chunk)<bs:
            s=samp[idx[i]]; L2=max(L,len(s[0]))
            if chunk and L2*(len(chunk)+1)>tok_cap: break
            chunk.append(s); L=L2; i+=1
        xb=torch.zeros(len(chunk),L,chunk[0][0].shape[1]); yb=torch.zeros(len(chunk),L); mb=torch.zeros(len(chunk),L)
        for j,(X,y,h) in enumerate(chunk):
            l=len(X); xb[j,:l]=torch.from_numpy(X); yb[j,:l]=torch.from_numpy(y); mb[j,:l]=torch.from_numpy(h.astype("float32"))
        yield xb.to(dev),yb.to(dev),mb.to(dev)

def train_one(trs, dev, seed, epochs=25):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    net=BiGRU().to(dev); opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-4)
    steps=epochs*max(1,len(trs)//24); sch=torch.optim.lr_scheduler.OneCycleLR(opt,2e-3,total_steps=steps+epochs); net.train()
    for ep in range(epochs):
        for xb,yb,mb in batched(trs,24,dev,True,rng):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                pr=net(xb); loss=((pr-yb)**2*mb).sum()/mb.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval(); return net

def predict(net, d, dev):
    ps=d["real_ps"]; X,_,hid,tvtps=feats(d,ps,tvt_src=d["tvti"])
    xb=torch.from_numpy(X[None]).to(dev)
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        pr=net(xb).float().cpu().numpy()[0]
    tvt=tvtps+pr                      # dtvt -> absolute TVT
    return tvt, hid

def cv(data_dir, dev):
    D={}
    for hp in sorted(glob.glob(os.path.join(data_dir,"train","*__horizontal_well.csv"))):
        w=os.path.basename(hp).split("__")[0]; hw=pd.read_csv(hp)
        d=load_well(hw)
        if d and d["tvt"] is not None and d["n"]>=80: D[w]=d
    folds=pd.read_csv("config/well_field_folds.csv").set_index("well").field_fold.to_dict()
    wells=[w for w in D if w in folds]
    P=[];T=[]; t0=time.time()
    for vf in range(5):
        tr=[w for w in wells if folds[w]!=vf]; va=[w for w in wells if folds[w]==vf]
        net=train_one(make_samples(D,tr,True),dev,0)
        for w in va:
            tvt,hid=predict(net,D[w],dev); P.append(tvt[hid]); T.append((D[w]["tvt"])[hid])
        print(f"fold{vf} cum={np.sqrt(np.mean((np.concatenate(P)-np.concatenate(T))**2)):.3f} ({time.time()-t0:.0f}s)",flush=True)
    print("FULL-LENGTH field-CV RMSE:", np.sqrt(np.mean((np.concatenate(P)-np.concatenate(T))**2)))

def submit(data_dir, dev, n_seeds=10):
    D={}
    for hp in sorted(glob.glob(os.path.join(data_dir,"train","*__horizontal_well.csv"))):
        w=os.path.basename(hp).split("__")[0]; d=load_well(pd.read_csv(hp))
        if d and d["tvt"] is not None and d["n"]>=80: D[w]=d
    trs=make_samples(D,list(D),True)
    T={}
    for hp in sorted(glob.glob(os.path.join(data_dir,"test","*__horizontal_well.csv"))):
        w=os.path.basename(hp).split("__")[0]; d=load_well(pd.read_csv(hp))
        if d: T[w]=d
    ngpu=torch.cuda.device_count(); rows={}
    t0=time.time()
    for s in range(n_seeds):
        dv=f"cuda:{s%max(ngpu,1)}" if torch.cuda.is_available() else "cpu"
        net=train_one(trs,dv,s)
        for w,d in T.items():
            tvt,hid=predict(net,d,dv); pos=np.where(hid)[0]
            for i,p in enumerate(pos):
                rows.setdefault(f"{w}_{p}",[]).append(tvt[p])
        print(f"seed{s} done ({time.time()-t0:.0f}s)",flush=True)
    out=pd.DataFrame({"id":list(rows),"tvt":[float(np.mean(v)) for v in rows.values()]})
    out.to_csv("submission.csv",index=False)
    print("wrote submission.csv", out.shape)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",default="cv"); ap.add_argument("--seeds",type=int,default=10)
    a=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"; dd=find_data()
    print("data_dir",dd,"device",dev,"gpus",torch.cuda.device_count())
    (cv if a.mode=="cv" else lambda x,y: submit(x,y,a.seeds))(dd,dev)
