"""train_visual_pretrain.py — the data-rich vision plan: PRETRAIN the material-CNN on unlimited synthetic
material-cross-section images (physics_simulator), then FINE-TUNE on real (PS-augmented) well images, field-CV.
Tests whether curing the data-starvation (773 images -> synthetic + augmented) lets vision beat the PF. Pure GPU."""
import os, glob, time, numpy as np, pandas as pd, torch
import geology_visual as V
import physics_simulator as S
DEV = V.DEV

def img_from_frame(hw, tw):
    r = V.make_image(hw, tw)
    return None if r is None else (r[0], r[1], r[4])   # img(4,H,W), ridx, grid

def gen_synth(n, seed=0):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        hw, tw = S.to_frame(S.simulate(rng))
        r = img_from_frame(hw, tw)
        if r is not None: out.append(r)
    return out

def real_images(ps_fracs=(0.35,0.5,0.65)):
    folds = pd.read_csv("config/well_field_folds.csv").set_index("well").field_fold.to_dict()
    data = []
    for hp in sorted(glob.glob("input/train/*__horizontal_well.csv")):
        w = os.path.basename(hp).split("__")[0]
        if w not in folds: continue
        try: hw = pd.read_csv(hp); tw = pd.read_csv(f"input/train/{w}__typewell.csv")
        except Exception: continue
        if "TVT" not in hw.columns: continue
        n = len(hw); tvt = hw.TVT.to_numpy(float)
        for f in ps_fracs:                                   # PS-augmentation -> multiple images/well
            ps = int(f*n)
            if ps < 30 or n-ps < 20: continue
            h2 = hw.copy(); ti = tvt.copy(); ti[ps:] = np.nan; h2["TVT_input"] = ti
            r = img_from_frame(h2, tw)
            if r is not None: data.append((r[0], r[1], r[2], folds[w]))
    return data

def batches(items, bs, rng, has_fold=False):
    idx = np.arange(len(items)); rng.shuffle(idx)
    for k in range(0, len(idx), bs):
        ch = [items[i] for i in idx[k:k+bs]]
        xb = torch.tensor(np.stack([c[0] for c in ch]), device=DEV)
        yb = torch.tensor(np.stack([c[1] for c in ch]), device=DEV)
        yield xb, yb, ch

def train(net, data, iters_epochs, rng, lr=2e-3):
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4); net.train()
    for ep in range(iters_epochs):
        for xb, yb, _ in batches(data, 32, rng):
            pred = V.soft_argmax_row(net(xb)); loss = ((pred - yb)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

def rmse_on(net, items):
    net.eval(); P=[]; T=[]
    with torch.no_grad():
        for k in range(0, len(items), 32):
            ch = items[k:k+32]; xb = torch.tensor(np.stack([c[0] for c in ch]), device=DEV)
            pr = V.soft_argmax_row(net(xb)).cpu().numpy()
            for j,c in enumerate(ch):
                grid=c[2]; rr=np.clip(np.nan_to_num(pr[j],nan=(V.H-1)/2),0,V.H-1)
                lo=np.floor(rr).astype(int); fr=rr-lo; hi=np.clip(lo+1,0,V.H-1)
                tp=grid[lo,np.arange(V.W)]*(1-fr)+grid[hi,np.arange(V.W)]*fr
                tt=grid[np.clip(c[1],0,V.H-1).astype(int),np.arange(V.W)]
                P.append(tp);T.append(tt)
    return float(np.sqrt(np.mean((np.concatenate(P)-np.concatenate(T))**2)))

def main():
    t0=time.time(); rng=np.random.default_rng(0)
    print("generating synthetic images...", flush=True)
    synth = gen_synth(4000, seed=1); print(f"synthetic {len(synth)} imgs ({time.time()-t0:.0f}s)", flush=True)
    real = real_images(); print(f"real (PS-aug) {len(real)} imgs ({time.time()-t0:.0f}s)", flush=True)
    P=[];T=[]
    for vf in range(5):
        tr=[d for d in real if d[3]!=vf]; va=[d for d in real if d[3]==vf]
        net=V.PathCNN().to(DEV)
        train(net, synth, 3, rng)                            # PRETRAIN on synthetic (independent data)
        train(net, tr, 12, rng, lr=1e-3)                     # FINE-TUNE on real augmented
        P.append(rmse_on(net, va))
        print(f"fold{vf} vision-pretrained CV = {np.mean(P):.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"PRETRAINED VISION field-CV = {np.mean(P):.3f}  (material-CNN no-pretrain 37, PF 11.13, blend 10.5)")

if __name__ == "__main__":
    main()
