"""train_statespace.py — train the proof-grounded StateSpaceNet (nn_statespace) on real rogii data, field-CV.
Pure GPU torch. Features per row (leak-free, available at inference): GR, Z, dZ, md_since_ps, is_known,
known-dtvt(0 in hidden). Per-well FiLM context: heel GR mean/std, dip0, and pf_noise_id log q_s, log q_v.
IRW integration head predicts increments cumulatively summed from the heel anchor; Student-t NLL on hidden rows.
Full-length sequences (no truncation). Reports field-grouped OOF RMSE."""
import os, sys, glob, time, types, importlib.util
import pandas as pd, torch
import nn_statespace as M
DEV = M.DEV

FA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet_agents")
if "fleet_agents" not in sys.modules:
    pkg = types.ModuleType("fleet_agents"); pkg.__path__ = [FA]; sys.modules["fleet_agents"] = pkg
sp = importlib.util.spec_from_file_location("fleet_agents.pf_noise_id", os.path.join(FA, "pf_noise_id.py"))
nid = importlib.util.module_from_spec(sp); sys.modules["fleet_agents.pf_noise_id"] = nid; sp.loader.exec_module(nid)

def T(x): return torch.tensor(x, dtype=torch.float32, device=DEV)

def load(n=None):
    W = []
    for hp in sorted(glob.glob("input/train/*__horizontal_well.csv"))[:n]:
        w = os.path.basename(hp).split("__")[0]
        hw = pd.read_csv(hp)
        if "TVT" not in hw.columns: continue
        tw = pd.read_csv(f"input/train/{w}__typewell.csv")
        kn = hw.TVT_input.notna().to_numpy()
        if kn.sum() < 40 or (~kn).sum() < 1: continue
        est = nid.identify(hw, tw)                       # per-well noise (heel only, leak-free)
        MD = hw.MD.to_numpy(float); Z = hw.Z.to_numpy(float); GR = hw.GR.to_numpy(float)
        TVT = hw.TVT.to_numpy(float)
        ps = int((~kn).argmax()); tvtps = float(TVT[ps-1]) if ps > 0 else float(hw.TVT_input.dropna().iloc[-1])
        import numpy as _np
        mdps = MD[ps]
        grn = _np.nan_to_num((GR - _np.nanmean(GR[kn])) / (_np.nanstd(GR[kn]) + 1e-6))
        zc = (Z - _np.nanmean(Z[kn])) / (_np.nanstd(Z[kn]) + 1e-6); dz = _np.gradient(zc)
        mds = (MD - mdps) / 1000.0
        dtvt = TVT - tvtps
        ki = _np.where(kn)[0]
        dip0 = _np.polyfit(MD[ki[-min(400,len(ki)):]], hw.TVT_input.to_numpy(float)[ki[-min(400,len(ki)):]], 1)[0]
        baseline = (dip0 * (MD - mdps)).astype("float32")     # physical dip-continuation baseline
        x = _np.stack([grn, zc, dz, mds, kn.astype(float), _np.where(kn, dtvt, 0.0)/50.0], 1).astype("float32")
        ctx = _np.array([_np.nanmean(grn[kn]), _np.nanstd(grn[kn]), dip0,
                         _np.log(est["pn"]**2 + 1e-9), _np.log(est["vn"]**2 + 1e-9)], "float32")
        W.append(dict(w=w, x=T(x), ctx=T(ctx), anchor=float(dtvt[0]), base=T(baseline),
                      y=T(dtvt.astype("float32")), mask=T((~kn).astype("float32")),
                      truey=T((TVT).astype("float32")), kn=kn))
    return W

def batches(items, bs, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle: rng.shuffle(idx)
    for k in range(0, len(idx), bs):
        chunk = [items[i] for i in idx[k:k+bs]]
        L = max(it["x"].shape[0] for it in chunk)
        Fdim = chunk[0]["x"].shape[1]
        xb = torch.zeros(len(chunk), L, Fdim, device=DEV)
        yb = torch.zeros(len(chunk), L, device=DEV); mb = torch.zeros(len(chunk), L, device=DEV)
        ctx = torch.zeros(len(chunk), chunk[0]["ctx"].numel(), device=DEV)
        anc = torch.zeros(len(chunk), device=DEV); bs_ = torch.zeros(len(chunk), L, device=DEV)
        for j, it in enumerate(chunk):
            l = it["x"].shape[0]; xb[j, :l] = it["x"]; yb[j, :l] = it["y"]; mb[j, :l] = it["mask"]
            ctx[j] = it["ctx"]; anc[j] = it["anchor"]; bs_[j, :l] = it["base"]
        yield xb, ctx, anc, bs_, yb, mb, chunk

def rmse(P, Ttrue): return float(((torch.cat(P) - torch.cat(Ttrue))**2).mean().sqrt())

def main():
    import numpy as _np
    t0 = time.time(); W = load()
    print(f"loaded {len(W)} wells ({time.time()-t0:.0f}s)", flush=True)
    folds = pd.read_csv("config/well_field_folds.csv").set_index("well").field_fold.to_dict()
    for it in W: it["fold"] = folds.get(it["w"], -1)
    W = [it for it in W if it["fold"] >= 0]
    P, Tt = [], []
    for vf in range(5):
        tr = [it for it in W if it["fold"] != vf]; va = [it for it in W if it["fold"] == vf]
        net = M.build_residual(feat_dim=6, ctx_dim=5, hid=96)
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
        rng = _np.random.default_rng(0); net.train()
        for ep in range(20):
            for xb, ctx, anc, bs_, yb, mb, _ in batches(tr, 16, True, rng):
                pred, logv, r = net(xb, ctx, bs_)
                loss = M.student_t_nll(pred, logv, yb, mb) + 5.0 * M.irw_smoothness(r, mb)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for xb, ctx, anc, bs_, yb, mb, chunk in batches(va, 16, False, rng):
                pred, _, _ = net(xb, ctx, bs_)
                for j, it in enumerate(chunk):
                    l = it["x"].shape[0]; hid = it["mask"][:l].bool()
                    P.append((pred[j, :l][hid]) + 0.0); Tt.append(it["y"][:l][hid])
        print(f"fold{vf} cum field-CV = {rmse(P,Tt):.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"StateSpaceNet field-CV = {rmse(P,Tt):.3f}  (blend_AB 10.75, gs130 10.56, PF 11.13)")

if __name__ == "__main__":
    main()
