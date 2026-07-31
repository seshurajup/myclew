"""geology_visual.py — reframe geosteering as VISION (hengck23 CNN-MTP idea). For each well build a 2D
correlation IMAGE: columns = MD along the hidden zone, rows = candidate TVT (centered on the dip-continuation
prior ± band), pixel = GR-match score between the horizontal GR window and the typewell GR at that TVT. The
true TVT trajectory is a PATH through this image; a CNN predicts it, using 2D context to resolve the local
multimodality (eda proof 8, XAI driver = twspan) the PF cannot. Pure GPU torch. Leak-free (GR + typewell + heel).
"""
import os, glob, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W, BAND = 96, 192, 60.0                       # image rows (TVT), cols (MD), TVT half-band (ft)

def make_image(hw, tw):
    tws = tw.sort_values("TVT"); T = tws.TVT.to_numpy(float); G = tws.GR.to_numpy(float)
    m = np.isfinite(T) & np.isfinite(G); T, G = T[m], G[m]
    kn = hw.TVT_input.notna().to_numpy(); ev = hw.TVT_input.isna().to_numpy() & hw.TVT.notna().to_numpy()
    if ev.sum() < 20 or kn.sum() < 30 or len(T) < 20: return None
    MD = hw.MD.to_numpy(float); GR = hw.GR.to_numpy(float); ki = np.where(kn)[0]
    mdps = MD[ki[-1]]; tvtps = hw.TVT_input.to_numpy(float)[ki[-1]]
    tw_at_k = np.interp(hw.TVT_input.to_numpy(float)[ki], T, G)
    a, b = np.polyfit(GR[ki][np.isfinite(GR[ki])], tw_at_k[np.isfinite(GR[ki])], 1) if np.isfinite(GR[ki]).sum() > 10 else (1.0, 0.0)
    Gc = a * GR + b
    kk = ki[-min(400, len(ki)):]; dip0 = np.polyfit(MD[kk], hw.TVT_input.to_numpy(float)[kk], 1)[0]
    ei = np.where(ev)[0]
    cols = np.linspace(0, len(ei) - 1, W).astype(int); rows_ei = ei[cols]        # resample hidden rows to W cols
    prior = tvtps + dip0 * (MD[rows_ei] - mdps)                                   # per-column TVT prior (dip cont.)
    tvt_grid = prior[None, :] + np.linspace(-BAND, BAND, H)[:, None]              # (H,W) candidate TVTs
    gtw = np.interp(tvt_grid.ravel(), T, G).reshape(H, W)                          # CH0: typewell GR = MATERIAL layers
    bit = np.repeat(Gc[rows_ei][None, :], H, axis=0)                                # CH1: bit's GR reading
    # CH2: WINDOWED NCC pattern match (captures GR SHAPE, not a single value) -> the path becomes a ridge.
    K = 12                                                                          # +-K rows window (MD)
    off = np.arange(-K, K+1)
    win_md = np.clip(rows_ei[None, :] + off[:, None], 0, len(Gc)-1)                 # (2K+1, W) HW-GR window rows
    hw_win = Gc[win_md]                                                             # (2K+1, W)
    hw_win = (hw_win - hw_win.mean(0)) / (hw_win.std(0) + 1e-6)
    match = np.zeros((H, W))
    for h in range(H):
        tvt_off = tvt_grid[h][None, :] + dip0 * (MD[win_md] - MD[rows_ei][None, :]) # map MD-window to TVT via dip
        tw_win = np.interp(tvt_off.ravel(), T, G).reshape(2*K+1, W)                 # typewell GR pattern at candidate
        tw_win = (tw_win - tw_win.mean(0)) / (tw_win.std(0) + 1e-6)
        match[h] = (hw_win * tw_win).mean(0)                                        # normalized cross-correlation
    # CH3: formation/material id at each candidate TVT (the layered structure from typewell Geology)
    gcol = tw.sort_values("TVT")["Geology"].fillna("").astype(str).str.strip().to_numpy()[m]
    labs = [x for x in np.unique(gcol) if x]; lut = {l:i+1 for i,l in enumerate(labs)}
    gid = np.array([lut.get(x,0) for x in gcol], float)
    fid = np.interp(tvt_grid.ravel(), T, gid).reshape(H, W)                          # CH3: material-id band
    def _n(a): return (a - a.mean())/(a.std()+1e-6)
    img = np.stack([_n(gtw), _n(bit), _n(match), _n(fid)], 0)                        # (4,H,W) material cross-section
    # target: true TVT as a row-index in the grid (0..H-1), per column
    true_tvt = hw.TVT.to_numpy(float)[rows_ei]
    row_idx = (true_tvt - prior + BAND) / (2 * BAND) * (H - 1)                     # continuous row target
    return img.astype("float32"), row_idx.astype("float32"), prior.astype("float32"), rows_ei, tvt_grid.astype("float32")

class PathCNN(nn.Module):
    """2D CNN over the correlation image → per-column distribution over TVT rows; soft-argmax = the path."""
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, 3, padding=1))
    def forward(s, x):                                # x:(B,1,H,W) -> logits (B,H,W) over rows
        return s.net(x).squeeze(1)

def soft_argmax_row(logits):                          # (B,H,W)->(B,W) expected row index
    p = torch.softmax(logits, dim=1)
    rows = torch.arange(logits.shape[1], device=logits.device, dtype=torch.float32)[None, :, None]
    return (p * rows).sum(1)

def load(limit=None):
    data = []
    for hp in sorted(glob.glob("input/train/*__horizontal_well.csv"))[:limit]:
        w = os.path.basename(hp).split("__")[0]
        try: hw = pd.read_csv(hp); tw = pd.read_csv(f"input/train/{w}__typewell.csv")
        except Exception: continue
        if "TVT" not in hw.columns: continue
        r = make_image(hw, tw)
        if r is None: continue
        img, ridx, prior, rows_ei, grid = r
        data.append(dict(w=w, img=img, ridx=ridx, prior=prior, grid=grid))
    return data

def main():
    folds = pd.read_csv("config/well_field_folds.csv").set_index("well").field_fold.to_dict()
    t0 = time.time(); D = [d for d in load() if d["w"] in folds]
    print(f"built {len(D)} correlation images ({time.time()-t0:.0f}s)", flush=True)
    P, Tt = [], []
    for vf in range(5):
        tr = [d for d in D if folds[d["w"]] != vf]; va = [d for d in D if folds[d["w"]] == vf]
        net = PathCNN().to(DEV); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
        rng = np.random.default_rng(0); net.train()
        for ep in range(30):
            rng.shuffle(tr)
            for k in range(0, len(tr), 32):
                ch = tr[k:k+32]
                xb = torch.tensor(np.stack([d["img"] for d in ch]), device=DEV)
                yb = torch.tensor(np.stack([d["ridx"] for d in ch]), device=DEV)
                logits = net(xb); pred_row = soft_argmax_row(logits)
                loss = ((pred_row - yb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for k in range(0, len(va), 32):
                ch = va[k:k+32]
                xb = torch.tensor(np.stack([d["img"] for d in ch]), device=DEV)
                pr = soft_argmax_row(net(xb)).cpu().numpy()
                for j, d in enumerate(ch):
                    # row index -> TVT via the grid; compare to true TVT
                    grid = d["grid"]; rr = np.nan_to_num(pr[j], nan=(H-1)/2); rr = np.clip(rr, 0, H-1)
                    lo = np.floor(rr).astype(int); frac = rr - lo; hi = np.clip(lo+1, 0, H-1)
                    tvt_pred = grid[lo, np.arange(W)]*(1-frac) + grid[hi, np.arange(W)]*frac
                    true_tvt = grid[np.clip(d["ridx"],0,H-1).astype(int), np.arange(W)]  # approx via target row
                    P.append(tvt_pred); Tt.append(true_tvt)
        P_=np.concatenate(P); T_=np.concatenate(Tt)
        print(f"fold{vf} cum image-CNN CV = {np.sqrt(np.mean((P_-T_)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    P_=np.concatenate(P); T_=np.concatenate(Tt)
    print(f"IMAGE-CNN field-CV = {np.sqrt(np.mean((P_-T_)**2)):.3f}  (PF 11.13, blend 10.75)")

if __name__ == "__main__":
    main()
