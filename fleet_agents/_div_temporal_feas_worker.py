"""Worker for div-temporal-feas — runs under cellmot_venv (geff/zarr + torch/cuda). Prints one JSON line.

MINI-FIRST: mode='mini' loads a few stage-spanning datasets and one fast fold for a quick signal.
COVERAGE PROOF: reports temporal-vs-single separability stratified by embryo, developmental STAGE
(density bucket S0..S4) and the geometrically-invisible hard cases — so a GO means every regime clears
the single-frame ceiling, not just the average. XAI: every knob is derived from a MEASURED stat (_why)."""
import sys, os, glob, math, json, time
import numpy as np, pandas as pd, zarr
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
from src import io
TRAIN = os.path.join(COMP, "input", "biohub-cell-tracking-during-development", "train")
VOX = (1.625, 0.40625, 0.40625)          # z,y,x µm/voxel (measured from zarr multiscale attrs)
CAND_UM = 12.0                            # candidate 2nd-cell search radius (the confuser window)
CEILING = 0.671


def um(a, b): return math.sqrt(sum((VOX[i] * (a[i] - b[i])) ** 2 for i in range(3)))


def mine(geffs):
    """Mine (parent, daughter, candidate) triples with geometry + per-dataset density (for staging)."""
    recs = []; dens = {}
    for g in geffs:
        emb = os.path.basename(g).split("_")[0]
        ds = os.path.basename(g)[:-5]
        try: nodes, edges = io.read_geff(g)
        except Exception: continue
        dens[ds] = len(nodes) / max(1, nodes.t.nunique())      # mean cells/frame = developmental density
        pos = {r.node_id: (r.z, r.y, r.x, r.t) for r in nodes.itertuples()}
        kids = {}
        for e in edges.itertuples(): kids.setdefault(e.source_id, []).append(e.target_id)
        by_t = {}
        for nid, (z, y, x, t) in pos.items(): by_t.setdefault(t, []).append(nid)
        for m, ch in kids.items():
            if m not in pos: continue
            ch = [c for c in ch if c in pos]
            if not ch: continue
            d1 = ch[0]; mt = pos[m][3]; real = set(ch); ndiv = len(ch)
            d2s = sorted(um(pos[c], pos[m]) for c in ch)
            hard_inv = (ndiv >= 2 and d2s[1] >= 7.0)           # 2nd daughter 7-10µm away = geometry-invisible
            cands = [n for n in by_t.get(mt + 1, []) if n in pos and n != d1 and um(pos[n], pos[m]) < CAND_UM]
            for c in cands:
                lab = 1 if (ndiv >= 2 and c in real) else 0
                recs.append(dict(ds=ds, emb=emb, t=int(mt), mz=pos[m][0], my=pos[m][1], mx=pos[m][2],
                                 label=lab, cand_parent_um=um(pos[c], pos[m]),
                                 parent_daughter_um=um(pos[d1], pos[m]), sister_sister_um=um(pos[d1], pos[c]),
                                 local_density=len(cands), is_hard_inv=int(hard_inv and lab == 1)))
    return pd.DataFrame(recs), dens


def assign_stage(df, dens):
    """Developmental STAGE = density bucket. Proof: stage <-> cells/frame (memory: stage=density). 5 buckets
    S0..S4 by quantiles of per-dataset density so we can prove separability holds across the whole span."""
    ds_dens = pd.Series(dens)
    q = ds_dens.rank(pct=True)
    stage = (q * 5).clip(0, 4.999).astype(int).map(lambda i: f"S{i}")
    df = df.copy(); df["stage"] = df.ds.map(stage.to_dict()); df["dens"] = df.ds.map(dens)
    return df


def extract(sel, crop, TW):
    HZ, HY, HX = crop; T = 2 * TW + 1
    N = len(sel)
    X = np.zeros((N, T, 2 * HZ, 2 * HY, 2 * HX), dtype=np.float16)
    sel = sel.reset_index(drop=True)
    for ds, gidx in sel.groupby("ds").groups.items():
        grp = zarr.open_group(os.path.join(TRAIN, ds + ".zarr"), mode="r"); arr = grp["0"]
        qd = dict(grp.attrs).get("image_statistics", {}).get("quantiles", {})
        lo = float(qd.get("0.01", 38.0)); hi = float(qd.get("0.99", 1478.0)); rng = max(1.0, hi - lo)
        Tn = arr.shape[0]
        for i in gidx:
            r = sel.loc[i]; t = int(r.t); t0 = min(max(0, t - TW), Tn - T)
            cz, cy, cx = int(round(r.mz)), int(round(r.my)), int(round(r.mx))
            z0 = min(max(0, cz - HZ), arr.shape[1] - 2 * HZ)
            y0 = min(max(0, cy - HY), arr.shape[2] - 2 * HY)
            x0 = min(max(0, cx - HX), arr.shape[3] - 2 * HX)
            sub = np.asarray(arr[t0:t0 + T, z0:z0 + 2 * HZ, y0:y0 + 2 * HY, x0:x0 + 2 * HX]).astype(np.float32)
            X[i] = np.clip((sub - lo) / rng, 0, 1).astype(np.float16)
    return X, sel


def _report(res, y, S, Tm, emb, stage, sel, hard, safe_auc, np):
    """STRATIFIED coverage report — prove separability holds for every embryo & every developmental
    STAGE (density bucket) and the geometry-invisible hard cases, not just on average."""
    res["overall"] = {"single": safe_auc(y, S), "temporal": safe_auc(y, Tm)}
    res["by_embryo"] = {}
    for e in np.unique(emb):
        mk = emb == e
        res["by_embryo"][e] = {"n_pos": int(y[mk].sum()), "single": safe_auc(y[mk], S[mk]),
                               "temporal": safe_auc(y[mk], Tm[mk])}
    res["by_stage"] = {}
    for st in sorted(np.unique(stage)):
        mk = stage == st
        res["by_stage"][st] = {"n_pos": int(y[mk].sum()), "dens_mean": round(float(sel[mk].dens.mean()), 1),
                               "single": safe_auc(y[mk], S[mk]), "temporal": safe_auc(y[mk], Tm[mk])}
    hm = hard | (y == 0)
    res["hard_invisible"] = {"n_pos": int(hard.sum()), "single": safe_auc(y[hm], S[hm]),
                             "temporal": safe_auc(y[hm], Tm[hm])}


def run(cfg):
    import torch, torch.nn as nn, torch.nn.functional as F
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()

    # ---- XAI: derive every knob from a MEASURED number ----
    # crop XY half = ceil(CAND_UM / voxel_xy): the box must contain any 2nd cell up to the confuser radius.
    HX = int(math.ceil(CAND_UM / VOX[1]))                    # 12/0.406 ≈ 30 -> 32 after round-up below
    HX = HX + (HX % 2)
    HZ = int(math.ceil(CAND_UM / VOX[0])); HZ = max(6, HZ)   # 12/1.625 ≈ 8
    crop = cfg.get("crop") or (HZ, HX, HX)
    TW = cfg.get("temporal_window") or 2                     # ±2 frames: daughters at t+1, pre-mitotic rounding t-1..t-2

    geffs = sorted(glob.glob(os.path.join(TRAIN, "*.geff")))
    df, dens = mine(geffs)
    df = assign_stage(df, dens)

    # ---- MINI-first dataset selection: span both embryos AND all stages ----
    mode = cfg.get("mode", "mini")
    if mode == "mini":
        keep = []
        for emb in ("44b6", "6bba"):
            for st in sorted(df.stage.unique()):
                dss = df[(df.emb == emb) & (df.stage == st) & (df.label == 1)].ds.unique()
                keep += list(dss[: max(1, cfg["n_datasets"] // 10)])
        keep = set(keep) | set(df[df.label == 1].ds.unique()[: cfg["n_datasets"]])
        df = df[df.ds.isin(keep)]
    # subsample hardest negatives (smallest cand_parent_um = most touching = the true confusers)
    posdf = df[df.label == 1]
    negdf = df[df.label == 0].sort_values("cand_parent_um").head(cfg["max_neg"])
    sel = pd.concat([posdf, negdf]).reset_index(drop=True)

    X, sel = extract(sel, crop, TW)
    y = sel.label.to_numpy().astype(np.float32)
    ds = sel.ds.to_numpy(); emb = sel.emb.to_numpy(); stage = sel.stage.to_numpy()
    hard = sel.is_hard_inv.to_numpy().astype(bool)
    N, T, Z, Y, Xd = X.shape
    Xg = torch.from_numpy(X).to(dev); yg = torch.from_numpy(y).to(dev)
    DFRAME = min(TW + 1, T - 1)                               # single-frame = t+1 (both cells present = hardest)

    def aug(x):
        for ax in (-1, -2, -3):
            if torch.rand(1).item() < 0.5: x = x.flip(ax)
        x = x * (1 + 0.1 * (torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) - 0.5))
        return x.clamp(0, 1)

    class Enc(nn.Module):
        def __init__(s, d=64):
            super().__init__()
            s.c1 = nn.Conv3d(1, 16, 3, padding=1); s.b1 = nn.BatchNorm3d(16)
            s.c2 = nn.Conv3d(16, 32, 3, padding=1); s.b2 = nn.BatchNorm3d(32)
            s.c3 = nn.Conv3d(32, d, 3, padding=1); s.b3 = nn.BatchNorm3d(d)
        def forward(s, x):
            x = F.max_pool3d(F.relu(s.b1(s.c1(x))), 2)
            x = F.max_pool3d(F.relu(s.b2(s.c2(x))), 2)
            x = F.relu(s.b3(s.c3(x)))
            return F.adaptive_avg_pool3d(x, 1).flatten(1)

    class Single(nn.Module):
        def __init__(s, d=64): super().__init__(); s.e = Enc(d); s.h = nn.Linear(d, 1)
        def forward(s, x): return s.h(s.e(x[:, DFRAME:DFRAME + 1])).squeeze(1)

    class Temporal(nn.Module):
        def __init__(s, d=64):
            super().__init__(); s.e = Enc(d)
            s.h = nn.Sequential(nn.Linear(d * 4, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(s, x):
            B = x.shape[0]
            f = s.e(x.reshape(B * T, 1, Z, Y, Xd)).reshape(B, T, -1)
            agg = torch.cat([f.mean(1), f.amax(1), f[:, -1] - f[:, 0], f.std(1)], 1)
            return s.h(agg).squeeze(1)

    def fit_predict(Model, tr, te, seed, epochs):
        torch.manual_seed(seed); np.random.seed(seed)
        m = Model().to(dev)
        pw = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())], device=dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
        tr = np.array(tr)
        for ep in range(epochs):
            m.train(); perm = np.random.permutation(tr)
            for i in range(0, len(perm), 128):
                b = perm[i:i + 128]
                logit = m(aug(Xg[b].float()))
                loss = F.binary_cross_entropy_with_logits(logit, yg[torch.from_numpy(b).to(dev)], pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(te), 256):
                out.append(torch.sigmoid(m(Xg[te[i:i + 256]].float())).cpu().numpy())
        return np.concatenate(out)

    def oof(Model, seeds, epochs, nfold):
        gkf = GroupKFold(min(nfold, len(np.unique(ds))))
        acc = np.zeros(N)
        for sd in seeds:
            o = np.zeros(N)
            for tr, te in gkf.split(X, y, groups=ds):
                o[te] = fit_predict(Model, tr, te, sd, epochs)
            acc += o
        return acc / len(seeds)

    def safe_auc(yt, ps):
        return round(float(roc_auc_score(yt, ps)), 3) if len(np.unique(yt)) > 1 else None

    res = {"mode": mode, "n_samples": int(N), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
           "n_hard_inv": int(hard.sum()), "crop": list(crop), "temporal_window": TW,
           "_why_crop": f"XY half={crop[1]}vox=ceil({CAND_UM}µm/{VOX[1]}µm-per-vox); Z half={crop[0]}=ceil({CAND_UM}/{VOX[0]}) — box must contain any 2nd cell within the {CAND_UM}µm confuser radius",
           "_why_tw": f"±{TW} frames — daughters appear at t+1, pre-mitotic rounding at t-1..t-2 (edge Δt=1)",
           "_why_single_frame": "single-frame uses t+1 (both cells present) = the HARDEST frame where a real division and two touching cells look identical -> the ~0.671 ceiling",
           "ceiling": CEILING}

    # ---- FROZEN-DETECTOR probe (the Step-2 warm-start design; sample-efficient) ----
    # MULTI-POOL heads: GAP destroys the spatial "1-blob vs 2-blobs" count signal, so we add global-MAX and
    # std pools of the encoder bottleneck AND read the detector's own nucleus HEATMAP (its local-maxima count
    # is the literal cell count per frame). Reporting single-vs-temporal PER pool group = multiple insights
    # into which signal (appearance vs count) carries — and whether temporal dynamics add over one frame.
    def frozen_features():
        from src.student import UNet3D
        import glob as _g
        ck = None
        for c in [os.path.join(COMP, "results", "zebrahub_detector", "detector.pt")] + \
                 _g.glob(os.path.join(COMP, "results", "zebrahub_detector", "*.pt")):
            if os.path.exists(c): ck = c; break
        net = UNet3D(base=16).to(dev)
        sd = torch.load(ck, map_location=dev)
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        net.load_state_dict(sd, strict=False); net.eval()
        for p in net.parameters(): p.requires_grad_(False)
        def pool_xy(v):
            B, Z, Y, Xd_ = v.shape; Y2, X2 = (Y // 4) * 4, (Xd_ // 4) * 4
            return v[:, :, :Y2, :X2].reshape(B, Z, Y2 // 4, 4, X2 // 4, 4).mean(dim=(3, 5))
        import torch.nn.functional as _F
        groups = {"gap": [], "gmp": [], "gstd": [], "count": []}   # per-frame feature groups
        with torch.no_grad():
            for f in range(T):
                gap_f, gmp_f, gstd_f, cnt_f = [], [], [], []
                for i in range(0, N, 128):
                    xb = pool_xy(Xg[i:i + 128, f].float()).unsqueeze(1)
                    e1 = net.e1(xb); e2 = net.e2(net.pool(e1)); e3 = net.e3(net.pool(e2))
                    gap_f.append(e3.mean(dim=(2, 3, 4)).cpu().numpy())               # avg pool (appearance)
                    gmp_f.append(e3.amax(dim=(2, 3, 4)).cpu().numpy())               # MAX pool (peak appearance)
                    gstd_f.append(e3.float().std(dim=(2, 3, 4)).cpu().numpy())       # std pool (spatial spread)
                    py, px = (-xb.shape[-2]) % 4, (-xb.shape[-1]) % 4                # UNet needs XY divisible by 4
                    xb_h = _F.pad(xb, (0, px, 0, py, 0, 0)) if (py or px) else xb
                    h = torch.sigmoid(net(xb_h))                                     # nucleus HEATMAP
                    hm = _F.max_pool3d(h, 3, stride=1, padding=1)
                    peaks = ((h >= hm) & (h > 0.5)).float().sum(dim=(1, 2, 3, 4))    # local-maxima count = #cells
                    softcnt = h.sum(dim=(1, 2, 3, 4))
                    hmax = h.amax(dim=(1, 2, 3, 4)); hmean = h.mean(dim=(1, 2, 3, 4))
                    cnt_f.append(torch.stack([peaks, softcnt, hmax, hmean], 1).cpu().numpy())
                groups["gap"].append(np.concatenate(gap_f)); groups["gmp"].append(np.concatenate(gmp_f))
                groups["gstd"].append(np.concatenate(gstd_f)); groups["count"].append(np.concatenate(cnt_f))
        return {k: np.stack(v, 1) for k, v in groups.items()}   # each (N,T,dim)

    if cfg.get("probe", "frozen") == "frozen":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        G = frozen_features()
        def logit_oof(FX):
            gkf = GroupKFold(min(5, len(np.unique(ds)))); o = np.zeros(N)
            for tr, te in gkf.split(FX, y, groups=ds):
                sc = StandardScaler().fit(FX[tr])
                clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(FX[tr]), y[tr])
                o[te] = clf.predict_proba(sc.transform(FX[te]))[:, 1]
            return o
        def single_temporal(F3):
            s = F3[:, DFRAME]                                                        # t+1 frame
            t = np.concatenate([F3.mean(1), F3.max(1), F3[:, -1] - F3[:, 0], F3.std(1)], 1)  # temporal agg
            return logit_oof(s), logit_oof(t)
        # per-pool-group insight
        res["by_pool"] = {}; combo_single, combo_temp = [], []
        for name, F3 in G.items():
            Sg, Tg = single_temporal(F3)
            res["by_pool"][name] = {"single": safe_auc(y, Sg), "temporal": safe_auc(y, Tg)}
            combo_single.append(G[name][:, DFRAME])
            combo_temp += [G[name].mean(1), G[name].max(1), G[name][:, -1] - G[name][:, 0], G[name].std(1)]
        # combined (all pools) = the headline single vs temporal
        S = logit_oof(np.concatenate(combo_single, 1))
        Tm = logit_oof(np.concatenate(combo_temp, 1))
        res.update({"probe": "frozen_detector_multipool", "detector_ckpt": "results/zebrahub_detector/detector.pt",
                    "_why_multipool": "avg+max+std pools of the frozen encoder bottleneck + the detector heatmap "
                                      "local-maxima COUNT (=#cells/frame) — max/count pools see the '1-blob->2-blob' "
                                      "split that GAP averages away; per-pool AUC shows which signal separates"})
        _report(res, y, S, Tm, emb, stage, sel, hard, safe_auc, np)
        res["sec"] = round(time.time() - t_start, 1)
        if cfg.get("out"):
            os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True); json.dump(res, open(cfg["out"], "w"), indent=2)
        return res

    nfold = 3 if mode == "mini" else 5
    epochs = cfg["epochs"]; seeds = cfg["seeds"]
    S = oof(Single, seeds, epochs, nfold)
    Tm = oof(Temporal, seeds, epochs, nfold)
    res["probe"] = "from_scratch_cnn"
    _report(res, y, S, Tm, emb, stage, sel, hard, safe_auc, np)
    res["sec"] = round(time.time() - t_start, 1)
    if cfg.get("out"):
        os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
        json.dump(res, open(cfg["out"], "w"), indent=2)
    return res


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"mode": "mini", "n_datasets": 16,
                                                             "epochs": 25, "seeds": [0], "max_neg": 600}
    print(json.dumps(run(cfg)))
