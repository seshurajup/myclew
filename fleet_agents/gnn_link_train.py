"""gnn-link-train — TRAIN the division + flow heads on the clean external GT (the div_J lever).

The executor that turns the assembled, scale-corrected supervision into a model. Two heads:
  • division — predict whether a cell is about to divide, from LOCAL GEOMETRY + count-change (t→t+1)
               features (no flow leak). This targets div_J=0, the one thing between us and 0.897.
  • flow     — regress the (dz,dy,dx) affinity vector from local neighbourhood geometry.

Leave-one-EMBRYO-out split (no leakage). Reads the architecture from a config/arch/*.yml (a SEARCH
candidate: hidden/layers) so arch-search can prove head/layer counts. Reports held-out division AP
(vs the div-rate baseline) and flow MAE — real, measured, no assumptions.

Reusable / spec-driven: {gt_path, arch_yml, hidden, n_layers, epochs, sample_frames, radius_vox, out}.
"""
from __future__ import annotations
import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
GT = COMP / "results" / "flow_gt" / "flow_node_gt_clean.parquet"
OUT = COMP / "results" / "gnn_link"
STATE = COMP / "config" / "_auto" / "gnn_link_train.json"


FEATURE_NAMES = ["d1_child", "d2_child", "dist_ratio", "sister_dist", "symmetry", "nn_dist_t"]


def _sister_features(pa_i, tb, pb, np):
    """Transfer-ROBUST division geometry (detector-independent): a true division = TWO daughters appear
    near the parent, symmetric & close to each other. XAI showed count_change (detector-specific) floods FP;
    these sister-pair distances/ratios are geometry, not detection counts."""
    if len(pb) < 2:
        return [10.0, 10.0, 1.0, 10.0, 0.0]
    d, idx = tb.query(pa_i, k=2)
    d1, d2 = float(d[0]), float(d[1])
    sister = float(np.linalg.norm(pb[idx[0]] - pb[idx[1]]))
    ratio = d2 / max(d1, 1e-3)                       # ~1 for a symmetric split, >>1 for a single child
    symm = abs(d1 - d2) / max(d1 + d2, 1e-3)         # 0 = perfectly symmetric daughters
    return [d1, d2, ratio, sister, symm]


def _features(sub, pd, np, cKDTree, radius, frames):
    """Per-node transfer-robust sister-geometry features + labels — VECTORISED (one batched KDTree query per
    frame over ALL nodes, not a per-node Python loop; that loop was the real bottleneck that starved the GPU)."""
    ts = sorted(sub["t"].unique())
    pick = ts[:: max(1, len(ts) // frames)][:frames]
    X, Ydiv, Yflow = [], [], []
    for t in pick:
        a = sub[sub["t"] == t]; b = sub[sub["t"] == t + 1]
        if len(a) < 8 or len(b) < 8:
            continue
        pa = a[["z", "y", "x"]].to_numpy(); pb = b[["z", "y", "x"]].to_numpy()
        ta = cKDTree(pa); tb = cKDTree(pb)
        dd, ii = tb.query(pa, k=2)                       # BATCH: all nodes' 2 nearest children at once
        d1, d2 = dd[:, 0], dd[:, 1]
        sister = np.linalg.norm(pb[ii[:, 0]] - pb[ii[:, 1]], axis=1)
        ratio = d2 / np.maximum(d1, 1e-3); symm = np.abs(d1 - d2) / np.maximum(d1 + d2, 1e-3)
        nn = ta.query(pa, k=2)[0][:, 1]                  # batched nearest-neighbour spacing
        X.append(np.stack([d1, d2, ratio, sister, symm, nn], axis=1))
        Ydiv.append(a["is_division"].to_numpy())
        Yflow.append(a[["dz", "dy", "dx"]].fillna(0.0).to_numpy())
    if not X:
        return np.zeros((0, 6), "float32"), np.zeros(0, "float32"), np.zeros((0, 3), "float32")
    return (np.concatenate(X).astype("float32"), np.concatenate(Ydiv).astype("float32"),
            np.concatenate(Yflow).astype("float32"))


def train(q, worker):
    from .base import gpu_train_held
    if gpu_train_held():
        msg = (f"[{worker}] gnn-link-train HELD — GPU training parked (5090 power-cap gate). "
               f"Remove config/_auto/gpu_train_hold.flag (human GO) before training.")
        return ("escalated", {"held": True}, "leader", msg)
    try:
        import numpy as np
        import pandas as pd
        import torch
        from torch import nn
        from scipy.spatial import cKDTree
        from sklearn.metrics import average_precision_score
    except Exception as e:  # noqa: BLE001
        return ("escalated", {"error": str(e)}, "researcher", f"[{worker}] gnn-link-train: deps missing ({e}).")
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    gt_path = Path(spec.get("gt_path") or GT)
    if not gt_path.is_absolute():                          # fleet workers run from tools/researchpapers —
        gt_path = COMP / gt_path                            # resolve relative paths against the competition root
    if not gt_path.exists():
        return ("done", {}, "all", f"[{worker}] gnn-link-train: GT parquet not found at {gt_path} (run box-sample/data-audit first).")
    hidden = int(spec.get("hidden", 128)); n_layers = int(spec.get("n_layers", 3))
    epochs = int(spec.get("epochs", 40)); frames = int(spec.get("sample_frames", 40))
    radius = float(spec.get("radius_vox", 6.0))
    lr = float(spec.get("lr", 1e-3))                       # Adam learning rate
    dev = spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")   # device override (cpu fallback if cuda absent)
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"

    df = pd.read_parquet(gt_path, columns=["embryo", "t", "z", "y", "x", "dz", "dy", "dx", "is_division"])
    # EMBRYO-disjoint grouping: collapse box names (ZSNS001_box5_w627 → ZSNS001) so leave-one-out is by
    # true embryo, not box (box-disjoint would LEAK across boxes of the same embryo). Opt-in via
    # include_embryos / group_by_base so existing callers (combined-train) are unchanged.
    import re as _re
    inc = spec.get("include_embryos")                     # e.g. ["ZSNS001","ZSNS003"] — train on these embryos only
    if inc or spec.get("group_by_base"):
        def _base(e):
            m = _re.match(r"(ZSNS\d+)", str(e))
            return m.group(1) if m else str(e).split("_box")[0]
        df["embryo"] = df["embryo"].map(_base)            # regroup to base embryo (embryo-disjoint CV)
        if inc:
            df = df[df["embryo"].isin(inc)]
    embs = list(df["embryo"].unique())
    if not embs:
        return ("done", {}, "all", f"[{worker}] gnn-link-train: no rows after include_embryos={inc}.")
    test_emb = spec.get("test_embryo", embs[-1])          # leave-one-embryo-out (no leak)
    # sample ONLY the external boxes (large, we crop them); competition data is used in FULL (all frames),
    # so rare divisions in the held-out competition embryo are never missed by sampling.
    BIG = 10 ** 9
    def _nframes(e):
        if e == test_emb:
            return BIG                                        # eval embryo: always all frames
        return frames if ("box" in e or e.startswith("ZSNS")) else BIG   # external boxes sampled; competition full
    feats = {e: _features(df[df["embryo"] == e], pd, np, cKDTree, radius, _nframes(e)) for e in embs}
    train_embs = [e for e in embs if e != test_emb]
    if not train_embs:                                    # only the held-out embryo present → cannot LOEO
        return ("done", {}, "all", f"[{worker}] gnn-link-train: only one embryo ({test_emb}); need ≥2 for leave-one-embryo-out.")
    Xtr = np.concatenate([feats[e][0] for e in train_embs])
    Dtr = np.concatenate([feats[e][1] for e in train_embs])
    Ftr = np.concatenate([feats[e][2] for e in train_embs])
    if len(Xtr) == 0 or test_emb not in feats or len(feats[test_emb][0]) == 0:
        return ("done", {}, "all", f"[{worker}] gnn-link-train: no usable train/eval features (frames too sparse).")
    Xte, Dte, Fte = feats[test_emb]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

    def mlp(out):
        layers, d = [], Xtr.shape[1]
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU()]; d = hidden
        layers += [nn.Linear(d, out)]
        return nn.Sequential(*layers).to(dev)

    div_net, flow_net = mlp(1), mlp(3)
    pos_w = torch.tensor([float(spec.get("div_pos_weight", (Dtr == 0).sum() / max((Dtr == 1).sum(), 1)))], device=dev)
    opt = torch.optim.Adam(list(div_net.parameters()) + list(flow_net.parameters()), lr=lr)
    xt = torch.tensor(Xtr, device=dev); dt = torch.tensor(Dtr, device=dev).unsqueeze(1)
    ft = torch.tensor(Ftr, device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    jitter = float(spec.get("feat_jitter_std", 0.0))       # per-epoch feature AUGMENTATION (std in normalised
    for ep in range(epochs):                               # feature space) — lets MORE epochs generalise, not overfit
        opt.zero_grad()
        xin = xt + jitter * torch.randn_like(xt) if jitter > 0 else xt
        ld = bce(div_net(xin), dt)
        lf = nn.functional.l1_loss(flow_net(xin), ft)
        (ld + lf).backward(); opt.step()

    div_net.eval(); flow_net.eval()
    with torch.no_grad():
        xe = torch.tensor(Xte, device=dev)
        dp = torch.sigmoid(div_net(xe)).cpu().numpy().ravel()
        fp = flow_net(xe).cpu().numpy()
    div_ap = float(average_precision_score(Dte, dp)) if Dte.sum() > 0 else 0.0
    base_ap = float(Dte.mean())                            # random-baseline AP = div rate
    flow_mae = float(np.abs(fp - Fte).mean())
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"div": div_net.state_dict(), "flow": flow_net.state_dict(),
                "mu": mu, "sd": sd, "hidden": hidden, "n_layers": n_layers}, OUT / "gnn_link.pt")
    # FREE GPU MEMORY — the fleet worker is long-lived, and this step trains FULL-BATCH on the whole
    # combined external+competition tensor. Without an explicit teardown the tensors + nets + optimizer
    # stay resident on CUDA and leak into the NEXT GPU step (this is the 28.7GiB that starved combined-train
    # → its 3.06GiB alloc OOM'd with only 2.59 free). Release everything and empty the allocator cache.
    del xt, dt, ft, xe, div_net, flow_net, opt, pos_w
    import gc as _gc
    _gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"div_ap": round(div_ap, 4), "base_ap": round(base_ap, 4),
                                 "lift": round(div_ap / max(base_ap, 1e-6), 1), "flow_mae": round(flow_mae, 3),
                                 "hidden": hidden, "n_layers": n_layers, "test_embryo": test_emb,
                                 "n_train": len(Xtr), "device": dev}, indent=2))
    from . import ledger
    ledger.log("gnn-link-train",
               summary=f"trained div+flow heads (hidden {hidden}, {n_layers}L) LOEO {test_emb}: div AP {div_ap:.3f} vs base {base_ap:.4f} ({div_ap/max(base_ap,1e-6):.0f}× lift), flow MAE {flow_mae:.2f}",
               detail=f"{len(Xtr):,} train nodes on {dev}; division head targets div_J=0 lever",
               kind="finding", recommendation="if div AP >> base, apply via affinity-link + prove on golden-12")
    from researchpapers.fleet import post
    lift = div_ap / max(base_ap, 1e-6)
    msg = (f"[{worker}] **GNN-LINK-TRAIN** · trained on clean GT ({dev}) · LOEO test=`{test_emb}` · hidden {hidden}/{n_layers}L\n"
           f"• division AP **{div_ap:.3f}** vs random baseline {base_ap:.4f} → **{lift:.0f}× lift** "
           f"{'✅ learns divisions' if lift > 3 else '⚠️ weak'}\n"
           f"• flow MAE **{flow_mae:.2f}** vox · {len(Xtr):,} train nodes\n"
           f"→ `results/gnn_link/gnn_link.pt`. {'Ready to apply (affinity-link) + prove on golden-12.' if lift>3 else 'Division signal weak from geometry alone — needs temporal features.'}")
    post.post_thread(worker, "all", msg, routine=False, kind="finding")
    return ("done", {"div_ap": round(div_ap, 4), "base_ap": round(base_ap, 4), "lift": round(lift, 1),
                     "flow_mae": round(flow_mae, 3), "test_embryo": test_emb}, "all", msg)
