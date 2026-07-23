"""gnn-probe — does a GRAPH neural net help, or is pairwise geometry already enough?

The problem is a spatiotemporal graph (cells=nodes, candidate links=edges). Our learned PAIRWISE edge
head was saturated (Δ0.000 vs geometry). A GNN's extra power is NEIGHBOURHOOD message-passing: a link's
correctness depends on how nearby cells move (local flow coherence), which a pairwise model can't see.

This agent quantifies that hypothesis CHEAPLY (sklearn, CPU, no GPU) on the external dense GT:
  • build true edges (same-track t→t+1) + hard negatives (other cells within radius at t+1)
  • model A = pairwise features only (displacement, distance)
  • model B = A + NEIGHBOURHOOD context (candidate displacement minus the local mean flow of the node's
    spatial neighbours — the exact signal GNN message-passing would learn)
  • compare held-out ROC-AUC. If B > A by a real margin, a GNN is worth training; if not, geometry wins.

Grounded, reward-aligned: we only build the GNN if the context signal it needs actually exists.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
TRACKS = COMP / "input" / "zebrahub" / "tracks"
STATE = COMP / "config" / "_auto" / "gnn_probe.json"
VOX = (1.0, 4.0, 4.0)
# defaults — every one is overridable via q["spec"] so the SAME agent works on any track set / geometry
DEFAULTS = {"n_frames": 24, "radius_um": 12.0, "k_neigh": 8,
            "tracks_glob": str(TRACKS / "*.csv"), "file_filter": ["003", "004"], "vox": list(VOX)}


def report(q, worker):
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    cfg = {**DEFAULTS, **{k: spec[k] for k in DEFAULTS if k in spec}}
    N_FRAMES = int(cfg["n_frames"]); RADIUS_UM = float(cfg["radius_um"]); K_NEIGH = int(cfg["k_neigh"])
    vox = tuple(cfg["vox"]); filt = cfg["file_filter"]

    files = sorted(glob.glob(cfg["tracks_glob"]))
    if filt:
        files = [f for f in files if any(s in f for s in filt)]
    if not files:
        return ("done", {}, "all", f"[{worker}] gnn-probe: no usable external track CSVs.")
    try:
        df = pd.read_csv(files[0], usecols=lambda c: c in ("track_id", "t", "z", "y", "x"))
    except Exception as e:  # noqa: BLE001
        return ("done", {"error": str(e)[:100]}, "all", f"[{worker}] gnn-probe: could not read {files[0]} ({str(e)[:60]}).")
    if not {"track_id", "t", "z", "y", "x"}.issubset(df.columns):
        return ("done", {}, "all", f"[{worker}] gnn-probe: {Path(files[0]).name} missing track_id/t/z/y/x columns.")
    df = df.dropna(subset=["z", "y", "x", "t", "track_id"])
    ts = sorted(df["t"].unique())
    # sample frames spread across the movie
    pick = ts[:: max(1, len(ts) // N_FRAMES)][:N_FRAMES]

    Xa, Xb, Y = [], [], []
    for t in pick:
        a = df[df["t"] == t]; b = df[df["t"] == t + 1]
        if len(a) < 20 or len(b) < 20:
            continue
        pa = a[["z", "y", "x"]].to_numpy() * vox
        pb = b[["z", "y", "x"]].to_numpy() * vox
        ta = a["track_id"].to_numpy(); tb = b["track_id"].to_numpy()
        tree_b = cKDTree(pb)
        # true child position per node-a (same track in b)
        b_by_track = {tk: i for i, tk in enumerate(tb)}
        # local flow: each node-a's true displacement (if it has a child), then average over K spatial neighbours
        true_disp = np.full((len(a), 3), np.nan)
        for i, tk in enumerate(ta):
            j = b_by_track.get(tk)
            if j is not None:
                true_disp[i] = pb[j] - pa[i]
        tree_a = cKDTree(pa)
        for i in range(len(a)):
            # candidate children within radius
            cand = tree_b.query_ball_point(pa[i], RADIUS_UM)
            if not cand:
                continue
            # local mean flow from spatial neighbours in frame a (exclude self); message-passing signal
            kn = tree_a.query(pa[i], k=min(K_NEIGH + 1, len(a)))[1]
            nb = [n for n in np.atleast_1d(kn) if n != i]
            lf = np.nanmean(true_disp[nb], axis=0) if nb else np.array([np.nan] * 3)
            if np.any(np.isnan(lf)):
                lf = np.nanmean(true_disp, axis=0)
            truej = b_by_track.get(ta[i])
            for j in cand:
                disp = pb[j] - pa[i]
                dist = float(np.linalg.norm(disp))
                Xa.append([disp[0], disp[1], disp[2], dist])
                coher = disp - lf                                # deviation from local flow (GNN context)
                Xb.append([disp[0], disp[1], disp[2], dist,
                           coher[0], coher[1], coher[2], float(np.linalg.norm(coher))])
                Y.append(1 if (truej is not None and j == truej) else 0)

    Y = np.array(Y)
    if len(Y) < 500 or Y.sum() < 20:
        return ("done", {"samples": int(len(Y))}, "all",
                f"[{worker}] gnn-probe: too few samples ({len(Y)}) to judge; skipping.")
    Xa = np.array(Xa); Xb = np.array(Xb)
    # held-out split (last 30%)
    n = len(Y); k = int(n * 0.7)
    idx = np.argsort(np.arange(n))     # deterministic; frames already time-ordered
    tr, te = idx[:k], idx[k:]

    def auc(X):
        m = LogisticRegression(max_iter=400, class_weight="balanced").fit(X[tr], Y[tr])
        return float(roc_auc_score(Y[te], m.predict_proba(X[te])[:, 1]))

    auc_pair = auc(Xa)
    auc_ctx = auc(Xb)
    gain = auc_ctx - auc_pair
    verdict = ("GNN worth training — neighbourhood context adds real edge signal" if gain >= 0.01
               else "pairwise geometry already captures it — GNN unlikely to beat geometry on linking")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"auc_pairwise": round(auc_pair, 4), "auc_context": round(auc_ctx, 4),
                                 "gain": round(gain, 4), "samples": int(n), "pos": int(Y.sum())}, indent=2))
    from . import ledger
    ledger.log("gnn-probe",
               summary=f"GNN context probe: pairwise AUC {auc_pair:.3f} → +context {auc_ctx:.3f} (Δ{gain:+.3f})",
               detail=f"{n} candidate edges, {int(Y.sum())} true; neighbourhood-flow features = the GNN signal",
               kind="finding", recommendation=verdict)
    from researchpapers.fleet import post
    msg = (f"[{worker}] 🕸️ **GNN-PROBE** — does neighbourhood context (the GNN signal) beat pairwise geometry?\n\n"
           f"| model | held-out edge AUC |\n|---|--:|\n"
           f"| pairwise geometry (disp+dist) | {auc_pair:.3f} |\n"
           f"| **+ neighbourhood flow context** | **{auc_ctx:.3f}** |\n\n"
           f"Δ = **{gain:+.3f}** on {n:,} candidate edges. **Verdict: {verdict}.**")
    post.post_thread(worker, "all", msg, routine=False, kind="finding")
    return ("done", {"auc_pairwise": auc_pair, "auc_context": auc_ctx, "gain": round(gain, 4),
                     "worth_gnn": gain >= 0.01}, "all", msg)
