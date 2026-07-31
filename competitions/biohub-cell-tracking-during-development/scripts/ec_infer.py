"""Self-contained edge-error consensus INFERENCE (Kaggle-safe: numpy/scipy/sklearn/tracksdata only).

No ground truth, no tracking_cellmot, no score_* imports. Computes the SAME features as
edge_consensus.build_features(with_labels=False) and removes edges the shipped HGB model is
>~(1-tau) confident are false. Remove-only + division-protected (source out-degree < 2).
"""
import numpy as np
from collections import Counter

SCALE = np.array([1.625, 0.40625, 0.40625])
DENSITY_R_UM = 10.0
FEATNAMES = ["prob", "dist", "adz", "ady", "adx", "dt", "s_out", "t_in", "s_in", "t_out",
             "dens_s", "dens_t", "comp", "comp_ratio", "prob_rank", "cos_ang", "speed_ratio",
             "dist_q95", "flow_res", "flow_cos", "flow_n", "chain"]


def _features(node_df, edge_df):
    """node_df: columns node_id,t,z,y,x (pandas). edge_df: source_id,target_id,edge_prob."""
    na = node_df.set_index("node_id")
    coord = na[["z", "y", "x"]].to_numpy() * SCALE
    tvec = na["t"].to_numpy()
    idpos = {int(nid): i for i, nid in enumerate(na.index.to_numpy())}
    src = edge_df["source_id"].to_numpy(); tgt = edge_df["target_id"].to_numpy()
    prob = edge_df["edge_prob"].to_numpy().astype(float)
    prob = np.nan_to_num(prob, nan=0.0)
    si = np.array([idpos[int(s)] for s in src]); ti = np.array([idpos[int(t)] for t in tgt])
    ps = coord[si]; pt = coord[ti]
    disp = pt - ps
    dist = np.linalg.norm(disp, axis=1)
    dt = (tvec[ti] - tvec[si]).astype(float)
    out_deg = Counter(int(s) for s in src); in_deg = Counter(int(t) for t in tgt)
    s_out = np.array([out_deg[int(s)] for s in src], float)
    t_in = np.array([in_deg[int(t)] for t in tgt], float)
    s_in = np.array([in_deg[int(s)] for s in src], float)
    t_out = np.array([out_deg[int(t)] for t in tgt], float)
    from scipy.spatial import cKDTree
    frames = {}
    for f in np.unique(tvec):
        idx = np.where(tvec == f)[0]
        frames[int(f)] = (idx, cKDTree(coord[idx]))
    def density(pos, f):
        if f not in frames: return 0.0
        return len(frames[f][1].query_ball_point(pos, DENSITY_R_UM)) - 1
    dens_s = np.array([density(ps[i], int(tvec[si[i]])) for i in range(len(src))], float)
    dens_t = np.array([density(pt[i], int(tvec[ti[i]])) for i in range(len(src))], float)
    def nearest_other(pos, f):
        if f not in frames: return 1e3
        d, _ = frames[f][1].query(pos, k=2)
        return float(d[1]) if np.ndim(d) and len(d) > 1 else float(d)
    comp = np.array([nearest_other(ps[i], int(tvec[ti[i]])) for i in range(len(src))], float)
    comp_ratio = dist / np.maximum(comp, 1e-3)
    prob_rank = np.ones(len(src))
    by_src = {}
    for i, s in enumerate(src): by_src.setdefault(int(s), []).append(i)
    for s, idxs in by_src.items():
        for r, i in enumerate(sorted(idxs, key=lambda i: -prob[i])): prob_rank[i] = r
    pred_of = {}
    for s, t in zip(src, tgt): pred_of.setdefault(int(t), []).append(int(s))
    cos_ang = np.zeros(len(src)); speed_ratio = np.ones(len(src))
    for i in range(len(src)):
        preds = pred_of.get(int(src[i]), [])
        if len(preds) == 1:
            vp = coord[idpos[int(src[i])]] - coord[idpos[preds[0]]]
            nvp = np.linalg.norm(vp); nd = dist[i]
            if nvp > 1e-6 and nd > 1e-6:
                cos_ang[i] = float(np.dot(vp, disp[i]) / (nvp * nd)); speed_ratio[i] = nd / nvp
    dsel = dist[dt == 1]
    q95 = float(np.quantile(dsel, 0.95)) if dsel.size else float(np.quantile(dist, 0.95) if dist.size else 1.0)
    q95 = max(q95, 1e-3)
    dist_q95 = dist / q95
    rep_disp = np.full((len(coord), 3), np.nan)
    best_p = np.full(len(coord), -1.0)
    for i in range(len(src)):
        sp = si[i]
        if prob[i] > best_p[sp]:
            best_p[sp] = prob[i]; rep_disp[sp] = disp[i]
    FLOW_R = 15.0
    flow_res = np.zeros(len(src)); flow_cos = np.zeros(len(src)); flow_n = np.zeros(len(src))
    for i in range(len(src)):
        f = int(tvec[si[i]])
        if f not in frames:
            continue
        idx, tree = frames[f]
        nb = tree.query_ball_point(ps[i], FLOW_R)
        if not nb:
            continue
        gd = rep_disp[idx[nb]]
        gd = gd[~np.isnan(gd[:, 0])]
        if gd.shape[0] < 2:
            continue
        lf = np.median(gd, axis=0)
        flow_n[i] = gd.shape[0]
        flow_res[i] = float(np.linalg.norm(disp[i] - lf))
        nlf = np.linalg.norm(lf); nd = dist[i]
        if nlf > 1e-6 and nd > 1e-6:
            flow_cos[i] = float(np.dot(disp[i], lf) / (nlf * nd))
    chain = ((s_in > 0).astype(float) + (t_out > 0).astype(float))
    feat = np.column_stack([prob, dist, np.abs(disp[:, 0]), np.abs(disp[:, 1]), np.abs(disp[:, 2]),
                            dt, s_out, t_in, s_in, t_out, dens_s, dens_t, comp, comp_ratio,
                            prob_rank, cos_ang, speed_ratio,
                            dist_q95, flow_res, flow_cos, flow_n, chain])
    return feat, src, tgt, s_out


def removal_keys(node_df, edge_df, model, tau):
    """Return set of (source_id,target_id) edges to REMOVE (division-protected, remove-only)."""
    if len(edge_df) == 0:
        return set()
    feat, src, tgt, s_out = _features(node_df, edge_df)
    proba = model.predict_proba(feat)[:, 1]
    rem = (proba < tau) & (s_out < 2)
    return {(int(src[i]), int(tgt[i])) for i in np.where(rem)[0]}
