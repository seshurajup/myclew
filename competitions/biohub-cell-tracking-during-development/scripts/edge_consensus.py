"""Cross-architecture edge-error consensus (arnav lever), built from scratch, LEAKAGE-FREE.

A second architecture (HistGradientBoosting) is trained on OUR predicted edges of ONE embryo
(features = geometry + local-competition + motion consistency + transformer edge_prob; label =
official metric's matched_edge_mask on GT-matched valid-pred edges), then applied to the HELD-OUT
embryo to flag learned edges it strongly disagrees with. REMOVE-ONLY, division-protected.

No test-aligned labels: each embryo's edges are judged only by a model trained on the *other*
embryo. Threshold is chosen on the TRAIN embryo (max train-adjJ gain), then frozen for the held-out
embryo. Score impact only comes through GT-matched (valid_pred) edges; removals elsewhere are graph
edits with zero metric effect.

Outputs modified geffs into --out-dir (one per dataset) that score_by_embryo can re-score.
"""
import argparse, sys, warnings, glob, os, json
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP / "research/official_repo/src"))
sys.path.insert(0, str(COMP / "scripts"))
sys.path.insert(0, str(COMP / "learning/ensemble_work"))

import tracksdata as td  # noqa
from tracking_cellmot.io import open_dataset  # noqa
from tracking_cellmot.metrics import evaluate as compute_metric  # noqa
from score_golden12_official import write_geff  # noqa
from score_pilkwang import geff_to_dicts  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa

K = td.DEFAULT_ATTR_KEYS
SCALE = np.array([1.625, 0.40625, 0.40625])
TRAIN = COMP / "input/biohub-cell-tracking-during-development/train"
DENSITY_R_UM = 10.0


def load_graph(path):
    g = td.graph.IndexedRXGraph.from_geff(path)
    return g[0] if isinstance(g, tuple) else g


def build_features(base_geff: Path, ds: str, with_labels: bool):
    """Return (feat[n_edges, F], edge_src, edge_tgt, edge_prob, labels_or_None, valid_or_None).

    Features are computable at inference WITHOUT GT. Labels/valid come from the official matcher
    (used for training + train-threshold selection only)."""
    g = load_graph(str(base_geff))
    na = g.node_attrs().to_pandas().set_index("node_id")
    ea = g.edge_attrs().to_pandas()
    # node coord arrays in um
    coord = na[["z", "y", "x"]].to_numpy() * SCALE
    tvec = na["t"].to_numpy()
    idpos = {nid: i for i, nid in enumerate(na.index.to_numpy())}
    src = ea["source_id"].to_numpy()
    tgt = ea["target_id"].to_numpy()
    prob = ea["edge_prob"].to_numpy().astype(float)
    si = np.array([idpos[s] for s in src]); ti = np.array([idpos[t] for t in tgt])
    ps = coord[si]; pt = coord[ti]
    disp = pt - ps
    dist = np.linalg.norm(disp, axis=1)
    dt = (tvec[ti] - tvec[si]).astype(float)
    # degrees
    from collections import Counter
    out_deg = Counter(src.tolist()); in_deg = Counter(tgt.tolist())
    s_out = np.array([out_deg[s] for s in src], float)
    t_in = np.array([in_deg[t] for t in tgt], float)
    s_in = np.array([in_deg[s] for s in src], float)
    t_out = np.array([out_deg[t] for t in tgt], float)
    # per-frame node index for density / competition (KDTree per frame)
    from scipy.spatial import cKDTree
    frames = {}
    for f in np.unique(tvec):
        idx = np.where(tvec == f)[0]
        frames[int(f)] = (idx, cKDTree(coord[idx]))
    def density(pos, f):
        if f not in frames: return 0.0
        _, tree = frames[f]
        return len(tree.query_ball_point(pos, DENSITY_R_UM)) - 1
    dens_s = np.array([density(ps[i], int(tvec[si[i]])) for i in range(len(src))], float)
    dens_t = np.array([density(pt[i], int(tvec[ti[i]])) for i in range(len(src))], float)
    # competition: distance from source to nearest OTHER node in target frame; ratio to this edge dist
    def nearest_other(pos, f, exclude_pos):
        if f not in frames: return 1e3
        idx, tree = frames[f]
        d, _ = tree.query(pos, k=2)
        return float(d[1]) if np.ndim(d) and len(d) > 1 else float(d)
    comp = np.array([nearest_other(ps[i], int(tvec[ti[i]]), pt[i]) for i in range(len(src))], float)
    comp_ratio = dist / np.maximum(comp, 1e-3)
    # edge_prob rank among source's outgoing edges (1 = best)
    prob_rank = np.ones(len(src))
    by_src = {}
    for i, s in enumerate(src): by_src.setdefault(s, []).append(i)
    for s, idxs in by_src.items():
        order = sorted(idxs, key=lambda i: -prob[i])
        for r, i in enumerate(order): prob_rank[i] = r
    # motion consistency: source single-predecessor velocity
    pred_of = {}   # node -> its single predecessor node
    for s, t in zip(src, tgt):
        pred_of.setdefault(t, []).append(s)
    cos_ang = np.zeros(len(src)); speed_ratio = np.ones(len(src))
    for i in range(len(src)):
        s = src[i]
        preds = pred_of.get(s, [])
        if len(preds) == 1:
            vp = coord[idpos[s]] - coord[idpos[preds[0]]]
            nvp = np.linalg.norm(vp); nd = dist[i]
            if nvp > 1e-6 and nd > 1e-6:
                cos_ang[i] = float(np.dot(vp, disp[i]) / (nvp * nd))
                speed_ratio[i] = nd / nvp
    # --- STRONGER edge-precision features (all GT-free, leakage-safe) ---
    # (a) per-movie physical plausibility: dist vs this movie's q95 single-step displacement
    dsel = dist[dt == 1]
    q95 = float(np.quantile(dsel, 0.95)) if dsel.size else float(np.quantile(dist, 0.95) if dist.size else 1.0)
    q95 = max(q95, 1e-3)
    dist_q95 = dist / q95
    # (b) local flow field residual: disagreement of this edge's displacement with the coherent
    #     local tissue motion (median representative displacement of source-frame neighbours).
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
    # (c) temporal persistence / link embeddedness: source has a predecessor AND target has a successor
    chain = ((s_in > 0).astype(float) + (t_out > 0).astype(float))
    feat = np.column_stack([
        prob, dist, np.abs(disp[:, 0]), np.abs(disp[:, 1]), np.abs(disp[:, 2]), dt,
        s_out, t_in, s_in, t_out, dens_s, dens_t, comp, comp_ratio, prob_rank,
        cos_ang, speed_ratio,
        dist_q95, flow_res, flow_cos, flow_n, chain,
    ])
    labels = valid = None
    if with_labels:
        gt = open_dataset(TRAIN / f"{ds}.zarr", require_tracks=True, load_image=False, device="cpu")
        compute_metric(g, gt.tracks, scale=tuple(SCALE), max_distance=7.0)
        ea2 = g.edge_attrs().to_pandas()
        # align by (source_id,target_id)
        key2mask = {(int(a), int(b)): bool(m) for a, b, m in
                    zip(ea2["source_id"], ea2["target_id"], ea2[K.MATCHED_EDGE_MASK])}
        na2 = g.node_attrs().to_pandas()
        matched = set(int(n) for n, mm in zip(na2["node_id"], na2[K.MATCHED_NODE_ID]) if int(mm) >= 0)
        labels = np.array([key2mask.get((int(s), int(t)), False) for s, t in zip(src, tgt)])
        valid = np.array([(int(s) in matched) or (int(t) in matched) for s, t in zip(src, tgt)])
    return feat, src, tgt, prob, labels, valid, s_out


FEATNAMES = ["prob","dist","adz","ady","adx","dt","s_out","t_in","s_in","t_out",
             "dens_s","dens_t","comp","comp_ratio","prob_rank","cos_ang","speed_ratio",
             "dist_q95","flow_res","flow_cos","flow_n","chain"]
