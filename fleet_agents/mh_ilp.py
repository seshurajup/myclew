"""mh_ilp — Multi-Hypothesis DoG + ultrack/tracksdata ILP tracker (research #1 DETECTION lever).

WHY (metric-linked): golden-CV is detection-bound and node_recall is the SQUARED lever
(R_edge ≈ R_node² · Q_link, see [[biohub_node_recall_lever]]). The kill-criterion (2026-07-11)
PASSED: a multi-threshold DoG UNION raised node_recall 0.715→0.741 (+0.026) on 12 golden full
movies at 7µm — the over-complete pool carries real cells the single threshold drops. But a GREEDY
union also adds FPs (union was WORSE on the full metric, [[biohub_detectors_complementary]]).

THE FIX = the ILP disambiguates with TEMPORAL evidence instead of a threshold. It selects a globally
flow-consistent subset: a candidate that links cheaply across frames is kept; an isolated FP that must
pay appearance+disappearance cost is dropped. THRESHOLD-FREE — no fixed threshold decides a detection;
the ILP costs (CV-tuned via embryo-disjoint LOEO) do (user: "thresholds as a trainable param is not good").

PIPELINE (per dataset, FULL movie):
  1. candidate pool  : union of DoG at multi (sigma×thresh_rel) combos; peak intensity = confidence.
                       light physical_nms(DEDUP_UM) collapses identical peaks from overlapping combos;
                       genuine near-neighbours stay and become graph OVERLAPS (mutually exclusive).
  2. graph (physical): tracksdata RustWorkXGraph; node x,y,z in µm so distance/gates are physical.
                       same-frame overlaps within CONFLICT_UM; DistanceEdges up to GAP frames < LINK_UM.
  3. ILP             : ILPSolver minimises  Σ node_cost + Σ edge_cost(distance) + appearance +
                       disappearance + division, s.t. flow conservation + overlap (≤1 per conflict set).
                       node_cost = −NODE_REWARD·score_norm (negative = reward for a confident detection).
  4. score           : full official metric per dataset (src/metric.official_counts, 7µm, count-calibrated
                       via estN); reported PER-EMBRYO (44b6 / 6bba), never just the mean.

Params are ALL tunable (config/_auto/mh_ilp.json or spec) and CV-tuned by embryo-disjoint LOEO.
"""
from __future__ import annotations
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseAgent, COMP

SRC = COMP / "src"
TRAIN = COMP / "input" / "biohub-cell-tracking-during-development" / "train"

# --- default ILP knobs (all CV-tunable; NOT thresholds — costs the solver trades off) ---
DEFAULTS = {
    # AGGRESSIVE pool (XAI: 44b6 is GENERATION-bound; aggressive ~2x pool_recall 0.196→0.392, 0.465→0.803).
    # More candidates = more FPs, which the ILP is designed to filter via temporal consistency.
    "combos": [[1.0, 1.0], [1.0, 0.6], [1.0, 0.4], [1.0, 0.2], [1.0, 0.1], [1.4, 0.4], [0.7, 0.4]],
    "min_peak_dist": 1,     # denser local-maxima grid (recovers close/dim cells DoG@dist2 merges away)
    "dedup_um": 1.0,        # collapse identical peaks from overlapping combos (NOT the conflict merge)
    "conflict_um": 3.0,     # same-frame candidates closer than this are mutually exclusive (ILP picks ≤1)
    "link_um": 11.0,        # temporal edge gate (== MAX_LINK_DIST_UM)
    "gap": 1,               # DistanceEdges delta_t (bridge up to `gap` missed frames)
    "knn": 5,               # temporal neighbours per node
    "node_reward": 20.0,    # node_cost = -node_reward * (floor + (1-floor)*score_norm).
                            # MUST dominate appearance+disappearance or the ILP under-selects (recall<0.5,
                            # count_ratio~0.6 on full 44b6/6bba). Tuned via XAI-guided sweep, not one slice.
    "reward_floor": 0.2,    # baseline reward every candidate gets even at score_norm=0 — else min-max norm
                            # gives the DIMMEST real cell reward 0 → always dropped (44b6 dim-cell pathology).
    "appearance": 3.0,      # cost of a track birth (kills isolated FP candidates; too-low fragments tracks)
    "disappearance": 3.0,   # cost of a track death
    "division": 8.0,        # cost of a split (keep divisions conservative)
    "edge_weight": "distance",  # edge COST attr. "distance"=prefer short links. A negative float (e.g. -1.0)
                                # = constant REWARD per edge (the public 0.90 field's balance: edge=-1,
                                # app/disapp=0.1, div=1 — rewards linking, cheap birth/death).
    # LEARNED candidate generator (detector_arch winner) — complementary to DoG on dim cells. None = off.
    "learned_ckpt_44b6": None,   # e.g. "model_scratch/results/winner_44b6/best.pt"
    "learned_ckpt_6bba": None,   # e.g. "model_scratch/results/winner_6bba/best.pt"
    "learned_config": "model_scratch/config/exp_det_winner.yml",
    "learned_topk": 400,         # threshold-free topk peaks per frame (over-complete; ILP filters)
    "use_cellpose": False,       # Cellpose-SAM cpsam centroids in the pool (VERIFIED recall ~0.96, external)
    "num_threads": 4,            # ILP solver threads
    "time_limit": None,          # optional ILP wall-clock cap in seconds (None = solve to optimality)
    "max_candidates": None,      # optional cap on per-frame candidates (keep top-score N; None = keep all)
}


def _learned_for(ds, params):
    """Build the {model, device, topk} learned-detector spec for this dataset (embryo-routed), or None."""
    import sys
    sys.path.insert(0, str(COMP))
    from src.io import embryo_id
    emb = embryo_id(ds)
    ckpt = params.get(f"learned_ckpt_{emb}")
    if not ckpt:
        return None
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_detector(str(COMP / ckpt) if not str(ckpt).startswith("/") else ckpt,
                           str(COMP / params["learned_config"]), device)
    return {"model": model, "device": device, "topk": params.get("learned_topk")}


def _cfg():
    import sys
    sys.path.insert(0, str(COMP))
    from src.config import Config
    return Config()


_DET_CACHE = {}


def _load_detector(ckpt, config, device):
    """Load + cache a trained cellmot detector (the detector_arch winner) for candidate generation."""
    key = (ckpt, config)
    if key not in _DET_CACHE:
        import torch
        from model_scratch.cellmot import load_config, build_model
        m = build_model(load_config(config)).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
        _DET_CACHE[key] = m
    return _DET_CACHE[key]


_CP_CACHE = {}


def _load_cellpose():
    """Cellpose-SAM cpsam — VERIFIED best external detector (training-CV recall 44b6=0.972 6bba=0.951,
    zero-shot, no leakage). 2D+stitch mode = Kaggle-feasible ~4s/frame."""
    if "cpsam" not in _CP_CACHE:
        from cellpose import models as cpm
        _CP_CACHE["cpsam"] = cpm.CellposeModel(gpu=True, pretrained_model="cpsam")
    return _CP_CACHE["cpsam"]


def _cellpose_centroids(vol):
    """Cellpose-SAM 2D+stitch instance masks → nucleus centroids (voxel z,y,x) + uniform score."""
    import numpy as _np
    from scipy import ndimage as _ndi
    m = _load_cellpose()
    masks, _, _ = m.eval(vol, do_3D=False, z_axis=0, stitch_threshold=0.5, normalize=True, batch_size=64)
    lbls = _np.unique(masks); lbls = lbls[lbls > 0]
    if not len(lbls):
        return _np.zeros((0, 3), _np.int32), _np.zeros((0,), _np.float32)
    c = _np.array(_ndi.center_of_mass(_np.ones_like(masks), masks, lbls), float)
    return _np.rint(c).astype(_np.int32), _np.ones(len(c), _np.float32)


def _candidate_pool(vol, cfg, combos, dedup_um, scale, min_peak_dist=None, learned=None, use_cellpose=False):
    """Candidate pool for the ILP: DoG (multi sigma×thresh) ∪ optional LEARNED peaks ∪ optional CELLPOSE-SAM
    centroids → (coords voxel, scores). Cellpose-SAM is the VERIFIED best external detector (recall ~0.96)
    → use_cellpose=True makes it the dominant, near-complete pool. The ILP filters FPs by temporal
    consistency + count-calibration, so an over-complete high-recall pool is exactly right."""
    import sys
    sys.path.insert(0, str(COMP))
    from src import detect as D
    allc, alls = [], []
    for sg_mult, th_mult in combos:
        c2 = copy.copy(cfg)
        c2.SMOOTH_SIGMA = cfg.SMOOTH_SIGMA * sg_mult
        c2.THRESH_REL = cfg.THRESH_REL * th_mult
        if min_peak_dist is not None:
            c2.MIN_PEAK_DIST = int(min_peak_dist)   # denser peaks (=1) recover close/dim cells
        c2.NMS_RADIUS_UM = 0.0            # NMS handled below (union first, then dedup)
        co, sc = D.detect_cells(vol, c2)
        if len(co):
            allc.append(co); alls.append(sc)
    dog_med = float(np.median(np.concatenate(alls))) if alls else 1.0
    if learned is not None:                          # add the learned detector's peaks (topk = threshold-free)
        from model_scratch.train_v0 import model_detect
        co_l = model_detect(learned["model"], vol, learned["device"], topk=learned.get("topk"))
        if len(co_l):
            allc.append(np.asarray(co_l).reshape(-1, 3))
            alls.append(np.full(len(co_l), dog_med, np.float32))  # mid-confidence score → fair node reward
    if use_cellpose:                                  # VERIFIED best external detector (recall ~0.96)
        co_c, sc_c = _cellpose_centroids(vol)
        if len(co_c):
            allc.append(co_c)
            alls.append(np.full(len(co_c), dog_med, np.float32))
    if not allc:
        return np.zeros((0, 3), np.int32), np.zeros((0,), np.float32)
    coords = np.vstack(allc); scores = np.concatenate(alls)
    if len(coords) > 1 and dedup_um > 0:  # collapse exact/near duplicates from overlapping combos + detectors
        keep = D.physical_nms(coords, scores, np.asarray(scale), dedup_um)
        coords, scores = coords[keep], scores[keep]
    return coords, scores


def _ilp_track(ds, params, frames=None, cfg=None):
    """Full-movie multi-hypothesis + ILP → (pred_nodes DataFrame voxel, pred_edges DataFrame)."""
    import sys
    sys.path.insert(0, str(COMP))
    from src import io
    from tracksdata.graph import RustWorkXGraph
    from tracksdata.edges import DistanceEdges
    from tracksdata.solvers import ILPSolver
    cfg = cfg or _cfg()
    scale = np.asarray(cfg.SCALE)
    ad, shape, dtype, T = _frames_of(ds, frames)
    learned = _learned_for(ds, params)
    per_frame_pts = []
    for t in range(T):
        vol = io.load_volume(ad, shape, dtype, t)
        per_frame_pts.append(_candidate_pool(vol, cfg, params["combos"], params["dedup_um"], scale,
                                             params.get("min_peak_dist"), learned,
                                             params.get("use_cellpose", False)))
    return _solve_from_points(per_frame_pts, scale, params)


def _solve_from_points(per_frame_pts, scale, params):
    """Core ILP: per_frame_pts=[(coords Nx3 voxel, scores N), ...] → (pred_nodes voxel df, pred_edges df).

    Threshold-free selection: over-complete candidates → same-frame overlaps (mutually exclusive) +
    temporal DistanceEdges → ILP keeps the flow-consistent subset, drops isolated FPs. This is the whole
    hypothesis, so it is what the data-wise test exercises on synthetic points."""
    from tracksdata.graph import RustWorkXGraph
    from tracksdata.edges import DistanceEdges
    from tracksdata.solvers import ILPSolver
    import polars as pl
    scale = np.asarray(scale)
    empty = (pd.DataFrame(columns=["node_id", "t", "z", "y", "x"]),
             pd.DataFrame(columns=["source_id", "target_id"]))

    g = RustWorkXGraph()
    for key in ("z", "y", "x", "score", "nw", "zv", "yv", "xv"):
        if key not in g.node_attr_keys():
            g.add_node_attr_key(key, pl.Float64, 0.0)
    vox = {}                                   # node_id -> (t, zv, yv, xv)
    all_scores = [s for _, s in per_frame_pts if len(s)]
    smax = float(np.concatenate(all_scores).max()) if all_scores else 1.0
    smin = float(np.concatenate(all_scores).min()) if all_scores else 0.0
    rng = max(smax - smin, 1e-6)

    max_cand = params.get("max_candidates")
    for t, (co, sc) in enumerate(per_frame_pts):
        co = np.asarray(co).reshape(-1, 3)
        sc = np.asarray(sc).reshape(-1)
        if max_cand is not None and len(co) > int(max_cand):    # keep only the top-score candidates this frame
            top = np.argsort(sc)[::-1][:int(max_cand)]
            co, sc = co[top], sc[top]
        ids = []
        for (z, y, x), s in zip(co, sc):
            snorm = (float(s) - smin) / rng                          # ∈ [0,1]
            floor = params.get("reward_floor", 0.0)
            reward = params["node_reward"] * (floor + (1.0 - floor) * snorm)  # baseline + confidence
            nid = g.add_node({"t": int(t),
                              "z": float(z) * scale[0], "y": float(y) * scale[1], "x": float(x) * scale[2],
                              "score": snorm, "nw": -reward,
                              "zv": float(z), "yv": float(y), "xv": float(x)})
            vox[nid] = (int(t), int(round(z)), int(round(y)), int(round(x)))
            ids.append(nid)
        if len(co) > 1 and params["conflict_um"] > 0:  # same-frame overlaps → ILP picks ≤1 per conflict set
            P = co.astype(np.float64) * scale[None, :]
            pairs = []
            for i in range(len(P)):
                d = np.sqrt(((P[i + 1:] - P[i]) ** 2).sum(axis=1))
                for jrel in np.where(d < params["conflict_um"])[0]:
                    pairs.append((ids[i], ids[i + 1 + jrel]))
            if pairs:
                g.bulk_add_overlaps(pairs)

    if g.num_nodes == 0:
        return empty

    DistanceEdges(distance_threshold=params["link_um"], n_neighbors=params["knn"],
                  delta_t=params["gap"]).add_edges(g)
    solver_kw = dict(node_weight="nw", edge_weight=params.get("edge_weight", "distance"),
                     appearance_weight=params["appearance"], disappearance_weight=params["disappearance"],
                     division_weight=params["division"], num_threads=int(params.get("num_threads", 4)))
    tl = params.get("time_limit")
    if tl is not None:
        solver_kw["time_limit"] = float(tl)
    try:
        sol = ILPSolver(**solver_kw).solve(g)
    except TypeError:                                   # older ILPSolver without time_limit → drop it
        solver_kw.pop("time_limit", None)
        sol = ILPSolver(**solver_kw).solve(g)
    if sol is None:
        return empty

    nd = sol.node_attrs().to_pandas()
    sel_ids = [int(i) for i in nd["node_id"].to_numpy()]
    pn = pd.DataFrame([(nid, *vox[nid]) for nid in sel_ids if nid in vox],
                      columns=["node_id", "t", "z", "y", "x"])
    ed = sol.edge_attrs().to_pandas()
    pe = ed[["source_id", "target_id"]].astype(int) if len(ed) else \
        pd.DataFrame(columns=["source_id", "target_id"])
    return pn, pe


def _frames_of(ds, frames):
    import sys
    sys.path.insert(0, str(COMP))
    from src import io
    ad, shape, dtype = io.read_array_meta(TRAIN / f"{ds}.zarr")
    T = shape[0] if frames is None else min(frames, shape[0])
    return ad, shape, dtype, T


def _pool_recall(ds, params, frames=None, cfg=None):
    """XAI: does a CANDIDATE even exist within 7µm of each GT node (pool coverage BEFORE the ILP)?
    Separates a generation gap (pool_recall low) from a selection gap (pool high, selected low)."""
    import sys
    sys.path.insert(0, str(COMP))
    from src import io, metric
    cfg = cfg or _cfg()
    scale = np.asarray(cfg.SCALE)
    gn, _ = io.read_geff(TRAIN / f"{ds}.geff")
    ad, shape, dtype, T = _frames_of(ds, frames)
    learned = _learned_for(ds, params)
    tp = gt = npool = 0
    for t in range(T):
        gf = gn[gn["t"] == t]
        vol = io.load_volume(ad, shape, dtype, t)
        co, sc = _candidate_pool(vol, cfg, params["combos"], params["dedup_um"], scale,
                                 params.get("min_peak_dist"), learned)
        npool += len(co); gt += len(gf)
        if len(gf) and len(co):
            pf = pd.DataFrame({"node_id": range(len(co)), "t": t,
                               "z": co[:, 0], "y": co[:, 1], "x": co[:, 2]})
            tp += len(metric._match_nodes(gf, pf, scale, 7.0))
    return {"pool_recall": tp / max(gt, 1), "pool_per_frame": npool / max(T, 1)}


def _node_xai(gn, pn, scale, T):
    """XAI: node recall/precision of the SELECTED set vs sparse GT (which cost is starving/flooding?)."""
    import sys
    sys.path.insert(0, str(COMP))
    from src import metric
    tp = gt = pr = 0
    for t in range(T):
        gf = gn[gn["t"] == t]; pf = pn[pn["t"] == t]
        gt += len(gf); pr += len(pf)
        if len(gf) and len(pf):
            tp += len(metric._match_nodes(gf, pf, scale, 7.0))
    return {"node_recall": tp / max(gt, 1), "node_prec": tp / max(pr, 1)}


def _score_ds(ds, params, frames=None, cfg=None, xai=True):
    import sys
    sys.path.insert(0, str(COMP))
    from src import io, metric
    from src.io import embryo_id
    cfg = cfg or _cfg()
    scale = np.asarray(cfg.SCALE)
    gn, ge = io.read_geff(TRAIN / f"{ds}.geff")
    estN = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
    pn, pe = _ilp_track(ds, params, frames=frames, cfg=cfg)
    c = metric.official_counts(gn, ge, pn, pe, scale, 7.0, t_true=estN)
    c["dataset"] = ds; c["embryo"] = embryo_id(ds)
    if xai:                                        # XAI: recall/precision + count ratio guide WHICH cost to move
        T = frames if frames is not None else int(gn["t"].max()) + 1
        c.update(_node_xai(gn, pn, scale, T))
        c["count_ratio"] = c.get("t_pred", 0) / max(c.get("t_true", 1), 1)
    return c


def _per_embryo(rows):
    """rows: official_counts dicts with embryo/w/adj_jaccard → per-embryo weighted adjJ + XAI, and mean."""
    df = pd.DataFrame(rows)
    out, xai = {}, {}
    for emb, g in df.groupby("embryo"):
        w = g["w"].to_numpy(); a = g["adj_jaccard"].to_numpy()
        out[emb] = float((w * a).sum() / w.sum()) if w.sum() > 0 else float("nan")
        xai[emb] = {"node_recall": round(float(g.get("node_recall", pd.Series([np.nan])).mean()), 3),
                    "node_prec": round(float(g.get("node_prec", pd.Series([np.nan])).mean()), 3),
                    "count_ratio": round(float(g.get("count_ratio", pd.Series([np.nan])).mean()), 2)}
    mean = float(np.nanmean(list(out.values()))) if out else float("nan")
    return out, mean, xai


def _xai_recommendation(xai):
    """Translate the XAI decomposition into WHICH ILP cost to move (guides the tune; no eyeballing)."""
    tips = []
    for emb, x in xai.items():
        r, p, cr = x.get("node_recall"), x.get("node_prec"), x.get("count_ratio")
        if r is not None and r < 0.9:
            tips.append(f"{emb} recall {r} low → RAISE node_reward / LOWER appearance+disappearance (under-select)")
        elif cr is not None and cr > 1.3:
            tips.append(f"{emb} count_ratio {cr} high → RAISE appearance/conflict_um (over-select FPs, count penalty)")
        elif p is not None and p < 0.5:
            tips.append(f"{emb} precision {p} low → RAISE conflict_um / appearance (FP flooding)")
        else:
            tips.append(f"{emb} balanced (recall {r}, prec {p}, count {cr})")
    return "; ".join(tips)


class MhIlp(BaseAgent):
    name = "mh-ilp"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        params = dict(DEFAULTS); params.update(self.load_state({}).get("params", {}))
        params.update({k: v for k, v in spec.items() if k in DEFAULTS})
        datasets = spec.get("datasets")
        frames = spec.get("frames")            # None = full movie (the real judge)
        if not datasets:
            from model_scratch.train_v0 import split_datasets
            _, te = split_datasets(); datasets = te[:spec.get("n_eval", 4)]

        t0 = time.time()
        rows = []
        for ds in datasets:
            try:
                rows.append(_score_ds(ds, params, frames=frames))
            except Exception as e:  # noqa: BLE001
                self.log(f"mh-ilp {ds} FAILED: {type(e).__name__}: {str(e)[:120]}", kind="finding")
        if not rows:
            return self.escalate(worker, "researcher", "mh-ilp: no datasets scored (see log)")
        per_emb, mean, xai = _per_embryo(rows)
        a44 = per_emb.get("44b6", float("nan")); a6b = per_emb.get("6bba", float("nan"))
        x44 = xai.get("44b6", {}); x6b = xai.get("6bba", {})
        dt = time.time() - t0
        scope = f"frames={frames or 'FULL'} n={len(rows)}"
        summary = (f"2-CV[44b6={a44:.4f} 6bba={a6b:.4f}] mean={mean:.4f} — multi-hyp DoG + ILP "
                   f"(reward={params['node_reward']} appear={params['appearance']} "
                   f"disappear={params['disappearance']} div={params['division']} conflict={params['conflict_um']}) "
                   f"| XAI 44b6[rec={x44.get('node_recall')} prec={x44.get('node_prec')} cnt={x44.get('count_ratio')}] "
                   f"6bba[rec={x6b.get('node_recall')} prec={x6b.get('node_prec')} cnt={x6b.get('count_ratio')}] "
                   f"[{scope}, {dt:.0f}s]")
        self.record(change=f"mh_ilp_r{params['node_reward']}_a{params['appearance']}_d{params['division']}",
                    cv=round(mean, 4), description=summary, script="fleet_dispatch mh-ilp",
                    train_set="golden12" if (frames is None and len(rows) <= 12) else "loeo")
        # XAI-guided next move (NOT eyeballed): the decomposition names which cost to change; math_master
        # owns the paired-delta/Wilcoxon vs DoG when judging whether a change is a real per-embryo win.
        self.log(summary, kind="verdict", recommendation=_xai_recommendation(xai))
        return self.done({"cv": round(mean, 4), "per_embryo": per_emb, "xai": xai,
                          "params": params, "rows": rows}, summary)


_AGENT = MhIlp()


def run(q, worker):
    return _AGENT.run(q, worker)
