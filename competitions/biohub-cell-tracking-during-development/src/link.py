"""Temporal linking: Hungarian assignment in physical µm, with V3 velocity prior,
multi-frame gap closing, and a conservative division (mitosis) pass.

Operates on a sequence of per-frame detections and produces a node table + edge list.
All distances are computed in scaled micrometres.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import Config

BIG = 1e9


def _phys(coords: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return coords.astype(np.float64) * scale[None, :]


def _assign(src_phys: np.ndarray, dst_phys: np.ndarray, gate_um: float):
    """One-to-one Hungarian matching under a distance gate. Returns list of (i,j,dist)."""
    if len(src_phys) == 0 or len(dst_phys) == 0:
        return []
    D = np.sqrt(((src_phys[:, None, :] - dst_phys[None, :, :]) ** 2).sum(axis=2))
    cost = np.where(D <= gate_um, D, BIG)
    ri, ci = linear_sum_assignment(cost)
    return [(int(r), int(c), float(D[r, c])) for r, c in zip(ri, ci) if cost[r, c] < BIG]


def track_dataset(frames: List[np.ndarray], cfg: Config):
    """frames[t] = (N_t,3) voxel coords for timepoint t.

    Returns (nodes, edges):
      nodes: list of dicts {node_id,t,z,y,x}
      edges: list of (source_id, target_id)
    """
    scale = np.asarray(cfg.SCALE)
    nodes: List[dict] = []
    edges: List[Tuple[int, int]] = []

    next_id = 1
    # per-frame: ids and voxel coords of accepted detections
    frame_ids: List[np.ndarray] = []
    frame_xyz: List[np.ndarray] = []
    # velocity (phys µm/frame) per node_id, for the inertial prior
    velocity: Dict[int, np.ndarray] = {}

    for t, coords in enumerate(frames):
        coords = np.asarray(coords, dtype=np.int32).reshape(-1, 3)
        ids = np.arange(next_id, next_id + len(coords), dtype=np.int64)
        next_id += len(coords)
        for nid, (z, y, x) in zip(ids, coords):
            nodes.append({"node_id": int(nid), "t": int(t),
                          "z": int(z), "y": int(y), "x": int(x)})
        frame_ids.append(ids)
        frame_xyz.append(coords)

        if t == 0:
            continue

        prev_ids, prev_xyz = frame_ids[t - 1], frame_xyz[t - 1]
        cur_ids, cur_xyz = ids, coords
        prev_phys = _phys(prev_xyz, scale)
        cur_phys = _phys(cur_xyz, scale)

        # GLOBAL-MOTION COMPENSATION (ultrack Tribolium 0.443→0.623): the setup can jump the WHOLE volume by
        # ≫ the link gate (our 47%-jump embryo), breaking every edge. Estimate the global/local shift via a COARSE
        # large-gate match's ROBUST-MEDIAN displacement, and add it to src ONLY for the assignment — the true match
        # falls back inside the gate. Output node positions are UNCHANGED (only the matching is compensated; this is
        # why the earlier position-distorting attempt failed).
        _gmc = os.environ.get("USE_GLOBAL_MOTION_COMP")
        gmc_on = (_gmc == "1") if _gmc is not None else getattr(cfg, "USE_GLOBAL_MOTION_COMP", False)
        # FROZEN-FRAME check (research): a duplicate/frozen step has ~zero global motion — carrying the velocity
        # prior across it injects PHANTOM velocity → a 2× jump next frame. Detect it up front and skip the prior.
        frozen = False; coarse = []
        if gmc_on:
            coarse = _assign(prev_phys, cur_phys, cfg.MAX_LINK_DIST_UM * float(getattr(cfg, "GMC_GATE_MULT", 4.0)))
            if len(coarse) >= int(getattr(cfg, "GMC_MIN_MATCHES", 5)):
                _g0 = np.median([cur_phys[j] - prev_phys[i] for i, j, _ in coarse], axis=0)
                frozen = (np.linalg.norm(_g0) < 0.5) and (len(coarse) >= 0.8 * min(len(prev_phys), len(cur_phys)))

        # inertial prior: predict each prev node forward (SKIP on a frozen frame — no phantom velocity)
        src_phys = prev_phys.copy()
        if cfg.USE_VELOCITY_PRIOR and not frozen:
            for k, pid in enumerate(prev_ids):
                if pid in velocity:
                    src_phys[k] = prev_phys[k] + cfg.VELOCITY_INERTIA * velocity[int(pid)]

        if gmc_on:
            if len(coarse) >= int(getattr(cfg, "GMC_MIN_MATCHES", 5)):
                midx = np.array([i for i, j, _ in coarse]); mdisp = np.array([cur_phys[j] - prev_phys[i] for i, j, _ in coarse])
                if getattr(cfg, "GMC_LOCAL_FLOW", False):      # per-REGION flow: each node shifted by its LOCAL neighbourhood
                    from scipy.spatial import cKDTree as _KD
                    tree = _KD(prev_phys[midx]); K = min(int(getattr(cfg, "GMC_FLOW_K", 8)), len(midx))
                    _, nn = tree.query(prev_phys, k=K)         # K nearest coarse-matched neighbours per source node
                    nn = nn[:, None] if nn.ndim == 1 else nn
                    local_g = np.median(mdisp[nn], axis=1)     # (N,3) per-node local displacement
                    big = np.linalg.norm(local_g, axis=1) > float(getattr(cfg, "GMC_MIN_SHIFT_UM", 2.0))
                    src_phys[big] = src_phys[big] + local_g[big]
                else:
                    g = np.median(mdisp, axis=0)               # single global shift (GMC_LOCAL_FLOW off = the K=all case)
                    if np.linalg.norm(g) > float(getattr(cfg, "GMC_MIN_SHIFT_UM", 2.0)):
                        src_phys = src_phys + g

        matches = _assign(src_phys, cur_phys, cfg.MAX_LINK_DIST_UM)
        parent_children = defaultdict(list)
        matched_cur = set()
        for i, j, _d in matches:
            pid, cid = int(prev_ids[i]), int(cur_ids[j])
            edges.append((pid, cid))
            parent_children[i].append(j)
            matched_cur.add(j)
            if not frozen:                              # don't overwrite real velocity with ~0 on a frozen frame
                velocity[cid] = cur_phys[j] - prev_phys[i]
            elif int(prev_ids[i]) in velocity:          # carry the pre-frozen velocity forward through the frozen step
                velocity[cid] = velocity[int(prev_ids[i])]

        # conservative division pass: a parent with exactly one child adopts a nearby 2nd daughter
        if cfg.DETECT_DIVISIONS and (len(cur_ids) - len(prev_ids) >= cfg.DIV_MIN_COUNT_GAIN):
            D = np.sqrt(((prev_phys[:, None, :] - cur_phys[None, :, :]) ** 2).sum(axis=2))
            for j in range(len(cur_ids)):
                if j in matched_cur:
                    continue
                for i in range(len(prev_ids)):
                    if len(parent_children[i]) != 1 or D[i, j] > cfg.DIV_PARENT_DIST_UM:
                        continue
                    sister = parent_children[i][0]
                    sdist = float(np.linalg.norm(cur_phys[j] - cur_phys[sister]))
                    if sdist <= cfg.DIV_SISTER_DIST_UM:
                        edges.append((int(prev_ids[i]), int(cur_ids[j])))
                        parent_children[i].append(j)
                        matched_cur.add(j)
                        break

    # multi-frame gap closing: T-1 unmatched -> T+1 unmatched (skip the missing frame)
    if cfg.USE_GAP_CLOSING:
        edges = _gap_close(frame_ids, frame_xyz, edges, scale, cfg)

    return nodes, edges


def _gap_close(frame_ids, frame_xyz, edges, scale, cfg: Config):
    has_out = set(s for s, _ in edges)   # node ids that already start an edge
    has_in = set(d for _, d in edges)    # node ids that already end an edge
    T = len(frame_ids)
    max_skip = int(getattr(cfg, "GAP_CLOSE_MAX_SKIP", 1))            # bridge over 1..max_skip missed frames
    use_gmc = getattr(cfg, "USE_GLOBAL_MOTION_COMP", False)
    for skip in range(1, max_skip + 1):                             # T -> T+1+skip (skip missed frames), shortest first
        step = skip + 1
        for t in range(T - step):
            a_ids, a_xyz = frame_ids[t], frame_xyz[t]
            c_ids, c_xyz = frame_ids[t + step], frame_xyz[t + step]
            # candidates: a-nodes with no outgoing, c-nodes with no incoming
            ai = [k for k, nid in enumerate(a_ids) if int(nid) not in has_out]
            ci = [k for k, nid in enumerate(c_ids) if int(nid) not in has_in]
            if not ai or not ci:
                continue
            ap = _phys(a_xyz[ai], scale)
            cp = _phys(c_xyz[ci], scale)
            if use_gmc and len(ai) >= int(getattr(cfg, "GMC_MIN_MATCHES", 5)):   # compensate the global shift OVER the gap
                coarse = _assign(ap, cp, cfg.GAP_CLOSE_DIST_UM * float(getattr(cfg, "GMC_GATE_MULT", 4.0)))
                if len(coarse) >= int(getattr(cfg, "GMC_MIN_MATCHES", 5)):
                    g = np.median(np.array([cp[j] - ap[i] for i, j, _ in coarse]), axis=0)
                    if np.linalg.norm(g) > float(getattr(cfg, "GMC_MIN_SHIFT_UM", 2.0)):
                        ap = ap + g
            for i, j, _d in _assign(ap, cp, cfg.GAP_CLOSE_DIST_UM * step):   # gate scales with the gap length
                sid, did = int(a_ids[ai[i]]), int(c_ids[ci[j]])
                edges.append((sid, did))
                has_out.add(sid)
                has_in.add(did)
    return edges


def prune_isolated(nodes: List[dict], edges: List[Tuple[int, int]]):
    """Drop nodes that never participate in any edge (cuts the count penalty)."""
    used = set()
    for s, d in edges:
        used.add(s)
        used.add(d)
    kept = [n for n in nodes if n["node_id"] in used]
    return kept, edges
