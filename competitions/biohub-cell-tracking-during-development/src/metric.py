"""LOCAL metric replica for offline, reward-aligned iteration.

Reconstructed from the public EDA "metric mental model" + data contract. The OFFICIAL
aggregation weights are not public (the Kaggle Evaluation page, likely the `traccuracy`
library); treat `score` as a *proxy* and CALIBRATE component weights against real LB feedback.
What IS faithful and load-bearing:
  * node match = one-to-one within MATCH_GATE_UM (7µm) in scaled µm,
  * an EDGE is a true positive only if BOTH endpoints are matched to the GT endpoints of a
    GT edge (=> R_edge ~ R_node^2 * Q_link),
  * over-prediction is penalised (node precision / count ratio),
  * divisions (out-degree >= 2) are scored separately.

Inputs are node/edge tables in the submission schema (node_id unique within a dataset).
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def official_counts(gt_nodes, gt_edges, pred_nodes, pred_edges, scale, gate_um=7.0, t_true=None):
    """EXACT official metric counts for one dataset (from metrics.md / tracking_cellmot).

    Returns dict with edge_tp/fp/fn, div_tp/fp/fn, t_pred, t_true, and the per-sample
    adjusted edge jaccard. Node match = one-to-one within 7µm; edge TP needs both endpoints
    matched to GT nodes joined by a GT edge; FP = a 'valid' pred edge (endpoint matches a GT
    node that has degree) that isn't TP; FN = GT edge with no match; others ignored.
    adjusted = max(0, J·(1 − 0.1·(T_pred − T_true)/T_true)), T_true = estimated_number_of_nodes.
    """
    # pred_node_id -> gt_node_id  (invert the gt->pred matching)
    g2p = _match_nodes(gt_nodes, pred_nodes, scale, gate_um)
    p2g = {p: g for g, p in g2p.items()}

    gt_edge_set = set(map(tuple, gt_edges[["source_id", "target_id"]].to_numpy())) if len(gt_edges) else set()
    # GT node degree: which GT nodes have an outgoing / incoming edge
    gt_out = set(int(s) for s in gt_edges["source_id"]) if len(gt_edges) else set()
    gt_in = set(int(t) for t in gt_edges["target_id"]) if len(gt_edges) else set()

    tp = fp = 0
    for ps, pt in pred_edges[["source_id", "target_id"]].to_numpy() if len(pred_edges) else []:
        A = p2g.get(int(ps)); B = p2g.get(int(pt))
        # 'valid' predicted edge: source matched a GT node with out-degree, OR target matched one with in-degree
        valid = (A is not None and A in gt_out) or (B is not None and B in gt_in)
        if not valid:
            continue  # ignored
        if A is not None and B is not None and (A, B) in gt_edge_set:
            tp += 1
        else:
            fp += 1
    fn = len(gt_edges) - tp

    # divisions: GT node with out-degree>=2; matched if its mapped pred node also has out-degree>=2
    def outdeg(edges):
        from collections import Counter
        return Counter(int(s) for s in edges["source_id"]) if len(edges) else Counter()
    gt_od = outdeg(gt_edges); pr_od = outdeg(pred_edges)
    gt_div = {n for n, c in gt_od.items() if c >= 2}
    pred_div_nodes = {n for n, c in pr_od.items() if c >= 2}
    dtp = sum(1 for g in gt_div if g2p.get(g) in pred_div_nodes)
    dfn = len(gt_div) - dtp
    # FP divisions: predicted forks whose matched GT node exists but isn't a GT division
    dfp = sum(1 for p in pred_div_nodes if (p in p2g) and (p2g[p] not in gt_div))

    t_pred = len(pred_nodes)
    if t_true is None or not np.isfinite(t_true) or t_true <= 0:
        t_true = max(len(gt_nodes), 1)  # fallback (sparse GT) — prefer passing real est
    jac = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    adj = max(0.0, jac * (1 - 0.1 * (t_pred - t_true) / t_true))
    return dict(edge_tp=tp, edge_fp=fp, edge_fn=fn, div_tp=dtp, div_fp=dfp, div_fn=dfn,
                t_pred=t_pred, t_true=float(t_true), jaccard=jac, adj_jaccard=adj,
                w=tp + fp + fn)


def official_score(per_dataset_counts) -> dict:
    """Aggregate per-dataset official counts into the run-level LB score.

    adj_edge_jaccard = Σ w_i·adj_jaccard_i / Σ w_i   (weight-avg by w=TP+FP+FN)
    division_jaccard = Σdtp / (Σdtp+Σdfp+Σdfn)        (micro)
    score = adj_edge_jaccard + 0.1·division_jaccard
    """
    rows = list(per_dataset_counts)
    W = sum(r["w"] for r in rows)
    adj_edge = sum(r["w"] * r["adj_jaccard"] for r in rows) / W if W else 0.0
    dtp = sum(r["div_tp"] for r in rows); dfp = sum(r["div_fp"] for r in rows); dfn = sum(r["div_fn"] for r in rows)
    has_div = (dtp + dfp + dfn) > 0
    div_j = dtp / (dtp + dfp + dfn) if has_div else 0.0
    score = adj_edge + 0.1 * div_j if has_div else adj_edge
    return dict(adj_edge_jaccard=adj_edge, division_jaccard=div_j if has_div else float("nan"), score=score)


def _match_nodes(gt: pd.DataFrame, pred: pd.DataFrame, scale, gate_um: float) -> Dict[int, int]:
    """Per-timepoint one-to-one matching within the gate. Returns {gt_node_id: pred_node_id}."""
    scale = np.asarray(scale)
    mapping: Dict[int, int] = {}
    for t in sorted(set(gt["t"].unique())):
        g = gt[gt["t"] == t]
        p = pred[pred["t"] == t]
        if len(g) == 0 or len(p) == 0:
            continue
        G = g[["z", "y", "x"]].to_numpy(np.float64) * scale[None, :]
        P = p[["z", "y", "x"]].to_numpy(np.float64) * scale[None, :]
        D = np.sqrt(((G[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
        cost = np.where(D <= gate_um, D, 1e9)
        ri, ci = linear_sum_assignment(cost)
        gid = g["node_id"].to_numpy()
        pid = p["node_id"].to_numpy()
        for r, c in zip(ri, ci):
            if cost[r, c] < 1e9:
                mapping[int(gid[r])] = int(pid[c])
    return mapping


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_dataset(gt_nodes, gt_edges, pred_nodes, pred_edges, scale, gate_um=7.0,
                  weights=(1 / 3, 1 / 3, 1 / 3)) -> dict:
    """Score one dataset. Returns component metrics + a (proxy) combined score."""
    m = _match_nodes(gt_nodes, pred_nodes, scale, gate_um)  # gt_id -> pred_id
    n_gt, n_pred, n_match = len(gt_nodes), len(pred_nodes), len(m)
    node_p, node_r, node_f = _prf(n_match, n_pred - n_match, n_gt - n_match)

    # edges: TP iff both endpoints matched and the mapped pred edge exists
    pred_edge_set = set(map(tuple, pred_edges[["source_id", "target_id"]].to_numpy())) \
        if len(pred_edges) else set()
    e_tp = 0
    for s, d in gt_edges[["source_id", "target_id"]].to_numpy():
        ps, pd_ = m.get(int(s)), m.get(int(d))
        if ps is not None and pd_ is not None and (ps, pd_) in pred_edge_set:
            e_tp += 1
    e_fn = len(gt_edges) - e_tp
    e_fp = len(pred_edges) - e_tp
    edge_p, edge_r, edge_f = _prf(e_tp, e_fp, e_fn)

    # divisions: nodes with out-degree >= 2
    def div_set(edges):
        out = pd.Series(edges["source_id"]).value_counts()
        return set(int(i) for i, c in out.items() if c >= 2)
    gt_div = div_set(gt_edges) if len(gt_edges) else set()
    pred_div = div_set(pred_edges) if len(pred_edges) else set()
    d_tp = sum(1 for g in gt_div if m.get(g) in pred_div)
    d_fn = len(gt_div) - d_tp
    d_fp = len(pred_div) - d_tp
    div_p, div_r, div_f = _prf(d_tp, d_fp, d_fn)

    wn, we, wd = weights
    combined = wn * node_f + we * edge_f + wd * div_f
    return {
        "n_gt": n_gt, "n_pred": n_pred, "n_match": n_match,
        "count_ratio": (n_pred / n_gt) if n_gt else float("nan"),
        "node_p": node_p, "node_r": node_r, "node_f": node_f,
        "edge_p": edge_p, "edge_r": edge_r, "edge_f": edge_f,
        "div_p": div_p, "div_r": div_r, "div_f": div_f,
        "score_proxy": combined,
    }


def score_submission(gt: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                     pred_df: pd.DataFrame, scale, gate_um=7.0) -> pd.DataFrame:
    """Score a full submission DataFrame against {dataset: (gt_nodes, gt_edges)}."""
    rows = []
    for ds, (gn, ge) in gt.items():
        sub = pred_df[pred_df["dataset"] == ds]
        pn = sub[sub["row_type"] == "node"][["node_id", "t", "z", "y", "x"]]
        pe = sub[sub["row_type"] == "edge"][["source_id", "target_id"]]
        r = score_dataset(gn, ge, pn, pe, scale, gate_um)
        r["dataset"] = ds
        rows.append(r)
    df = pd.DataFrame(rows)
    return df
