"""Metric-decomposition adapter — deterministic, from real MLflow component metrics.

Decomposes the official score (edge-Jaccard + 0.1*division-Jaccard) into failure buckets using
adjJ ≈ node_recall^2 * edge_precision and div_J = the +0.1 lever. Uses only urllib (venv-agnostic).
The scoring itself is the competition's src.metric.official_score, run by the training pipeline and
logged to MLflow — we read those component metrics back here.
"""
from __future__ import annotations

import json
import urllib.request

MLFLOW = "http://127.0.0.1:5000"
EXPERIMENT = "kaggle-biohub-cell-tracking"


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _scored_runs():
    try:
        exp = json.load(urllib.request.urlopen(
            f"{MLFLOW}/api/2.0/mlflow/experiments/get-by-name?experiment_name={EXPERIMENT}", timeout=6
        ))["experiment"]["experiment_id"]
        req = urllib.request.Request(
            f"{MLFLOW}/api/2.0/mlflow/runs/search",
            data=json.dumps({"experiment_ids": [exp], "max_results": 100,
                             "order_by": ["attributes.start_time DESC"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        runs = json.load(urllib.request.urlopen(req, timeout=6)).get("runs", [])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for r in runs:
        m = {x["key"]: float(x["value"]) for x in r.get("data", {}).get("metrics", []) if _isnum(x["value"])}
        if "adj_edge_jaccard" in m or "official_score" in m:
            out.append({"name": r["info"].get("run_name", r["info"]["run_id"][:8]), "metrics": m})
    return out


def decompose(q, worker):
    runs = _scored_runs()
    if not runs:
        return ("escalated", {"reason": "no scored runs in MLflow"}, "researcher",
                f"[{worker}] THREAD-A blocked: no scored runs in MLflow yet. Need one baseline scored "
                f"(src.metric.official_score → component metrics logged). Trigger a mini baseline and "
                f"I'll decompose it deterministically.")
    best = max(runs, key=lambda r: r["metrics"].get("official_score",
                                                    r["metrics"].get("adj_edge_jaccard", 0.0)))
    m = best["metrics"]
    nr, ep, dj, cr = (m.get("mean_node_recall"), m.get("adj_edge_jaccard"),
                      m.get("division_jaccard"), m.get("mean_count_ratio"))
    parts, buckets = [], []
    if nr is not None:
        parts.append(f"node_recall={nr:.4f}")
        buckets.append(("node_recall", nr, "detection recall is the bottleneck"
                        if nr < 0.95 else "node recall ~saturated (not the bottleneck)"))
    if ep is not None:
        parts.append(f"adj_edge_J={ep:.4f}")
    if dj is not None:
        parts.append(f"division_J={dj:.4f}")
        buckets.append(("division", dj, "division term (+0.1) almost entirely UNCLAIMED = biggest headroom"
                        if dj < 0.05 else "division term partially claimed"))
    if cr is not None:
        parts.append(f"count_ratio={cr:.3f}")
        if cr > 1.15 or cr < 0.85:
            buckets.append(("count_calibration", cr, "count mis-calibrated (over/under-detecting)"))
    ranked = sorted(buckets, key=lambda b: (b[0] != "division", b[1]))
    top = ranked[0][2] if ranked else "need more component metrics"
    result = {"run": best["name"], "node_recall": nr, "edge_J": ep, "div_J": dj, "count_ratio": cr,
              "weakest_bucket": ranked[0][0] if ranked else None, "buckets": [b[2] for b in ranked]}
    msg = (f"[{worker}] THREAD-A metric decomposition (run {best['name'][:26]}): " + ", ".join(parts) +
           f". Weakest bucket → {top}. (adjJ≈node_rec²·edge_prec; div_J is the +0.1 lever.)")
    return ("done", result, "all", msg)
