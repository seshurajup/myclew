"""Stage-1 metric anatomy — decompose the FULL official score into failure buckets.

The headline stays `official_score = adj_edge_jaccard + 0.1*division_jaccard`
(fleet_agents.official_scorer is THE scorer). This tool does NOT replace it — it reads
WHERE a baseline loses, so the leader can pick the next lever. Buckets (micro-aggregated
over a fold's test datasets, tied to the same GT geffs + estN the official scorer uses):

  R_node  = Σ n_match / Σ n_gt                      node recall (detection)
  R_edge  = Σ edge_tp / Σ (edge_tp + edge_fn)       edge recall (both endpoints matched + linked)
  Q_link  = R_edge / R_node**2                      implied per-edge link quality GIVEN detection
                                                    (R_edge ≈ R_node^2 · Q_link; isolates linking from detection)
  count   = Σ t_pred / Σ estN  (ratio)              over/under-prediction; estN = estimated_number_of_nodes.
            penalty = weighted mean adj_jaccard/jaccard  (realized count penalty factor; <1 over-pred, >1 under-pred bonus)
  edge_P  = Σ edge_tp / Σ (edge_tp + edge_fp)        edge precision (FP-edge flooding)
  div_J   = Σdtp / (Σdtp+Σdfp+Σdfn)                 division jaccard (micro; the rare 0.1 term)

Same interface as the scorer:
    research/cellmot_venv/bin/python -m fleet_agents.metric_anatomy \
        --split learning/ensemble_work/finetune/fleet_loeo_mini.json --fold 0 --pred-dir <preds>
    research/cellmot_venv/bin/python -m fleet_agents.metric_anatomy --verify-pilk   # golden-12 SECONDARY smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fleet_agents.official_scorer import (
    COMP, TRAIN, PILK, GOLDEN12, SCALE, GATE,
    _load_pred_geff, _load_pilk_solution,
)
from src import io, metric  # noqa: E402


def anatomy(datasets, pred_dir, loader=_load_pred_geff, gt_dir=TRAIN):
    """Return (aggregate_buckets, per_dataset_rows). Reuses the official counts + estN."""
    rows = []
    for ds in datasets:
        gn, ge = io.read_geff(Path(gt_dir) / f"{ds}.geff")
        estN = io.geff_estimated_nodes(Path(gt_dir) / f"{ds}.geff")
        pn, pe = loader(pred_dir, ds)
        c = metric.official_counts(gn, ge, pn, pe, SCALE, GATE, t_true=estN)   # edge/div counts + estN penalty
        d = metric.score_dataset(gn, ge, pn, pe, SCALE, GATE)                  # node recall / n_match / n_gt
        rows.append({
            "dataset": ds,
            "n_gt": d["n_gt"], "n_match": d["n_match"], "t_pred": c["t_pred"], "estN": c["t_true"],
            "edge_tp": c["edge_tp"], "edge_fp": c["edge_fp"], "edge_fn": c["edge_fn"],
            "div_tp": c["div_tp"], "div_fp": c["div_fp"], "div_fn": c["div_fn"],
            "jaccard": c["jaccard"], "adj_jaccard": c["adj_jaccard"],
            "R_node": d["node_r"], "R_edge": d["edge_r"], "count_ratio": (c["t_pred"] / c["t_true"]) if c["t_true"] else float("nan"),
        })

    def _sum(k):
        return sum(r[k] for r in rows)

    sum_gt, sum_match = _sum("n_gt"), _sum("n_match")
    e_tp, e_fp, e_fn = _sum("edge_tp"), _sum("edge_fp"), _sum("edge_fn")
    d_tp, d_fp, d_fn = _sum("div_tp"), _sum("div_fp"), _sum("div_fn")
    R_node = sum_match / sum_gt if sum_gt else float("nan")
    R_edge = e_tp / (e_tp + e_fn) if (e_tp + e_fn) else float("nan")
    Q_link = (R_edge / (R_node ** 2)) if R_node else float("nan")
    edge_P = e_tp / (e_tp + e_fp) if (e_tp + e_fp) else float("nan")
    div_J = d_tp / (d_tp + d_fp + d_fn) if (d_tp + d_fp + d_fn) else float("nan")
    # realized count-penalty factor: weight-avg adj/jac by edge weight w = tp+fp+fn (same weighting as the headline)
    W = sum((r["edge_tp"] + r["edge_fp"] + r["edge_fn"]) for r in rows)
    pen = (sum((r["edge_tp"] + r["edge_fp"] + r["edge_fn"]) * (r["adj_jaccard"] / r["jaccard"] if r["jaccard"] else 1.0)
               for r in rows) / W) if W else float("nan")
    agg = metric.official_score(rows_for_score(rows))
    return {
        "official_score": agg["score"], "adj_edge_jaccard": agg["adj_edge_jaccard"],
        "division_jaccard": agg["division_jaccard"],
        "R_node": R_node, "R_edge": R_edge, "Q_link": Q_link, "edge_P": edge_P,
        "count_ratio": (_sum("t_pred") / _sum("estN")) if _sum("estN") else float("nan"),
        "count_penalty": pen, "div_J": div_J, "n_datasets": len(rows),
    }, rows


def rows_for_score(rows):
    """Re-shape anatomy rows into the shape metric.official_score expects (needs w + adj_jaccard + div)."""
    out = []
    for r in rows:
        w = r["edge_tp"] + r["edge_fp"] + r["edge_fn"]
        out.append({"w": w, "adj_jaccard": r["adj_jaccard"],
                    "div_tp": r["div_tp"], "div_fp": r["div_fp"], "div_fn": r["div_fn"]})
    return out


def _load_fold(split_path, fold):
    folds = json.loads(Path(split_path).read_text())
    return folds[fold]["test"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage-1 metric anatomy (R_node/R_edge/Q_link/count).")
    ap.add_argument("--split", default=str(COMP / "learning/ensemble_work/finetune/fleet_loeo_mini.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--pred-dir", default=None, help="dir of final-format <ds>.geff predictions")
    ap.add_argument("--verify-pilk", action="store_true",
                    help="golden-12 SECONDARY smoke: decompose pilkwang raw-ILP preds (proves the tool runs)")
    ap.add_argument("--json", action="store_true", help="emit the aggregate buckets as JSON")
    args = ap.parse_args(argv)

    if args.verify_pilk:
        datasets, pred_dir, loader, scope = GOLDEN12, PILK, _load_pilk_solution, "golden-12 (SECONDARY/leaky) pilk raw-ILP"
    else:
        if not args.pred_dir:
            ap.error("--pred-dir required unless --verify-pilk")
        datasets = _load_fold(args.split, args.fold)
        pred_dir, loader, scope = args.pred_dir, _load_pred_geff, f"fold {args.fold} ({len(datasets)} test ds)"

    agg, rows = anatomy(datasets, pred_dir, loader=loader)

    if args.json:
        print(json.dumps(agg, indent=2))
        return

    print(f"METRIC ANATOMY — {scope}")
    print(f"  official_score   = {agg['official_score']:.4f}   (= adj_edge_jaccard + 0.1*division_jaccard)")
    print(f"  adj_edge_jaccard = {agg['adj_edge_jaccard']:.4f}   division_jaccard = {agg['division_jaccard']}")
    print( "  ── anatomy (where the score is lost) ──")
    print(f"  R_node (detection recall)     = {agg['R_node']:.4f}")
    print(f"  R_edge (edge recall)          = {agg['R_edge']:.4f}   [R_edge ≈ R_node^2 · Q_link]")
    print(f"  Q_link (link quality|matched) = {agg['Q_link']:.4f}")
    print(f"  edge_P (edge precision)       = {agg['edge_P']:.4f}")
    print(f"  count_ratio (t_pred/estN)     = {agg['count_ratio']:.4f}   penalty_factor = {agg['count_penalty']:.4f}")
    print(f"  div_J  (division jaccard)     = {agg['div_J']:.4f}")
    print(f"  n_datasets                    = {agg['n_datasets']}")


if __name__ == "__main__":
    main()
