"""Stage-0 canonical scorer — THE metric every EXP row is scored by: the FULL official score.

score = adj_edge_jaccard + 0.1*division_jaccard  (src.metric.official_score), NEVER a component
recall/F1. Node match = 1-to-1 Hungarian <=7um; an edge TP needs BOTH endpoints matched to a GT edge;
over-prediction penalised via t_true = estimated_number_of_nodes. This is the single source of truth;
the training pipeline logs THIS number to MLflow and fleet_agents/scorer.py surfaces the trajectory.

Scores a fold's predictions (final-format geffs, one <ds>.geff per test dataset) against the GT train
geffs, using the frozen embryo-disjoint split. Prints the full metric + components + per-dataset table.

    # score fold 0's predictions:
    research/cellmot_venv/bin/python -m fleet_agents.official_scorer \
        --split learning/ensemble_work/finetune/fleet_loeo_mini.json --fold 0 --pred-dir <preds>
    # wiring smoke-test (reproduce the pilkwang golden-12 anchor ~0.8708; golden-12 = SECONDARY/leaky):
    research/cellmot_venv/bin/python -m fleet_agents.official_scorer --verify-pilk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP))
sys.path.insert(0, str(COMP / "learning" / "ensemble_work"))

from src import io, metric  # noqa: E402
from src.config import Config  # noqa: E402

SRC = Config()
SCALE = SRC.SCALE
GATE = SRC.MATCH_GATE_UM
TRAIN = COMP / "input" / "biohub-cell-tracking-during-development" / "train"
PILK = COMP / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
GOLDEN12 = ["44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"]


def _load_pred_geff(pred_dir: Path, ds: str):
    """Final-format prediction: <ds>.geff with nodes[node_id,t,z,y,x] + edges[source_id,target_id]."""
    p = Path(pred_dir) / f"{ds}.geff"
    if not p.is_dir():
        raise FileNotFoundError(f"no prediction geff for {ds} at {p}")
    n, e = io.read_geff(p)
    return n[["node_id", "t", "z", "y", "x"]], e[["source_id", "target_id"]]


def _load_pilk_solution(pred_dir: Path, ds: str):
    """pilkwang RAW ILP geff (<ds>.zarr.geff): filter solution==True (for the anchor smoke-test)."""
    import pilk_post as P
    for name in (f"{ds}.zarr.geff", f"{ds}.geff"):
        if (Path(pred_dir) / name).is_dir():
            g = P.graph_from_geff(Path(pred_dir) / name)
            nd = g.node_attrs().to_pandas()
            ed = g.edge_attrs().to_pandas()
            if "solution" in nd.columns:
                nd = nd[nd["solution"] == True]  # noqa: E712
            if "solution" in ed.columns:
                ed = ed[ed["solution"] == True]  # noqa: E712
            return (nd[["node_id", "t", "z", "y", "x"]],
                    ed[["source_id", "target_id"]])
    raise FileNotFoundError(f"no pilkwang geff for {ds} under {pred_dir}")


def score_datasets(datasets, pred_dir, loader=_load_pred_geff, gt_dir=TRAIN, gate_um=None,
                   skip_missing=False):
    """Return (official_score_dict, per_dataset_rows). Full metric only.

    gate_um: override the 7um Hungarian match gate (default None → the config's MATCH_GATE_UM).
    skip_missing: if True, datasets whose GT/prediction geff is absent are skipped instead of raising.
    """
    try:
        gate = float(gate_um) if gate_um is not None else GATE
    except (TypeError, ValueError):
        gate = GATE
    rows = []
    for ds in datasets:
        try:
            gn, ge = io.read_geff(Path(gt_dir) / f"{ds}.geff")
            estN = io.geff_estimated_nodes(Path(gt_dir) / f"{ds}.geff")
            pn, pe = loader(pred_dir, ds)
        except Exception:  # noqa: BLE001
            if skip_missing:
                continue
            raise
        c = metric.official_counts(gn, ge, pn, pe, SCALE, gate, t_true=estN)
        c["dataset"] = ds
        rows.append(c)
    return metric.official_score(rows), rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(COMP / "learning/ensemble_work/finetune/fleet_loeo_mini.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--pred-dir", default=None, help="dir of final-format <ds>.geff predictions")
    ap.add_argument("--verify-pilk", action="store_true",
                    help="wiring smoke-test: score pilkwang golden-12 (SECONDARY/leaky) -> expect ~0.8708")
    ap.add_argument("--gate-um", type=float, default=None,
                    help="override the Hungarian node-match gate in um (default: config MATCH_GATE_UM)")
    ap.add_argument("--skip-missing", action="store_true",
                    help="skip datasets whose GT/prediction geff is absent instead of erroring")
    args = ap.parse_args(argv)

    if args.verify_pilk:
        datasets, pred_dir, loader = GOLDEN12, PILK, _load_pilk_solution
        scope = "golden-12 (SECONDARY/leaky wiring smoke-test)"
    else:
        if not args.pred_dir:
            ap.error("--pred-dir required unless --verify-pilk")
        folds = json.load(open(args.split))
        datasets, pred_dir, loader = folds[args.fold]["test"], Path(args.pred_dir), _load_pred_geff
        scope = f"fold{args.fold} test ({len(datasets)} datasets)"

    agg, rows = score_datasets(datasets, pred_dir, loader, gate_um=args.gate_um,
                               skip_missing=args.skip_missing)
    print(f"OFFICIAL SCORE — {scope}")
    print(f"  official_score   = {agg['score']:.4f}   (= adj_edge_jaccard + 0.1*division_jaccard)")
    print(f"  adj_edge_jaccard = {agg['adj_edge_jaccard']:.4f}")
    print(f"  division_jaccard = {agg['division_jaccard']}")
    print(f"  n_datasets       = {len(rows)}")
    return agg


if __name__ == "__main__":
    main()
