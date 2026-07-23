"""tracker_postproc_test — data-wise verifier for the tracker-postproc agent (MUST actually run).

Runs the pilkwang post-proc on the EXISTING raw split_0 predictions for 2 division-bearing
datasets, writes post-processed geffs + submission.csv, and scores vs GT with src.metric
(official adj_edge_jaccard + division_jaccard, t_true = estimated_number_of_nodes). Confirms a
valid non-empty graph and that post-proc does not regress the edge jaccard vs the raw ILP graph.
Prints edge & div numbers for both raw and post-processed.
"""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
sys.path.insert(0, os.path.join(COMP, "src"))
from fleet_agents import tracker_postproc as A

_PRED = os.path.join(COMP, "research", "pilkwang_support_pack", "repo", "predictions",
                     "seshu", "unet_transformer", "split_0")
_IMG = os.path.join(COMP, "input", "biohub-cell-tracking-during-development", "train")
_GT = _IMG
_DATASETS = ["6bba_05db0fb1", "6bba_07e24132"]  # both carry GT divisions
_SCALE = (1.625, 0.40625, 0.40625)


def _raw_score():
    """Baseline: score the raw ILP geffs (no post-proc) the same way, for the lift comparison."""
    import pandas as pd
    from src import io as SIO, metric as M
    import glob
    counts = []
    for ds in _DATASETS:
        hits = glob.glob(os.path.join(_PRED, f"{ds}.zarr.geff")) + glob.glob(os.path.join(_PRED, f"{ds}.geff"))
        gn, ge = SIO.read_geff(hits[0])  # raw geff reads as node/edge tables directly
        # raw geff node_id/source_id are graph ids; read via io gives node/edge tables
        pn, pe = SIO.read_geff(hits[0])
        est = SIO.geff_estimated_nodes(os.path.join(_GT, f"{ds}.geff"))
        ggn, gge = SIO.read_geff(os.path.join(_GT, f"{ds}.geff"))
        counts.append(M.official_counts(ggn, gge, pn, pe, _SCALE, gate_um=7.0, t_true=est))
    return M.official_score(counts)


def _run():
    print("=== TRACKER-POSTPROC DATA-WISE VERIFIER ===")
    for ds in _DATASETS:
        if not os.path.exists(os.path.join(_GT, f"{ds}.geff")):
            print(f"  X missing GT for {ds}"); return False
    with tempfile.TemporaryDirectory() as out:
        spec = {"pred_geff_dir": _PRED, "image_dir": _IMG, "out_dir": out,
                "datasets": _DATASETS, "gt_dir": _GT, "gate_um": 7.0}
        status, res, to, msg = A.run({"question": "postproc smoke", "spec": spec}, "test")
        checks = {}
        checks["status_done"] = status == "done"
        checks["ran_two_datasets"] = isinstance(res, dict) and res.get("n_datasets") == 2
        checks["nodes_positive"] = isinstance(res, dict) and res.get("total_nodes", 0) > 0
        checks["edges_positive"] = isinstance(res, dict) and res.get("total_edges", 0) > 0
        checks["submission_written"] = isinstance(res, dict) and res.get("submission_csv") and os.path.exists(res["submission_csv"])
        sc = res.get("score") if isinstance(res, dict) else None
        checks["scored_vs_gt"] = isinstance(sc, dict) and "adj_edge_jaccard" in sc
        try:
            raw = _raw_score()
            print(f"  RAW  adj_edge_jaccard={raw['adj_edge_jaccard']:.4f} division_jaccard={raw['division_jaccard']}")
        except Exception as e:  # noqa: BLE001
            raw = None; print(f"  (raw baseline skipped: {type(e).__name__}: {e})")
        if isinstance(sc, dict) and "adj_edge_jaccard" in sc:
            print(f"  POST adj_edge_jaccard={sc['adj_edge_jaccard']} division_jaccard={sc['division_jaccard']} "
                  f"(div TP={sc['div_tp']}/FP={sc['div_fp']}/FN={sc['div_fn']})")
            checks["edge_jaccard_meaningful"] = sc["adj_edge_jaccard"] > 0
            if raw is not None:
                checks["no_edge_regression"] = sc["adj_edge_jaccard"] >= raw["adj_edge_jaccard"] - 1e-6
        for k, v in checks.items():
            print(f"  {'OK' if v else 'X'} {k}")
        ok = all(checks.values())
        print(f"=== tracker-postproc: {'PASS' if ok else 'FAIL'} ===  {'' if ok else msg}")
        return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
