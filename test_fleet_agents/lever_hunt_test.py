"""lever_hunt_test — data-wise verifier for the lever-hunt agent (MUST actually run the loop).

Runs the real metric-driven mini-experiment on a 2-dataset slice (one per embryo) with max_k=1 and confirms:
XAI produces a goal + subset + gap-length histogram, gap_fill runs, and the OFFICIAL patched metric returns a
real before/after Δ (edge + penalised adj) with a SOLID/DEAD verdict. Nothing is mocked — it scores geffs with
the host tracking_cellmot metric.
"""
import os, sys, tempfile, glob
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import lever_hunt as A

_PRED = os.path.join(COMP, "research", "official_repo", "predictions", "seshu", "cv_flip", "split_0")


def _run():
    print("=== LEVER-HUNT DATA-WISE VERIFIER ===")
    if not glob.glob(os.path.join(_PRED, "*.geff")):
        print(f"  X no prediction geffs at {_PRED}"); return False
    out = tempfile.mkdtemp(prefix="leverhunt_test_")
    spec = {"mode": "both", "lever": "gap_fill", "max_k": 1,
            "datasets": ["44b6_c50204e0", "6bba_05db0fb1"], "out_dir": out, "timeout": 1200}
    status, res, to, msg = A.run({"question": "lever-hunt smoke", "spec": spec}, "test")
    checks = {}
    checks["status_done"] = status == "done"
    checks["goal_present"] = isinstance(res, dict) and bool(res.get("goal"))
    checks["subset_present"] = isinstance(res, dict) and bool(res.get("subset"))
    checks["gap_hist_present"] = isinstance(res, dict) and res.get("gap_length_hist") is not None
    checks["delta_edge_present"] = isinstance(res, dict) and res.get("d_edge_jaccard") is not None
    checks["delta_adj_present"] = isinstance(res, dict) and res.get("d_adj_edge_jaccard") is not None
    checks["verdict_present"] = isinstance(res, dict) and res.get("verdict") in ("SOLID CLUE", "DEAD")
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    if isinstance(res, dict):
        print(f"  -> goal={res.get('goal')}")
        print(f"  -> verdict={res.get('verdict')} Δedge={res.get('d_edge_jaccard')} "
              f"Δadj={res.get('d_adj_edge_jaccard')} @k≤{res.get('chosen_max_k')} ({res.get('recommended_mode')})")
    ok = all(checks.values())
    print(f"=== lever-hunt: {'PASS' if ok else 'FAIL'} ===  {'' if ok else msg}")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
