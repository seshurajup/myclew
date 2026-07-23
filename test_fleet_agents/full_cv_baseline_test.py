"""full_cv_baseline_test — data-wise verifier for the full-cv-baseline agent (MUST actually run).

Runs the real predict+ILP+score pipeline on a 1-dataset slice (CELLMOT_MAX_DATASETS=1) into an
isolated scratch out_dir so the real full_cv results are never touched. Confirms a real
(non-degenerate) full-CV score comes back through the fleet contract.
"""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import full_cv_baseline as A

_DS = os.path.join(COMP, "input", "biohub-cell-tracking-during-development", "train", "44b6_0113de3b.zarr")
_W = os.path.join(COMP, "research", "pilkwang_support_pack", "weights", "unet_transformer", "split_0", "edge_predictor_best.pth")


def _run():
    print("=== FULL-CV-BASELINE DATA-WISE VERIFIER ===")
    if not os.path.exists(_DS) or not os.path.exists(_W):
        print(f"  X missing dataset/weights ({_DS} / {_W})"); return False
    out = tempfile.mkdtemp(prefix="fullcv_base_test_")
    spec = {"max_datasets": 1, "out_dir": out, "deadline_h": 0.5, "timeout": 1200}
    status, res, to, msg = A.run({"question": "baseline smoke", "spec": spec}, "test")
    checks = {}
    checks["status_done"] = status == "done"
    checks["score_real"] = isinstance(res, dict) and 0.3 < float(res.get("cv", 0) or 0) <= 1.0
    checks["n_positive"] = isinstance(res, dict) and int(res.get("n_datasets", 0) or 0) >= 1
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    if isinstance(res, dict):
        print(f"  -> full-CV={res.get('cv')} edge={res.get('edge')} n={res.get('n_datasets')}")
    ok = all(checks.values())
    print(f"=== full-cv-baseline: {'PASS' if ok else 'FAIL'} ===  {'' if ok else msg}")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
