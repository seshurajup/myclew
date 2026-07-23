"""tracker_predict_test — data-wise verifier for the tracker-predict agent (MUST actually run).

Runs the real UNet+transformer detect+edge+ILP pipeline on ONE dataset, capped to a few frames
(via the runner's --max-frames), writing to a SELF-TEST method dir so the real split_0
predictions are never touched. Confirms a non-empty predicted graph (nodes>0).
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import tracker_predict as A

_DS = os.path.join(COMP, "input", "biohub-cell-tracking-during-development", "train", "44b6_0113de3b.zarr")
_W = os.path.join(COMP, "research", "pilkwang_support_pack", "weights", "unet_transformer", "split_0", "edge_predictor_best.pth")


def _run():
    print("=== TRACKER-PREDICT DATA-WISE VERIFIER ===")
    if not os.path.exists(_DS) or not os.path.exists(_W):
        print(f"  X missing dataset/weights ({_DS} / {_W})"); return False
    spec = {"method": "unet_transformer_selftest", "split": "0", "weights": _W,
            "debug_video": _DS, "det_threshold": 0.5, "use_ilp": True,
            "max_frames": 3, "timeout": 600}
    status, res, to, msg = A.run({"question": "predict smoke", "spec": spec}, "test")
    checks = {}
    checks["status_done"] = status == "done"
    checks["nodes_positive"] = isinstance(res, dict) and res.get("total_nodes", 0) > 0
    checks["geff_written"] = isinstance(res, dict) and res.get("n_geffs", 0) >= 1
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    if isinstance(res, dict):
        print(f"  -> predicted nodes={res.get('total_nodes')} geffs={res.get('n_geffs')} "
              f"dir={res.get('predictions_dir')}")
    ok = all(checks.values())
    print(f"=== tracker-predict: {'PASS' if ok else 'FAIL'} ===  {'' if ok else msg}")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
