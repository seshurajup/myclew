"""lora_validate_test — data-wise verifier for the lora-validate agent (MUST actually run).

Loads a REAL trained adapter (det_v3/adapter_best) + PEFT + base, scores a 1-dataset slice
(CELLMOT_MAX_DATASETS=1) into an isolated scratch out_dir, and confirms a real full-CV score
plus a parsed delta come back through the fleet contract. Does NOT assert a WIN (small slice) —
only that the honest validation machinery runs end-to-end and returns a delta.
"""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import lora_validate as A

_DS = os.path.join(COMP, "input", "biohub-cell-tracking-during-development", "train", "44b6_0113de3b.zarr")
_ADAPTER = os.path.join(COMP, "research", "lora_finetune", "runs", "det_v3", "adapter_best")


def _run():
    print("=== LORA-VALIDATE DATA-WISE VERIFIER ===")
    if not os.path.exists(_DS) or not os.path.exists(_ADAPTER):
        print(f"  X missing dataset/adapter ({_DS} / {_ADAPTER})"); return False
    out = tempfile.mkdtemp(prefix="fullcv_lora_test_")
    spec = {"adapter": _ADAPTER, "max_datasets": 1, "out_dir": out, "baseline": 0.8675, "timeout": 1200}
    status, res, to, msg = A.run({"question": "validate smoke", "spec": spec}, "test")
    checks = {}
    checks["status_done"] = status == "done"
    checks["score_real"] = isinstance(res, dict) and 0.3 < float(res.get("cv", 0) or 0) <= 1.0
    checks["delta_parsed"] = isinstance(res, dict) and res.get("delta") is not None
    checks["n_positive"] = isinstance(res, dict) and int(res.get("n_datasets", 0) or 0) >= 1
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    if isinstance(res, dict):
        print(f"  -> full-CV={res.get('cv')} Δ={res.get('delta')} improved={res.get('improved')} "
              f"n={res.get('n_datasets')}")
    ok = all(checks.values())
    print(f"=== lora-validate: {'PASS' if ok else 'FAIL'} ===  {'' if ok else msg}")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
