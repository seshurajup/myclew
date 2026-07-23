"""detect_quality_test — data-wise verifier for the detect-quality agent (MUST actually run).

Runs the real pilkwang detector on ONE external dense crop / few frames into an isolated scratch
out_dir, and confirms real per-embryo recall+precision come back through the fleet contract.
"""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import detect_quality as A

_EXT = os.path.join(COMP, "research", "zebrahub", "geff_trainset", "ZSNS001_c0.zarr")


def _run():
    print("=== DETECT-QUALITY DATA-WISE VERIFIER ===")
    if not os.path.exists(_EXT):
        print(f"  X missing external crop ({_EXT})"); return False
    out = tempfile.mkdtemp(prefix="detq_test_")
    spec = {"embryos": "ZSNS001", "max_crops": 1, "max_frames": 4, "det_threshold": 0.99,
            "out_dir": out, "timeout": 900}
    status, res, to, msg = A.run({"question": "detect-quality smoke", "spec": spec}, "test")
    checks = {}
    checks["status_done"] = status == "done"
    per = res.get("per_embryo", {}) if isinstance(res, dict) else {}
    z = per.get("ZSNS001", {})
    checks["recall_real"] = 0.3 < float(z.get("recall", 0) or 0) <= 1.0
    checks["precision_real"] = 0.1 < float(z.get("precision", 0) or 0) <= 1.0
    checks["counts_positive"] = int(z.get("n_gt", 0) or 0) > 0 and int(z.get("n_pred", 0) or 0) > 0
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    if z:
        print(f"  -> ZSNS001 recall={z.get('recall')} precision={z.get('precision')} "
              f"GT={z.get('n_gt')} pred={z.get('n_pred')} matched={z.get('n_match')}")
    ok = all(checks.values())
    print(f"=== detect-quality: {'PASS' if ok else 'FAIL'} ===  {'' if ok else msg}")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
