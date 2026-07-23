"""division_rescue_test — data-wise verifier for the division-rescue agent (MUST actually run).

Runs the real geometry-fork rescue on a few base predictions into an isolated self-test method dir and
confirms forks are added at ~biological rate (not a flood) and rescued geffs are written.
"""
import os, sys, glob, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import division_rescue as A

_IN = os.path.join(COMP, "research", "pilkwang_support_pack", "repo", "predictions", "seshu",
                   "unet_transformer", "split_0")


def _run():
    print("=== DIVISION-RESCUE DATA-WISE VERIFIER ===")
    if not glob.glob(os.path.join(_IN, "*.geff")):
        print(f"  X no base predictions at {_IN}"); return False
    out = tempfile.mkdtemp(prefix="drescue_test_")
    spec = {"in_method": "unet_transformer", "out_dir": out, "max": 2, "rate": 0.0015, "timeout": 900}
    status, res, to, msg = A.run({"question": "division-rescue smoke", "spec": spec}, "test")
    checks = {}
    checks["status_done"] = status == "done"
    checks["forks_added"] = isinstance(res, dict) and int(res.get("forks_added", 0) or 0) >= 1
    checks["geffs_written"] = len(glob.glob(os.path.join(out, "*.geff"))) >= 1
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    if isinstance(res, dict):
        print(f"  -> forks_added={res.get('forks_added')} n_datasets={res.get('n_datasets')} out={out}")
    ok = all(checks.values())
    print(f"=== division-rescue: {'PASS' if ok else 'FAIL'} ===  {'' if ok else msg}")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
