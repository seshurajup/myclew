"""official_score_test — data-wise verifier for the official-score agent (MUST actually run).

Scores a real predictions dir (base unet_transformer geffs) with the ORGANIZER metric via the fleet
contract and confirms official edge/division jaccard + counts come back.
"""
import os, sys, tempfile, glob
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import official_score as A

_PRED = os.path.join(COMP, "research", "pilkwang_support_pack", "repo", "predictions", "seshu",
                     "unet_transformer", "split_0")


def _run():
    print("=== OFFICIAL-SCORE DATA-WISE VERIFIER ===")
    if not glob.glob(os.path.join(_PRED, "*.geff")):
        print(f"  X no prediction geffs at {_PRED}"); return False
    out = tempfile.mkdtemp(prefix="oscore_test_")
    spec = {"method": "unet_transformer", "max": 4, "out_dir": out, "timeout": 900}
    status, res, to, msg = A.run({"question": "official-score smoke", "spec": spec}, "test")
    checks = {}
    checks["status_done"] = status == "done"
    ov = res.get("overall", {}) if isinstance(res, dict) else {}
    checks["edge_jaccard_present"] = isinstance(res, dict) and res.get("edge_jaccard") is not None
    checks["division_jaccard_present"] = isinstance(res, dict) and res.get("division_jaccard") is not None
    checks["scored_positive"] = isinstance(res, dict) and int(res.get("n_scored", 0) or 0) >= 1
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    if isinstance(res, dict):
        print(f"  -> OFFICIAL edge_J={res.get('edge_jaccard')} div_J={res.get('division_jaccard')} "
              f"n={res.get('n_scored')}")
    ok = all(checks.values())
    print(f"=== official-score: {'PASS' if ok else 'FAIL'} ===  {'' if ok else msg}")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
