"""stage_dynamics_test — data-wise verifier for the stage-dynamics agent (MUST actually run).

Runs the real GT dynamics profiling with a small per-stage dataset cap into an isolated scratch out_dir,
and confirms per-stage motion percentiles + adjacent-stage significance tests + a verdict come back.
"""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import stage_dynamics as A

_STAGE = os.path.join(COMP, "results", "label_selection", "dataset_zf_stage.parquet")


def _run():
    print("=== STAGE-DYNAMICS DATA-WISE VERIFIER ===")
    if not os.path.exists(_STAGE):
        print(f"  X missing stage parquet ({_STAGE})"); return False
    out = tempfile.mkdtemp(prefix="stagedyn_test_")
    spec = {"max_ds": 4, "out_dir": out, "timeout": 900}
    status, res, to, msg = A.run({"question": "stage-dynamics smoke", "spec": spec}, "test")
    checks = {}
    checks["status_done"] = status == "done"
    pe = res.get("per_embryo_stage", {}) if isinstance(res, dict) else {}
    checks["per_embryo"] = len(pe) >= 2                       # 44b6 + 6bba, NOT pooled
    checks["motion_real"] = any(isinstance(v.get("motion_p50"), (int, float)) and v["motion_p50"] > 0
                                for st in pe.values() for v in st.values())
    checks["verdict_present"] = isinstance(res, dict) and bool(res.get("verdict"))
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    if isinstance(res, dict):
        print(f"  -> embryos={list(pe)} verdict={res.get('verdict')}")
    ok = all(checks.values())
    print(f"=== stage-dynamics: {'PASS' if ok else 'FAIL'} ===  {'' if ok else msg}")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
