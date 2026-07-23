"""det_sweep_test — plant a recall/count surface where the HIGHEST-recall point OVER-detects (count out
of range) and must be rejected in favour of the best recall AT a calibrated count. Asserts det-sweep
picks recall-at-calibrated-count, not raw recall. No GPU — eval_fn injected."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import det_sweep


def _run():
    print("=== DET-SWEEP DATA-WISE VERIFIER ===")
    # surface: lower det → higher recall but over-detection (count_ratio blows up).
    # det 0.90 has the HIGHEST recall (0.99) but count 1.9 (floods FP) → must be REJECTED.
    # det 0.97 / pool 3.0 = recall 0.988 @ count 1.05 → the calibrated optimum (abhijith's point).
    SURF = {
        (0.90, 3.0): {"node_recall": 0.990, "count_ratio": 1.90, "cv": 0.70},   # floods → reject
        (0.95, 3.0): {"node_recall": 0.989, "count_ratio": 1.30, "cv": 0.88},   # slightly over → reject (>1.25)
        (0.97, 3.0): {"node_recall": 0.988, "count_ratio": 1.05, "cv": 0.9257}, # THE pick
        (0.99, 3.0): {"node_recall": 0.950, "count_ratio": 0.90, "cv": 0.90},   # calibrated but lower recall
        (0.97, 5.0): {"node_recall": 0.970, "count_ratio": 0.95, "cv": 0.89},   # calibrated, lower recall
    }
    def ev(det, pool):
        return SURF[(det, pool)]

    s, data, to, msg = det_sweep.DetSweep().run(
        {"question": "sweep", "spec": {"det_grid": [0.90, 0.95, 0.97, 0.99], "pool_grid": [3.0, 5.0],
                                       "eval_fn": ev, "count_lo": 0.8, "count_hi": 1.25}}, "test")
    pick = data.get("pick", {})
    checks = {
        "picked_det_0.97": pick.get("det") == 0.97,
        "picked_pool_3.0": pick.get("pool") == 3.0,
        "rejected_flooding_0.90": pick.get("count_ratio", 9) <= 1.25,
        "not_just_max_recall": pick.get("node_recall") == 0.988,   # 0.990 (det .90) was higher but rejected
    }
    # note: (0.90,5.0),(0.95,5.0),(0.99,5.0) not in SURF → agent records them as errors; ensure it still picks
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== det-sweep: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
    except KeyError as e:
        print(f"  ❌ surface missing point {e}"); sys.exit(1)
