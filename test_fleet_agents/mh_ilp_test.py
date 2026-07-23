"""mh_ilp_test — DATA-WISE verifier of the ILP selection MECHANISM on synthetic points (no volumes, no GPU).

The whole hypothesis: an over-complete candidate pool + a threshold-free ILP KEEPS a temporally-consistent
track (a real cell that links cheaply frame-to-frame) and DROPS an isolated high-nothing candidate that would
have to pay appearance+disappearance. We build exactly that graph and assert the solver does it.

  • a REAL cell: 3 frames, high score, small displacement (links cheaply) → all 3 nodes + 2 edges kept
  • an isolated FP: 1 frame, LOW score, far away (cannot link) → dropped (reward < birth+death cost)

Also asserts overlaps are mutually exclusive (two near-duplicate candidates in one frame → the ILP keeps ≤1).
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import mh_ilp

SCALE = np.array([1.625, 0.40625, 0.40625])


def _run():
    print("=== MH-ILP MECHANISM VERIFIER (synthetic, no volumes) ===")
    params = dict(mh_ilp.DEFAULTS)

    # ---- case 1: real 3-frame track (high score) vs isolated 1-frame FP (low score) ----
    per_frame = [
        (np.array([[10, 100, 100]]), np.array([1.0])),                       # t0: real cell
        (np.array([[10, 102, 102], [10, 300, 300]]), np.array([1.0, 0.05])), # t1: real + isolated FP (low)
        (np.array([[10, 104, 104]]), np.array([1.0])),                       # t2: real cell
    ]
    pn, pe = mh_ilp._solve_from_points(per_frame, SCALE, params)
    far_kept = ((pn["y"] > 200).sum() if len(pn) else 0)
    track_nodes = ((pn["y"] < 200).sum() if len(pn) else 0)

    # ---- case 2: same-frame overlap (two near-duplicates within conflict_um) → ILP keeps <= 1 ----
    ov = [
        (np.array([[10, 100, 100]]), np.array([1.0])),
        (np.array([[10, 102, 102], [10, 102, 104]]), np.array([1.0, 0.9])),  # 2 dupes ~0.8um apart (< conflict 3um)
        (np.array([[10, 104, 104]]), np.array([1.0])),
    ]
    pn2, _ = mh_ilp._solve_from_points(ov, SCALE, params)
    t1_kept = ((pn2["t"] == 1).sum() if len(pn2) else 0)

    checks = {
        "real_track_kept": track_nodes == 3,          # all 3 real nodes selected
        "two_edges": len(pe) == 2,                     # linked across the 3 frames
        "isolated_fp_dropped": far_kept == 0,          # low-score singleton dropped
        "overlap_at_most_one": t1_kept <= 1,           # mutual-exclusion constraint honoured
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"    (track_nodes={track_nodes} edges={len(pe)} far_kept={far_kept} overlap_t1_kept={t1_kept})")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
