"""pattern_tune_test — mock box-sample (no-op) + a verify that starts MISMATCHED and becomes all_match
after the agent nudges the pools. Assert pattern-tune loops, adjusts params toward the target, and stops
on convergence. No GPU."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import pattern_tune


def _run():
    print("=== PATTERN-TUNE WORKFLOW VERIFIER ===")
    state = {"round": 0, "tracks_seen": []}
    COMP_T = {"tracks_per_crop_med": 15, "track_len_frames_med": 33, "full_span_pct": 17, "divisions_per_crop_med": 0}

    def box(q, w):
        state["tracks_seen"].append(q["spec"].get("tracks_per_crop_pool")); return ("done", {}, "all", "boxed")

    def verify(q, w):
        # round 0/1: tracks too low (3) → mismatch; round >=2: matches (the nudges worked)
        state["round"] += 1
        boxed = {"tracks_per_crop_med": 3 if state["round"] < 3 else 15, "track_len_frames_med": 33,
                 "full_span_pct": 17, "divisions_per_crop_med": 0, "broken_tracks": 0}
        allm = boxed["tracks_per_crop_med"] == 15
        checks = {"broken_is_0": True, "tracks_per_crop_med_match": allm}
        return ("done", {"boxed": boxed, "competition": COMP_T, "checks": checks, "all_match": allm}, "all", "v")

    agent = pattern_tune.PatternTune()
    agent._agents = lambda: {"box-sample": box, "ext-label-stats": verify}
    s, d, to, msg = agent.run({"question": "tune", "spec": {"max_rounds": 6, "tracks_per_crop_pool": [3]}}, "test")

    checks = {
        "converged_matched": d.get("matched") is True,
        "stopped_when_matched": d.get("rounds") == 3,                 # r1 mismatch, r2 mismatch, r3 match → stop
        "nudged_tracks_pool_up": state["tracks_seen"][-1][0] > state["tracks_seen"][0][0],  # pool increased
    }
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== pattern-tune: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
