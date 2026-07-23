"""gpu_best_practices_test — assert the catalog ranks by acc×speed, buckets free-wins vs search, and
respects hardware (T4/Turing blocks FP8/FP4/Blackwell-only practices)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import gpu_best_practices as g


def test_free_wins_and_hardware_gating():
    # Blackwell (5090): FP8/NVFP4 are usable search candidates
    s, d, to, m = g.report({"question": "t", "spec": {"hardware": "blackwell"}}, "test")
    assert s == "done", m
    assert "torch.compile" in d["adopt"], f"torch.compile should be a free-win: {d['adopt']}"
    assert "FP8 (E4M3)" in d["search"], f"FP8 should be a search candidate on 5090: {d['search']}"
    # Turing (T4): FP8/NVFP4 must be BLOCKED (hardware lacks them)
    s2, d2, _, _ = g.report({"question": "t", "spec": {"hardware": "turing"}}, "test")
    assert "FP8 (E4M3)" in d2["blocked"] and "NVFP4" in d2["blocked"], f"FP8/FP4 must be blocked on T4: {d2['blocked']}"
    return {"free_win_ok": True, "blackwell_fp8_search": True, "t4_blocks_fp4": True}


def _run():
    print("=== GPU-BEST-PRACTICES DATA-WISE VERIFIER ===")
    try:
        r = test_free_wins_and_hardware_gating()
        for k, v in r.items(): print(f"  {'OK' if v else 'X'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  X FAILED: {e}"); ok = False
    print(f"=== gpu-best-practices: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
