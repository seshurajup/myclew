"""config_ablate_test — stub scorer + temp state; assert it measures a per-block delta (block OFF vs base)."""
import os, sys, tempfile
from pathlib import Path
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import config_ablate


def test_config_ablate_measures_block_delta():
    assert len(config_ablate.BLOCKS) >= 5, "expected several ablatable blocks"
    with tempfile.TemporaryDirectory() as d:
        orig_state, orig_score = config_ablate.STATE, config_ablate._score
        config_ablate.STATE = Path(d) / "state.json"
        # base scores 0.88; disabling motion_relink drops it (load-bearing)
        def fake(env):
            return {"score": 0.876 if env.get("BIOHUB_OUTPUT_MOTION_RELINK") == "0" else 0.880, "adjE": 0.88, "n": 4}
        config_ablate._score = fake
        try:
            r1 = config_ablate.report({"question": "t", "spec": {}}, "test")   # scores base first
            r2 = config_ablate.report({"question": "t", "spec": {}}, "test")   # ablates first block
        finally:
            config_ablate.STATE, config_ablate._score = orig_state, orig_score
        assert r1[0] == "done" and r2[0] == "done", (r1[3], r2[3])
        assert "base_score" in r1[1] or "block" in r2[1], f"no base/block measured: {r1[1]} {r2[1]}"
        return {"blocks_defined": True, "measures_delta": True}


def _run():
    print("=== CONFIG-ABLATE DATA-WISE VERIFIER ===")
    try:
        r = test_config_ablate_measures_block_delta()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== config-ablate: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
