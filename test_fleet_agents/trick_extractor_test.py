"""trick_extractor_test — plant a notebook with KNOWN trick strings; assert the agent detects them."""
import os, sys, json, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import trick_extractor


def test_trick_extractor_detects_known_tricks():
    with tempfile.TemporaryDirectory() as d:
        nb = os.path.join(d, "known.ipynb")
        # a notebook whose code contains ILP, motion_relink, gap_close, safe_divisions, linear_sum_assignment
        src = ("linear_sum_assignment(cost)\nmotion_relink(edges)\ngap_close(g)\n"
               "add_safe_divisions(nodes)\nILP solver\nfilter_short_tracks(x)\n")
        json.dump({"cells": [{"cell_type": "code", "source": src}]}, open(nb, "w"))
        s, res, to, msg = trick_extractor.report(
            {"question": "t", "spec": {"notebook_glob": [os.path.relpath(nb, COMP)]}}, "test")
        assert s == "done", msg
        # linking + division + post-proc stages must have detected tricks
        stages = res["stages"]
        assert stages.get("linking", 0) >= 1, f"missed linking tricks: {stages}"
        assert stages.get("division", 0) >= 1, f"missed division tricks: {stages}"
        assert stages.get("post-proc", 0) >= 1, f"missed post-proc tricks: {stages}"
        return {"linking": True, "division": True, "postproc": True}


def _run():
    print("=== TRICK-EXTRACTOR DATA-WISE VERIFIER ===")
    try:
        r = test_trick_extractor_detects_known_tricks()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== trick-extractor: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
