"""paper_research_test — assert innovations are ranked by accuracy×speed and bucketed (adopt vs search)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import paper_research


def test_paper_research_ranks_and_buckets():
    catalog = {
        "fast_free":  ("backbone", "0", "++", "cheap speed win", "adopt-cheap"),
        "risky":      ("attention", "-", "-", "worse both", "search"),
        "acc_win":    ("backbone", "++", "0", "accuracy win to prove", "search"),
    }
    s, d, to, msg = paper_research.report({"question": "t", "spec": {"catalog": catalog, "speed_weight": 1.0}}, "test")
    assert s == "done", msg
    assert "fast_free" in d["adopt_cheap"], d
    assert "risky" in d["search_candidates"] and "acc_win" in d["search_candidates"], d
    return {"adopt_bucket_ok": True, "search_bucket_ok": True}


def _run():
    print("=== PAPER-RESEARCH DATA-WISE VERIFIER ===")
    try:
        r = test_paper_research_ranks_and_buckets()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== paper-research: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
