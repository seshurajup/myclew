"""combo_search_test — stub the scorer + temp state; assert coordinate-descent RECORDS a scored combo and
tracks the running best (its core search logic on known scores)."""
import os, sys, tempfile
from pathlib import Path
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import combo_search


def test_combo_search_seed_and_score():
    # deterministic candidate generation
    q = combo_search._seed_queue(dict(combo_search.BASE), {})
    assert len(q) > 0, "seed queue empty"
    assert all(combo_search._key(c) for c in q), "keys not generated"
    # one stubbed tick with temp state
    with tempfile.TemporaryDirectory() as d:
        orig_state, orig_score = combo_search.STATE, combo_search._score
        combo_search.STATE = Path(d) / "state.json"
        combo_search._score = lambda env, worker: ({"score": 0.87, "adjE": 0.87}, None)
        try:
            s, res, to, msg = combo_search.search({"question": "t", "spec": {}}, "test")
        finally:
            combo_search.STATE, combo_search._score = orig_state, orig_score
        assert s == "done", msg
        assert "running_best" in res, f"no running best tracked: {res}"
        return {"seed_nonempty": True, "records_score": "running_best" in res}


def _run():
    print("=== COMBO-SEARCH DATA-WISE VERIFIER ===")
    try:
        r = test_combo_search_seed_and_score()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== combo-search: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
