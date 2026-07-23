"""fullconfig_search_test — assert it parses the yaroslav base env (53 vars) and seeds a multi-axis descent."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import fullconfig_search


def test_fullconfig_parses_base_and_seeds():
    base = fullconfig_search._base_env()
    assert len(base) >= 40, f"base env should parse ~53 vars, got {len(base)}"
    assert "BIOHUB_ILP_DIVISION_WEIGHT" in base, "ILP division weight missing from base"
    q = fullconfig_search._seed_queue(dict(base), {})
    assert len(q) >= len(fullconfig_search.AXES), f"seed queue too small: {len(q)} for {len(fullconfig_search.AXES)} axes"
    return {"base_parsed": len(base) >= 40, "seeds_all_axes": len(q) >= len(fullconfig_search.AXES)}


def _run():
    print("=== FULLCONFIG-SEARCH DATA-WISE VERIFIER ===")
    try:
        r = test_fullconfig_parses_base_and_seeds()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== fullconfig-search: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
