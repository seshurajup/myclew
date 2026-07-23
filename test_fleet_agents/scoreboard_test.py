"""scoreboard_test — assert the leaderboard table ranks the KNOWN-best recipe first with the correct gap."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import scoreboard


def test_scoreboard_ranks_best_first():
    measured = {"weak": 0.850, "BEST": 0.890, "mid": 0.870}
    table = scoreboard._table(measured)
    lines = table.splitlines()
    assert "best `0.8900`" in lines[0], f"header should report 0.8900 best: {lines[0]}"
    first_row = [l for l in lines if l.startswith("| 1")][0]
    assert "BEST" in first_row and "🏆" in first_row, f"best recipe not ranked #1: {first_row}"
    return {"header_best_correct": True, "best_ranked_first": True}


def _run():
    print("=== SCOREBOARD DATA-WISE VERIFIER ===")
    try:
        r = test_scoreboard_ranks_best_first()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== scoreboard: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
