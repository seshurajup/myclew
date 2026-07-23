"""tracker_select_test — DATA-WISE verifier of the tracker-CHOICE logic (_choose) + the compare-JSON fold.

Asserts the agent picks the tracker best on BOTH embryos (max of MIN per-embryo full-metric score),
excludes leaky trackers, and folds the tracker_compare_cv.py raw output correctly.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import tracker_select as T


def _run():
    print("=== TRACKER-SELECT CHOICE-LOGIC VERIFIER ===")
    # 1) fold raw compare output (list of per-dataset {edge,div,score}) → per-embryo mean score
    raw = {
        "trackastra": {"44b6": [{"edge": 0.80, "div": 0.1, "score": 0.79}, {"edge": 0.82, "div": 0.1, "score": 0.81}],
                       "6bba": [{"edge": 0.88, "div": 0.2, "score": 0.87}]},
        "ultrack":    {"44b6": [{"edge": 0.86, "div": 0.3, "score": 0.85}],
                       "6bba": [{"edge": 0.90, "div": 0.3, "score": 0.89}]},
        "mh-ilp":     {"44b6": [{"edge": 0.50, "div": 0.0, "score": 0.50}],
                       "6bba": [{"edge": 0.70, "div": 0.0, "score": 0.70}]},
    }
    folded = T._results_from_compare(raw)
    folded["leaky-oracle"] = {"44b6": 0.99, "6bba": 0.99, "leaky": True}   # must be ignored
    winner, ranked = T._choose(folded)
    names = [d["name"] for d in ranked]
    checks = {
        "fold_mean_score": abs(folded["trackastra"]["44b6"] - 0.80) < 1e-9,   # mean(0.79,0.81)=0.80
        "leaky_excluded": "leaky-oracle" not in names,
        "winner_is_ultrack": winner == "ultrack",       # min(0.85,0.89)=0.85 > trackastra min(0.80,0.87)=0.80
        "ranked_by_min_score": ranked == sorted(ranked, key=lambda d: -d["min_score"]),
        "mh_ilp_last": names[-1] == "mh-ilp",            # weakest linker → lowest min score
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"    winner={winner}  ranked={[(d['name'], d['min_score']) for d in ranked]}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
