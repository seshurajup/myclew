"""compress_select_test — DATA-WISE verifier of the prune-CHOICE logic (_choose), no GPU/weights.

Feeds the real 2D+stitch K-sweep numbers and asserts the agent picks the FASTEST prune (smallest K =
most time-margin) that KEEPS quality (min per-embryo recall ≥ bar) AND fits the time budget — and that it
excludes an over-budget K and a quality-losing K.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import compress_select as P


def _run():
    print("=== PRUNE-SELECT CHOICE-LOGIC VERIFIER ===")
    # real measured sweep (2D+stitch, per-embryo recall + s/frame)
    results = {
        24: {"44b6": 1.000, "6bba": 1.000, "spf": 3.96},   # over budget (3.9)
        20: {"44b6": 1.000, "6bba": 1.000, "spf": 3.48},   # fits + full quality
        16: {"44b6": 1.000, "6bba": 0.933, "spf": 2.97},   # fits but recall<0.95 → quality lost
        12: {"44b6": 1.000, "6bba": 0.967, "spf": 2.50},   # fits + keeps quality + fastest
    }
    bestK, best, ranked = P._choose(results, recall_bar=0.95, budget_spf=3.9)
    checks = {
        "picks_smallest_quality_keeping_K": bestK == 12,            # fastest that keeps recall≥0.95 within budget
        "over_budget_excluded": not any(d["K"] == 24 and d["fits_budget"] for d in ranked),
        "quality_loss_flagged": next(d for d in ranked if d["K"] == 16)["keeps_quality"] is False,
        "best_keeps_quality": best["keeps_quality"] is True,
        "best_fits_budget": best["fits_budget"] is True,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"    bestK={bestK}  best={best}")

    trace = [
        {"n_kept": 24, "min_recall": 1.00, "spf": 3.96},
        {"n_kept": 20, "min_recall": 1.00, "spf": 3.48},
        {"n_kept": 16, "min_recall": 0.93, "spf": 2.97},
        {"n_kept": 12, "min_recall": 0.97, "spf": 2.50},
        {"n_kept": 10, "min_recall": 0.80, "spf": 2.10},
    ]
    floor_k, floor = P._iterative_select(trace, recall_bar=0.95)
    checks["floor_is_fewest_above_bar"] = floor_k == 12
    checks["floor_holds_recall"] = floor["min_recall"] >= 0.95
    print(f"  [{'PASS' if checks['floor_is_fewest_above_bar'] else 'FAIL'}] floor_is_fewest_above_bar (SLEB) -> {floor_k}")
    print(f"  [{'PASS' if checks['floor_holds_recall'] else 'FAIL'}] floor_holds_recall")

    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
