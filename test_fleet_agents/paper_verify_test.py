"""paper_verify_test — DATA-WISE verifier of the PURE claim logic (verify_claims), no file I/O. Asserts each
claim's verdict responds correctly to synthetic measurements: monotone cell growth → C1 MATCH; our narrow
cell-count range → C2 PARTIAL (early sub-window); divisions present but sparse → C3 PARTIAL; sparse continuous
labels → C4 MATCH; labeled cells less-dense → C5 MATCH; long tracks → C6 MATCH. Also a MISMATCH branch."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import paper_verify as P


def _run():
    print("=== PAPER-VERIFY CLAIM-LOGIC VERIFIER ===")
    good = {
        "cells_per_frame_by_stage": {"S0": 5000, "S1": 9000, "S2": 17000, "S3": 30000, "S4": 59000},
        "cells_per_frame_min": 3783, "cells_per_frame_max": 78644,
        "total_div": 151, "movies_with_div": 87, "n_movies": 199, "div_per_track": 0.066,
        "label_pct": 2.6, "gap_frac": 0.0, "median_span": 38.0, "local_density_cohens_d": -0.58,
    }
    rows = {r["id"]: r for r in P.verify_claims(good)}
    checks = {
        "C1_match_growth": rows["C1"]["verdict"] == "MATCH",
        "C2_partial_window": rows["C2"]["verdict"] == "PARTIAL",
        "C3_partial_div": rows["C3"]["verdict"] == "PARTIAL",
        "C4_match_sparse": rows["C4"]["verdict"] == "MATCH",
        "C5_match_lessdense": rows["C5"]["verdict"] == "MATCH",
        "C6_match_longtracks": rows["C6"]["verdict"] == "MATCH",
        "all_have_source": all(r.get("source") for r in rows.values()),
    }
    # MISMATCH branches: no divisions → C3 MISMATCH; non-monotone growth → C1 MISMATCH; dense labels → C5 MISMATCH
    bad = dict(good, total_div=0,
               cells_per_frame_by_stage={"S0": 9000, "S1": 5000, "S2": 17000},
               local_density_cohens_d=+0.3)
    brows = {r["id"]: r for r in P.verify_claims(bad)}
    checks["C3_mismatch_no_div"] = brows["C3"]["verdict"] == "MISMATCH"
    checks["C1_mismatch_nonmono"] = brows["C1"]["verdict"] == "MISMATCH"
    checks["C5_mismatch_dense"] = brows["C5"]["verdict"] == "MISMATCH"

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("   good verdicts:", {k: r["verdict"] for k, r in rows.items()})
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
