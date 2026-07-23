"""submit_verify_test — DATA-WISE verifier of the submission-VALIDITY logic (_valid_submission), no GPU/ILP.
Feeds synthetic submission frames and asserts: a well-formed node+edge set passes; a dangling edge
(references a non-existent node) is caught; a header-only frame fails has_rows/has_nodes.
"""
import os, sys
import pandas as pd
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents.submit_verify import SubmitVerify, COLS, _valid_submission


def _mk(rows):
    return pd.DataFrame(rows, columns=COLS)


def _run():
    print("=== SUBMIT-VERIFY VALIDITY-LOGIC VERIFIER ===")
    good = _mk([
        [0, "44b6_x", "node", 0, 0, 5, 5, 5, -1, -1],
        [1, "44b6_x", "node", 1, 1, 6, 6, 6, -1, -1],
        [2, "44b6_x", "edge", -1, -1, -1, -1, -1, 0, 1],
    ])
    dangling = _mk([
        [0, "44b6_x", "node", 0, 0, 5, 5, 5, -1, -1],
        [1, "44b6_x", "edge", -1, -1, -1, -1, -1, 0, 99],   # 99 is not an emitted node
    ])
    empty = _mk([]).astype(object)

    cg = _valid_submission(good)
    cd = _valid_submission(dangling)
    ce = _valid_submission(empty)
    checks = {
        "good_all_pass": all(cg.values()),
        "dangling_edge_caught": cd["edges_reference_nodes"] is False,
        "empty_fails_rows": ce["has_rows"] is False,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("    good=", cg)
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
