"""ledger_guard_test — the anti-fabrication provenance gate. Reproduces the EXP_153 bug (a division
THRESHOLD of 0.9 recorded as cv=0.9) and asserts ledger._verify_cv now REFUSES any winning/sentinel
score that lacks a measured artifact literally containing that number. Regressions still pass freely."""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import ledger


def _raises(cv, artifact):
    try:
        ledger._verify_cv(cv, artifact); return False
    except ValueError:
        return True


def _run():
    print("=== LEDGER PROVENANCE-GATE VERIFIER ===")
    checks = {}
    # the exact EXP_153 bug: threshold sentinel 0.9 recorded as a score, no proof
    checks["blocks_fake_sentinel_0.9"] = _raises(0.9, None)
    # a fabricated new-best with no artifact
    checks["blocks_fake_new_best"] = _raises(0.999, None)
    # a claimed win with a missing artifact path
    checks["blocks_missing_artifact"] = _raises(0.98, os.path.join(tempfile.gettempdir(), "nope_xyz.json"))
    # a claimed win whose artifact does NOT contain the number
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); f.write('{"score": 0.10}'); f.close()
    checks["blocks_mismatched_artifact"] = _raises(0.98, f.name)
    # a REAL win backed by a matching artifact → allowed
    f2 = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); f2.write('{"score": 0.9123}'); f2.close()
    checks["allows_real_win_with_proof"] = not _raises(0.9123, f2.name)
    # an honest regression (well below any best) needs no proof
    checks["allows_regression_no_proof"] = not _raises(0.10, None)
    # None / status strings are always fine
    checks["allows_none"] = not _raises(None, None)
    os.unlink(f.name); os.unlink(f2.name)
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== ledger-guard: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
