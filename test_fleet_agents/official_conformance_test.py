"""official_conformance_test — DATA-WISE verifier of the PURE helpers (no network, no pytest). Asserts the
pytest-summary parser, the official-COLUMNS extractor, the schema-equality (with 'id' normalisation), and the
git-blob hash."""
import os, sys, hashlib
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import official_conformance as OC


def _run():
    print("=== OFFICIAL-CONFORMANCE HELPER VERIFIER ===")
    checks = {}
    # pytest summary parser
    checks["pytest_pass_fail"] = OC._parse_pytest("13 failed, 106 passed, 2 warnings in 3.67s") == (106, 13)
    checks["pytest_pass_only"] = OC._parse_pytest("29 passed in 1.68s") == (29, 0)
    checks["pytest_empty"] = OC._parse_pytest("collected 0 items") == (0, 0)
    # official COLUMNS extractor
    src = 'COLUMNS: tuple[str, ...] = (\n  "dataset", "row_type", "node_id", "t", "z", "y", "x",\n  "source_id", "target_id",\n)'
    cols = OC._official_columns(src)
    checks["cols_extract"] = cols == ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    # schema equality — 'id' on either side is normalised away
    ours = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    off = ["id"] + ours
    ok, detail = OC._schema_equal(ours, off)
    checks["schema_equal_with_id"] = ok and not detail["missing"] and not detail["extra"]
    ok2, d2 = OC._schema_equal(["dataset", "row_type"], ["dataset", "row_type", "t"])
    checks["schema_detects_missing"] = (not ok2) and d2["missing"] == ["t"]
    # git blob sha matches git's algorithm for a known input
    b = b"hello\n"
    expect = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
    checks["gitblob"] = OC._gitblob(b) == expect

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
