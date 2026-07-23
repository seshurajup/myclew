"""coverage_audit_test — verify every agent is placed in a pack (0 unclassified) and biohub is its own pack."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import fleet_agents as F
from fleet_agents import coverage_audit as CA


def _run():
    print("=== COVERAGE-AUDIT VERIFIER ===")
    placed = CA.audit(list(F.HANDLERS))
    counts = {k: len(v) for k, v in placed.items()}
    checks = {}
    checks["zero_unclassified"] = len(placed.get("UNCLASSIFIED", [])) == 0
    # CONSERVATION: every registered agent is placed exactly once — NOTHING is lost in the split
    all_placed = [n for names in placed.values() for n in names]
    checks["no_agent_lost"] = sorted(all_placed) == sorted(F.HANDLERS)
    checks["no_duplicates"] = len(all_placed) == len(set(all_placed))
    checks["biohub_split_out"] = counts.get("BIOHUB (3D+time)", 0) >= 15   # smaller, truly-biology-only
    checks["reusable_packs_grew"] = all(counts.get(p, 0) > 0 for p in
                                        ["Detection & Tracking", "Arch/Config search", "External-data transfer"])
    checks["matrix_writes"] = os.path.exists(CA.write_matrix(placed, len(F.HANDLERS)))
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    if placed.get("UNCLASSIFIED"):
        print("  unclassified:", placed["UNCLASSIFIED"])
    print(f"  packs: { {k: counts[k] for k in sorted(counts)} }")
    ok = all(checks.values())
    print(f"=== coverage-audit: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
