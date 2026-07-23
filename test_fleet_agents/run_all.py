"""run_all — the fleet's DATA-WISE test runner (lesson from the sandboxed-AutoML verify_execution.py:
run each verifier as a subprocess, parse its result, tabulate PASS/FAIL). One `*_test.py` per agent; each
plants known ground-truth data and asserts the agent RECOVERS it. This gives every agent the same quality
gate — "does it produce the correct output on known data", not "does it run".

    research/cellmot_venv/bin/python test_fleet_agents/run_all.py

Add coverage by dropping a new `<agent>_test.py` here; it is auto-discovered. Exit code 0 iff all pass.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.dirname(HERE)
PY = os.path.join(COMP, "research", "cellmot_venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable

# roster of agents → whether a data-wise test exists yet (so gaps are visible, not hidden)
FLEET_AGENTS = [
    "xai", "data_audit", "decision_audit", "arch_builder", "gnn_link_train", "tracker_consensus",
    "trick_gate", "trick_extractor", "prior_art", "paper_research", "gnn_probe", "flow_gt_builder",
    "ext_label_stats", "fullconfig_search", "config_ablate", "combo_search", "scoreboard", "heal",
]


def main():
    tests = sorted(glob.glob(os.path.join(HERE, "*_test.py")))
    have = {os.path.basename(t).replace("_test.py", "") for t in tests}
    env = dict(os.environ); env["PYTHONPATH"] = os.path.join(COMP, "tools", "researchpapers") + ":" + env.get("PYTHONPATH", "")
    print(f"=== FLEET AGENT DATA-WISE TEST SUITE ({len(tests)} verifiers) ===\n")
    results = {}
    for t in tests:
        name = os.path.basename(t).replace("_test.py", "")
        r = subprocess.run([PY, t], capture_output=True, text=True, cwd=COMP, env=env, timeout=600)
        tail = [l for l in r.stdout.splitlines() if "===" in l or "PASS" in l or "pass" in l]
        summary = tail[-1] if tail else "(no summary)"
        ok = r.returncode == 0
        results[name] = ok
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name:20s} {summary.strip()}")
        if not ok:
            for l in r.stdout.splitlines():
                if "❌" in l or "FAILED" in l:
                    print(f"        {l.strip()}")

    print("\n--- COVERAGE (agents without a data-wise test yet) ---")
    missing = [a for a in FLEET_AGENTS if a not in have]
    print(f"  tested: {len(have)}/{len(FLEET_AGENTS)}  ·  missing: {', '.join(missing) or 'none — full coverage'}")

    n_pass = sum(results.values())
    print(f"\n=== {n_pass}/{len(results)} agent verifiers PASS ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
