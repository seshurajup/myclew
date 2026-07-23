"""deep_research_test — mock the Claude CLI (no real web call): assert the agent builds a DOMAIN-grounded prompt,
saves the report to research/deep_research/<slug>.md, journals a decision with the SHORTLIST as recommendation,
and returns the report path. No network, no LLM."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import deep_research as D


def _run():
    print("=== DEEP-RESEARCH AGENT VERIFIER ===")
    calls = {"prompt": None, "journaled": []}
    canned = "## focal loss\nmechanism...\nverdict ADOPT.\n\nSHORTLIST:\n1. Varifocal loss — swap the head loss.\n2. nnPU negative branch."

    agent = D.DeepResearch()
    import subprocess
    class _R:  # fake CompletedProcess
        def __init__(s): s.stdout = canned
    def fake_run(cmd, **kw):
        calls["prompt"] = cmd[2]      # claude -p <prompt>
        return _R()
    orig = subprocess.run; subprocess.run = fake_run
    # capture journal
    from fleet_agents import ledger
    orig_log = ledger.log
    ledger.log = lambda *a, **k: calls["journaled"].append(k.get("recommendation", ""))
    agent.post = lambda *a, **k: None
    try:
        s, d, to, msg = agent.run({"question": "loss for dim nuclei", "spec": {"question": "loss for dim nuclei", "family": "loss"}}, "test")
    finally:
        subprocess.run = orig; ledger.log = orig_log

    path = d.get("report_path", "")
    checks = {
        "domain_grounded_prompt": calls["prompt"] and "biohub-cell-tracking" in calls["prompt"] and "DIM" in calls["prompt"],
        "asks_2024_2026_and_verdicts": "2024-2026" in calls["prompt"] and "ADOPT" in calls["prompt"] and "SKIP" in calls["prompt"],
        "report_saved": path and os.path.exists(path) and "SHORTLIST" in open(path).read(),
        "journaled_shortlist": calls["journaled"] and "Varifocal" in calls["journaled"][0],
        "returns_shortlist": "Varifocal" in (d.get("shortlist") or ""),
    }
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== deep-research: {'PASS' if ok else 'FAIL'} · {sum(bool(v) for v in checks.values())}/{len(checks)} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); print(f"  ❌ ERROR: {e}"); sys.exit(1)
