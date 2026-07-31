"""coworker_backend_test — pure logic: openworker provider routing + risk classify + approval gating."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import coworker_backend as C


def _run():
    print("=== COWORKER-BACKEND LOGIC VERIFIER ===")
    r_prov = C.route(provider="anthropic")
    r_full = C.route("openai:gpt-4o")
    r_bad = C.route("mistral:foo")                 # unknown → anthropic fallback
    checks = {
        "route_provider_default": r_prov["provider"] == "anthropic" and r_prov["resolved"],
        "route_colon_form": r_full["provider"] == "openai" and r_full["model"] == "gpt-4o",
        "unknown_falls_back": r_bad["provider"] == "anthropic" and r_bad["resolved"] is False,
        "classify_external": C.classify("send email to team") == "EXTERNAL",
        "classify_exec": C.classify("run pip install torch") == "EXEC",
        "classify_write": C.classify("save the config file") == "WRITE_LOCAL",
        "classify_read": C.classify("look at the data") == "READ",
        "external_beats_exec": C.classify("run script then post results") == "EXTERNAL",
        # gating: EXTERNAL always needs approval even at full autonomy
        "external_always_approved": C.needs_approval("delete the repo", autonomy=3)["needs_approval"] is True,
        "exec_auto_at_tier2": C.needs_approval("run tests", autonomy=2)["auto_run"] is True,
        "exec_gated_at_tier1": C.needs_approval("run tests", autonomy=1)["needs_approval"] is True,
        "read_always_auto": C.needs_approval("read logs", autonomy=0)["auto_run"] is True,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"    route_bad={r_bad} | gate_ext={C.needs_approval('send email', 3)}")
    ok = all(checks.values()); print("RESULT:", "PASS" if ok else "FAIL"); return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
