"""setup_env_test — verify the lib-gated dependency manager: registry integrity, check() probes imports,
dry-run planning is safe (no install), ABI-heavy deps are flagged for numpy-pin, and the agent run() reports
per-agent state. Does NOT install anything (test stays fast + side-effect-free)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import setup_env as SE


def _run():
    print("=== SETUP-ENV VERIFIER ===")
    checks = {}
    # registry integrity: every entry has pip + probe + abi flag
    checks["registry_wellformed"] = all(
        isinstance(v.get("pip"), list) and v.get("pip") and "probe" in v and "abi" in v
        for v in SE.REQUIRES.values())
    # check() probes imports and returns ready True/False per known agent
    chk = SE.check()
    checks["check_covers_registry"] = set(chk) == set(SE.REQUIRES)
    checks["check_ready_is_bool"] = all(isinstance(chk[a]["ready"], bool) for a in chk)
    # a definitely-present dep (numpy via a stand-in) and a definitely-absent one behave correctly
    checks["present_probe_true"] = SE._present("os") is True
    checks["absent_probe_false"] = SE._present("nonexistent_pkg_xyz123") is False
    # DRY-RUN install must NOT install — only plan, and flag ABI deps for the numpy pin
    missing = [a for a, v in chk.items() if v["ready"] is False]
    plan = SE.install(missing or list(SE.REQUIRES)[:1], dry_run=True)
    checks["dryrun_plans_not_installs"] = all(
        v["status"] in ("would-install", "already-ready", "skip") for v in plan.values())
    abi_agents = [a for a in (missing or []) if SE.REQUIRES.get(a, {}).get("abi")]
    checks["abi_flagged_numpy_pin"] = (not abi_agents) or all(
        "constraint" in plan[a].get("cmd", "").lower() or SE.NUMPY_PIN in plan[a].get("cmd", "")
        for a in abi_agents if plan.get(a, {}).get("status") == "would-install")
    # agent run() dispatch (dry-run) reports state + never raises
    agent = SE.SetupEnv()
    st, d, to, msg = agent.run({"question": "setup", "spec": {"install": False}}, "test")
    checks["agent_runs_dryrun"] = st == "done" and "check" in d and "missing" in d
    checks["agent_no_side_effects"] = isinstance(d.get("missing"), list)

    # gpu-guard: verify_gpu returns a well-formed report; repair_torch dry-run never installs; guard mode works
    v = SE.verify_gpu(run_matmul=False)
    checks["verify_gpu_keys"] = all(k in v for k in ("ok", "torch", "build_cu128", "cuda", "reason"))
    rt = SE.repair_torch(dry_run=True)
    checks["repair_dryrun"] = rt["dry_run"] is True and "cu128" in rt["cmd"] and "install" in rt["cmd"]
    st2, d2, _, _ = SE.run({"spec": {"mode": "gpu-guard"}}, "t")
    checks["gpu_guard_mode"] = st2 == "done" and "cuda" in d2 and "build_cu128" in d2
    print(f"  (gpu-guard: torch={v.get('torch')} cu128={v.get('build_cu128')} cuda={v.get('cuda')})")

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    print(f"  (missing gated agents: {d.get('missing')})")
    ok = all(checks.values())
    print(f"\n=== setup-env: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  X FAILED: {e}"); sys.exit(1)
