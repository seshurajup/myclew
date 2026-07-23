"""detector_arch_search_test — mock the trainer (no GPU): a fake 2-CV scorer that rewards ONE specific axis
value (kernel=[5,5,5]) and is flat elsewhere. Assert the agent (1) starts from the simplest baseline, (2) grows
one axis at a time, (3) KEEPS only the improving value, (4) records every step. No torch, no train_v0."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import detector_arch_search as D


def _run():
    print("=== DETECTOR-ARCH-SEARCH LOGIC VERIFIER ===")
    calls = {"train": 0, "records": []}

    agent = D.DetectorArchSearch()
    calls["warm_inits"] = []
    # fake scorer: baseline 0.50; kernel=[5,5,5] → 0.60 (a real improvement); everything else flat 0.50.
    # Also records the warm-start init it was handed, to prove the best checkpoints thread forward.
    def fake_score(cfg_path, spec, tag, init_ckpts=None):
        import yaml
        cfg = yaml.safe_load(open(cfg_path)); k = cfg["model"]["backbone"].get("kernel")
        calls["train"] += 1; calls["warm_inits"].append(dict(init_ckpts or {}))
        cv = 0.60 if k == [5, 5, 5] else 0.50
        return cv, {"44b6": cv, "6bba": cv}, {"44b6": f"{tag}_44b6/best.pt", "6bba": f"{tag}_6bba/best.pt"}
    agent._score_2cv = fake_score
    agent.record = lambda **kw: calls["records"].append(kw.get("change"))
    agent.post = lambda *a, **k: None
    agent.save_state = lambda st: calls.__setitem__("state", st)

    s, d, to, msg = agent.run({"question": "arch", "spec": {"scratch": "/tmp/det_arch_test",
                              "axes": ["padding_mode", "stem_kernel", "kernel", "activation"]}}, "test")

    best = d["best_cfg"]; trail = d["trail"]
    checks = {
        "baseline_first": trail[0]["step"] == "baseline" and trail[0]["cfg"]["kernel"] == [3, 3, 3],
        # each candidate changes exactly ONE axis from the running best (coordinate descent) → the step label names it
        "one_axis_per_step": all("=" in t["step"] and t["step"].split("=")[0] in
                                 ("padding_mode", "stem_kernel", "kernel", "activation") for t in trail[1:]),
        # after kernel=[5,5,5] is KEPT, later candidates build on it (carry the improvement forward)
        "builds_on_improvement": all(t["cfg"]["kernel"] == [5, 5, 5] for t in trail
                                     if t["step"].startswith("activation")),
        "kept_the_improvement": best["kernel"] == [5, 5, 5] and d["best_cv"] == 0.60,
        "baseline_no_warmstart": calls["warm_inits"][0] == {},                       # baseline trains from random
        "candidates_warmstart": all(w for w in calls["warm_inits"][1:]),             # every candidate warm-starts
        "warmstart_follows_best": any("kernel" in str(w.get("44b6", "")) for w in calls["warm_inits"][-1:]),  # after kernel win, later inits carry it
        "recorded_every_step": len(calls["records"]) == len(trail),
        "trained_each_candidate": calls["train"] == len(trail),
    }
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== detector-arch-search: {'PASS' if ok else 'FAIL'} · {sum(checks.values())}/{len(checks)} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); print(f"  ❌ ERROR: {e}"); sys.exit(1)
