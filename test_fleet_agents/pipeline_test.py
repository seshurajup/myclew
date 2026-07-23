"""pipeline_test — data-wise verifier for the orchestrator. Plants FAKE stand-ins named like the real
fleet agents and asserts: ordered run + carry, HALT on escalate (GPU blocker), `when` gating,
`parallel` fan-out (no mid-group halt), `on_fail:skip`, `loop_until` a CV target, and template expansion.
No Claude, no board — pure deterministic chain."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import pipeline


def _handlers(order):
    def data_audit(q, w):                      # measures the data → emits a scale fact to carry forward
        order.append("data-audit"); return ("done", {"seed_count": 42}, "all", "data measured")

    def combined_train(q, w):                  # needs the carried data fact, then emits a measured CV
        order.append("combined-train")
        assert q["spec"].get("seed_count") == 42, "carry failed"
        return ("done", {"cv": 0.873}, "all", "trained")

    def gpu_train(q, w):                        # the known GPU-CUDA blocker → escalates (halt case)
        order.append("gpu-train"); return ("escalated", {}, "researcher", "cuda unavailable")

    def submission_build(q, w):                 # downstream step (must not run after a halt / when gated off)
        order.append("submission-build"); return ("done", {}, "all", "packed")

    def fullconfig_search(q, w):                # CV producer with no carry dependency
        order.append("fullconfig-search"); return ("done", {"cv": 0.873}, "all", "searched")

    def config_ablate(q, w):                    # parallel member (succeeds)
        order.append("config-ablate"); return ("done", {"score": 0.5}, "all", "ablated")

    def box_sample(q, w):                       # parallel member (fails — group must still finish)
        order.append("box-sample"); return ("failed", {"error": "no external cache"}, "all", "boxed")

    def broken_detector(q, w):                  # a failing step used to test on_fail:skip
        order.append("broken-detector"); return ("failed", {"error": "boom"}, "all", "broken")

    return {"data-audit": data_audit, "combined-train": combined_train, "gpu-train": gpu_train,
            "submission-build": submission_build, "fullconfig-search": fullconfig_search,
            "config-ablate": config_ablate, "box-sample": box_sample, "broken-detector": broken_detector,
            "heal": lambda q, w: ("done", {}, "all", "healed")}


def _pipe(spec, order):
    pipeline.Pipeline._handlers = lambda self: _handlers(order)
    return pipeline.Pipeline().run({"question": "t", "spec": spec}, "test")


def _run():
    print("=== PIPELINE ORCHESTRATOR VERIFIER ===")
    checks = {}

    # 1) ordered run + carry (data-audit → combined-train) + HALT at the GPU blocker (submission-build must not run)
    o = []; _s, d, _t, _m = _pipe({"steps": [
        {"kind": "data-audit"}, {"kind": "combined-train"}, {"kind": "gpu-train"}, {"kind": "submission-build"}]}, o)
    checks["ordered_and_halts_at_blocker"] = o == ["data-audit", "combined-train", "gpu-train"] and d["halted_at"] == "escalated"
    checks["surfaced_cv_0.873"] = d.get("best_cv") == 0.873

    # 2) `when` gate: don't submit unless CV clears 0.897; do run a step gated at >=0.5
    o = []; _s, d, _t, _m = _pipe({"steps": [
        {"kind": "fullconfig-search"},
        {"kind": "submission-build", "when": "best_cv>=0.897"},
        {"kind": "config-ablate", "when": "best_cv>=0.5"}]}, o)
    checks["when_gate_blocks_submit_runs_ablate"] = ("submission-build" not in o) and ("config-ablate" in o)

    # 3) parallel fan-out runs BOTH members even though box-sample fails, then continues to submission-build
    o = []; _s, d, _t, _m = _pipe({"steps": [
        {"parallel": [{"kind": "config-ablate"}, {"kind": "box-sample"}]}, {"kind": "submission-build"}]}, o)
    checks["parallel_runs_all_and_continues"] = {"config-ablate", "box-sample", "submission-build"}.issubset(set(o))

    # 4) on_fail:skip lets the chain continue past a failed detector
    o = []; _s, d, _t, _m = _pipe({"steps": [
        {"kind": "broken-detector", "on_fail": "skip"}, {"kind": "submission-build"}]}, o)
    checks["on_fail_skip_continues"] = "submission-build" in o and d["halted_at"] is None

    # 5) loop_until repeats to a target (CV 0.873 < 0.9 → runs all 3 rounds), caps rounds
    o = []; _s, d, _t, _m = _pipe({"steps": [{"kind": "fullconfig-search"}],
                                   "loop_until": {"metric": "best_cv", "target": 0.9, "max_rounds": 3}}, o)
    checks["loop_until_caps_rounds"] = d["rounds"] == 3 and o == ["fullconfig-search"] * 3

    # 6) loop stops early once the target is met (0.873 >= 0.8 → 1 round)
    o = []; _s, d, _t, _m = _pipe({"steps": [{"kind": "fullconfig-search"}],
                                   "loop_until": {"metric": "best_cv", "target": 0.8, "max_rounds": 3}}, o)
    checks["loop_stops_on_target"] = d["rounds"] == 1

    # 7) templates exist and cover external-data + training
    ext = pipeline._TEMPLATES.get("external_train", [])
    ext_kinds = {s.get("kind") for s in ext}
    checks["external_train_template_covers_it"] = {"box-sample", "sample-match", "combined-train"}.issubset(ext_kinds)

    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== pipeline: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
