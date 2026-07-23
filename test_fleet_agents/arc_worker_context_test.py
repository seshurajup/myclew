"""arc_worker_context_test — data-wise verifier for the arc-worker-context BUILDER tool. Feeds STUB task
state and asserts the assembled context (rewrite-first target, idioms pulled from patterns.md, history +
similar-task transfer) and that the rendered prompt carries the load-bearing sections. No LLM, no onnx.
"""
import os
import sys
import tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))


def _run():
    print("=== ARC-WORKER-CONTEXT DATA-WISE VERIFIER ===")
    from fleet_agents import arc_worker_context as W
    checks = {}

    state = {
        "task_id": "task179", "agi_id": "abc123", "baseline_score": 20.4,
        "rule": "Reflect the square grid across the main diagonal.",
        "signature": {"tags": ["same shape", "square", "mirror reflection transpose"]},
        "best": {"source": "optimize_v1", "transform": "recolor_conv", "ops": ["Conv"],
                 "memory": 0, "params": 100, "cost": 100, "score": 20.395,
                 "builder": "Conv(input, W[10,10,1,1]) -> output"},
        "history": [{"id": "A02", "idea": "Slice per-cell copy", "decision": "reject", "score": 14.1,
                     "cost": 36000, "reason": "charged intermediate"}],
        "similar_tasks": [{"task": "task241", "transform": "transpose", "cost": 0, "score": 25.0}],
    }

    # target: rewrite-first = baseline + margin, with the +0.5 rewrite headroom note
    tgt = W.target_for(20.4, margin=1.5)
    checks["target_is_rewrite_first"] = tgt["target"] == 21.9
    checks["target_cost_budget"] = tgt["cost_budget"] > 0
    checks["target_note_rewrite"] = "REWRITE" in tgt["note"] and "+0.5" in tgt["note"]

    ctx = W.build_context(state, margin=1.5)
    checks["ctx_target"] = ctx["target"]["target"] == 21.9
    checks["ctx_best_carried"] = ctx["best"]["transform"] == "recolor_conv"
    checks["ctx_history"] = len(ctx["history"]) == 1
    checks["ctx_similar"] = ctx["similar_tasks"][0]["task"] == "task241"
    # idioms pulled from the bundled patterns.md via the signature (transpose family)
    checks["ctx_idioms_pulled"] = len(ctx["idioms"]) >= 1
    checks["ctx_idioms_relevant"] = any("Transpose" in (i.get("title", "") + " ".join(i.get("ops") or []))
                                        or "transpose" in i.get("title", "").lower() for i in ctx["idioms"])

    prompt = W.build_prompt(ctx)
    for section in ["REWRITE-FIRST", "OFFICIAL SCORER", "cost = memory_bytes + params",
                    "arc-onnx-golf", "arc-idioms", "agent-author", "VALIDATION GATES",
                    "output > 0.0", "ATTEMPT LOG", "task179"]:
        checks[f"prompt_has::{section[:22]}"] = section in prompt

    # attempt-log recording (MEMORY.md-style)
    with tempfile.TemporaryDirectory() as td:
        f = W.record_attempt(td, "task179", {"id": "A03", "idea": "terminal Transpose", "change": "Transpose perm",
                                             "valid": True, "public": "4/1/262", "score": 25.0, "cost": 0,
                                             "delta": 4.6, "decision": "promote", "reason": "cost 0",
                                             "next": "done"}, best=state["best"], rule=state["rule"])
        txt = open(f).read()
        checks["attempt_logged"] = "A03" in txt and "terminal Transpose" in txt and "Best known:" in txt
        mf = W.update_memory(td, "task179: terminal Transpose perm=[0,1,3,2] → cost 0 (exact 25)")
        checks["memory_updated"] = "terminal Transpose" in open(mf).read()

    # agent contract (empty spec → demo context+prompt)
    status, res, to, msg = W.run({"question": "smoke", "spec": {}}, "test")
    checks["agent_contract"] = (status == "done" and isinstance(res, dict)
                                and isinstance(res.get("prompt"), str) and "REWRITE-FIRST" in res["prompt"])

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    print(f"  -> baseline 20.4 → target {ctx['target']['target']} (budget≈{ctx['target']['cost_budget']}); "
          f"{len(ctx['idioms'])} idioms, {len(ctx['history'])} prior attempts, prompt {len(prompt)} chars")
    ok = all(checks.values())
    print(f"=== arc-worker-context: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
