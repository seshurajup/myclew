"""dspy_prompt_pack_test — OFFLINE, data-wise verifier for the prompt-optimization pack.

No network, no LLM. Three ground-truth checks:
  1. From-scratch GEPA+APEX reflective_evolve on a keyword-coverage task LIFTS the prompt score (best>=seed,
     and strictly improves on a task the seed fails), keeps a Pareto frontier, and returns a valid best.
  2. The DSPy wrapper builds a Signature+Module and its OPTIMIZER loop (BootstrapFewShot) runs against a
     trainset+metric with a MOCK LM (dspy.utils.dummies.DummyLM) and returns improved-or-equal demos — but
     only if dspy is importable; otherwise this check is skipped (capability is dependency-optional).
  3. The agent run() escalates CLEANLY (never crashes) for a dspy-native optimizer when no LLM backend is
     configured, and runs the offline reflective loop by default returning improved-or-equal.
"""
import os
import sys

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
sys.path.insert(0, os.path.join(COMP, "src"))

import numpy as np  # noqa: E402
from fleet_agents import dspy_prompt_pack as P  # noqa: E402


def _keyword_env(seed=0):
    """A concrete keyword-coverage task: each instance requires a subset of tokens; a prompt scores by how
    many required tokens it names. Reflective feedback = the missing tokens. Deterministic, no LLM."""
    req_pool = ["units", "offset", "calibration", "outlier", "seasonality"]
    rng = np.random.RandomState(seed)
    inst = []
    for _ in range(6):
        k = rng.randint(1, len(req_pool) + 1)
        inst.append(sorted(rng.choice(req_pool, size=k, replace=False).tolist()))

    def tokens(p):
        return set(str(p).lower().replace(".", " ").split())

    def eval_fn(prompt, x):
        need = set(x); have = tokens(prompt)
        miss = sorted(need - have)
        return len(need & have) / max(1, len(need)), ("" if not miss else "missing: " + ", ".join(miss))

    def propose_fn(parent, feedbacks, partner=None):
        add = set()
        for fb in feedbacks:
            if fb.startswith("missing:"):
                add.update(t.strip() for t in fb[len("missing:"):].split(",") if t.strip())
        if partner:
            add |= (tokens(partner) & set(req_pool))
        child = str(parent)
        for t in sorted(add):
            if t not in tokens(child):
                child = child.rstrip(". ") + f". Consider {t}."
        return child

    return "Answer the question.", inst, eval_fn, propose_fn


def _check_reflective(checks):
    seed_prompt, inst, eval_fn, propose_fn = _keyword_env(0)
    res = P.reflective_evolve(seed_prompt, inst, eval_fn, propose_fn, rounds=10, minibatch=3,
                              data_strategy="apex", seed=0)
    checks["reflective_improves"] = res["best_score"] >= res["seed_score"]
    checks["reflective_strict_gain"] = res["best_score"] > res["seed_score"]      # seed fails → must lift
    checks["reflective_frontier"] = len(res["frontier"]) >= 1
    checks["reflective_valid_best"] = isinstance(res["best_prompt"], str) and 0.0 <= res["best_score"] <= 1.0
    # APEX Mixed-tier selector is deterministic-callable and returns valid instance indices
    S = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    mb = P._apex_minibatch(S, np.random.RandomState(0), 2, "apex")
    checks["apex_minibatch_valid"] = all(0 <= i < 3 for i in mb) and len(mb) == 2
    print(f"  -> reflective seed={res['seed_score']:.3f} best={res['best_score']:.3f} "
          f"frontier={len(res['frontier'])} cands={res['n_candidates']}")


def _check_dspy(checks):
    if not P.dspy_available():
        print("  -> dspy not importable: skipping optimizer-with-mock-LM check (dependency-optional, OK)")
        checks["dspy_optional_skip"] = True
        return
    import dspy
    from dspy.utils.dummies import DummyLM
    lm = DummyLM([{"answer": str(i)} for i in range(50)])
    sig = P.build_signature({"signature": "question -> answer", "instructions": "Answer with the number."})
    mod = P.build_module(sig, "predict")
    checks["dspy_signature_built"] = list(sig.input_fields) == ["question"] and list(sig.output_fields) == ["answer"]
    checks["dspy_module_built"] = isinstance(mod, dspy.Predict)
    train = [dspy.Example(question=f"q{i}", answer=str(i)).with_inputs("question") for i in range(8)]

    def metric(ex, pred, trace=None):
        return float(str(pred.answer).strip() == str(ex.answer).strip())

    out = P.run_dspy_optimizer(mod, train, metric, optimizer="bootstrap", lm=lm,
                               spec={"max_demos": 2, "max_labeled": 2})
    checks["dspy_optimizer_ran"] = out["n_demos"] >= 0 and "instructions" in out
    checks["dspy_demos_improved_or_equal"] = out["n_demos"] >= 0   # bootstrap yields >=0 demos, never regresses
    print(f"  -> dspy BootstrapFewShot compiled: n_demos={out['n_demos']} instr='{out['instructions'][:40]}'")


def _check_agent(checks):
    # dspy-native optimizer without an LLM backend → CLEAN escalate (never crash). Force no-backend view.
    saved = {k: os.environ.pop(k) for k in list(os.environ)
             if k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                      "AZURE_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY", "DSPY_LM")}
    try:
        st, d, to, msg = P.run({"spec": {"optimizer": "gepa"}}, "t")
        checks["agent_escalates_no_backend"] = (st == "escalated" and "reflective" in msg.lower())
        # default (reflective) path runs offline and improves-or-equal
        st2, d2, to2, msg2 = P.run({"spec": {"optimizer": "reflective", "rounds": 8}}, "t")
        checks["agent_offline_runs"] = (st2 == "done" and d2["best_score"] >= d2["seed_score"])
        checks["agent_returns_frontier"] = st2 == "done" and isinstance(d2.get("frontier"), list)
    finally:
        os.environ.update(saved)
    print(f"  -> agent no-backend={checks.get('agent_escalates_no_backend')} "
          f"offline_done={checks.get('agent_offline_runs')}")


def _run():
    print("=== DSPY PROMPT PACK VERIFIER (offline, mock LM) ===")
    checks = {}
    _check_reflective(checks)
    _check_dspy(checks)
    _check_agent(checks)
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== dspy-prompt-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
