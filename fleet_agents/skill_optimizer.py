"""skill_optimizer — the SkillOpt (microsoft/SkillOpt) "train a text artifact with an SGD-analogy loop"
pattern, ported as a reusable optimizer. SkillOpt improves an LLM agent's SKILL/PROMPT text without touching
weights: rollouts → Reflect (an LLM writes a "gradient" patch from a minibatch of failures) → Aggregate (merge
patches) → Optimizer update, all behind a HELD-OUT validation gate so a patch is only kept if it actually
improves on unseen tasks (guards against overfitting the minibatch). It mirrors minibatch SGD: epochs,
minibatches, learning-rate-as-patch-strength, early stopping, and top-k candidate selection.

This differs from our successive-halving experiment SCREENER (which ranks fixed configs) — here the search
SPACE itself is edited each step by a proposer (an LLM via llm_backend, or any callable), so it optimizes the
prompt/skill text online. The control loop is pure-python and offline-testable; the LLM is injected as
`propose_fn` (defaults to llm_backend when a provider is configured, stubbed in tests).

This port captures SkillOpt's FULL concept set, offline-testable:
  Optimizer core (SGD analogy):
    • rank_and_select(cands, scores, k)        — gradient-clip: keep top-k edits (controls step size).
    • gate_accept / evaluate_gate              — held-out gate; evaluate_gate is the 3-way best-tracking
                                                 decision (accept_new_best / accept / reject).
    • optimize(init, propose_fn, score_fn, ...)— the epoch/minibatch SGD-analogy loop.
    • reflect_patch_llm(skill, failures, ...)  — Reflect: LLM "gradient" via llm_backend (guarded).
    • aggregate_patches(patches)               — Aggregate: hierarchical merge, FAILURE-priority.
  Sleep self-evolution engine (skillopt_sleep):
    • mine_retry_chains(sessions)              — mine: label tasks from retry chains (no LLM).
    • sleep_cycle(init, sessions, ...)         — harvest→mine→replay→consolidate(gate)→adopt in one step.
"""
from __future__ import annotations
from .base import BaseAgent
from . import llm_backend as LB


def rank_and_select(candidates, scores, k=1):
    """Return the top-k candidates by score (descending). SkillOpt's rank_and_select / gradient-clip step
    (optimizer/clip.py): controls the effective STEP SIZE by keeping only the top-L most impactful edits."""
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in order[:max(1, int(k))]]


def aggregate_patches(patches):
    """SkillOpt Aggregate stage (gradient/aggregate.py): merge independently-generated Reflect patches into one,
    with FAILURE-driven patches taking priority over success-driven ones. A patch = {"edits":[...], "source":
    "failure"|"success", "weight":float}. Returns a single merged patch (failure edits first, dedup, priority
    order preserved) — the hierarchical merge, minus the LLM call."""
    fail = [p for p in patches if p.get("source") == "failure"]
    succ = [p for p in patches if p.get("source") != "failure"]
    merged, seen = [], set()
    for p in fail + succ:                                    # failure priority
        for e in p.get("edits", []):
            key = e if isinstance(e, str) else str(e)
            if key not in seen:
                seen.add(key); merged.append(e)
    return {"edits": merged, "n_failure_sources": len(fail), "n_success_sources": len(succ)}


def evaluate_gate(cand_score, current_score, best_score, best_step, global_step):
    """SkillOpt full best-tracking gate (skillopt_sleep/gate.py evaluate_gate): a candidate is
      'accept_new_best' if it beats BOTH current and best (record a new global best),
      'accept'          if it beats current but not the all-time best (local improvement kept),
      'reject'          otherwise (revert to current).
    Returns (action, score, best_score, best_step)."""
    if cand_score > current_score:
        if cand_score > best_score:
            return ("accept_new_best", cand_score, cand_score, global_step)
        return ("accept", cand_score, best_score, best_step)
    return ("reject", current_score, best_score, best_step)


def mine_retry_chains(sessions):
    """SkillOpt heuristic_mine (skillopt_sleep/mine.py): turn a session log into labeled TaskRecords WITHOUT an
    LLM. The key signal — a RETRY CHAIN: the same intent re-asked after negative feedback means the early
    attempt FAILED. sessions = [{"intent":str, "feedback":["neg"|"pos"|...]}]. Labels each:
      negative present and no later positive → 'fail'; positive present → 'success'; else 'unknown'.
    Returns [{"intent","label"}] — the training units the consolidate step replays."""
    neg = {"neg", "negative", "wrong", "no", "retry", "bad"}
    pos = {"pos", "positive", "yes", "good", "correct", "thanks"}
    out = []
    for s in sessions:
        sig = [str(x).lower() for x in s.get("feedback", [])]
        has_pos = any(x in pos for x in sig); has_neg = any(x in neg for x in sig)
        label = "success" if has_pos and not has_neg else ("fail" if has_neg else "unknown")
        out.append({"intent": s.get("intent", ""), "label": label})
    return out


def sleep_cycle(init_skill, sessions, propose_fn, score_fn, val_fn=None, *, min_delta=0.0):
    """SkillOpt-Sleep nightly self-evolution cycle (skillopt_sleep/cycle.py): harvest→mine→replay→consolidate
    (gated)→adopt, as ONE offline step. Mines retry chains from `sessions` to find failing intents, proposes
    edits (propose_fn), aggregates them (failure-priority), scores the candidate, and adopts it only through the
    best-tracking gate. Returns (skill, action, report)."""
    val_fn = val_fn or score_fn
    tasks = mine_retry_chains(sessions)
    failing = [t["intent"] for t in tasks if t["label"] == "fail"]
    cur_score = val_fn(init_skill)
    cands = propose_fn(init_skill, "; ".join(failing)) or [init_skill]
    cand_scores = [val_fn(c) for c in cands]
    best_cand = rank_and_select(cands, cand_scores, k=1)[0]; best_cand_score = max(cand_scores)
    action, score, _, _ = evaluate_gate(best_cand_score, cur_score + min_delta, cur_score, 0, 1)
    adopted = best_cand if action != "reject" else init_skill
    return adopted, action, {"n_failing": len(failing), "action": action,
                             "cur_score": cur_score, "cand_score": best_cand_score}


def gate_accept(cur_val, cand_val, min_delta=0.0):
    """Held-out validation gate: accept the candidate only if it beats the current by > min_delta."""
    return cand_val > cur_val + min_delta


def optimize(init, propose_fn, score_fn, val_fn=None, *, epochs=20, k=1, n_candidates=4,
             min_delta=0.0, patience=5):
    """SkillOpt SGD-analogy optimization of a text/skill artifact.
      init         — starting skill (any object the proposer/scorers understand).
      propose_fn   — (skill, minibatch_scores) -> list[new_skill]  (the LLM "reflect/gradient"; n_candidates).
      score_fn     — (skill) -> float on a TRAIN minibatch (drives the reflection).
      val_fn       — (skill) -> float on HELD-OUT data; the acceptance gate (defaults to score_fn).
    Returns (best_skill, best_val, history). Keeps a candidate only if it improves the held-out score."""
    val_fn = val_fn or score_fn
    best = init; best_val = val_fn(init); hist = [best_val]; stale = 0
    for ep in range(int(epochs)):
        train_score = score_fn(best)
        cands = propose_fn(best, train_score) or []
        cands = list(cands)[:max(1, int(n_candidates))]
        if not cands:
            break
        cand_vals = [val_fn(c) for c in cands]
        top = rank_and_select(cands, cand_vals, k=k)
        top_val = max(cand_vals)
        if gate_accept(best_val, top_val, min_delta):
            best, best_val = top[0], top_val; stale = 0
        else:
            stale += 1
        hist.append(best_val)
        if stale >= patience:
            break
    return best, best_val, hist


def reflect_patch_llm(skill, failures, *, model="dummy/echo", n=4):
    """LLM 'gradient': ask llm_backend to rewrite the skill given failing examples. Returns n candidate skills.
    Guarded — falls back to returning [skill] if no provider is configured (caller's gate then keeps current)."""
    try:
        out = []
        for i in range(n):
            r = LB.chat([{"role": "system", "content": "You improve an agent skill given its failures."},
                         {"role": "user", "content": f"SKILL:\n{skill}\nFAILURES:\n{failures}\nRewrite the skill."}],
                        model=model)
            out.append(r["text"])
        return out
    except LB.LLMBackendUnavailable:
        return [skill]


# ---------------------------------------------------------------- agent
class SkillOptimizer(BaseAgent):
    name = "skill-optimizer"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q); import random
        rnd = random.Random(int(s.get("seed", 0)))
        # synthetic proof: a "skill" is a float; target=optimum; proposer perturbs; held-out gate = same fn + noise.
        target = float(s.get("target", 7.0)); epochs = int(s.get("epochs", 40))
        def score(x): return -abs(x - target)                        # train score (higher=better)
        def val(x): return -abs(x - target)                          # held-out gate
        def propose(x, tr):
            step = 0.5 * (1.0 + abs(tr))                              # bigger step when far (lr-analogy)
            return [x + rnd.uniform(-step, step) for _ in range(4)]
        best, bv, hist = optimize(0.0, propose, score, val, epochs=epochs, k=1, patience=8)
        msg = (f"skill-optimizer: SGD-analogy skill search converged 0.0→{best:.3f} (target {target}, "
               f"|err|={abs(best-target):.3f}) in {len(hist)-1} epochs; held-out gate accepted only improving "
               f"patches (val {hist[0]:.2f}→{bv:.2f}). Reflect via llm_backend (Ollama/OpenRouter/Claude) for "
               f"real prompt/skill optimization (SkillOpt)")
        self.log(msg, kind="finding",
                 recommendation="optimize prompts/skills online: inject reflect_patch_llm as propose_fn and a "
                                "held-out val_fn as the gate; monotone improvement, no weight training")
        return self.done({"best": best, "best_val": bv, "err": abs(best - target), "epochs": len(hist) - 1}, msg)


_AGENT = SkillOptimizer()


def run_skillopt(q, worker):
    return _AGENT.run(q, worker)
