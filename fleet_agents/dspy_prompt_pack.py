"""dspy_prompt_pack — PROMPT-OPTIMIZATION as a first-class fleet capability.

The existing Prompt-program pack (skill-build / agent-author / agent-config-eval / prompt-optimize)
tunes the DETERMINISTIC skill floor and picks the best skill VARIANT by hidden-label AUC. It does NOT do
prompt-program optimization in the DSPy sense: declaring a Signature+Module and having an OPTIMIZER rewrite
the instruction and bootstrap few-shot demos against a trainset+metric. This pack adds exactly that, plus a
from-scratch reflective-prompt-evolution loop so the capability exists even with no dspy / no LLM backend.

Two layers, both dependency-optional and degrade cleanly (never crash):

  (A) DSPy wrapper  — thin, lazy-imported. Build a Signature from a spec, wrap it in a Module
      (Predict / ChainOfThought / ReAct / ProgramOfThought), and run an OPTIMIZER
      (BootstrapFewShot / BootstrapFewShotWithRandomSearch / MIPROv2 / COPRO / GEPA / BootstrapFinetune)
      against a trainset + metric. Returns the optimized instruction + bootstrapped demos. Requires dspy AND
      an LLM backend; if either is missing the agent ESCALATES with a clear message instead of failing.

  (B) reflective_evolve — a pure-numpy, LM-AGNOSTIC re-implementation of GEPA's core loop
      (Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning",
      arXiv:2507.19457, ICLR 2026 Oral): sample per-instance scores, use natural-language reflection as the
      mutation operator, keep a PARETO FRONTIER of complementary prompts, and crossover frontier lessons.
      It additionally implements APEX dynamic data selection (Wang et al., arXiv:2606.11459) — stratify the
      trainset into Easy / Hard / MIXED tiers by frontier pass-rate and draw the reflection minibatch from the
      informative Mixed tier — the 2026 method that beats GEPA on data efficiency. Both `eval_fn` and
      `propose_fn` are pluggable callbacks, so the loop runs fully OFFLINE with a mock LM and always exists.

Grounded verdict baked in: GEPA remains the strongest *published, implementable, DSPy-native* prompt
optimizer (beats MIPROv2 by ~13% and RL/GRPO by up to 20% with ~35x fewer rollouts). APEX is a data-efficiency
wrapper on TOP of the evolutionary paradigm, not a different optimizer — so we adopt its Mixed-tier selector
as an option INSIDE the reflective loop rather than as a separate method.
"""
from __future__ import annotations

import os

import numpy as np

from .base import BaseAgent

# DSPy-native optimizers this pack knows how to drive (need dspy + an LLM backend).
DSPY_OPTIMIZERS = {
    "bootstrap", "bootstrapfewshot", "bfs",
    "bootstrap-rs", "random-search", "rs", "bootstrapfewshotwithrandomsearch",
    "mipro", "miprov2",
    "copro",
    "gepa",
    "finetune", "bootstrapfinetune",
}


# ---------------------------------------------------------------- (A) DSPy wrapper (lazy, optional)
def dspy_available() -> bool:
    """True if dspy is importable. Lazy — never imported at module load, so `import fleet_agents` stays cheap
    and the pack works when dspy is absent."""
    try:
        import dspy  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def lm_backend_configured(spec=None) -> bool:
    """True if an LLM backend looks available: an explicit `lm` in spec, or a provider API key in env."""
    if spec and spec.get("lm") is not None:
        return True
    keys = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
            "AZURE_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY", "DSPY_LM")
    return any(os.environ.get(k) for k in keys)


def build_signature(spec: dict):
    """Build a dspy.Signature from a spec: {'signature':'q -> a'} or {'inputs':[...],'outputs':[...]} plus
    optional 'instructions'. Returns the Signature class (instructions live in its docstring)."""
    import dspy
    base = spec.get("signature")
    if not base:
        ins = ", ".join(spec.get("inputs") or ["question"])
        outs = ", ".join(spec.get("outputs") or ["answer"])
        base = f"{ins} -> {outs}"
    instr = spec.get("instructions")
    return dspy.Signature(base, instr) if instr else dspy.Signature(base)


def build_module(sig, kind: str = "predict", tools=None):
    """Wrap a Signature in a DSPy Module. kind ∈ predict|cot|react|pot (aliases accepted)."""
    import dspy
    k = (kind or "predict").lower().replace("_", "").replace("-", "")
    if k in ("cot", "chainofthought"):
        return dspy.ChainOfThought(sig)
    if k in ("react",):
        return dspy.ReAct(sig, tools=tools or [])
    if k in ("pot", "programofthought"):
        return dspy.ProgramOfThought(sig)
    return dspy.Predict(sig)


def _build_optimizer(optimizer: str, metric, spec: dict):
    import dspy
    opt = (optimizer or "bootstrap").lower()
    s = spec or {}
    if opt in ("bootstrap", "bootstrapfewshot", "bfs"):
        return dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=int(s.get("max_demos", 4)),
                                     max_labeled_demos=int(s.get("max_labeled", 4)))
    if opt in ("bootstrap-rs", "random-search", "rs", "bootstrapfewshotwithrandomsearch"):
        return dspy.BootstrapFewShotWithRandomSearch(
            metric=metric, max_bootstrapped_demos=int(s.get("max_demos", 4)),
            max_labeled_demos=int(s.get("max_labeled", 4)), num_candidate_programs=int(s.get("candidates", 4)))
    if opt in ("mipro", "miprov2"):
        return dspy.MIPROv2(metric=metric, auto=s.get("auto", "light"))
    if opt in ("copro",):
        return dspy.COPRO(metric=metric, breadth=int(s.get("breadth", 4)), depth=int(s.get("depth", 2)))
    if opt in ("gepa",):
        return dspy.GEPA(metric=metric, reflection_lm=s.get("reflection_lm"),
                         max_metric_calls=int(s.get("max_metric_calls", 30)),
                         reflection_minibatch_size=int(s.get("reflection_minibatch", 3)))
    if opt in ("finetune", "bootstrapfinetune"):
        return dspy.BootstrapFinetune(metric=metric)
    raise ValueError(f"unknown dspy optimizer '{optimizer}'")


def run_dspy_optimizer(module, trainset, metric, optimizer="bootstrap", lm=None, spec=None, valset=None):
    """Compile `module` with a DSPy optimizer against trainset+metric. Returns the optimized instruction +
    bootstrapped demos. Requires dspy + a configured LM (caller passes `lm` or configures dspy globally)."""
    import dspy
    if lm is not None:
        dspy.configure(lm=lm)
    tele = _build_optimizer(optimizer, metric, spec or {})
    kw = {}
    if valset is not None and (optimizer or "").lower() == "gepa":
        kw["valset"] = valset
    compiled = tele.compile(module, trainset=trainset, **kw)
    pred = compiled.predictors()[0]
    demos = list(getattr(pred, "demos", []) or [])
    return {
        "compiled": compiled,
        "instructions": pred.signature.instructions,
        "demos": demos,
        "n_demos": len(demos),
    }


# ---------------------------------------------------------------- (B) reflective_evolve (from-scratch GEPA + APEX)
def _pareto_frontier(scores: np.ndarray) -> list:
    """Indices of candidates NOT strictly dominated on the per-instance score matrix scores[cand, inst].
    A candidate is on the frontier if no other candidate is >= on every instance and > on at least one."""
    n = scores.shape[0]
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j == i:
                continue
            if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def _apex_minibatch(scores_frontier: np.ndarray, rng, k: int, strategy: str = "apex") -> list:
    """Pick k instance indices for the reflection minibatch. APEX: prefer the MIXED tier (0<pass<1 across
    frontier candidates) — the informative frontier — then Hard, then Easy. 'uniform' = plain GEPA sampling."""
    m = scores_frontier.shape[1]
    if strategy != "apex" or scores_frontier.shape[0] == 0:
        return list(rng.choice(m, size=min(k, m), replace=False)) if m else []
    passrate = (scores_frontier >= 1.0).mean(axis=0)           # per-instance pass fraction across frontier
    mixed = np.where((passrate > 0.0) & (passrate < 1.0))[0]
    hard = np.where(passrate <= 0.0)[0]
    easy = np.where(passrate >= 1.0)[0]
    order = np.concatenate([rng.permutation(mixed), rng.permutation(hard), rng.permutation(easy)])
    return list(order[:min(k, m)])


def reflective_evolve(seed_prompt, instances, eval_fn, propose_fn, *,
                      rounds=8, minibatch=3, data_strategy="apex", crossover=True, seed=0,
                      max_pop=12):
    """GEPA-style reflective prompt evolution (Agrawal et al. 2025, arXiv:2507.19457) with APEX dynamic data
    selection (Wang et al. 2026, arXiv:2606.11459). Pure-numpy, LM-AGNOSTIC.

      eval_fn(prompt, instance)   -> (score in [0,1], feedback_str)   — the (mockable) rollout+metric
      propose_fn(parent, feedbacks, partner=None) -> child_prompt      — the (mockable) reflective mutation

    Maintains a Pareto frontier of complementary prompts, mutates a frontier parent from reflective feedback on
    an APEX-selected minibatch, optionally crosses over two frontier prompts, and keeps every non-dominated
    candidate. Returns the best prompt by mean score plus the frontier and history. best_score >= seed_score
    always (we only ADD candidates and report the max), so it degrades to a no-op, never a regression.
    """
    rng = np.random.RandomState(int(seed))
    inst = list(instances)

    def row(prompt):
        return np.array([float(eval_fn(prompt, x)[0]) for x in inst], dtype=float)

    prompts = [seed_prompt]
    scores = [row(seed_prompt)]
    seed_score = float(scores[0].mean())

    for _ in range(int(rounds)):
        S = np.vstack(scores)
        front = _pareto_frontier(S)
        # candidate selection: sample a frontier parent weighted by aggregate score
        agg = S[front].mean(axis=1)
        w = agg - agg.min() + 1e-6
        parent_idx = front[int(rng.choice(len(front), p=w / w.sum()))]
        parent = prompts[parent_idx]
        # APEX minibatch from the frontier's mixed tier → reflective feedback on FAILING instances
        mb = _apex_minibatch(S[front], rng, int(minibatch), data_strategy)
        feedbacks = []
        for i in mb:
            sc, fb = eval_fn(parent, inst[i])
            if sc < 1.0 and fb:
                feedbacks.append(fb)
        partner = None
        if crossover and len(front) > 1:
            partner = prompts[front[int(rng.choice(len(front)))]]
        child = propose_fn(parent, feedbacks, partner) if feedbacks or partner else parent
        if child is not None and child not in prompts:
            prompts.append(child)
            scores.append(row(child))
            # bound population: keep frontier + top-agg fillers
            if len(prompts) > max_pop:
                S2 = np.vstack(scores)
                fr = set(_pareto_frontier(S2))
                order = sorted(range(len(prompts)),
                               key=lambda i: (i in fr, S2[i].mean()), reverse=True)[:max_pop]
                order = sorted(order)
                prompts = [prompts[i] for i in order]
                scores = [scores[i] for i in order]

    S = np.vstack(scores)
    aggr = S.mean(axis=1)
    best = int(aggr.argmax())
    front = _pareto_frontier(S)
    return {
        "best_prompt": prompts[best],
        "best_score": float(aggr[best]),
        "seed_score": seed_score,
        "improved": float(aggr[best]) >= seed_score,
        "gain": float(aggr[best] - seed_score),
        "frontier": [prompts[i] for i in front],
        "n_candidates": len(prompts),
    }


# ---------------------------------------------------------------- general optimizer: dataset + metric + runner
def _default_propose(parent, feedbacks, partner=None):
    """LM-FREE fallback proposer: append the reflective feedback as a directive. Works on a single prompt (str)
    OR a prompt BUNDLE (dict{node: prompt}) — for a bundle it appends to every node. For real reflection pass an
    LM-backed proposer (or use the DSPy/GEPA path with reflection_lm). Deterministic; never regresses (loop keeps
    the max)."""
    hint = " ".join(dict.fromkeys(fb for fb in feedbacks if fb))[:200]

    def _amend(text):
        t = str(text)
        if hint and hint not in t:
            t = t.rstrip(". ") + f". Note: {hint}."
        return t

    if isinstance(parent, dict):                       # multi-node flow bundle
        child = dict(parent)
        if isinstance(partner, dict):                  # crossover: pull partner's longer node prompt
            for k, v in partner.items():
                if len(str(v)) > len(str(child.get(k, ""))):
                    child[k] = v
        # amend the node with the most failing feedback (here: all nodes, cheaply)
        for k in child:
            child[k] = _amend(child[k])
        return child
    return _amend(parent)


def optimize_prompts(examples, metric="norm_exact", runner=None, seed_prompts="Answer the question.",
                     propose=None, metric_spec=None, rounds=8, minibatch=3, data_strategy="apex",
                     crossover=True, seed=0):
    """Black-box prompt/flow optimizer — the reusable core. Ties the three inputs together:
       • examples  : [{input, gold}] from prompt-dataset
       • metric    : a named prompt-metric ('norm_exact','token_f1','numeric',... ) → score+feedback
       • runner    : runner(prompt, input) -> prediction. This is what EXECUTES the system under a candidate
                     prompt — a single LLM call, a DSPy module, OR a LANGGRAPH node-flow (pass the graph's
                     invoke). seed_prompts may be a str (one prompt) or dict{node: prompt} (a multi-node flow,
                     each node evolved as a bundle).
    Returns reflective_evolve's result (best_prompt / best_score / gain / frontier). runner is REQUIRED and is
    in-process (a callable can't cross the JSON board)."""
    from . import prompt_metric as PM
    if runner is None:
        raise ValueError("optimize_prompts needs runner(prompt, input)->prediction — an LLM call, a DSPy module, "
                         "or a LangGraph flow's invoke. It executes the system under each candidate prompt.")
    score, feedback = PM.build_metric(metric, metric_spec)

    def eval_fn(prompt, ex):
        pred = runner(prompt, ex["input"])
        return float(score(pred, ex["gold"])), feedback(pred, ex["gold"])

    return reflective_evolve(seed_prompts, examples, eval_fn, propose or _default_propose,
                             rounds=rounds, minibatch=minibatch, data_strategy=data_strategy,
                             crossover=crossover, seed=seed)


# ---------------------------------------------------------------- self-contained offline demo task
def _keyword_task(spec: dict):
    """Build a JSON-safe, LM-free demo: a keyword-coverage task where a prompt is judged by whether it names
    the tokens each instance requires. Reflective feedback = the missing tokens. Deterministic; proves the
    evolution loop lifts the prompt without any LLM. Returns (seed_prompt, instances, eval_fn, propose_fn)."""
    req_pool = spec.get("required") or ["units", "offset", "calibration", "outlier", "seasonality"]
    n = int(spec.get("n_instances", 6))
    rng = np.random.RandomState(int(spec.get("seed", 0)))
    instances = []
    for _ in range(n):
        k = rng.randint(1, len(req_pool) + 1)
        instances.append(sorted(rng.choice(req_pool, size=k, replace=False).tolist()))
    seed_prompt = spec.get("seed_prompt", "Answer the question.")

    def tokens(p):
        return set(str(p).lower().replace(".", " ").replace(",", " ").split())

    def eval_fn(prompt, inst):
        have = tokens(prompt)
        need = set(inst)
        hit = len(need & have)
        score = hit / max(1, len(need))
        missing = sorted(need - have)
        fb = "" if not missing else "missing: " + ", ".join(missing)
        return score, fb

    def propose_fn(parent, feedbacks, partner=None):
        add = set()
        for fb in feedbacks:
            if fb.startswith("missing:"):
                add.update(t.strip() for t in fb[len("missing:"):].split(",") if t.strip())
        child = str(parent)
        if partner:                                    # crossover: absorb partner's named tokens too
            add |= (tokens(partner) & set(req_pool))
        for t in sorted(add):
            if t not in tokens(child):
                child = child.rstrip(". ") + f". Consider {t}."
        return child

    return seed_prompt, instances, eval_fn, propose_fn


# ---------------------------------------------------------------- the agent
class DspyPromptOptimize(BaseAgent):
    name = "dspy-prompt-optimize"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        optimizer = (spec.get("optimizer") or "reflective").lower()

        # BOARD-DRIVEN real task: dataset (examples/file/synthetic) + named metric arrive as JSON; the RUNNER
        # (executes a prompt on an input — an LLM call / DSPy module / LangGraph flow) is a callable passed
        # in-process via q['runner']. With a runner we optimise for real; without one we still build+report the
        # dataset+metric and escalate naming exactly what's missing (the runner/LM), never crash.
        if any(k in spec for k in ("examples", "file", "synthetic")) and (spec.get("metric") or spec.get("runner") or q.get("runner")):
            from . import prompt_dataset as PD
            try:
                ts = PD.build_trainset(spec)
            except (ValueError, FileNotFoundError) as e:
                return self.escalate(worker, "researcher", f"dspy-prompt-optimize: dataset error — {e}")
            metric_name = spec.get("metric", "norm_exact")
            runner = q.get("runner") or spec.get("runner")
            if not callable(runner):
                return self.escalate(
                    worker, "researcher",
                    f"dspy-prompt-optimize: dataset ready ({ts['n']} examples from {ts['source']}) + metric "
                    f"'{metric_name}', but no RUNNER. Pass q['runner']=fn(prompt,input)->prediction in-process "
                    f"(an LLM call, a DSPy module, or a LangGraph flow's .invoke). Then it optimises the prompt/"
                    f"bundle against the metric. (A runner is a callable — it can't cross the JSON board.)")
            try:
                res = optimize_prompts(
                    ts["train"], metric=metric_name, runner=runner,
                    seed_prompts=spec.get("seed_prompt", spec.get("seed_prompts", "Answer the question.")),
                    metric_spec=spec, rounds=int(spec.get("rounds", 8)),
                    minibatch=int(spec.get("minibatch", 3)), data_strategy=spec.get("data_strategy", "apex"),
                    crossover=bool(spec.get("crossover", True)), seed=int(spec.get("seed", 0)))
            except Exception as e:  # noqa: BLE001
                return self.escalate(worker, "researcher", f"dspy-prompt-optimize: runner/metric failed — {type(e).__name__}: {str(e)[:120]}")
            msg = (f"dspy-prompt-optimize: optimised over {ts['n']} examples ({ts['source']}) by metric "
                   f"'{metric_name}' → score {res['seed_score']:.3f}→{res['best_score']:.3f} (+{res['gain']:.3f}), "
                   f"{res['n_candidates']} candidates, {len(res['frontier'])} on frontier. Runner executed each "
                   f"candidate prompt{'/bundle' if isinstance(res['best_prompt'], dict) else ''}.")
            self.log(msg, kind="finding", recommendation="ship best_prompt; supply an LM-backed proposer for stronger reflection")
            return self.done({"best_prompt": res["best_prompt"], "best_score": res["best_score"],
                              "seed_score": res["seed_score"], "gain": res["gain"], "improved": res["improved"],
                              "frontier": res["frontier"], "n_candidates": res["n_candidates"],
                              "metric": metric_name, "n_examples": ts["n"]}, msg)

        # DSPy-native path needs dspy AND an LLM backend — degrade with a CLEAN escalate, never crash.
        if optimizer in DSPY_OPTIMIZERS:
            if not dspy_available():
                return self.escalate(
                    worker, "researcher",
                    f"dspy-prompt-optimize: optimizer='{optimizer}' needs the `dspy` package (not importable). "
                    f"Install ABI-safely (`pip install dspy` leaves numpy/torch untouched) or use "
                    f"optimizer='reflective' (offline, no dependency).")
            if not lm_backend_configured(spec):
                return self.escalate(
                    worker, "researcher",
                    f"dspy-prompt-optimize: optimizer='{optimizer}' needs an LLM backend (no provider API key "
                    f"in env and no `lm` in spec). Set one, or use optimizer='reflective' (offline).")
            # A live trainset+metric are Python callables that can't arrive over the JSON board; the DSPy path
            # is driven in-process (tests / a Python caller pass module+trainset+metric to run_dspy_optimizer).
            return self.escalate(
                worker, "researcher",
                f"dspy-prompt-optimize: optimizer='{optimizer}' is ready (dspy + LM present) but needs an "
                f"in-process trainset+metric — call run_dspy_optimizer(module, trainset, metric, "
                f"optimizer='{optimizer}') from Python. Board-driven runs use optimizer='reflective'.")

        # Default: OFFLINE reflective GEPA+APEX evolution on a JSON-safe keyword task — always available.
        seed_prompt, instances, eval_fn, propose_fn = _keyword_task(spec)
        res = reflective_evolve(
            seed_prompt, instances, eval_fn, propose_fn,
            rounds=int(spec.get("rounds", 8)), minibatch=int(spec.get("minibatch", 3)),
            data_strategy=spec.get("data_strategy", "apex"), crossover=bool(spec.get("crossover", True)),
            seed=int(spec.get("seed", 0)))
        msg = (f"dspy-prompt-optimize: reflective GEPA+APEX evolution lifted prompt score "
               f"{res['seed_score']:.3f}→{res['best_score']:.3f} (+{res['gain']:.3f}) over "
               f"{res['n_candidates']} candidates, {len(res['frontier'])} on the Pareto frontier. "
               f"(GEPA arXiv:2507.19457 + APEX arXiv:2606.11459; offline, LM-agnostic. For real prompt-program "
               f"tuning call run_dspy_optimizer with optimizer=bootstrap|mipro|gepa when a LM backend is set.)")
        self.log(msg, kind="finding",
                 recommendation="use DSPy BootstrapFewShot/MIPROv2/GEPA in-process for LLM tasks; reflective "
                                "loop is the offline fallback and the from-scratch GEPA implementation")
        return self.done({"best_prompt": res["best_prompt"], "best_score": res["best_score"],
                          "seed_score": res["seed_score"], "gain": res["gain"], "improved": res["improved"],
                          "frontier": res["frontier"], "n_candidates": res["n_candidates"]}, msg)


_DSPY = DspyPromptOptimize()


def run(q, worker):
    return _DSPY.run(q, worker)
