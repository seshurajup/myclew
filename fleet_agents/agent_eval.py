"""agent-eval — score a chosen policy OFFLINE before any submit: mean reward over N episodes, budget
compliance, and (if the env exposes it) fraction-of-optimal. This is the JED lesson operationalized — the
budget (~step/time/replay ceiling) is usually what decides the score, so we check it here, locally, instead
of burning a submission to discover a timeout.

Returns a verdict the submit gate can trust. Reusable across all agentic comps; sec-eval specializes it
for the attack-budget model.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent
from . import comp_config as CC
from . import agent_common as AC


def _discounted(trace, gamma):
    """Sum of per-step rewards discounted by gamma^t (undiscounted when gamma>=1)."""
    return float(sum((gamma ** t) * float(step.get("reward", 0.0)) for t, step in enumerate(trace)))


def evaluate(env_factory, policy, episodes=10, budget=None, gamma=1.0):
    """Mean reward + budget compliance over `episodes` fresh rollouts.
    gamma: discount factor for the per-episode return (gamma>=1 → plain undiscounted sum = default)."""
    episodes = max(1, int(episodes))
    rolls = [AC.rollout(env_factory(), policy, step_budget=budget) for _ in range(episodes)]
    if gamma is not None and gamma < 1.0:
        rewards = [_discounted(r["trace"], float(gamma)) for r in rolls]
    else:
        rewards = [r["total_reward"] for r in rolls]
    steps = [r["steps"] for r in rolls]
    out = {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
           "mean_steps": float(np.mean(steps)), "budget": budget,
           "budget_ok": all(r["budget_ok"] for r in rolls)}
    env0 = env_factory()
    if hasattr(env0, "optimal_reward"):
        opt = env0.optimal_reward()
        out["frac_optimal"] = float(np.mean(rewards) / opt) if opt else None
    return out


_POLICIES = {"greedy_right": AC.greedy_right_policy, "random0": AC.random_policy(0)}


class AgentEval(BaseAgent):
    name = "agent-eval"
    thread = "M"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else None
        ef = spec.get("env_factory") or (lambda: AC.ToyCollectEnv(**(spec.get("env_kwargs") or {})))
        pol = spec.get("policy") or _POLICIES.get(spec.get("policy_name", "greedy_right"), AC.greedy_right_policy)
        res = evaluate(ef, pol, episodes=int(spec.get("episodes", 10)), budget=spec.get("budget"),
                       gamma=float(spec.get("gamma", 1.0)))
        ok = res["budget_ok"]
        msg = (f"agent-eval{(' ' + cfg.slug) if cfg else ''}: mean_reward={res['mean_reward']:.3f} "
               f"budget_ok={ok}" + (f" frac_optimal={res['frac_optimal']:.2f}" if res.get('frac_optimal') is not None else ""))
        self.log(msg, kind="verdict",
                 recommendation="SAFE to submit (budget met)" if ok else "OVER BUDGET — will time out, do NOT submit")
        return self.done({"eval": res, "safe": ok}, msg)


_AGENT = AgentEval()


def run(q, worker):
    return _AGENT.run(q, worker)
