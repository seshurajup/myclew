"""agent-env — onboard an AGENTIC competition's environment: identify the action space, the reward signal,
and (critically) the BUDGET (max steps / time / replay length) that is usually the real constraint. Runs a
smoke rollout with a random policy to confirm the env is wired and returns its characteristics so the
policy/eval agents can work against it.

The env is supplied by the comp (a Python object satisfying agent_common.Env) via spec['env'] or a factory
in cfg.extra; fixtures use agent_common.ToyCollectEnv. Reusable across any agentic comp.
"""
from __future__ import annotations
from .base import BaseAgent
from . import comp_config as CC
from . import agent_common as AC


def characterize(env, budget=None, seed=0, episodes=1):
    """Smoke-characterize an env with a random policy. episodes>1 averages the random reward/steps over
    that many fresh rollouts for a more stable baseline (single rollout on the same env by default)."""
    obs = env.reset()
    legal = env.action_space()
    rolls = [AC.rollout(env, AC.random_policy(seed + i), step_budget=budget) for i in range(max(1, int(episodes)))]
    import numpy as _np
    info = {"n_actions": len(legal), "sample_actions": list(legal)[:8],
            "random_reward": float(_np.mean([r["total_reward"] for r in rolls])),
            "random_steps": float(_np.mean([r["steps"] for r in rolls])),
            "budget": budget, "budget_ok": all(r["budget_ok"] for r in rolls)}
    if hasattr(env, "optimal_reward"):
        info["optimal_reward"] = env.optimal_reward()
    return info


class AgentEnv(BaseAgent):
    name = "agent-env"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        env = spec.get("env") or AC.ToyCollectEnv(**(spec.get("env_kwargs") or {}))
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else None
        info = characterize(env, budget=spec.get("budget"), seed=int(spec.get("seed", 0)))
        msg = (f"agent-env{(' ' + cfg.slug) if cfg else ''}: {info['n_actions']} actions, "
               f"random-policy reward={info['random_reward']} in {info['random_steps']} steps"
               + (f" (optimal={info['optimal_reward']})" if 'optimal_reward' in info else ""))
        self.log(msg, kind="finding", recommendation="search a policy with agent-policy; watch the budget")
        return self.done({"env_info": info}, msg)


_AGENT = AgentEnv()


def run(q, worker):
    return _AGENT.run(q, worker)
