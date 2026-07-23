"""agent-policy — search for a good policy in an agentic competition's environment. The reusable core is a
CANDIDATE TOURNAMENT: evaluate a pool of candidate policies by rollout and keep the best by mean reward
(the GM staple — beat a strong heuristic before anything fancy). An optional evolutionary refinement layer
perturbs a parameterized policy; the tournament alone already beats random and is the honest baseline.

Candidates default to {random(seed), greedy-right, greedy-left}; a comp supplies its own via spec['candidates'].
Reusable across pokemon-tcg (game policy), autonomous-agent (control), and — via sec-* — attack sequencing.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent
from . import comp_config as CC
from . import agent_common as AC


def _greedy_left(obs, legal):
    return 2 if 2 in legal else legal[0]


def default_candidates():
    return {
        "random0": AC.random_policy(0),
        "random1": AC.random_policy(1),
        "greedy_right": AC.greedy_right_policy,
        "greedy_left": _greedy_left,
    }


def tournament(env_factory, candidates=None, episodes=3, budget=None, on_error="skip"):
    """Evaluate each candidate over `episodes` rollouts on FRESH envs. Returns (best_name, scores dict).
    on_error: 'skip' scores a crashing candidate -inf (tournament survives one bad policy); 'raise' re-raises."""
    candidates = candidates or default_candidates()
    episodes = max(1, int(episodes))
    scores = {}
    for name, pol in candidates.items():
        try:
            rs = [AC.rollout(env_factory(), pol, step_budget=budget)["total_reward"] for _ in range(episodes)]
            scores[name] = float(np.mean(rs))
        except Exception:  # noqa: BLE001
            if on_error == "raise":
                raise
            scores[name] = float("-inf")
    if not scores:
        return None, {}
    best = max(scores, key=lambda n: scores[n])
    return best, scores


class AgentPolicy(BaseAgent):
    name = "agent-policy"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else None
        ef = spec.get("env_factory") or (lambda: AC.ToyCollectEnv(**(spec.get("env_kwargs") or {})))
        best, scores = tournament(ef, candidates=spec.get("candidates"),
                                  episodes=int(spec.get("episodes", 3)), budget=spec.get("budget"))
        msg = (f"agent-policy{(' ' + cfg.slug) if cfg else ''}: best='{best}' reward={scores[best]:.3f} "
               f"(pool={ {k: round(v,2) for k,v in scores.items()} })")
        self.log(msg, kind="finding", recommendation=f"eval '{best}' vs budget with agent-eval before submit")
        return self.done({"best_policy": best, "scores": scores}, msg)


_AGENT = AgentPolicy()


def run(q, worker):
    return _AGENT.run(q, worker)
