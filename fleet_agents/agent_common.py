"""agent_common — shared, REUSABLE scaffolding for AGENTIC competitions (pokemon-tcg, autonomous-agent,
ai-agent-security). Agentic comps don't fit predict-a-column: you act in an ENVIRONMENT under a budget and
are scored on the trajectory. This module gives the pack a common contract so no agent-* agent re-invents
the env loop, the rollout, or the budget accounting.

  • Env protocol: reset() → obs; step(action) → (obs, reward, done, info); action_space(); is a pure Python
    object the comp provides (or a toy for fixtures). No RL framework dependency.
  • rollout(env, policy, max_steps, budget) — run one episode, return total reward + a replay trace + the
    step/time budget spent (the JED lesson: the BUDGET is usually the real constraint, not cleverness).
  • ToyCollectEnv — a deterministic grid-collect env for fixtures (agents move to collect reward cells;
    optimal policy is computable) so the whole agentic spine is testable offline with no external simulator.

Pure stdlib + numpy. No torch, no biohub.
"""
from __future__ import annotations
import numpy as np


class Env:
    """Minimal environment protocol. A real comp wraps its simulator to satisfy this; fixtures use ToyCollectEnv."""
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        """Returns (obs, reward, done, info)."""
        raise NotImplementedError

    def action_space(self):
        """List/range of legal actions in the current state."""
        raise NotImplementedError


def rollout(env, policy, max_steps=100, step_budget=None, record_trace=True):
    """Run one episode. policy(obs, legal_actions) → action. Returns dict with total_reward, trace, steps.
    step_budget caps steps (models the comp's action/time budget — the real ceiling).
    record_trace: keep the per-step (action,reward) trace (set False to save memory on long episodes)."""
    obs = env.reset()
    total, trace, steps = 0.0, [], 0
    cap = max_steps if step_budget is None else min(max_steps, step_budget)
    cap = max(0, int(cap))
    for _ in range(cap):
        legal = env.action_space()
        if legal is None or len(legal) == 0:       # no legal moves → episode is effectively over
            break
        a = policy(obs, legal)
        obs, r, done, info = env.step(a)
        total += float(r)
        if record_trace:
            trace.append({"action": a, "reward": r})
        steps += 1
        if done:
            break
    return {"total_reward": float(total), "steps": steps, "trace": trace, "budget_ok": steps <= cap}


# ---------------------------------------------------------------- deterministic toy env for fixtures
class ToyCollectEnv(Env):
    """A 1-D line of length n with reward cells. Agent starts at 0, actions = {0:stay,1:right,2:left}.
    Stepping onto an uncollected reward cell yields its value once. Deterministic → optimal reward known:
    the agent that walks right collecting every cell gets sum(rewards). Used to prove a policy is non-trivial."""
    def __init__(self, rewards=None, n=10):
        self.rewards = list(rewards) if rewards is not None else [0, 0, 1, 0, 2, 0, 0, 3, 0, 0]
        self.n = len(self.rewards)
        self.pos = 0
        self.collected = set()

    def reset(self):
        self.pos = 0; self.collected = set()
        return {"pos": self.pos, "n": self.n, "collected": tuple(sorted(self.collected))}

    def action_space(self):
        return [0, 1, 2]

    def step(self, action):
        if self.n == 0:                            # empty board → immediately done
            return {"pos": 0, "n": 0, "collected": ()}, 0.0, True, {}
        if action == 1:
            self.pos = min(self.n - 1, self.pos + 1)
        elif action == 2:
            self.pos = max(0, self.pos - 1)
        r = 0.0
        if self.pos not in self.collected and self.rewards[self.pos] > 0:
            r = float(self.rewards[self.pos]); self.collected.add(self.pos)
        n_targets = sum(1 for v in self.rewards if v > 0)
        done = len(self.collected) == n_targets   # all reward cells collected (True at once if none exist)
        return {"pos": self.pos, "n": self.n, "collected": tuple(sorted(self.collected))}, r, done, {}

    def optimal_reward(self):
        return float(sum(v for v in self.rewards if v > 0))


# ---------------------------------------------------------------- baseline policies
def random_policy(seed=0):
    rng = np.random.RandomState(seed)
    return lambda obs, legal: (legal[rng.randint(len(legal))] if legal is not None and len(legal) else None)


def greedy_right_policy(obs, legal):
    """Heuristic: keep moving right to sweep reward cells (optimal for ToyCollectEnv)."""
    return 1 if 1 in legal else legal[0]
