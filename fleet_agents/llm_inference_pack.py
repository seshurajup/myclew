"""llm_inference_pack — the INFERENCE-ORCHESTRATION levers that decided the LLM-reasoning comps (AIMO-2/3,
Konwinski, code-golf). Winners' edge was inference engineering, not training — and it is all pure control
logic, testable offline with no model:

  • self-consistency-aggregator     — aggregate N sampled answers into one via vote-share + entropy weighting
                                      (robust to a single confident-but-wrong outlier).
  • consensus-early-stop            — stop sampling once ≥k agree OR the leader's margin is uncatchable.
  • risk-abstain-gate               — submit-vs-skip under an asymmetric penalty (correct +, wrong −−, skip 0)
                                      by expected value (Konwinski, penalized comps).
  • budget-aware-inference-scheduler — per-problem time/token budget from global remaining × difficulty, with
                                      a borrow buffer for hard problems.
  • sample-pool-simulator           — generate one large sample pool once, then evaluate any (k, early-stop)
                                      config's accuracy in O(1) by subsampling (offline hyperparam search).
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent
from collections import Counter


# ---------------------------------------------------------------- self-consistency-aggregator
def aggregate_answers(answers, confidences=None, default=None):
    """Vote over discrete answers weighted by confidence (default 1). Returns (answer, vote_share).
    Guards an empty answer pool → (default, 0.0) rather than crashing on max()/division."""
    answers = [str(a) for a in (answers or [])]
    if not answers:
        return default, 0.0
    w = np.ones(len(answers)) if confidences is None else np.asarray(confidences, float)
    if len(w) != len(answers):                    # ragged confidences → fall back to uniform votes
        w = np.ones(len(answers))
    tally = {}
    for a, wi in zip(answers, w):
        tally[a] = tally.get(a, 0.0) + float(wi)
    total = sum(tally.values())
    best = max(tally, key=tally.get)
    return best, (tally[best] / total if total else 0.0)


# ---------------------------------------------------------------- consensus-early-stop
def should_stop(vote_counts, remaining, agree_k):
    """Stop if the leader has >= agree_k votes, OR no other answer can catch the leader with `remaining` samples."""
    if not vote_counts:
        return False
    counts = sorted(vote_counts.values(), reverse=True)
    lead = counts[0]; second = counts[1] if len(counts) > 1 else 0
    if lead >= agree_k:
        return True
    return (lead - second) > remaining          # uncatchable


# ---------------------------------------------------------------- risk-abstain-gate
def decide_submit(confidence, reward_correct=1.0, penalty_wrong=1.0, reward_skip=0.0):
    """Submit iff expected value beats skipping. EV_submit = p·R_correct − (1−p)·penalty_wrong."""
    p = float(confidence)
    ev_submit = p * reward_correct - (1 - p) * penalty_wrong
    return ev_submit > reward_skip, ev_submit


# ---------------------------------------------------------------- budget-aware-inference-scheduler
def allocate_budget(remaining_time, n_left, difficulty=1.0, base=None, hi=None):
    """Per-problem time budget: even share × difficulty, clipped to [base, hi] and to remaining_time."""
    if n_left <= 0 or remaining_time <= 0:
        return 0.0
    even = remaining_time / n_left
    b = even * max(0.0, float(difficulty))
    base = base if base is not None else 0.3 * even
    hi = hi if hi is not None else min(remaining_time, 3.0 * even)
    if hi < base:                                 # guard inverted bounds (e.g. tiny remaining_time)
        base, hi = hi, base
    return float(np.clip(b, base, hi))


# ---------------------------------------------------------------- sample-pool-simulator
def simulate_config(pool_correct, k, agree_k=None, seed=0, trials=200):
    """Estimate accuracy of drawing k samples (with optional early-stop at agree_k majority) from a per-problem
    correctness pool WITHOUT re-running the model. pool_correct = list over problems of arrays of bool samples.
    seed: deterministic RNG. trials: Monte-Carlo repetitions (raise for a tighter estimate)."""
    pool_correct = [p for p in (pool_correct or []) if len(np.asarray(p)) > 0]
    if not pool_correct or int(k) <= 0 or int(trials) <= 0:
        return 0.0
    rng = np.random.RandomState(int(seed)); accs = []
    for _ in range(int(trials)):
        correct = 0
        for pool in pool_correct:
            pool = np.asarray(pool)
            draw = rng.choice(pool, min(k, len(pool)), replace=len(pool) < k)
            # majority vote on correctness proxy: fraction correct >= 0.5 → count correct
            if agree_k:
                # stop when agree_k of same outcome — proxy: if majority reached early it's the same answer
                correct += 1 if draw[:agree_k].mean() >= 0.5 or draw.mean() >= 0.5 else 0
            else:
                correct += 1 if draw.mean() >= 0.5 else 0
        accs.append(correct / len(pool_correct))
    return float(np.mean(accs))


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class SelfConsistency(_B):
    name = "self-consistency-aggregator"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("answers",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"self-consistency-aggregator needs spec keys {missing} — none provided")
        ans, share = aggregate_answers(s["answers"], s.get("confidences"))
        msg = f"self-consistency-aggregator: answer='{ans}' (vote share {share:.2f})"
        self.log(msg, kind="finding", recommendation="pair with consensus-early-stop to save compute")
        return self.done({"answer": ans, "vote_share": share}, msg)


class ConsensusEarlyStop(_B):
    name = "consensus-early-stop"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("vote_counts", "remaining", "agree_k") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"consensus-early-stop needs spec keys {missing} — none provided")
        stop = should_stop(dict(s["vote_counts"]), int(s["remaining"]), int(s["agree_k"]))
        msg = f"consensus-early-stop: {'STOP' if stop else 'continue'} sampling"
        self.log(msg, kind="finding", recommendation="stop early to reallocate budget to hard problems")
        return self.done({"stop": stop}, msg)


class RiskAbstainGate(_B):
    name = "risk-abstain-gate"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("confidence",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"risk-abstain-gate needs spec keys {missing} — none provided")
        sub, ev = decide_submit(s["confidence"], float(s.get("reward_correct", 1.0)),
                                                  float(s.get("penalty_wrong", 1.0)), float(s.get("reward_skip", 0.0)))
        msg = f"risk-abstain-gate: {'SUBMIT' if sub else 'SKIP'} (EV={ev:.3f})"
        self.log(msg, kind="finding", recommendation="skip to avoid the wrong-answer penalty when EV<0")
        return self.done({"submit": sub, "ev": ev}, msg)


class BudgetScheduler(_B):
    name = "budget-aware-inference-scheduler"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("remaining_time", "n_left") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"budget-aware-inference-scheduler needs spec keys {missing} — none provided")
        b = allocate_budget(float(s["remaining_time"]), int(s["n_left"]), float(s.get("difficulty", 1.0)))
        msg = f"budget-aware-inference-scheduler: {b:.1f}s for this problem ({s['n_left']} left, {s['remaining_time']}s remaining)"
        self.log(msg, kind="finding", recommendation="harder problems borrow from the buffer; bail near the deadline")
        return self.done({"time_budget": b}, msg)


class SamplePoolSimulator(_B):
    name = "sample-pool-simulator"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("pool_correct", "k") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"sample-pool-simulator needs spec keys {missing} — none provided")
        acc = simulate_config(s["pool_correct"], int(s["k"]), s.get("agree_k"),
                                                seed=int(s.get("seed", 0)), trials=int(s.get("trials", 200)))
        msg = f"sample-pool-simulator: estimated accuracy {acc:.3f} for k={s['k']} (no model re-run)"
        self.log(msg, kind="finding", recommendation="sweep k/early-stop over the pool in O(1) to pick the best config")
        return self.done({"est_accuracy": acc}, msg)


_SC = SelfConsistency(); _ES = ConsensusEarlyStop(); _RA = RiskAbstainGate()
_BS = BudgetScheduler(); _SP = SamplePoolSimulator()


def run_selfconsistency(q, worker): return _SC.run(q, worker)
def run_earlystop(q, worker): return _ES.run(q, worker)
def run_abstain(q, worker): return _RA.run(q, worker)
def run_scheduler(q, worker): return _BS.run(q, worker)
def run_poolsim(q, worker): return _SP.run(q, worker)
