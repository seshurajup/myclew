"""harness-opt-gate — blind-holdout keep-if-improves gate for PROMPT / HARNESS self-optimization ONLY.

This is deepagents' `better-harness/core.py:run_experiment` acceptance rule ported as a tiny, deterministic,
dependency-free pure function. It exists for the PROMPT-OPTIMIZATION side of the fleet (the prompt-optimize
triad: prompt-metric / prompt-dataset / dspy-prompt-optimize / skill-build / agent-author) — the surfaces we
edit are PROMPTS / SKILLS / HARNESS text, not models.

    DELIBERATELY NOT for ML experiments. Our ML gating (lever-hunt / feasibility-gate / per-embryo paired
    Wilcoxon LOEO) is MORE rigorous and stays the sole judge of model changes. Do not route a model change
    through here.

The contract (from better-harness):
  * the optimizer sees ONLY the `train` split's failures; the `holdout` split is a BLIND accept/reject gate;
  * ACCEPT iff the COMBINED (train + holdout) pass-count STRICTLY improves over baseline
    (`cand_train.passed + cand_holdout.passed > base_train.passed + base_holdout.passed`) — this is what
    guards against a prompt edit that overfits the visible train failures while regressing the blind holdout;
  * results are taken as INPUT arrays (no LLM, no subprocess pytest here) so the gate is a pure, testable fn.

Inputs are permissive — each split's result may be given as:
  * an int            (already-counted passes),
  * a list            (booleans / 0-1 / pass-case-ids → truthy elements are counted),
  * a dict            ({"passed": int, "total": int} or {"results": [...]}).

`gate(...)` returns accept/reject + the per-split and combined deltas, and a `valid` flag that warns when the
baseline and candidate did not run on the SAME-SIZED split (better-harness `validate_experiment` forces train
and holdout to cover the same strata — a size mismatch means the gate would be comparing different eval sets).

Reusable across every competition: it's pure arithmetic over pass counts. Offline, no GPU, no network.
"""
from __future__ import annotations

from .base import BaseAgent


def _passed(x) -> tuple[int, int | None]:
    """Coerce a split result to (passed, total|None). Accepts int / list / dict."""
    if x is None:
        return 0, None
    if isinstance(x, bool):
        return int(x), 1
    if isinstance(x, int):
        return x, None
    if isinstance(x, (list, tuple)):
        return sum(1 for v in x if v), len(x)
    if isinstance(x, dict):
        if "passed" in x:
            p = int(x["passed"])
            t = int(x["total"]) if "total" in x else None
            return p, t
        if "results" in x and isinstance(x["results"], (list, tuple)):
            r = x["results"]
            return sum(1 for v in r if v), len(r)
    # last resort: truthiness
    return (1, 1) if x else (0, 1)


def gate(baseline_train, baseline_holdout, cand_train, cand_holdout) -> dict:
    """ACCEPT a prompt/harness edit iff combined (train+holdout) pass-count STRICTLY improves over baseline.

    Each arg is a split result (int / list / dict — see _passed). Returns a verdict dict:
      accept              : bool  — the keep-if-combined-improves decision.
      baseline_combined   : int   — base_train.passed + base_holdout.passed.
      candidate_combined  : int   — cand_train.passed + cand_holdout.passed.
      delta_combined      : int   — candidate_combined - baseline_combined (must be > 0 to accept).
      train_delta         : int   — cand_train.passed - base_train.passed (the VISIBLE signal).
      holdout_delta       : int   — cand_holdout.passed - base_holdout.passed (the BLIND gate signal).
      valid               : bool  — False if a per-split total mismatched between baseline and candidate
                                    (comparing different-sized eval sets → the combined count isn't comparable).
      reason              : str
    """
    bt, btt = _passed(baseline_train)
    bh, bht = _passed(baseline_holdout)
    ct, ctt = _passed(cand_train)
    ch, cht = _passed(cand_holdout)

    base_comb = bt + bh
    cand_comb = ct + ch
    delta = cand_comb - base_comb
    accept = delta > 0

    # same-strata guard: when totals are known, baseline & candidate must have run the SAME-SIZED split.
    mismatch = []
    if btt is not None and ctt is not None and btt != ctt:
        mismatch.append(f"train total {btt}!={ctt}")
    if bht is not None and cht is not None and bht != cht:
        mismatch.append(f"holdout total {bht}!={cht}")
    valid = not mismatch

    if not valid:
        accept = False
        reason = "REJECT: split-size mismatch (" + "; ".join(mismatch) + ") — not comparing the same eval set"
    elif accept:
        reason = (f"ACCEPT: combined train+holdout passes {base_comb}→{cand_comb} (+{delta}); "
                  f"holdout {bh}→{ch} confirms the edit generalizes past the visible train failures")
    else:
        reason = (f"REJECT: combined train+holdout passes {base_comb}→{cand_comb} ({delta:+d}); "
                  f"did not strictly improve (train {bt}→{ct}, holdout {bh}→{ch})")

    return {"accept": accept, "valid": valid,
            "baseline_combined": base_comb, "candidate_combined": cand_comb, "delta_combined": delta,
            "train_delta": ct - bt, "holdout_delta": ch - bh,
            "baseline_train": bt, "candidate_train": ct, "baseline_holdout": bh, "candidate_holdout": ch,
            "reason": reason}


def run(q, worker):
    """Fleet handler. spec keys: baseline_train, baseline_holdout, cand_train, cand_holdout (any of
    int/list/dict). Empty spec (smoke) → a harmless self-check on tiny fixtures that still exercises the gate.
    """
    self = BaseAgent(); self.name = "harness-opt-gate"
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}

    if not any(k in spec for k in ("baseline_train", "baseline_holdout", "cand_train", "cand_holdout")):
        # smoke: an accepting example (train improves, holdout holds) — proves the gate wiring end to end.
        res = gate([1, 1, 0], [1, 0], [1, 1, 1], [1, 0])
        return self.done(res, f"[{worker}] harness-opt-gate smoke: accept={res['accept']} "
                              f"(combined {res['baseline_combined']}→{res['candidate_combined']})", to="all")

    res = gate(spec.get("baseline_train", 0), spec.get("baseline_holdout", 0),
               spec.get("cand_train", 0), spec.get("cand_holdout", 0))
    self.log(summary=f"harness-opt-gate: accept={res['accept']} "
                     f"combined {res['baseline_combined']}→{res['candidate_combined']} ({res['delta_combined']:+d})",
             detail=res["reason"], kind="verdict",
             recommendation="keep prompt/harness edit" if res["accept"] else "discard prompt/harness edit")
    msg = f"[{worker}] {res['reason']}"
    return self.done(res, msg, to="all")
