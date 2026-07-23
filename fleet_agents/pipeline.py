"""pipeline — run a plan of fleet agents deterministically, NO Claude in the loop.

The (expensive) Claude leader/researcher think ONCE — name a template or emit a step plan — and the Python
fleet executes the whole thing itself: in order, fanning out where asked, gating on measured results,
healing or skipping failures, and looping until a CV target. Claude is not consulted between steps.

Spec (all fields optional except one of template/steps):
  {
    "template": "node_recall",             # expand a NAMED recipe (see _TEMPLATES) into steps
    "steps": [                              # explicit plan (overrides/extends a template's steps)
      {"kind": "data-audit"},
      {"parallel": [                        # FAN-OUT: run this group's steps, join all, don't halt on one
        {"kind": "fullconfig-search", "spec": {"focus": "node_recall"}},
        {"kind": "config-ablate"}
      ]},
      {"kind": "scorer"},
      {"kind": "submission-build", "when": "best_cv>=0.897"},   # GATED: only if the metric clears the bar
      {"kind": "combined-train", "on_fail": "heal"}             # heal|skip|halt (default halt)
    ],
    "loop_until": {"metric": "best_cv", "target": 0.897, "max_rounds": 3},  # repeat the chain to a target
    "stop_on_escalate": true,               # default true — halt the chain on escalate/fail (unless step on_fail)
    "carry": true                           # default true — merge prior scalar outputs into the next step's spec
  }

Backward-compatible: a plain {"steps":[{"kind":..},...]} still runs as a simple ordered chain.
A BaseAgent subclass with its own data-wise test.
"""
from __future__ import annotations
import re
from .base import BaseAgent

# ---- named recipes: "use all agents well" for the recurring goals (Claude names one, fleet runs it) ----
_TEMPLATES = {
    # raise node recall at fixed count on golden-12 (the real lever to 0.897)
    "node_recall": [
        {"kind": "data-audit"},
        {"parallel": [
            {"kind": "fullconfig-search", "spec": {"focus": "node_recall"}},
            {"kind": "config-ablate"},
        ]},
        {"kind": "scorer"},
    ],
    # EXTERNAL DATA + TRAINING done right: measure/correct data scale → build flow GT → box-sample the
    # dense external embryos to competition density → gate them to the author's scheme → train the
    # division/flow heads on box-sampled external + competition (FULL) → train the combined model → score.
    "external_train": [
        {"kind": "data-audit"},                       # measure + per-embryo scale-correct
        {"kind": "ext-label-stats"},                  # what the external labels actually contain
        {"kind": "flow-gt-build", "on_fail": "skip"},  # per-node flow+division GT
        {"kind": "box-sample"},                       # density-match external → competition crops
        {"kind": "sample-match"},                     # GATE: external must match author density/sister-ratio
        {"kind": "gnn-link-train", "on_fail": "skip"},  # train division/flow heads (LOEO)
        {"kind": "combined-train"},                   # train on box-sampled external + competition (FULL)
        {"kind": "scorer"},                           # official golden-12 score (needs JSON proof to record)
    ],
    # full end-to-end: external+training THEN config search + reproduce
    "full_e2e": [
        {"kind": "data-audit"},
        {"kind": "box-sample"},
        {"kind": "sample-match"},
        {"kind": "combined-train", "on_fail": "skip"},
        {"parallel": [
            {"kind": "fullconfig-search"},
            {"kind": "config-ablate"},
        ]},
        {"kind": "scorer"},
        {"kind": "reproduce-score"},
    ],
    # cheap sanity chain (diagnostics only)
    "diagnose": [
        {"kind": "perf-choice"},
        {"kind": "data-audit"},
        {"kind": "eda-stats"},
    ],
    # config-only search around the current best (no training)
    "config_search": [
        {"kind": "fullconfig-search"},
        {"kind": "config-ablate"},
        {"kind": "scorer"},
    ],
}

_METRIC_KEYS = ("cv", "combined_score", "score", "golden_cv", "official_score", "best_cv")


def _predicate(expr: str, ctx: dict) -> bool:
    """Safe 'when'/'until' evaluation: 'key' (truthy present) or 'key OP number' (OP in >=,<=,>,<,==,!=).
    Only reads numeric keys from the carried context — no eval, no arbitrary code."""
    expr = (expr or "").strip()
    if not expr:
        return True
    m = re.match(r"^([A-Za-z_][\w]*)\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)$", expr)
    if not m:
        v = ctx.get(expr)                                  # bare key → truthy presence
        return bool(v)
    key, op, num = m.group(1), m.group(2), float(m.group(3))
    v = ctx.get(key)
    if not isinstance(v, (int, float)):
        return False
    return {">=": v >= num, "<=": v <= num, ">": v > num, "<": v < num,
            "==": v == num, "!=": v != num}[op]


class Pipeline(BaseAgent):
    name = "pipeline"
    thread = "B"
    kind = "verdict"

    def _handlers(self):
        from . import _RAW_HANDLERS                        # unwrapped handlers — run directly, in order
        return _RAW_HANDLERS

    # ---- run ONE atomic step (kind + optional when/on_fail); returns a result dict ----
    def _run_step(self, step, raw, ctx, carry, worker, stop_on_escalate):
        kind = step.get("kind")
        when = step.get("when")
        if when is not None and not _predicate(when, ctx):
            return {"kind": kind, "status": "skipped_when", "data": {}}
        if kind not in raw:
            return {"kind": kind, "status": "unknown_kind", "data": {}}
        sspec = dict(step.get("spec") or {})
        if carry and ctx:
            # ctx holds prior outputs re-emitted under every INPUT alias (via flow) so this agent finds
            # the value under the key IT reads; the step's own spec always wins.
            sspec = {**ctx, **sspec}
        sq = {"question": step.get("question") or f"pipeline: {kind}", "spec": sspec}
        try:
            out = raw[kind](sq, worker)
            status = out[0] if isinstance(out, (list, tuple)) and out else "done"
            data = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 and isinstance(out[1], dict) else {}
        except Exception as e:  # noqa: BLE001
            status, data = "failed", {"error": str(e)[:200]}
        # per-step failure policy
        if status in ("escalated", "failed", "holding", "unknown_kind"):
            policy = step.get("on_fail", "halt" if stop_on_escalate else "skip")
            if policy == "heal" and "heal" in raw:
                try:
                    raw["heal"]({"question": f"heal {kind}", "spec": {"agent": kind, "error": data.get("error")}}, worker)
                except Exception:  # noqa: BLE001
                    pass
                status = status + "+healed"
        return {"kind": kind, "status": status, "data": data}

    def _expand(self, spec):
        steps = []
        tmpl = spec.get("template")
        if tmpl and tmpl in _TEMPLATES:
            steps.extend(_TEMPLATES[tmpl])
        steps.extend(spec.get("steps") or [])
        return steps

    def run(self, q, worker):
        spec = self.spec(q)
        steps = self._expand(spec)
        if not steps:
            return self.escalate(worker, "researcher",
                                 f"[{worker}] pipeline: no steps (give spec.steps or a known template: {list(_TEMPLATES)}).")
        # OPTIONAL dry_run: return the expanded plan WITHOUT executing any step (preview a template/chain).
        if spec.get("dry_run"):
            plan = [(s.get("kind") or ("parallel:" + ",".join(x.get("kind", "?") for x in s.get("parallel", []))))
                    for s in steps]
            return self.done({"dry_run": True, "n_steps": len(steps), "plan": plan,
                              "template": spec.get("template")},
                             f"[{worker}] pipeline [dry-run]: {len(steps)} steps → {' → '.join(plan)}")
        # 'continue_on_error' (default = stop_on_escalate's inverse) lets callers use the task's canonical name.
        if "continue_on_error" in spec:
            stop_on_escalate = not bool(spec["continue_on_error"])
        else:
            stop_on_escalate = bool(spec.get("stop_on_escalate", True))
        carry = bool(spec.get("carry", True))
        loop = spec.get("loop_until") or {}
        try:                                                  # cap rounds so a bad loop_until can't run unbounded
            max_rounds = min(50, max(1, int(loop.get("max_rounds", 1)))) if loop else 1
        except Exception:  # noqa: BLE001
            max_rounds = 1
        raw = self._handlers()

        all_results, ctx = [], {}
        rounds = 0
        halted = None
        for rnd in range(max(1, max_rounds)):
            rounds = rnd + 1
            round_halted = None
            for step in steps:
                if "parallel" in step:                     # FAN-OUT group: run all, join, never halt mid-group
                    group = []
                    for sub in step["parallel"]:
                        r = self._run_step(sub, raw, ctx, carry, worker, stop_on_escalate=False)
                        group.append(r)
                        self._absorb(ctx, r, carry)
                    all_results.append({"kind": "parallel", "status": "done",
                                        "data": {"group": [f"{g['kind']}:{g['status']}" for g in group]},
                                        "members": group})
                    continue
                r = self._run_step(step, raw, ctx, carry, worker, stop_on_escalate)
                all_results.append(r)
                self._absorb(ctx, r, carry)
                if r["status"] in ("escalated", "failed", "holding", "unknown_kind") and stop_on_escalate \
                        and step.get("on_fail", "halt") == "halt":
                    round_halted = r["status"]; break
            # loop-until check
            if loop:
                if _predicate(f"{loop.get('metric','best_cv')}>={loop.get('target',1.0)}", ctx):
                    break
            if round_halted:
                halted = round_halted; break

        best_cv = ctx.get("best_cv")
        ok = [r for r in all_results if r["status"] == "done"]
        self.save_state({"rounds": rounds, "ran": len(all_results), "ok": len(ok),
                         "halted_at": halted, "best_cv": best_cv, "template": spec.get("template"),
                         "chain": [f"{r['kind']}:{r['status']}" for r in all_results]})
        chain = " → ".join(f"`{r['kind']}`({r['status']})" for r in all_results)
        cvtxt = f" · best measured CV **{best_cv}**" if isinstance(best_cv, (int, float)) else ""
        rtxt = f" [{rounds} round{'s' if rounds > 1 else ''}]" if loop else ""
        msg = (f"[{worker}] **PIPELINE**{rtxt} — {len(ok)}/{len(all_results)} steps"
               + (f", HALTED at `{halted}`" if halted else " ok") + cvtxt
               + (f" (template `{spec.get('template')}`)" if spec.get("template") else "") + f"\n{chain}")
        self.log(summary=f"pipeline {len(ok)}/{len(all_results)} ok{cvtxt}{rtxt}", detail=chain, kind="verdict",
                 recommendation="inspect the halted step" if halted else "chain complete")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"rounds": rounds, "ran": len(all_results), "ok": len(ok), "halted_at": halted,
                          "best_cv": best_cv, "results": all_results}, msg, to="leader")

    @staticmethod
    def _absorb(ctx, result, carry):
        """Pull outputs into the carried context — via `flow` so canonical values (cv/config/nodes) are
        re-emitted under EVERY input alias the next agent might read (that's what lets ANY agent chain to
        ANY agent). Keeps raw scalars + running best_cv for `when`/`until` predicates."""
        data = result.get("data") or {}
        if not carry:
            return
        from . import flow
        for k, v in data.items():                            # raw scalars (for exact-key `when` predicates)
            if not isinstance(v, (list, dict)):
                ctx[k] = v
        for k, v in flow.carry_spec({}, data).items():       # canonical values under input aliases (composition glue)
            ctx[k] = v
        cv = flow.cv_of(data)
        if isinstance(cv, (int, float)):
            ctx["best_cv"] = cv if not isinstance(ctx.get("best_cv"), (int, float)) else max(ctx["best_cv"], cv)


_AGENT = Pipeline()


def run(q, worker):
    return _AGENT.run(q, worker)
