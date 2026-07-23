"""task_spec — a spec-driven, acceptance-gated task contract, the transferable orchestration concept from
GeminiLight/MindOS ("Think in one place, let agents work from shared context"). MindOS is a TS knowledge-base
app, but its load-bearing idea for MULTI-AGENT coordination is its spec template (wiki/specs/spec-*.md): every
task carries a structured contract with a goal, current-state analysis, an explicit DATA-FLOW (who reads / who
writes — MindOS credits this section for catching the "sidebar didn't update" class of bug), the plan, impact,
edge cases, and an objective ACCEPTANCE CHECKLIST. Agents then self-review against that checklist before a task
counts as done. This binds a fleet of agents to a shared goal and makes "done" objective instead of vibes.

We already have durable shared context (the :7777 knowledge hub, MEMORY.md, the board), so MindOS validates that
half — the NEW piece is this task contract: a lightweight, offline primitive that (a) defines the required spec
sections, (b) validates a spec is complete (with a data-flow readers/writers lint), and (c) gates task
completion on an acceptance checklist where every item is objectively pass/fail.

Primitives (stdlib, no deps):
  • REQUIRED_SECTIONS                    — the MindOS spec sections a task contract must fill.
  • validate_spec(spec)                  — (ok, missing, warnings) completeness + data-flow lint.
  • acceptance_status(criteria)          — {passed,total,done} for a list of {check, passed} checklist items.
  • gate(spec)                           — a task may proceed/complete only if spec complete AND acceptance done.
  • new_spec_template(title)             — an empty, correctly-sectioned spec dict to fill.
"""
from __future__ import annotations
from .base import BaseAgent

REQUIRED_SECTIONS = ("goal", "current_state", "data_flow", "plan", "impact", "edge_cases", "acceptance")


def new_spec_template(title="task"):
    """An empty spec dict with every required section, ready to fill. acceptance is a checklist of items."""
    return {"title": title, "goal": "", "current_state": "", "data_flow": {"readers": [], "writers": []},
            "plan": "", "impact": [], "edge_cases": [], "acceptance": []}


def validate_spec(spec):
    """Completeness check. Returns (ok, missing_sections, warnings). A section is 'filled' if truthy; data_flow
    additionally must name at least one reader AND one writer (the MindOS anti-stale-state lint); acceptance
    must have ≥1 checklist item; edge_cases should have ≥3 (MindOS rule) — a warning, not a hard failure."""
    missing = [s for s in REQUIRED_SECTIONS if not spec.get(s)]
    warnings = []
    df = spec.get("data_flow") or {}
    if "data_flow" not in missing:
        if not df.get("readers") or not df.get("writers"):
            missing.append("data_flow")  # incomplete data-flow counts as missing (its whole point)
            warnings.append("data_flow must name readers AND writers")
    acc = spec.get("acceptance") or []
    if "acceptance" not in missing and len(acc) < 1:
        missing.append("acceptance")
    if len(spec.get("edge_cases") or []) < 3:
        warnings.append("MindOS rule: list ≥3 edge cases")
    return (len(missing) == 0, missing, warnings)


def acceptance_status(criteria):
    """Summarize an acceptance checklist. criteria = [{"check": str, "passed": bool}, ...].
    Returns {passed, total, done, failing:[checks]}."""
    crit = list(criteria or [])
    passed = [c for c in crit if c.get("passed")]
    failing = [c.get("check", "?") for c in crit if not c.get("passed")]
    return {"passed": len(passed), "total": len(crit), "done": len(crit) > 0 and not failing, "failing": failing}


def gate(spec):
    """Whole-contract gate: a task may be marked DONE only if the spec is complete AND every acceptance item
    passes. Returns {allowed, reason, missing, acceptance}."""
    ok, missing, warns = validate_spec(spec)
    acc = acceptance_status(spec.get("acceptance"))
    allowed = ok and acc["done"]
    if allowed:
        reason = "spec complete + all acceptance criteria pass"
    elif not ok:
        reason = f"spec incomplete: missing {missing}"
    else:
        reason = f"acceptance not met: {acc['failing']}"
    return {"allowed": allowed, "reason": reason, "missing": missing, "warnings": warns, "acceptance": acc}


# ---------------------------------------------------------------- agent
class TaskSpec(BaseAgent):
    name = "task-spec"
    thread = "S"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        spec = s.get("task_spec")
        if not spec:                                            # demo: an incomplete then a complete contract
            incomplete = new_spec_template("demo")
            incomplete["goal"] = "raise node recall"
            g_bad = gate(incomplete)
            complete = new_spec_template("demo")
            complete.update(goal="raise node recall", current_state="recall 0.81",
                            data_flow={"readers": ["detector"], "writers": ["gap_fill"]},
                            plan="interpolate gap-recoverable misses", impact=["cv.py"],
                            edge_cases=["k>3", "empty frame", "boundary"],
                            acceptance=[{"check": "cv up on 44b6", "passed": True},
                                        {"check": "wilcoxon p<0.05", "passed": True}])
            g_good = gate(complete)
            msg = (f"task-spec: MindOS task contract — incomplete spec BLOCKED ({g_bad['reason']}); "
                   f"complete+accepted spec ALLOWED ({g_good['acceptance']['passed']}/"
                   f"{g_good['acceptance']['total']} criteria). Bind every fleet task to a spec "
                   f"(goal/data-flow/acceptance) so 'done' is objective and agents share the goal")
            data = {"blocked_reason": g_bad["reason"], "allowed": g_good["allowed"]}
        else:
            g = gate(spec)
            msg = f"task-spec: gate → {'ALLOWED' if g['allowed'] else 'BLOCKED'} ({g['reason']}); warnings={g['warnings']}"
            data = {"allowed": g["allowed"], "reason": g["reason"]}
        self.log(msg, kind="finding",
                 recommendation="wrap each task in new_spec_template, fill data_flow readers/writers, and gate() "
                                "completion on the acceptance checklist — no task is 'done' on vibes")
        return self.done(data, msg)


_AGENT = TaskSpec()


def run_taskspec(q, worker):
    return _AGENT.run(q, worker)
