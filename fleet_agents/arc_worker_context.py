"""arc-worker-context — DETERMINISTIC per-task context + prompt BUILDER for network-golf / grid-reasoning
(ARC-AGI-ONNX) competitions. PURE PYTHON — it contains NO LLM call and does NO ARC-solving. It ASSEMBLES the
rich per-task context block (best-known ONNX + builder + cost profile + target + promoted/rejected history +
similar solved tasks + relevant idioms from arc-idioms) and renders the rewrite-first / target-driven worker
PROMPT (adapted from the mined 9th-place prompt.txt). It also records per-task state (attempt_log/task<NNN>.md
and a shared MEMORY.md), MEMORY.md-style.

The OUTPUT (prompt string + context dict) is what the LIVE reasoning layer consumes: the researchpapers
`researcher` agent (the live-LLM brain) reads it and does the ARC-solving + architecture-rewrite reasoning,
calling the python tools (arc-onnx-golf to emit/verify/cost, arc-idioms to query constructions) via
fleet_dispatch; the `leader` orchestrates across the 400 tasks and the shared MEMORY.md. No LLM lives here.

Grounded in the mined winning workflow: architecture REWRITES average +0.5 per task vs +0.05 for pruning →
the prompt is REWRITE-FIRST with a concrete headstrong target ("beat baseline by ≥margin — it's possible");
the worker starts from the best-known state; idioms + similar-task transfer are supplied as memory.

Spec: {task_id, agi_id, state{...}, baseline_score, margin, patterns_path, out_dir, record, demo}.
Empty spec → a DEMO context+prompt from a stub task (no onnx/LLM). Data-wise test:
test_fleet_agents/arc_worker_context_test.py (stub state, asserts the assembled context + prompt sections).
"""
from __future__ import annotations
import math
import re
from pathlib import Path
from .base import BaseAgent
from . import arc_idioms as IDIOMS

COMP = Path(__file__).resolve().parent.parent

# the cost/score identity every prompt states (official scorer)
_COST_LINE = "cost = memory_bytes + params ;  score = max(1.0, 25.0 - ln(max(1.0, cost)))  ;  MACs are FREE"


def cost_for_score(score):
    """Inverse of the scorer: the cost budget you must hit for a target score. exp(25 - score)."""
    return math.exp(25.0 - score)


def target_for(baseline_score, margin=1.5):
    """Rewrite-first target: beat the baseline by `margin` (capped at 25), with the mined headroom note.
    Architecture rewrites averaged +0.5/task vs +0.05 for pruning → aim for a REWRITE, not a polish."""
    base = float(baseline_score or 0.0)
    tgt = min(25.0, round(base + float(margin), 3))
    return {"baseline": round(base, 3), "target": tgt, "cost_budget": round(cost_for_score(tgt), 1),
            "note": (f"Architecture REWRITES averaged +0.5/task vs +0.05 for pruning. Beat the baseline "
                     f"({base:.3f}) by ≥{margin} → target {tgt}. It is possible — explore a new "
                     f"representation, don't polish the current graph.")}


DEMO_STATE = {
    "task_id": "task179", "agi_id": "3c9b0459",
    "baseline_score": 21.0,
    "rule": "Reflect the square grid across the main diagonal (output[r,c] = input[c,r]).",
    "signature": {"tags": ["same shape", "square", "mirror reflection transpose"]},
    "best": {"source": "optimize_v1", "transform": "recolor_conv", "ops": ["Conv"],
             "memory": 0, "params": 100, "cost": 100, "score": 20.395,
             "builder": "Conv(input, W[10,10,1,1]) -> output  # 100-param full colour map"},
    "history": [
        {"id": "A01", "idea": "1x1 Conv colour map", "decision": "keep", "score": 20.395, "cost": 100},
        {"id": "A02", "idea": "Slice-based per-cell copy", "decision": "reject", "score": 14.1, "cost": 36000,
         "reason": "charged [1,10,30,30] intermediate blew memory"},
    ],
    "similar_tasks": [
        {"task": "task241", "transform": "transpose", "cost": 0, "score": 25.0},
        {"task": "task016", "transform": "recolor", "cost": 10, "score": 22.697},
    ],
}


def _fmt_best(best):
    if not best:
        return "  (none yet — cold start; propose a first formulation from the rule + idioms.)"
    return (f"  source={best.get('source')}  transform={best.get('transform')}  ops={best.get('ops')}\n"
            f"  cost={best.get('cost')} (memory={best.get('memory')} + params={best.get('params')})  "
            f"score={best.get('score')}\n  builder: {best.get('builder','')}")


def _fmt_history(history):
    if not history:
        return "  (no attempts logged yet)"
    lines = []
    for h in history[-8:]:
        r = f" — {h.get('reason')}" if h.get("reason") else ""
        lines.append(f"  {h.get('id','?')}: {h.get('idea','')} → {h.get('decision','?')} "
                     f"(score={h.get('score')} cost={h.get('cost')}){r}")
    return "\n".join(lines)


def _fmt_similar(similar):
    if not similar:
        return "  (no similar solved tasks supplied)"
    return "\n".join(f"  {s.get('task')}: {s.get('transform')} cost={s.get('cost')} score={s.get('score')}"
                     for s in similar[:8])


def _fmt_idioms(idioms):
    if not idioms:
        return "  (no idiom candidates — query arc-idioms with the task signature)"
    lines = []
    for r in idioms[:8]:
        band = f"band {r.get('band')}" if r.get("band") else "band ?"
        ex = f" [{r.get('task')} cost={r.get('cost')}]" if r.get("task") else ""
        lines.append(f"  [{band}] {r.get('title')}  ops={r.get('ops')}{ex}")
    return "\n".join(lines)


def build_context(state, patterns_path=None, margin=1.5, top_k=8):
    """Assemble the per-task context dict from `state` (stub or live). Pulls relevant idioms from arc-idioms
    using the task signature. Pure/deterministic."""
    state = dict(state or {})
    tgt = target_for(state.get("baseline_score"), margin=margin)
    sig = state.get("signature")
    idioms = []
    try:
        cat = IDIOMS.parse_patterns(patterns_path)
        if sig:
            idioms = IDIOMS.query(cat, sig, top_k=top_k)
    except Exception:  # noqa: BLE001
        idioms = []
    return {"task_id": state.get("task_id"), "agi_id": state.get("agi_id"), "rule": state.get("rule"),
            "target": tgt, "best": state.get("best"), "history": state.get("history") or [],
            "similar_tasks": state.get("similar_tasks") or [], "signature": sig,
            "idioms": [{"kind": r.get("kind"), "band": r.get("band"), "title": r.get("title"),
                        "ops": r.get("ops"), "task": r.get("task"), "cost": r.get("cost"),
                        "score": r.get("score")} for r in idioms]}


def build_prompt(context):
    """Render the rewrite-first / target-driven worker PROMPT (adapted from 9th-place prompt.txt) around the
    assembled context. Returns a string — the thing the live researcher agent consumes via fleet_dispatch."""
    c = context
    tid = c.get("task_id") or "<task>"
    tgt = c.get("target") or {}
    return f"""# NETWORK-GOLF WORKER TASK — {tid}
Rule (from the ARC-GEN generator, the oracle — not from a lookup):
  {c.get('rule') or '<derive from data/task JSON + generator; state it in 2-4 sentences>'}

OFFICIAL SCORER (ground truth — re-run it on every candidate; never trust a recorded number):
  {_COST_LINE}
  I/O: `input` and `output`, one-hot [1,10,30,30] float32, opset 10, ir_version 10.
  Correct = exact (output > 0.0) one-hot equality on EVERY train+test+arc-gen example ≤30×30.

TARGET (REWRITE-FIRST):
  baseline={tgt.get('baseline')}  ->  target={tgt.get('target')}  (cost budget ≈ {tgt.get('cost_budget')} bytes+params)
  {tgt.get('note')}

BEST-KNOWN STATE (start here, then try to BEAT it with a different representation):
{_fmt_best(c.get('best'))}

PROMOTED / REJECTED HISTORY (do not repeat rejected dead-ends):
{_fmt_history(c.get('history'))}

SIMILAR SOLVED TASKS (cross-task transfer — reuse their winning construction if the family matches):
{_fmt_similar(c.get('similar_tasks'))}

CANDIDATE IDIOMS (from arc-idioms / patterns.md — pick a target band + construction):
{_fmt_idioms(c.get('idioms'))}

TOOLS (call via fleet_dispatch — these are deterministic python, they do the emit/verify/cost, not the thinking):
  - arc-idioms       : query the construction catalogue for this task's rule family (returns band + ops).
  - arc-onnx-golf    : emit a candidate ONNX, VERIFY (output>0 equality on train+test+arc-gen), return
                       official cost (memory+params) + score. Trust ONLY its re-scored numbers.
  - arc-worker-context: (this tool) re-assemble state after each accepted attempt.
  If a python tool you need does not exist, call the `agent-author` agent (via fleet_dispatch) to DESIGN +
  register it, then use it. Do NOT do heavy deterministic compute inside your reasoning.

OPTIMIZATION LOOP (apply strongest levers FIRST):
  1. State the rule from the generator. Pick a target band + idiom via arc-idioms.
  2. Produce >=2 MATERIALLY DIFFERENT formulations; keep the best re-scored one.
  3. Synthesize the full [1,10,30,30] tensor ONLY at the final node named `output` (output memory is FREE).
  4. Reduce charged intermediates to scalar / short-vec / bool / uint8 / int8 / float16 EARLY.
  5. Put geometry in ATTRIBUTES (pads/strides/dilations/kernel_shape/axes/equation/perm), not initializers.
  6. Spend free MACs (Conv/Einsum/MatMul/Pool) to DELETE charged memory/params; end in Einsum/Gather/Conv/
     ConvTranspose/ScatterElements writing directly to `output` (recompute, don't store).
  7. If cost stays high, SWITCH REPRESENTATION rather than polishing the graph (rewrites >> pruning).

VALIDATION GATES (all required before promotion):
  - onnx.checker.check_model(full_check=True) + strict shape inference, concrete positive dims everywhere.
  - No banned ops (Loop/Scan/NonZero/Unique/Script/Function/Compress/*Sequence*), no subgraphs/functions,
    no custom domain, no initializer<->IO name collision, no `kernel_time` in any name.
  - Exact (output>0) one-hot equality on every train/test/arc-gen example ≤30×30.
  - 1000 fresh ARC-GEN samples (seed = task number): ZERO failures. No lookup tables / per-example dispatch.
  - Official score returns non-None, non-negative memory+params. File size <= 1.44 MiB.

RETURN: rule; re-scored baseline-to-beat; best formulation + why it lowers cost; the candidate builder;
public + fresh-arc-gen validation counts; memory/params/cost/score/delta; whether it was promoted.

ATTEMPT LOG (append to attempt_log/{tid}.md; update shared MEMORY.md with any new promoted idiom):
  ## A<NN> - <idea> - <ts>
  Change: <representation/graph change>  |  Result: valid=<y/n> public=<tr>/<te>/<gen> score=<s> cost=<c> delta=<d>
  Decision: <promote|keep|reject|revisit> because <reason>  |  Next: <one step>
"""


_ATTEMPT_TEMPLATE = """# {tid} Attempt Log

Best known: {source} score={score} cost={cost} memory={memory} params={params}
Rule: {rule}
"""


def record_attempt(out_dir, task_id, attempt, best=None, rule=""):
    """Append one attempt to attempt_log/<task_id>.md (create with template if missing). MEMORY.md-style."""
    d = Path(out_dir) / "attempt_log"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{task_id}.md"
    if not f.exists():
        b = best or {}
        f.write_text(_ATTEMPT_TEMPLATE.format(tid=task_id, source=b.get("source", "-"), score=b.get("score", "-"),
                                              cost=b.get("cost", "-"), memory=b.get("memory", "-"),
                                              params=b.get("params", "-"), rule=rule or "-"))
    a = attempt
    block = (f"\n## {a.get('id','A??')} - {a.get('idea','')} - {a.get('ts','')}\n"
             f"Change: {a.get('change','')}\n"
             f"Result: valid={a.get('valid')} public={a.get('public','-')} score={a.get('score')} "
             f"cost={a.get('cost')} delta={a.get('delta')}\n"
             f"Decision: {a.get('decision','')} because {a.get('reason','')}\n"
             f"Next: {a.get('next','')}\n")
    with f.open("a") as fh:
        fh.write(block)
    return str(f)


def update_memory(out_dir, note):
    """Append a cross-task lesson / promoted idiom to the shared MEMORY.md (the team's growing catalogue)."""
    f = Path(out_dir) / "MEMORY.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        f.write_text("# NETWORK-GOLF SHARED MEMORY\n\nPromoted idioms + cross-task lessons (append-only).\n")
    with f.open("a") as fh:
        fh.write(f"\n- {note}\n")
    return str(f)


class ArcWorkerContext(BaseAgent):
    name = "arc-worker-context"
    thread = "S"
    kind = "reason"

    def run(self, q, worker):
        spec = self.spec(q)
        state = spec.get("state")
        if not state and (spec.get("demo") or not spec.get("task_id")):
            state = DEMO_STATE                                   # empty-spec/demo path — no onnx/LLM
        else:
            state = dict(state or {})
            state.setdefault("task_id", spec.get("task_id"))
            state.setdefault("agi_id", spec.get("agi_id"))
            if spec.get("baseline_score") is not None:
                state["baseline_score"] = spec["baseline_score"]
        ctx = build_context(state, patterns_path=spec.get("patterns_path"),
                            margin=float(spec.get("margin", 1.5)), top_k=int(spec.get("top_k", 8)))
        prompt = build_prompt(ctx)
        recorded = None
        if spec.get("record") and spec.get("out_dir") and spec.get("attempt"):
            recorded = record_attempt(spec["out_dir"], ctx.get("task_id") or "task", spec["attempt"],
                                      best=ctx.get("best"), rule=ctx.get("rule"))
        msg = (f"[{worker}] arc-worker-context: assembled context for {ctx.get('task_id')} "
               f"(target {ctx['target']['target']} vs baseline {ctx['target']['baseline']}, "
               f"{len(ctx.get('idioms') or [])} idioms, {len(ctx.get('history') or [])} prior attempts). "
               f"Prompt is for the live researcher agent (via fleet_dispatch).")
        self.log(summary=msg, kind="reason")
        return self.done({"context": ctx, "prompt": prompt, "target": ctx["target"],
                          "recorded": recorded, "n_idioms": len(ctx.get("idioms") or [])}, msg, to="researcher")


_AGENT = ArcWorkerContext()


def run(q, worker):
    return _AGENT.run(q, worker)
