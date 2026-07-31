"""local-pilot — drive the fleet from a LOCAL model, with no Claude in the loop.

This is the payoff for the two-tool capability layer and the tool-call fine-tune: given a competition's
CURRENT state, a local Gemma-class model picks the next agent to run and the pilot dispatches it onto the
real fleet board, where the normal workers execute it and journal the result. The next tick reads that
result and decides again — the same loop the `next_agent` training examples were mined from.

Why the request is shaped the way it is: it is byte-for-byte the shape the model was TRAINED on
(`prompt_dataset.build_history_tool_dataset`, kind `next_agent`) — recent agent outputs as plain result
strings, then "Choose the next agent to run." Training and serving must not drift apart, so both are built
from the same decision log.

Safety, because this runs unattended:
  • `execute=False` is the DEFAULT — the pilot returns a PLAN and dispatches nothing unless asked. An LLM
    choosing to run a training job on a shared GPU is not something to enable silently;
  • all kinds are dispatchable by default (standing instruction) EXCEPT Kaggle submission, which posts
    outward and spends a daily quota — it needs `user_requested_submit=true` on that specific run;
    `allow`/`deny` remain available per dispatch when a run needs further narrowing;
  • dispatch goes to the BOARD, not to the handler — the real worker runs it under the fleet's own
    quarantine, timeout and journal, rather than inside this process;
  • every decision is written to `docs/pilot_decisions.jsonl` so the pilot's own choices become future
    training data (the standing rule: every run improves the dataset).

Spec:
    {"kind": "local-pilot", "spec": {"comp": "birdclef-2026", "model": "ollama/gemma3n:e4b",
                                     "history": 4, "execute": false, "adapter": null,
                                     "allow": [...], "deny": [...], "thread": "B"}}
"""
from __future__ import annotations

import json
import os
import time

from .base import BaseAgent, COMP

ROOT = "/home/seshu/kaggle/2026"
# ALL GATES OPEN by standing instruction: the pilot may dispatch any registered kind, training and
# submission included. The blanket family denylist that used to live here is gone — it was blocking the
# very agents that move a competition forward.
#
# What still applies, because these are correctness mechanisms rather than policy gates:
#   * `gpu_train_hold.flag` — the fleet's own single-GPU coordination, honoured by the trainers themselves;
#   * the board — dispatch goes to a queue that real workers drain under the fleet's quarantine/timeout,
#     so a bad pick fails like any other agent instead of running unsupervised in this process;
#   * `execute` still defaults to False, so a run only dispatches when explicitly asked.
# A caller can still pass `deny` (or `allow`) per-dispatch when a specific run needs narrowing.
DENY_DEFAULT = ()

# THE ONE CARVE-OUT, set explicitly by the user: a Kaggle submission never happens autonomously. It is
# outward-facing (it posts to the competition) and it spends a finite daily quota, so it fires only when
# the user asks for it — pass `user_requested_submit=True` in the spec for that single run.
SUBMIT_KINDS = ("kaggle-submit", "submit-verify", "submission-build", "submit-guard")


THREAD_LOG = os.path.join(COMP if isinstance(COMP, str) else str(COMP),
                          "tools", "researchpapers", ".research-mvp-data", "runtime", "thread.jsonl")


def pending_user_messages(limit=5, log=None):
    """Human messages from the :7788 runboard channel (`POST /api/runtime/messages`, sender='human').

    The runboard lets a person address the fleet directly, and a human message outranks anything the
    agents are doing — so the pilot reads this FIRST and treats it as the instruction, using agent history
    only as background. The log stores the sender under `from` (not `sender`), which is why this reads both.
    """
    path = log or THREAD_LOG
    if not os.path.exists(path):
        return []
    out = []
    for ln in open(path, errors="replace"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        who = str(r.get("from") or r.get("sender") or "").lower()
        if who in ("human", "user"):
            out.append({"ts": r.get("timestamp") or r.get("ts"),
                        "to": r.get("to") or r.get("recipient") or "all",
                        "content": str(r.get("content") or "")[:400]})
    return out[-limit:]


SELF_KINDS = ("local-pilot",)


def _decisions(comp, limit=12, drop_self=True):
    """Recent agent outputs, EXCLUDING the pilot's own.

    Measured defect: the pilot's decisions land in the same `experiment_decisions.jsonl` it reads, so after
    a few runs its history was mostly its own prior picks. Every run then changed the next run's prompt --
    at temperature 0, identical state gave three different answers and TWO runs produced no capability at
    all (3/5 usable), while the benchmark read 99.5% dispatchable. The model is trained on what REAL agents
    returned, so self-entries are out-of-distribution input as well as a feedback loop.
    """
    path = os.path.join(ROOT, comp, "docs", "experiment_decisions.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    for ln in open(path, errors="replace"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except ValueError:
            continue
    if drop_self:
        rows = [r for r in rows if str(r.get("agent") or "") not in SELF_KINDS]
    rows.sort(key=lambda r: str(r.get("ts", "")))
    return rows[-limit:]


def _ledger_state(comp):
    path = os.path.join(ROOT, comp, "docs", "experiment_ledger.jsonl")
    best, n = None, 0
    if not os.path.exists(path):
        return {"best_private": None, "experiments": 0}
    for ln in open(path, errors="replace"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        n += 1
        try:
            p = float(r.get("private"))
            best = p if best is None else max(best, p)
        except (TypeError, ValueError):
            pass
    return {"best_private": best, "experiments": n}


def build_state_request(comp, history=4, msg_limit=5):
    """The exact prompt shape the model was fine-tuned on (kind `next_agent`)."""
    rows = _decisions(comp, limit=history)
    st = _ledger_state(comp)
    lines = []
    for w in rows:
        if not w.get("agent"):
            continue
        out = str(w.get("finding") or w.get("summary") or "")[:340]
        lines.append(f"agent `{w['agent']}` returned: {out}")
    from . import prompt_dataset as _PD
    # TRAINING SHAPE, EXACTLY. Measured defect: this used to insert a "Competition `x`: N experiments
    # logged, best private score ..." line between the MODE marker and the body. Training contains no such
    # line, and that ONE unseen line pushed the model out of the protocol -- it answered FINAL instead of
    # SEARCH and the pilot got no capability at all, while the benchmark (which feeds training-shaped
    # prompts) read 85%. The competition state is still recorded in the decision log; it belongs in the
    # prompt only once `prompt_dataset` emits it too (see `state_line` there).
    head = _PD.MODE_DISCOVER
    body = ("Fleet run in progress. Recent agent output:\n" + "\n".join(lines)
            if lines else "Fleet run starting; no agent output yet.")

    # A HUMAN MESSAGE OUTRANKS EVERYTHING. The runboard's own channel lets a person address the fleet
    # mid-run; when one is pending it becomes the instruction and the agent history is only background.
    msgs = pending_user_messages(limit=int(msg_limit))
    if msgs:
        um = "\n".join(f"USER -> {m['to']}: {m['content']}" for m in msgs)
        ask = ("The USER has sent the instruction(s) above. Choose the next agent that carries it out; "
               "the agent output is background only.")
        return head + "\n" + um + "\n\n" + body + "\n\n" + ask, st, len(lines), len(msgs)
    return head + "\n" + body + "\n\nChoose the next agent to run.", st, len(lines), 0


def allowed(kind, allow=None, deny=None, user_requested_submit=False):
    """(ok, why) — the dispatch gate. Open by default; two things can still close it:
    a caller-supplied `deny`/`allow`, and the standing rule that a Kaggle SUBMISSION requires the user."""
    if kind in SUBMIT_KINDS and not user_requested_submit:
        return False, (f"`{kind}` posts to Kaggle and spends daily quota — it runs only when the user "
                       f"asks (set user_requested_submit=true for that run)")
    if allow and kind not in allow:
        return False, f"`{kind}` is not in the allow list"
    for pat in list(deny or ()) + list(DENY_DEFAULT):
        if pat == kind or pat in kind:
            return False, f"`{kind}` matches denied family `{pat}`"
    return True, ""


def record(comp, rec):
    """Append the pilot's own decision — it becomes training data for the next round."""
    d = os.path.join(ROOT, comp, "docs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "pilot_decisions.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


class LocalPilot(BaseAgent):
    name = "local-pilot"
    thread = "B"
    kind = "finding"

    def run(self, q, worker):
        from . import agent_routing as AR
        spec = self.spec(q)
        comp = spec.get("comp") or "biohub-cell-tracking-during-development"
        model = spec.get("model") or "ollama/gemma3n:e4b"
        do_exec = bool(spec.get("execute", False))

        request, state, n_hist, n_user = build_state_request(
            comp, int(spec.get("history", 4)), int(spec.get("user_msgs", 5)))
        t0 = time.time()
        try:
            out = AR.capability_loop(request, model=model, max_steps=int(spec.get("max_steps", 4)),
                                     execute=False,          # never run inside this process
                                     adapter=spec.get("adapter"), bits=int(spec.get("bits", 4)))
        except Exception as e:  # noqa: BLE001 — a local model being down is a RESULT, not a crash
            return self.escalate(worker, "leader",
                                 f"[{worker}] local-pilot: model `{model}` unreachable "
                                 f"({type(e).__name__}: {str(e)[:140]}). Is Ollama up?")

        searched = [s for s in out["steps"] if s.get("tool") == "search_capabilities"]
        planned = next((s for s in out["steps"] if s.get("tool") == "execute_capability"), None)
        cands = searched[0]["matches"][:6] if searched else []
        choice = (planned or {}).get("name") or (cands[0] if cands else None)

        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "comp": comp, "model": model,
               "history_shown": n_hist, "user_messages": n_user, "state": state,
               "adapter": spec.get("adapter"), "candidates": cands, "chose": choice,
               "seconds": round(time.time() - t0, 1), "dispatched": False, "gate": ""}

        if not choice:
            rec["gate"] = "the model named no capability"
            record(comp, rec)
            return self.escalate(worker, "leader",
                                 f"[{worker}] local-pilot ({model}) produced no capability from the state "
                                 f"({n_hist} prior outputs shown). Steps: "
                                 f"{[s.get('tool') for s in out['steps']]}")

        ok_gate, why = allowed(choice, spec.get("allow"), spec.get("deny"),
                               bool(spec.get("user_requested_submit", False)))
        rec["gate"] = why or "allowed"

        if do_exec and ok_gate:
            try:
                from researchpapers.fleet import board
                board.add(spec.get("thread", "B"), choice,
                          f"[local-pilot/{model}] next action from fleet state ({n_hist} prior outputs)",
                          (planned or {}).get("spec") or {})
                rec["dispatched"] = True
            except Exception as e:  # noqa: BLE001
                rec["gate"] = f"board unavailable: {type(e).__name__}"
        record(comp, rec)

        verb = ("DISPATCHED to the board" if rec["dispatched"]
                else ("PLAN only (execute=false)" if ok_gate else f"BLOCKED — {why}"))
        msg = (f"[{worker}] local-pilot [{model}] on `{comp}` ({n_user} user msg, "
               f"{n_hist} prior agent outputs, "
               f"best private {state['best_private']}): chose `{choice}` — {verb}. "
               f"Shortlist {cands[:4]}. {out['steps'][0].get('tool', '?')} in "
               f"{rec['seconds']}s, native_tools={out['used_native_tools']}.")
        self.log(msg, kind="finding",
                 recommendation=("review docs/pilot_decisions.jsonl, then re-run with execute=true once "
                                 "the choices look right; the log is also next round's training data"))
        return self.done({**rec, "steps": [s.get("tool") for s in out["steps"]]}, msg)


_AGENT = LocalPilot()


def run(q, worker):
    return _AGENT.run(q, worker)
