"""heal — the SELF-HEALING bridge. The Python fleet can run/score autonomously, but it cannot fix its
own CODE. So when a training job FAILS, this agent captures the exact error + a diagnosis and ESCALATES
it to the Claude leader/researcher (directed inbox) asking them to patch the config/runner and re-queue.

This is the user's principle: "even automl — when train fails it needs Claude agents to fix it." The
fleet keeps doing routine work; the moment it hits a code failure it can't resolve, it asks Claude.
In hybrid mode (start_all) Claude acts on the escalation; in pure-automl mode it's queued for when a
human/Claude is available. Deduped so one failure escalates once.
"""
from __future__ import annotations
import json
import re
import sqlite3
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
TRAIN_LOG = COMP / "train_log.txt"
STATE = COMP / "config" / "_auto" / "heal_state.json"

# error signature → (which file likely needs the fix, a one-line diagnosis for Claude)
DIAGNOSES = [
    (r"KeyError:\s*'paths'",
     ("fleet_agents/runner.py + scripts/train_from_config.py",
      "model_scratch/config/exp_det_*.yml has no `paths:` key — the runner submits it to train_from_config.py "
      "(which needs `paths`), but model_scratch detectors must run via `model_scratch/train_v0.py --config`. "
      "Route model_scratch/config/* to train_v0.py (or add a `paths` block), and point at the real base checkpoint.")),
    (r"checkpoint not found.*edge_predictor_best\.pth",
     ("config paths / weights",
      "base checkpoint edge_predictor_best.pth is missing at research/official_repo/weights/... — point the config "
      "at the shipped pilkwang weights (research/pilkwang_support_pack/weights/unet_transformer/split_0/).")),
    (r"CUDA out of memory|OutOfMemoryError",
     ("config batch_size / patch_size", "GPU OOM — lower batch_size or patch_size in the config.")),
    (r"num_workers|DataLoader worker.*(killed|exited)|deadlock",
     ("config train.num_workers", "DataLoader worker hang — set train.num_workers=0 (known noise-aug deadlock).")),
]


def _load():
    try:
        return json.loads(STATE.read_text())
    except Exception:  # noqa: BLE001
        return {"escalated": []}


def _recent_failures():
    """Recently-failed training jobs from the thread (system posts 'Training job X finished ... failed')."""
    from researchpapers.fleet import post
    thread = post.runtime_dir() / "thread.jsonl"
    out = []
    if thread.exists():
        for ln in thread.read_text(errors="replace").splitlines()[-200:]:
            try:
                d = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            c = d.get("content", "")
            m = re.search(r"Training job `([^`]+)` finished with status `failed`", c)
            if m:
                out.append((m.group(1), c))
    return out


def heal(q, worker):
    """OPTIONAL spec: strict (bool) — re-escalate even already-escalated failures (force re-notify) instead of
    deduping; useful when a fix was attempted but the job still fails."""
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    strict = bool(spec.get("strict"))
    st = _load()
    done = set() if strict else set(st.get("escalated", []))
    fails = [(jid, c) for jid, c in _recent_failures() if jid not in done]
    if not fails:
        return ("done", {"new_failures": 0}, "all",
                f"[{worker}] heal: no new training failures to escalate (fleet healthy).")

    err = ""
    if TRAIN_LOG.exists():
        tail = TRAIN_LOG.read_text(errors="replace")[-4000:]
        err = "\n".join(l for l in tail.splitlines()
                        if re.search(r"error|traceback|keyerror|not found|exception|cuda", l, re.I))[-500:]

    fix_where, diag = "the training config/runner", "inspect train_log.txt for the traceback."
    for pat, (where, d) in DIAGNOSES:
        if re.search(pat, err, re.I):
            fix_where, diag = where, d
            break

    jid, _ = fails[0]
    msg = (f"[{worker}] 🩹 HEAL — training `{jid}` FAILED and the Python fleet can't patch code. "
           f"**Fix needed in `{fix_where}`.** Diagnosis: {diag} "
           f"Error tail: `{err[-200:] or 'see train_log.txt'}` — please patch and re-queue the detector run.")
    # DIRECTED to the Claude leader (visible, non-routine) + also drop a researcher note
    from researchpapers.fleet import post
    post.post_thread(worker, "leader", msg, routine=False, kind="heal")
    from . import ledger
    ledger.log("heal", summary=f"training {jid} failed → escalated to Claude ({fix_where})",
               detail=diag[:180], kind="verdict", recommendation="Claude: patch the runner/config, then re-queue")
    st.setdefault("escalated", []).append(jid)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st))
    return ("escalated", {"job": jid, "fix_where": fix_where, "diagnosis": diag}, "leader", msg)
