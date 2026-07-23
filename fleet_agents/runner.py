"""Runner agent — deterministically submit an existing config to the train-service (:7799).

Removes the Claude trainer from the loop for standard mini experiments. Safety:
- dry-run gate: skips submission entirely when the fleet runs --dry-run.
- config-exists check before submitting.
- ONE-AT-A-TIME: only submits when the :7799 queue is idle; otherwise returns 'holding' so the
  question is retried on the next daemon cycle (no GPU flooding).
The train-service serializes jobs and each config writes its own MLflow run/output (no clobber).
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
TRAIN_SERVICE = "http://127.0.0.1:7799"
CALLBACK = "http://127.0.0.1:7788/api/runtime/training-callback"


def _queue_busy() -> bool:
    try:
        with urllib.request.urlopen(f"{TRAIN_SERVICE}/api/board", timeout=4) as r:
            q = json.loads(r.read()).get("queue", {})
        return int(q.get("running_count", 0)) + int(q.get("queued_count", 0)) > 0
    except Exception:  # noqa: BLE001
        return True  # if we can't tell, don't pile on


def submit(config: str, title: str, focus: list[str], approved: bool = False, timeout: int = 8):
    """Return (job_id | None, note). Deterministic; honors dry-run and one-at-a-time.

    `approved` = this is a DELIBERATE experiment (leader/manual POST via the endpoint) → it TRAINS even
    when FLEET_AUTO_SUBMIT=0. The auto-submit gate only stops the reactive BLIND seed sweep; a chosen
    experiment must always run, otherwise the whole leader-driven pipeline sits idle.
    `timeout` (s) caps the train-service submission request."""
    from researchpapers.fleet import post as _post  # framework flag (this pkg always runs under it)
    if not (COMP / config).exists():
        return None, f"config {config} missing"
    if getattr(_post, "DRY", False):
        return "DRY", "dry-run: not submitted"
    if not approved and os.environ.get("FLEET_AUTO_SUBMIT", "0") not in ("1", "true", "yes"):
        return "READY", "auto-submit OFF for un-approved seed sweep (leader/manual POSTs run regardless)"
    if _queue_busy():
        return None, "queue busy — holding (one at a time)"
    payload = {
        "title": title[:70],
        "script_path": str(COMP / "start_train.sh"),
        "script_args": [config],
        "workdir": str(COMP),
        "technical_focus": focus,
        "notify_agent": "leader",
        "callback_url": CALLBACK,
        "notes": f"fleet runner (deterministic): config-driven {config}",
    }
    req = urllib.request.Request(f"{TRAIN_SERVICE}/jobs", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        r = json.load(urllib.request.urlopen(req, timeout=max(1, int(timeout))))
        return (r.get("train_task_id") or r.get("id") or "queued"), "queued"
    except Exception as exc:  # noqa: BLE001
        return None, f"submit failed: {exc}"
