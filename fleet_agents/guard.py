"""Guard agent — the trainer's 'is this result reliable?' check, done deterministically.

Reads the latest :7799 job + validates liveness: succeeded, has a finish time, plausible wall-time.
This is the trainer role's post-run reliability judgement — now Python. (Component-metric plausibility
is covered by the metric-decomposition agent reading MLflow.)
"""
from __future__ import annotations

import json
import urllib.request

TRAIN_SERVICE = "http://127.0.0.1:7799"


def check(q, worker):
    """Fleet handler — validate the most recent training job's liveness.
    OPTIONAL spec: strict (bool) — also flag a missing start time + implausible wall-time as SUSPECT;
    timeout (s) — cap the train-service request (default 4)."""
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    strict = bool(spec.get("strict"))
    try:
        to = max(1, int(spec.get("timeout", 4)))
    except Exception:  # noqa: BLE001
        to = 4
    try:
        with urllib.request.urlopen(f"{TRAIN_SERVICE}/api/board", timeout=to) as r:
            b = json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        return ("failed", {"error": str(exc)}, "all", f"[{worker}] GUARD: train-service unreachable: {exc}")
    recent = (b.get("queue", {}).get("recent") or []) if isinstance(b, dict) else []
    if not recent:
        return ("escalated", {"reason": "no jobs"}, "researcher",
                f"[{worker}] GUARD: no training jobs yet to validate — nothing has run.")
    j = recent[0] if isinstance(recent[0], dict) else {}
    probs = []
    if j.get("status") != "succeeded":
        probs.append(f"status={j.get('status')}")
    if j.get("status") == "succeeded" and not j.get("finished_at"):
        probs.append("succeeded but no finish time (suspect silent exit)")
    if strict:                                              # tighter liveness checks (docstring's wall-time gate)
        if not j.get("started_at"):
            probs.append("no start time recorded")
        try:
            import datetime as _dt
            def _p(s):
                for f in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                    try:
                        return _dt.datetime.strptime(str(s)[:26], f)
                    except (ValueError, TypeError):
                        continue
                return None
            s0, s1 = _p(j.get("started_at")), _p(j.get("finished_at"))
            if s0 and s1:
                wall = (s1 - s0).total_seconds()
                if wall < 0:
                    probs.append("finish precedes start (clock/log corruption)")
                elif wall < 1:
                    probs.append(f"implausibly short wall-time ({wall:.0f}s)")
        except Exception:  # noqa: BLE001
            pass
    ok = not probs
    verdict = "RELIABLE" if ok else "SUSPECT — " + "; ".join(probs)
    return ("done", {"job": j.get("id"), "reliable": ok, "problems": probs}, "all",
            f"[{worker}] GUARD: latest job {j.get('id')} → {verdict} "
            f"(started {j.get('started_at')}, finished {j.get('finished_at')}).")
