"""train-heartbeat — REUSABLE training liveness + metric-stream contract, distilled from
chrishayuk/chuk-mcp-training (worker→controlplane wire: heartbeat / metric / job_started events,
heartbeat-loss ⇒ unreachable, watchdog gates like isnan(last(loss)) ⇒ stop_run).

Any trainer calls `hb = Heartbeat(run_id)` then `hb.beat(step=..., loss=..., **metrics)` once per
epoch/fold/step-window. Each beat (1) prints one grep-friendly `HEARTBEAT <run_id> step=N k=v ...`
stdout line so log monitors see progress, and (2) touches a heartbeat JSON file so an out-of-process
watchdog can detect stalls even when stdout is captured/buffered by a parent process.

`check(run_id)` is the watchdog side: returns a verdict dict — `stale` if the file is older than
`max_silence_s`, `nan` if the last loss was non-finite — so a Monitor/cron can kill or alert.
Spec-driven, no deps, competition-agnostic.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

HB_DIR = Path(os.environ.get("TRAIN_HB_DIR", "/tmp/train_heartbeats"))


class Heartbeat:
    def __init__(self, run_id: str, hb_dir=None, log=print):
        self.run_id = run_id
        self.dir = Path(hb_dir) if hb_dir else HB_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.json"
        self.log = log
        self.t0 = time.time()
        self._write({"event": "job_started", "step": 0})

    def _write(self, payload: dict):
        payload.update({"run_id": self.run_id, "ts": time.time(), "elapsed_s": round(time.time() - self.t0, 1)})
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.path)

    def beat(self, step=None, **metrics):
        kv = " ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        self.log(f"HEARTBEAT {self.run_id} step={step} {kv}", flush=True) if self.log is print \
            else self.log(f"HEARTBEAT {self.run_id} step={step} {kv}")
        self._write({"event": "metric", "step": step, **metrics})

    def done(self, **metrics):
        self._write({"event": "done", **metrics})


def check(run_id: str, hb_dir=None, max_silence_s: float = 900.0) -> dict:
    """Watchdog verdict for one run: {status: ok|stale|nan|missing|done, age_s, last}."""
    p = (Path(hb_dir) if hb_dir else HB_DIR) / f"{run_id}.json"
    if not p.exists():
        return {"status": "missing", "run_id": run_id}
    last = json.loads(p.read_text())
    age = time.time() - last.get("ts", 0)
    if last.get("event") == "done":
        return {"status": "done", "age_s": age, "last": last}
    loss = last.get("loss")
    if isinstance(loss, (int, float)) and not math.isfinite(loss):
        return {"status": "nan", "age_s": age, "last": last}
    if age > max_silence_s:
        return {"status": "stale", "age_s": age, "last": last}
    return {"status": "ok", "age_s": age, "last": last}
