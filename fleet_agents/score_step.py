"""Score step — the competition's PREDICT+SCORE, run AFTER training to produce the golden/official CV.

Closes the loop: reuses the existing predict_and_score (predict from the trained weights →
src.metric.official_score + src.golden_cv.golden_cv), which logs golden_cv/official_score/component
metrics to MLflow experiment 'kaggle-biohub-cell-tracking' (what the scorer/metric agents read) and
writes a score.json the journal keys off. Submitted to :7799 so the GPU predict serializes ONE AT A
TIME (after the training finishes). Competition-specific — the generic framework just dispatches 'score'.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
TRAIN_SERVICE = "http://127.0.0.1:7799"
CALLBACK = "http://127.0.0.1:7788/api/runtime/training-callback"
# GENERIC loop-closer: predict+score ANY training config (reads train.method from the config itself).
SCRIPT = COMP / "predict_and_score.sh"


def _queue_busy() -> bool:
    try:
        with urllib.request.urlopen(f"{TRAIN_SERVICE}/api/board", timeout=4) as r:
            q = json.loads(r.read()).get("queue", {})
        return int(q.get("running_count", 0)) + int(q.get("queued_count", 0)) > 0
    except Exception:  # noqa: BLE001
        return True


def score_after_train(q, worker):
    """Predict+score a trained config → golden_cv/official_score (MLflow + journal). One at a time."""
    from researchpapers.fleet import post as _post
    spec = q["spec"]
    cfg = spec.get("config")
    method = spec.get("method") or (Path(cfg).stem if cfg else None)
    # OPTIONAL script override (backward-compatible): a spec may point at a DEDICATED predict+score
    # script (e.g. predict_and_score_pilk.sh for the support-pack re-detect) and pass extra CLI args
    # (e.g. a det-threshold). Absent these keys the behaviour is byte-identical to before.
    script = (COMP / spec["script"]) if spec.get("script") else SCRIPT
    extra_args = [str(a) for a in (spec.get("extra_args") or [])]
    if not cfg or not (COMP / cfg).exists():
        return ("escalated", {"config": cfg}, "researcher",
                f"[{worker}] SCORE: config '{cfg}' not found — cannot locate trained weights.")
    if not script.exists():
        return ("escalated", {"missing": str(script)}, "researcher",
                f"[{worker}] SCORE: predict+score script missing at {script}.")
    if getattr(_post, "DRY", False):
        return ("done", {"method": method, "config": cfg, "dry": True}, "all",
                f"[{worker}] SCORE dry-run: would predict+score {cfg} → golden_cv/official_score.")
    if _queue_busy():
        return ("holding", {"method": method, "config": cfg}, "all",
                f"[{worker}] SCORE holding: GPU busy — will predict+score {method} when it frees (one at a time).")
    payload = {
        "title": f"score {method}"[:70],
        "script_path": str(script),
        "script_args": [cfg, *extra_args],
        "workdir": str(COMP),
        "technical_focus": ["score", method],
        "notify_agent": "leader",
        "callback_url": CALLBACK,
        "notes": f"predict+score {cfg} → golden_cv/official_score (kaggle-biohub-cell-tracking) + journal",
    }
    req = urllib.request.Request(f"{TRAIN_SERVICE}/jobs", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        r = json.load(urllib.request.urlopen(req, timeout=8))
        jid = r.get("train_task_id") or r.get("id") or "queued"
        # after scoring lands, hand the LEADER a COMPLETE metrics table (holds until golden_cv appears)
        from researchpapers.fleet import board
        board.add("A", "metrics-report", f"Complete metrics table for '{method}' → leader (job {jid})",
                  {"method": method, "config": cfg})
        return ("done", {"method": method, "job": jid}, "all",
                f"[{worker}] SCORE QUEUED: predict+score {method} → golden_cv/official_score to MLflow "
                f"(kaggle-biohub-cell-tracking) + journal backfill. job {jid}. Leader gets the full metrics table after. (no Kaggle)")
    except Exception as exc:  # noqa: BLE001
        return ("failed", {"error": str(exc)}, "all", f"[{worker}] SCORE submit failed: {exc}")
