"""Scorer agent — report the CV trajectory (official_score / golden_cv) across runs (deterministic).

Reuses the component metrics the training pipeline (src.metric.official_score / src.golden_cv) logs to
MLflow; surfaces the best + the trajectory for the journal. Higher is better. urllib-only (venv-agnostic).
"""
from __future__ import annotations

import json
import urllib.request

MLFLOW = "http://127.0.0.1:5000"
EXPERIMENT = "kaggle-biohub-cell-tracking"


def _finite(v):
    """True only for a real finite number (rejects None / nan / inf / bool)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _runs(mlflow=MLFLOW, experiment=EXPERIMENT):
    """mlflow/experiment: optional endpoint + experiment-name override (defaults to this comp's)."""
    try:
        exp = json.load(urllib.request.urlopen(
            f"{mlflow}/api/2.0/mlflow/experiments/get-by-name?experiment_name={experiment}", timeout=6
        ))["experiment"]["experiment_id"]
        req = urllib.request.Request(
            f"{mlflow}/api/2.0/mlflow/runs/search",
            data=json.dumps({"experiment_ids": [exp], "max_results": 100,
                             "order_by": ["attributes.start_time ASC"]}).encode(),
            headers={"Content-Type": "application/json"})
        runs = json.load(urllib.request.urlopen(req, timeout=6)).get("runs", [])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for r in runs:
        m = {}
        for x in r.get("data", {}).get("metrics", []):
            try:
                v = float(x["value"])
                if v == v and abs(v) != float("inf"):     # drop nan/inf metrics
                    m[x["key"]] = v
            except (TypeError, ValueError):
                pass
        s = m.get("official_score", m.get("golden_cv", m.get("adj_edge_jaccard")))
        if _finite(s):
            out.append((r["info"].get("run_name", r["info"]["run_id"][:8]), m.get("golden_cv", s), m))
    return out


def _backfill(scored):
    """Idempotent: ensure EVERY scored MLflow run has its golden_cv in the ledger (create the row if the
    experiment was submitted outside the fleet). This is the safety net so a run's CV can never silently
    stay None just because a one-shot metrics-report didn't fire. Returns count updated."""
    from . import ledger
    n = 0
    for name, cv, _m in scored:
        row = next((e for e in ledger.entries() if e.get("change") == name), None)
        if row is None:
            row = ledger.record(change=name, description=f"scored run {name}",
                                 script=f"bash start_train.sh config/aug_ablation/{name}.yml",
                                 train_set="screen_matched" if name[:2].isdigit() else "?", stage=3)
        if not isinstance(row.get("cv"), (int, float)) and isinstance(cv, (int, float)):
            ledger.set_scores(row["exp"], cv=round(float(cv), 4))
            n += 1
    return n


def report(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    scored = _runs(mlflow=spec.get("mlflow", MLFLOW), experiment=spec.get("experiment", EXPERIMENT))
    if not scored:
        return ("escalated", {"reason": "no scored runs"}, "researcher",
                f"[{worker}] SCORER: no scored runs in MLflow yet — need a baseline scored to start the CV trajectory.")
    updated = _backfill(scored)   # SAFETY NET: copy every golden_cv from MLflow → the journal (idempotent)
    traj = [(n, cv) for n, cv, _ in scored]
    best_name, best = max(traj, key=lambda t: t[1])
    tail = " → ".join(f"{n[:16]}:{s:.4f}" for n, s in traj[-5:])
    return ("done", {"n": len(traj), "best": best, "best_run": best_name, "backfilled": updated,
                     "trajectory": traj[-8:]}, "all",
            f"[{worker}] SCORER: {len(traj)} scored runs, best CV={best:.4f} ({best_name[:22]}); "
            f"backfilled {updated} CV(s) into the journal. Recent: {tail}.")
