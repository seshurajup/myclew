"""metrics-report — the LEADER-facing COMPLETE metrics table for a finished+scored run.

Split of concerns (what the user asked):
  * train-monitor → LIVE progress card for the USER (epoch/iter/ETA/GPU + live MLflow link), each cycle.
  * metrics-report → the FULL metrics TABLE for the LEADER, once, after the run is trained AND scored,
    so the leader decides kept/rejected on real numbers — training metrics (loss/recall from
    kaggle-biohub-loeo) + official golden-CV decomposition (golden_cv/adjJ/node_recall/div_J/count from
    kaggle-biohub-cell-tracking). Also backfills the journal CV.

Generic: pulls both experiments over the MLflow REST API by run_name; any competition that logs a train
run + a score run under the same run_name gets a complete table for free.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
MLFLOW_API = "http://127.0.0.1:5000/api/2.0/mlflow"
MLFLOW_UI = "http://gpu:5000"
TRAIN_EXP = "kaggle-biohub-loeo"
SCORE_EXP = "kaggle-biohub-cell-tracking"

# the metrics worth showing the leader, per experiment (key → display label)
TRAIN_KEYS = {"recall": "det recall", "acc": "det acc", "det_loss": "det loss",
              "edge_loss": "edge loss", "test_loss": "val loss", "best_score": "best val"}
SCORE_KEYS = {"golden_cv": "golden_cv", "official_score": "official (adjE+0.1·divJ)",
              "adj_edge_jaccard": "adj edge-J", "division_jaccard": "division-J",
              "mean_node_recall": "node recall", "mean_count_ratio": "count ratio",
              "adjJ_44b6": "adjJ 44b6", "adjJ_6bba": "adjJ 6bba"}


def _run(exp_name: str, method: str):
    try:
        exp = json.loads(urllib.request.urlopen(
            f"{MLFLOW_API}/experiments/get-by-name?experiment_name={exp_name}", timeout=5).read())
        eid = exp["experiment"]["experiment_id"]
        body = json.dumps({"experiment_ids": [eid], "filter": f"tags.`mlflow.runName` = '{method}'",
                           "max_results": 1, "order_by": ["attributes.start_time DESC"]}).encode()
        req = urllib.request.Request(f"{MLFLOW_API}/runs/search", data=body,
                                     headers={"Content-Type": "application/json"})
        runs = json.loads(urllib.request.urlopen(req, timeout=5).read()).get("runs", [])
        if runs:
            r = runs[0]
            mets = {m["key"]: m["value"] for m in r.get("data", {}).get("metrics", [])}
            return eid, r["info"]["run_id"], mets
    except Exception:  # noqa: BLE001
        pass
    return None, None, {}


REPORTED = COMP / "tools" / "researchpapers" / ".research-mvp-data" / "runtime" / ".metrics_reported.json"


def _already_reported(method):
    try:
        return method in set(json.loads(REPORTED.read_text()))
    except Exception:  # noqa: BLE001
        return False


def _mark_reported(method):
    try:
        cur = set(json.loads(REPORTED.read_text())) if REPORTED.exists() else set()
    except Exception:  # noqa: BLE001
        cur = set()
    cur.add(method)
    REPORTED.parent.mkdir(parents=True, exist_ok=True)
    REPORTED.write_text(json.dumps(sorted(cur)))


def report(q, worker):
    """Compose the COMPLETE metrics table for `method` and send it to the LEADER — ONCE per method."""
    method = q["spec"].get("method")
    if not method:
        return ("escalated", {}, "researcher", f"[{worker}] METRICS-REPORT: no method given.")
    if _already_reported(method):
        # already delivered this method's table → never re-post (kills the duplicate spam)
        return ("done", {"method": method, "duplicate": True}, "all",
                f"[{worker}] metrics-report {method}: already delivered (no repost).")
    teid, trid, tmet = _run(TRAIN_EXP, method)
    seid, srid, smet = _run(SCORE_EXP, method)
    if not smet:
        # score hasn't landed yet — wait (the score job is still predicting/scoring)
        return ("holding", {"method": method}, "all",
                f"[{worker}] METRICS-REPORT holding: {method} not yet scored (golden_cv pending).")

    def _num(v):  # a real, finite number (nan/inf/bool are NOT display-formattable as a score)
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")

    def row(label, val):
        return f"| {label} | {val:.4f} |" if _num(val) else f"| {label} | {val} |"

    def _fmtv(v):
        return f"{v:.4f}" if _num(v) else (str(v) if v not in (None, "") else "—")

    lines = [f"**COMPLETE METRICS — {method}**  (leader: kept or rejected?)", "", "| metric | value |", "| :-- | --: |"]
    for k, lbl in SCORE_KEYS.items():
        if k in smet:
            lines.append(row(lbl, smet[k]))
    for k, lbl in TRAIN_KEYS.items():
        if k in tmet:
            lines.append(row(lbl, tmet[k]))
    links = []
    if srid:
        links.append(f"score run: {MLFLOW_UI}/#/experiments/{seid}/runs/{srid}")
    if trid:
        links.append(f"train run: {MLFLOW_UI}/#/experiments/{teid}/runs/{trid}")
    table = "\n".join(lines) + ("\n\n" + " · ".join(links) if links else "")

    # backfill the journal CV so the ledger shows the real number
    cv = smet.get("golden_cv") or smet.get("official_score")
    try:
        from . import ledger
        rowe = next((e for e in ledger.entries() if e.get("change") == method), None)
        if rowe and _num(cv):
            ledger.set_scores(rowe["exp"], cv=round(float(cv), 4))
    except Exception:  # noqa: BLE001
        pass

    # FULL MARKDOWN TRAINING REPORT (Python writes it — no leader needed): docs/results/<method>_result.md
    import datetime
    rowe = next((e for e in ledger.entries() if e.get("change") == method), None)
    md = [f"# Result — {method}", "",
          f"*Auto-written by the fleet {datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]}Z. "
          f"No hand-writing needed.*", "",
          f"- **config / set:** {rowe.get('script') if rowe else '?'} · {rowe.get('trn_set') if rowe else '?'}",
          f"- **golden CV:** {_fmtv(cv)}  ·  **official (adjE+0.1·divJ):** {_fmtv(smet.get('official_score'))}", "",
          "## Score (golden CV, by embryo)", "", "| metric | value |", "| :-- | --: |"]
    for k, lbl in SCORE_KEYS.items():
        if k in smet:
            md.append(f"| {lbl} | {_fmtv(smet[k])} |")
    md += ["", "## Training", "", "| metric | value |", "| :-- | --: |"]
    for k, lbl in TRAIN_KEYS.items():
        if k in tmet:
            md.append(f"| {lbl} | {_fmtv(tmet[k])} |")
    md += ["", "## MLflow", ""] + [f"- {ln}" for ln in links]
    resdir = COMP / "docs" / "results"
    resdir.mkdir(parents=True, exist_ok=True)
    (resdir / f"{method}_result.md").write_text("\n".join(md))

    data = {"method": method, "score": smet, "train": tmet, "golden_cv": cv,
            "mlflow_score": srid, "mlflow_train": trid, "report_md": f"docs/results/{method}_result.md"}
    _mark_reported(method)   # deliver ONCE — never re-post this method's table
    return ("done", data, "leader",   # DIRECTED to the leader → shown (not routine)
            f"[{worker}] 📋 {method} — full report written → docs/results/{method}_result.md. Metrics:\n\n{table}")
