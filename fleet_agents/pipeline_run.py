"""pipeline-run — config-driven end-to-end: inference base + secondary support models → COMBINED golden CV.
Runs scripts/pipeline_combined_cv.py under cellmot_venv (base pilkwang + div-model divisions) and reports
base vs combined score. FAST-ish: base inference is fixed; this just re-applies post-proc + re-scores."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
COMP = Path(__file__).resolve().parent.parent
PY_ENV = COMP / "research" / "cellmot_venv" / "bin" / "python"
SCRIPT = COMP / "scripts" / "pipeline_combined_cv.py"


def run(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}   # tolerate a missing/None spec
    thr = str(spec.get("precision_threshold", 0.9))
    try:                                                          # OPTIONAL timeout (s) for the scorer subprocess
        timeout = max(1, int(spec.get("timeout", 1200)))
    except Exception:  # noqa: BLE001
        timeout = 1200
    if not PY_ENV.exists() or not SCRIPT.exists():
        return ("escalated", {}, "researcher", f"[{worker}] pipeline-run: cellmot_venv or combined-CV script missing.")
    if not (COMP / "models" / "div_clf.pkl").exists():
        return ("holding", {}, "all", f"[{worker}] pipeline-run holding: div-model not trained yet (need models/div_clf.pkl).")
    try:
        r = subprocess.run([str(PY_ENV), str(SCRIPT), thr], capture_output=True, text=True,
                           timeout=timeout, cwd=str(COMP))
    except Exception as e:  # noqa: BLE001 — timeout / launch failure → clean escalate, never crash
        return ("escalated", {"error": str(e)[:200]}, "researcher",
                f"[{worker}] pipeline-run: scorer subprocess failed ({type(e).__name__}: {str(e)[:120]}).")
    line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    if not line:
        return ("escalated", {"stderr": r.stderr[-200:]}, "researcher",
                f"[{worker}] pipeline-run: combined-CV failed: {r.stderr.strip()[-150:]}")
    try:
        d = json.loads(line[-1])
    except Exception as e:  # noqa: BLE001
        return ("escalated", {"error": str(e)[:150]}, "researcher",
                f"[{worker}] pipeline-run: scorer emitted invalid JSON ({str(e)[:100]}).")
    if "combined_score" not in d:
        return ("escalated", {"stderr": str(d)[:150]}, "researcher",
                f"[{worker}] pipeline-run: output missing combined_score ({str(d)[:120]}).")
    from . import ledger
    beat = d["combined_score"] > 0.891
    # PROVENANCE: write the scorer's own JSON so record() can verify the score is real (not a placeholder).
    proof = COMP / "results" / "pipeline_run_score.json"
    proof.parent.mkdir(parents=True, exist_ok=True)
    proof.write_text(json.dumps(d))
    ledger.record(change="winning_inference_div", description=f"inference base + div-model (combined golden CV)",
                  script="scripts/pipeline_combined_cv.py", cv=d["combined_score"], train_set="golden12", stage=8,
                  verify_json=str(proof))
    ledger.log("pipeline-run", kind="finding",
               summary=f"COMBINED golden CV = {d['combined_score']} (base {d['base_score']} + div_J {d['combined_divJ']}); "
                       f"delta {d['delta']}; {'BEATS' if beat else 'below'} 0.891",
               recommendation="ready for human Kaggle submission review" if beat else "raise div-model precision / try deep-sister")
    return ("done", d, "leader",
            f"[{worker}] PIPELINE-RUN combined golden CV = {d['combined_score']} "
            f"(base {d['base_score']}, adjE {d['combined_adjE']}, div_J {d['combined_divJ']}, Δ{d['delta']}). "
            f"{'🎯 BEATS 0.891 — human review for submission.' if beat else 'below 0.891 — tune div-model precision or use deep-sister.'}")
