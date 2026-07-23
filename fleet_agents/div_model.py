"""div-model — a SECONDARY SUPPORT MODEL (generic): small classifier that claims an unclaimed metric term.
Today: the sister/division term (div_J). Runs the fit under cellmot_venv (geff/zarr deps + src)."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
COMP = Path(__file__).resolve().parent.parent
PY_ENV = COMP / "research" / "cellmot_venv" / "bin" / "python"
SCRIPT = COMP / "scripts" / "fit_div_model.py"


def train(q, worker):
    if not PY_ENV.exists() or not SCRIPT.exists():
        return ("escalated", {}, "researcher", f"[{worker}] div-model: cellmot_venv or fit script missing.")
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    timeout = int(spec.get("timeout", 600))            # subprocess wall-clock cap in seconds
    try:
        r = subprocess.run([str(PY_ENV), str(SCRIPT)], capture_output=True, text=True, timeout=timeout, cwd=str(COMP))
    except Exception as e:  # noqa: BLE001 — timeout / OSError → escalate, never crash the fleet
        return ("escalated", {"error": str(e)[:150]}, "researcher", f"[{worker}] div-model: subprocess failed ({str(e)[:80]}).")
    line = [l for l in (r.stdout or "").strip().splitlines() if l.startswith("{")]
    if not line:
        return ("escalated", {"stderr": (r.stderr or "")[-200:]}, "researcher",
                f"[{worker}] div-model fit failed: {(r.stderr or '').strip()[-150:]}")
    try:
        d = json.loads(line[-1])
    except Exception as e:  # noqa: BLE001
        return ("escalated", {"error": str(e)[:120]}, "researcher", f"[{worker}] div-model: bad JSON output ({str(e)[:60]}).")
    if not d.get("ok"):
        return ("done", d, "all", f"[{worker}] div-model: only {d.get('pos')} positives — too few.")
    from . import ledger
    ledger.log("div-model", kind="finding",
               summary=f"SECONDARY MODEL (division term): {d['kind']} AUC={d['auc']} ({d['pos']} pos/{d['neg']} neg)",
               recommendation="apply high-precision on the inference base via pipeline-run → claim the division term")
    return ("done", d, "leader",
            f"[{worker}] DIV-MODEL (secondary support model, {d['kind']}): AUC={d['auc']} "
            f"precision={d['precision']} recall={d['recall']} ({d['pos']} pos/{d['neg']} neg). Apply high-precision on the base.")
