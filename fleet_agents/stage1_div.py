"""stage-1-div — STAGE-1 div_J verdict: run div-model on 36-event predicted-node split.
This is the powered test for whether learned div-model transfers to inference density."""
from __future__ import annotations
import subprocess
import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
SCRIPT = COMP / "scripts" / "stage1_div_j_verdict.py"
PY = COMP / "research" / "cellmot_venv" / "bin" / "python"


def run(q, worker):
    spec = (q or {}).get("spec", {}) or {}
    try:
        threshold = float(spec.get("threshold", 0.9))     # threshold: div-model probability cutoff
    except (TypeError, ValueError):
        threshold = 0.9
    try:
        timeout = int(spec.get("timeout", 600))           # timeout: verdict-script wall-clock cap (s)
    except (TypeError, ValueError):
        timeout = 600

    if not SCRIPT.exists() or not PY.exists():
        return ("escalated", {"status": "missing deps"}, "researcher",
                f"[{worker}] STAGE-1-DIV: script or cellmot_venv missing.")

    try:
        result = subprocess.run([str(PY), str(SCRIPT), str(threshold)],
                              capture_output=True, text=True, timeout=timeout, cwd=str(COMP))
        if result.returncode != 0:
            return ("escalated", {"error": (result.stderr or "")[:200]}, "researcher",
                    f"[{worker}] STAGE-1-DIV failed: {(result.stderr or '')[:100]}")

        out = json.loads((result.stdout or "").strip())
        base = out.get("base_score")
        combined = out.get("combined_score")
        div_j = out.get("div_j", 0) or 0
        divs_added = out.get("divisions_added", 0) or 0

        have = isinstance(base, (int, float)) and isinstance(combined, (int, float))
        delta = (combined - base) if have else 0
        _b = f"{base:.4f}" if isinstance(base, (int, float)) else "n/a"
        _c = f"{combined:.4f}" if isinstance(combined, (int, float)) else "n/a"
        message = (f"[{worker}] STAGE-1-DIV: base={_b}, combined={_c}, "
                  f"delta={delta:.4f}, div_J={float(div_j):.3f} (added {divs_added} high-prec). "
                  f"{'✓ Transfer OK' if divs_added > 0 else '⚠ No transfer (density caveat bites)'}.")

        return ("done", {
            "base_score": base, "combined_score": combined, "delta": delta,
            "div_j": div_j, "divisions_added": divs_added, "threshold": threshold
        }, "researcher", message)
    except Exception as exc:
        return ("escalated", {"error": str(exc)}, "researcher",
                f"[{worker}] STAGE-1-DIV exception: {exc}")
