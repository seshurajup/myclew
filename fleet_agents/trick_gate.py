"""trick-gate — the EVIDENCE GATE: a trick is adopted ONLY if it proves out on golden-12, never on
popularity. Enforces the user's rule: "any agent takes a choice only if data proves it, from analysis."

trick-extractor tells us which tricks the top solutions use; this agent then MEASURES each testable one
on our own golden-12 (vs the 0.8803 yaroslav base) and returns a verdict:
  • ADOPT   — measured Δ ≥ +threshold (proven on OUR metric)
  • NEUTRAL — |Δ| < threshold (no evidence either way → not adopted)
  • REJECT  — Δ < 0 (popular but hurts us — explicitly rejected)
  • NEEDS-GPU — the trick changes prediction-time (TTA / multi-scale / detection) so it cannot be proven
                on cached predictions; honestly flagged, NOT adopted on popularity.

Post-proc toggles that the base leaves OFF are the cheap, cached-pred-testable candidates. Reusable /
spec-driven: {candidates:[{name, env, prediction_time}], base_env_sh, threshold, screen_n}.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
PY_ENV = COMP / "research" / "cellmot_venv" / "bin" / "python"
SCRIPT = COMP / "scripts" / "score_postproc_golden12.py"
BASE_ENV_SH = COMP / "model_scratch" / "results" / "div_probe" / "yaroslav_env.sh"
STATE = COMP / "config" / "_auto" / "trick_gate.json"

# candidate tricks the base leaves OFF that top solutions use — each testable on CACHED predictions.
# prediction_time=True ⇒ can't be proven on cached preds ⇒ NEEDS-GPU (honest, not adopted on popularity).
DEFAULT_CANDIDATES = [
    {"name": "gap2-recovery",        "env": {"BIOHUB_OUTPUT_GAP2_RECOVERY": "1"}},
    {"name": "single-child-repair",  "env": {"BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR": "1"}},
    {"name": "division-geom-filter",  "env": {"BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER": "1"}},
    {"name": "rotate-TTA",           "env": {}, "prediction_time": True},
    {"name": "multi-scale-detect",   "env": {}, "prediction_time": True},
]
DEFAULTS = {"candidates": DEFAULT_CANDIDATES, "threshold": 0.001, "screen_n": 0, "base_cv": 0.8803,
            "timeout": 1200}   # timeout: per-score subprocess cap (seconds)


def _base_env():
    env = {}
    if BASE_ENV_SH.exists():
        for ln in BASE_ENV_SH.read_text().splitlines():
            m = re.match(r"\s*export\s+(BIOHUB_[A-Z0-9_]+)=(.*)", ln)
            if m:
                env[m.group(1)] = m.group(2).strip()
    return env


_SCORE_TIMEOUT = 1200   # per-score subprocess cap (s); run() may override from spec['timeout']


def _score(env, n):
    """Cap the (slow) golden-12 scorer subprocess at _SCORE_TIMEOUT; None/failed → None (caller marks FAILED)."""
    run_env = dict(os.environ); run_env.update(env)
    try:
        r = subprocess.run([str(PY_ENV), str(SCRIPT), str(n)], capture_output=True, text=True,
                           timeout=max(1, int(_SCORE_TIMEOUT)), cwd=str(COMP), env=run_env)
    except Exception:  # noqa: BLE001 — timeout / launch failure → treat as unscored, not a crash
        return None
    line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    if not line:
        return None
    try:
        d = json.loads(line[-1])
    except Exception:  # noqa: BLE001
        return None
    return d if d.get("n", 0) and d.get("score", 0) else None


def run(q, worker):
    if not PY_ENV.exists() or not SCRIPT.exists() or not BASE_ENV_SH.exists():
        return ("escalated", {}, "researcher", f"[{worker}] trick-gate: scorer or base env missing.")
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    cfg = {**DEFAULTS, **{k: spec[k] for k in DEFAULTS if k in spec}}
    try:
        thr = float(cfg["threshold"]); n = int(cfg["screen_n"])
    except Exception:  # noqa: BLE001 — bad numeric spec → fall back to safe defaults
        thr, n = 0.001, 0
    global _SCORE_TIMEOUT                                          # OPTIONAL spec['timeout'] caps each scorer call
    try:
        _SCORE_TIMEOUT = max(1, int(cfg["timeout"]))
    except Exception:  # noqa: BLE001
        _SCORE_TIMEOUT = 1200
    base = _base_env()
    # measure the base once (source of truth, not the assumed 0.8803)
    try:
        st = json.loads(STATE.read_text()) if STATE.exists() else {"results": {}}
    except Exception:  # noqa: BLE001
        st = {"results": {}}
    st.setdefault("results", {})
    base_d = _score(base, n)
    base_cv = base_d["score"] if base_d else float(cfg["base_cv"])

    verdicts = []
    for c in cfg["candidates"]:
        if not isinstance(c, dict) or "name" not in c:            # skip malformed candidate, don't crash
            continue
        if c.get("prediction_time"):
            verdicts.append((c["name"], None, "NEEDS-GPU"))
            st["results"][c["name"]] = {"delta": None, "verdict": "NEEDS-GPU"}
            continue
        d = _score({**base, **(c.get("env") or {})}, n)
        if not d:
            verdicts.append((c["name"], None, "FAILED")); continue
        delta = d["score"] - base_cv
        verdict = "ADOPT" if delta >= thr else ("REJECT" if delta < -1e-9 else "NEUTRAL")
        verdicts.append((c["name"], round(delta, 4), verdict))
        st["results"][c["name"]] = {"score": d["score"], "delta": round(delta, 4), "verdict": verdict}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"base_cv": base_cv, **st}, indent=2))

    adopted = [v for v in verdicts if v[2] == "ADOPT"]
    def cell(v):
        name, delta, verd = v
        icon = {"ADOPT": "✅", "REJECT": "❌", "NEUTRAL": "➖", "NEEDS-GPU": "🖥️", "FAILED": "⚠️"}[verd]
        dtxt = f"Δ{delta:+.4f}" if delta is not None else "unproven"
        return f"{icon} {name} `{dtxt}`"
    from . import ledger
    ledger.log("trick-gate",
               summary=f"evidence gate on {len(verdicts)} tricks (base {base_cv}): {len(adopted)} ADOPTED by proof",
               detail="; ".join(f"{nm}:{vd}({dl})" for nm, dl, vd in verdicts), kind="verdict",
               recommendation=("adopt: " + ", ".join(a[0] for a in adopted)) if adopted else "none proved out — base stands")
    from researchpapers.fleet import post
    msg = (f"[{worker}] **TRICK-GATE** · evidence-only adoption · base golden-12 `{base_cv}`\n"
           + " · ".join(cell(v) for v in verdicts)
           + f"\n**Adopted (proven Δ>0):** {', '.join(a[0] for a in adopted) or 'none — no trick beat the base'}")
    post.post_thread(worker, "all", msg, routine=False, kind="verdict")
    return ("done", {"base_cv": base_cv, "adopted": [a[0] for a in adopted],
                     "verdicts": {nm: vd for nm, dl, vd in verdicts}}, "all", msg)
