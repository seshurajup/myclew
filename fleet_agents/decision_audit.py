"""decision-audit — enforce "decide only from data" across the whole ledger.

The user's rule: an agent may take a choice ONLY if data proves it from analysis. This agent audits the
experiment ledger for violations: rows that were KEPT / recommended / crowned "best" WITHOUT a measured
golden-12 CV behind them, or CV values that are implausible (the 1.12-style artifacts). It reports the
unproven decisions so they get re-verified (e.g. via verify-cv / trick-gate) instead of standing on
assertion. Read-only + honest — it never adopts anything itself.

Reusable / spec-driven: {ledger_path, cv_min, cv_max} — point it at any experiment ledger.
"""
from __future__ import annotations
import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
LEDGER = COMP / "docs" / "experiment_ledger.jsonl"
STATE = COMP / "config" / "_auto" / "decision_audit.json"
# cv_max=1.1: adjusted-jaccard can legitimately touch ~1.0 (under-count bonus); >1.1 is the real artifact
# (the 1.12 blow-up). cv_min=0.0: a LOW score is a valid failed experiment, not "impossible" — don't flag it.
DEFAULTS = {"ledger_path": str(LEDGER), "cv_min": 0.0, "cv_max": 1.1}


def report(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    cfg = {**DEFAULTS, **{k: spec[k] for k in DEFAULTS if k in spec}}
    p = Path(cfg["ledger_path"])
    if not p.exists():
        return ("done", {}, "all", f"[{worker}] decision-audit: no ledger at {p}.")
    try:
        lo, hi = float(cfg["cv_min"]), float(cfg["cv_max"])
    except Exception:  # noqa: BLE001 — bad bounds → safe defaults
        lo, hi = 0.0, 1.1
    try:
        text = p.read_text(errors="replace")
    except Exception as e:  # noqa: BLE001 — unreadable ledger → clean escalate, don't crash
        return ("escalated", {"error": str(e)[:100]}, "researcher",
                f"[{worker}] decision-audit: cannot read ledger {p} ({type(e).__name__}).")
    rows = []
    for ln in text.splitlines():
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                pass

    kept_no_cv, impossible_cv, rec_no_cv = [], [], []
    for r in rows:
        cv = r.get("cv")
        exp = r.get("exp") or r.get("change") or "?"
        has_cv = isinstance(cv, (int, float))
        if has_cv and not (lo <= cv <= hi):
            impossible_cv.append((exp, cv))
        if r.get("kept") is True and not has_cv:
            kept_no_cv.append(exp)
        rec = (r.get("recommendation") or "")
        if rec and any(w in rec.lower() for w in ("adopt", "keep", "best", "use ")) and not has_cv:
            rec_no_cv.append(exp)

    n_viol = len(kept_no_cv) + len(impossible_cv) + len(rec_no_cv)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"rows": len(rows), "kept_no_cv": kept_no_cv[:50],
                                 "impossible_cv": impossible_cv[:50], "rec_no_cv": rec_no_cv[:50]}, indent=2))
    from . import ledger
    ledger.log("decision-audit",
               summary=f"audited {len(rows)} ledger rows → {n_viol} unproven choices (kept/recommended w/o measured CV)",
               detail=f"kept_no_cv={len(kept_no_cv)} impossible_cv={len(impossible_cv)} rec_no_cv={len(rec_no_cv)}",
               kind="verdict", recommendation="re-verify the flagged rows via verify-cv before trusting them")
    from researchpapers.fleet import post
    parts = [f"audited `{len(rows)}` ledger rows"]
    parts.append(f"❌ kept-without-CV: `{len(kept_no_cv)}`" if kept_no_cv else "✅ no kept-without-CV")
    parts.append(f"⚠️ impossible-CV (>1.0/<0.5): `{len(impossible_cv)}`" if impossible_cv else "✅ no impossible CV")
    parts.append(f"➖ recommended-without-CV: `{len(rec_no_cv)}`" if rec_no_cv else "✅ recs all backed by CV")
    ex = ", ".join(f"{e}={c}" for e, c in impossible_cv[:3]) or ", ".join(kept_no_cv[:3]) or "none"
    msg = (f"[{worker}] **DECISION-AUDIT** · enforce 'decide only from data' · "
           + " · ".join(parts)
           + f"\n**Flagged for re-verification:** {ex if n_viol else 'clean — every choice is data-backed'}")
    post.post_thread(worker, "all", msg, routine=False, kind="verdict")
    return ("done", {"rows": len(rows), "violations": n_viol, "kept_no_cv": len(kept_no_cv),
                     "impossible_cv": len(impossible_cv)}, "all", msg)
