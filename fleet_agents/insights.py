"""insights — the fleet's FINAL-INSIGHTS markdown for the super-leader / super-researcher handoff.

The workflow: the super-agents set DIRECTION, then step away; the deterministic fleet takes over and runs
the whole loop; when the super-agents come back they should NOT re-read every message — they read ONE
markdown that has the complete work + final insights. This agent regenerates that markdown each cycle
from the journal + decision trail + MLflow, so it is always current.

Writes docs/INSIGHTS.md (shown on the board at /insights).
"""
from __future__ import annotations

import json
import datetime
from pathlib import Path

from . import ledger, preanalysis

COMP = Path(__file__).resolve().parent.parent
OUT = COMP / "docs" / "INSIGHTS.md"

# ═══════════════════════════ LEVER FEASIBILITY MAP ═══════════════════════════
# Every GO/NO-GO feasibility gate we run (temporal-division NO-GO, detector-signal GO, gap_fill GO,
# edge-consensus CV-GO/LB-neutral, ...) becomes a STRUCTURED, DURABLE insight so we never re-run a
# killed lever. The map DERIVES from the ledger (single source of truth): a gate is journaled as a
# ledger DECISION (kind="verdict", agent="feasibility-gate") whose detail carries a GATE_TAG JSON
# payload. feasibility_gates() scans those rows back; feasibility_map_md() renders the ranked table.
GATE_TAG = "FEASGATE::"                                     # marker prefix in the decision `detail` field
# verdict → (rank, emoji, bucket-label). Lower rank sorts first (adopted GO on top, dead NO-GO last).
_VERDICT_RANK = {
    "GO": (0, "✅", "GO — adopted"),
    "WEAK-GO": (1, "🟡", "WEAK-GO — conditional"),
    "CV-GO/LB-NEUTRAL": (2, "🟠", "CV-GO / LB-neutral — honesty flag"),
    "NO-GO": (3, "⛔", "NO-GO — killed (do not re-run)"),
    "UNKNOWN": (4, "❔", "UNKNOWN"),
}


def _norm_verdict(v) -> str:
    """Canonicalize a free-text verdict into one of the map buckets."""
    s = str(v or "").strip().upper().replace(" ", "").replace("_", "-")
    if s in _VERDICT_RANK:
        return s
    if "LB-NEUTRAL" in s or ("CV-GO" in s and "LB" in s) or "LB-NEUTRAL" in s:
        return "CV-GO/LB-NEUTRAL"
    if s.startswith("WEAK"):
        return "WEAK-GO"
    if s.startswith("NO") or "KILL" in s or "DEAD" in s:
        return "NO-GO"
    if s.startswith("GO") or "ADOPT" in s or "SOLID" in s:
        return "GO"
    return "UNKNOWN"


def record_gate(lever_name, verdict, mechanism, delta="", significance="", evidence="",
                date=None, extra=None):
    """Record ONE feasibility gate as a durable insight — written THROUGH the ledger (single source of
    truth) as a verdict decision. Idempotent (ledger.log dedups identical agent+summary+recommendation).
    This is the write half of the map; feasibility_gates() reads them back.

    lever_name   : the lever being gated (e.g. 'temporal-division', 'gap_fill')
    verdict      : GO | NO-GO | WEAK-GO | CV-GO/LB-neutral (free text is normalized)
    mechanism    : the ONE-LINE WHY (the reason we never re-run it)
    delta        : patched-metric per-embryo delta, as a short string
    significance : math_master result (p-value / paired test verdict), short string
    evidence     : the ledger EXP id (or artifact) that proves it
    """
    v = _norm_verdict(verdict)
    gate = {"lever_name": str(lever_name), "verdict": v, "mechanism": str(mechanism),
            "delta": str(delta), "significance": str(significance), "evidence": str(evidence),
            "date": date or datetime.datetime.now(datetime.timezone.utc).date().isoformat()}
    if extra:
        gate["extra"] = extra
    emoji = _VERDICT_RANK.get(v, _VERDICT_RANK["UNKNOWN"])[1]
    ledger.log("feasibility-gate",
               summary=f"{emoji} {v}: {gate['lever_name']} — {gate['mechanism']}",
               detail=GATE_TAG + json.dumps(gate), kind="verdict",
               recommendation=(f"do NOT re-run '{gate['lever_name']}' — {gate['mechanism']}"
                               if v == "NO-GO" else f"'{gate['lever_name']}' is {v}: {gate['mechanism']}"))
    return gate


def feasibility_gates():
    """SCAN the ledger's decision trail for feasibility-gate rows → the current map (latest per lever).
    Derives from the ledger only; dedups by lever_name keeping the most recent record."""
    out = {}
    for d in ledger.decisions():
        det = d.get("detail") or ""
        if not det.startswith(GATE_TAG):
            continue
        try:
            g = json.loads(det[len(GATE_TAG):])
        except Exception:  # noqa: BLE001
            continue
        g["verdict"] = _norm_verdict(g.get("verdict"))
        g["_ts"] = d.get("ts", "")
        out[g.get("lever_name", "?")] = g                  # later rows overwrite → latest wins
    gates = list(out.values())
    gates.sort(key=lambda g: (_VERDICT_RANK.get(g["verdict"], _VERDICT_RANK["UNKNOWN"])[0],
                              g.get("lever_name", "")))
    return gates


def feasibility_map_md() -> str:
    """The Lever Feasibility Map as markdown — a ranked GO/NO-GO table with the one-line mechanism for
    each lever (the 'don't re-run dead levers' memory)."""
    gates = feasibility_gates()
    if not gates:
        return ("## 🧭 Lever Feasibility Map\n\n*No feasibility gates recorded yet — run the "
                "`feasibility-gate` orchestration (xai-diagnose → lever → math-master → official-score "
                "→ ledger) to populate it.*\n")
    n_go = sum(1 for g in gates if g["verdict"] == "GO")
    n_nogo = sum(1 for g in gates if g["verdict"] == "NO-GO")
    n_flag = sum(1 for g in gates if g["verdict"] == "CV-GO/LB-NEUTRAL")
    L = ["## 🧭 Lever Feasibility Map",
         "*Every GO/NO-GO gate we ran, derived from the ledger. This is the durable 'never re-run a "
         "killed lever' memory — each row is a structured feasibility insight (verdict · patched-metric "
         "delta · math-master significance · one-line mechanism · evidence EXP).*",
         f"- **{n_go} GO (adopted)** · **{n_nogo} NO-GO (killed)** · **{n_flag} CV-GO/LB-neutral "
         f"(honesty flag)** · {len(gates)} levers gated", "",
         "| lever | verdict | Δ (patched, per-embryo) | significance | mechanism (WHY) | evidence | date |",
         "| :-- | :-- | :-- | :-- | :-- | :-- | :-- |"]
    for g in gates:
        emoji = _VERDICT_RANK.get(g["verdict"], _VERDICT_RANK["UNKNOWN"])[1]
        L.append(f"| `{g.get('lever_name','?')}` | {emoji} {g['verdict']} | {g.get('delta') or '—'} "
                 f"| {g.get('significance') or '—'} | {g.get('mechanism') or '—'} "
                 f"| {g.get('evidence') or '—'} | {g.get('date') or '—'} |")
    L.append("")
    return "\n".join(L)


def _best_and_lever():
    try:
        from . import metric
        runs = metric._scored_runs()
        if not runs:
            return None, "", {}
        best = max(runs, key=lambda r: r["metrics"].get("official_score", r["metrics"].get("adj_edge_jaccard", 0.0)))
        return best["name"], preanalysis.next_lever(best["metrics"]), best["metrics"]
    except Exception:  # noqa: BLE001
        return None, "", {}


def _finite(v):
    """True only for a real finite number (rejects None / nan / inf / bool)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _fmt(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:.4f}" if _finite(v) else str(v)
    return str(v) if v not in (None, "") else "—"


def build_md() -> str:
    es = ledger.entries()
    scored = [e for e in es if _finite(e.get("cv"))]
    scored.sort(key=lambda e: e["cv"], reverse=True)
    best_run, lever, bm = _best_and_lever()
    kept = [e for e in es if e.get("kept")]
    decs = ledger.decisions()[-12:]

    L = ["# 🧠 Research Insights — biohub cell-tracking",
         "*Auto-generated by the deterministic fleet for the super-leader / super-researcher handoff. "
         "Read this instead of the message stream — it is the complete work + current direction.*", ""]
    L += ["## Where we are",
          f"- **Experiments run:** {len(es)}  ·  **scored:** {len(scored)}  ·  **kept:** {len(kept)}",
          f"- **Best scored:** `{best_run or '—'}`"
          + (f" (CV {_fmt(bm.get('golden_cv') or bm.get('official_score'))}, "
             f"adjJ_44b6 {_fmt(bm.get('adjJ_44b6'))}, adjJ_6bba {_fmt(bm.get('adjJ_6bba'))})" if bm else ""),
          f"- **Current weakest link → next lever:** {lever or '(needs a scored baseline)'}", ""]

    L += ["## Top results (by golden CV)", "", "| method | CV | adjJ_44b6 | adjJ_6bba | set |",
          "| :-- | --: | --: | --: | :-- |"]
    for e in scored[:12]:
        L.append(f"| `{e.get('change')}` | {_fmt(e.get('cv'))} | — | — | {e.get('trn_set','')} |")
    L += ["", "## Key findings (deterministic)", ""]
    # surface the loudest facts we can derive
    if scored:
        worst = min(scored, key=lambda e: e["cv"])
        L.append(f"- Best so far: `{scored[0].get('change')}` @ {_fmt(scored[0]['cv'])}; "
                 f"worst: `{worst.get('change')}` @ {_fmt(worst['cv'])}.")
    L.append(f"- Kept (survived verdict): {', '.join('`'+k.get('change','?')+'`' for k in kept) or 'none yet'}.")
    # the durable GO/NO-GO lever memory (derived from the ledger) — never re-run a killed lever
    L += ["", feasibility_map_md()]
    L += ["", "## Analysis & decision trail (latest)", ""]
    for d in reversed(decs):
        rec = f" → **{d.get('recommendation')}**" if d.get("recommendation") else ""
        L.append(f"- `{d.get('agent')}` [{d.get('kind','')}] {(d.get('summary') or d.get('finding') or '')[:160]}{rec}")

    L += ["", "## 👉 Recommended next direction (for the super-agents)",
          f"- The fleet is driving the deterministic search; the current weakest link is: **{lever or '—'}**.",
          "- If this lever has no deterministic recipe (e.g. division recovery), the super-researcher should "
          "provide the recipe; the fleet keeps running aug/detection search meanwhile.",
          "- Nothing here is Kaggle-submitted; submission remains a human decision.", ""]
    return "\n".join(L)


def report(q, worker):
    md = build_md()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md)
    n = md.count("\n")
    return ("done", {"out": str(OUT), "lines": n}, "all",
            f"[{worker}] INSIGHTS refreshed → docs/INSIGHTS.md ({n} lines) — the super-agent handoff report is current.")


def report_map(q, worker):
    """feasibility-map handler — refresh INSIGHTS.md (so the map renders on /insights) and return the
    current Lever Feasibility Map as one message. Comp-parameterized via the ledger's RP_COMP resolution."""
    gates = feasibility_gates()
    md = build_md()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md)
    n_go = sum(1 for g in gates if g["verdict"] == "GO")
    n_nogo = sum(1 for g in gates if g["verdict"] == "NO-GO")
    n_flag = sum(1 for g in gates if g["verdict"] == "CV-GO/LB-NEUTRAL")
    rows = " · ".join(f"{_VERDICT_RANK.get(g['verdict'], _VERDICT_RANK['UNKNOWN'])[1]}`{g['lever_name']}`"
                      for g in gates) or "(none yet)"
    return ("done", {"gates": gates, "n_go": n_go, "n_nogo": n_nogo, "n_cv_go_lb_neutral": n_flag,
                     "out": str(OUT)}, "all",
            f"[{worker}] **LEVER FEASIBILITY MAP** · {n_go} GO / {n_nogo} NO-GO / {n_flag} CV-GO-LB-neutral "
            f"({len(gates)} levers) → on /insights (docs/INSIGHTS.md). {rows}")
