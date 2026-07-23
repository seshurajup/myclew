"""feasibility-gate — the reusable ORCHESTRATION that turns every GO/NO-GO lever decision into a
structured, durable XAI insight on the Lever Feasibility Map (docs/INSIGHTS.md → /insights).

It is a THIN DRIVER that chains the EXISTING fleet agents in sequence — it reimplements none of them:

    xai-diagnose  (name the failing bucket / candidate lever)                  → fleet_agents.xai_diagnose
    → lever       (run the lever: lever-hunt or a named agent handler)         → fleet_agents.lever_hunt / HANDLERS
    → math-master (per-embryo PAIRED Δ + significance: Wilcoxon/sign/CI)        → fleet_agents.math_master
    → official-score (patched organizer metric, per embryo)                    → fleet_agents.official_score
    → ledger      (record kept + verdict + mechanism, single source of truth)  → fleet_agents.insights.record_gate
    → insights    (surface on the Lever Feasibility Map, auto-refresh)         → fleet_agents.insights.report_map

The map is the "don't re-run a killed lever" memory: each gate = {lever_name, verdict, delta, significance,
mechanism, evidence, date}. The gate itself is journaled as a ledger verdict-decision (single source of truth),
so the map DERIVES from the ledger — no parallel store. Comp-parameterized via the ledger's RP_COMP resolution.

Spec modes:
  mode="gate"     (default) — run/record ONE gate. Inputs (all optional, chained where present):
      lever_name, mechanism (required to record),
      before_scores / after_scores  : per-dataset patched-metric arrays (→ math-master paired Δ + significance)
      before_json  / after_json     : official_score.json paths (→ per-embryo patched-metric delta)
      lever_agent  / lever_spec      : a fleet handler to actually RUN the lever first (else consume provided deltas)
      diagnose_spec                  : passed to xai-diagnose to NAME the bucket
      verdict / delta / significance / evidence : explicit overrides (used for already-measured/backfilled gates)
      lb_note                        : e.g. "LB 0.888 neutral" → CV-GO/LB-neutral honesty flag
  mode="backfill" — record THIS SESSION's gates (division-post-proc, ILP-weight, temporal-division,
                    detector-signal, peak-retrain, gap_fill, edge-consensus, adaptive-decode, edge-precision-prune).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import insights

COMP = Path(__file__).resolve().parent.parent
PATCHED_DIV_WEIGHT = 0.1                                    # metric = edge_jaccard + 0.1·division_jaccard


# ── step helpers (each wraps an EXISTING agent — no reimplementation) ──────────────────────────────
def _patched(edge, div):
    if edge is None:
        return None
    return round(float(edge) + PATCHED_DIV_WEIGHT * float(div or 0.0), 4)


def _official_patched(json_path):
    """Read an official-score artifact → {embryo: patched_metric, overall: patched}. Reads the agent's
    output file (official_score.json); it does NOT reimplement the organizer metric."""
    p = Path(json_path)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}
    ov = d.get("overall", {}) if isinstance(d.get("overall"), dict) else {}
    out = {"overall": _patched(ov.get("edge_jaccard"), ov.get("division_jaccard"))}
    for emb, s in (d.get("per_embryo", {}) or {}).items():
        if isinstance(s, dict):
            out[emb] = _patched(s.get("edge_jaccard"), s.get("division_jaccard"))
    return out


def _significance(before, after):
    """Per-dataset PAIRED significance via math-master (reused, not reimplemented): Wilcoxon signed-rank +
    sign test + mean Δ + bootstrap CI. Returns a short human string + the raw stats."""
    from . import math_master as MM
    a = list(before or []); b = list(after or [])
    if len(a) != len(b) or len(a) < 3:
        return "n<3 paired — underpowered", {"n": min(len(a), len(b))}
    import numpy as np
    d = np.asarray(b, float) - np.asarray(a, float)
    p_w = MM.wilcoxon_p(b, a)
    p_s = MM.sign_test_p(b, a)
    ci = MM.bootstrap_ci(d.tolist(), stat="mean")
    mean_d = float(d.mean())
    sig = "SIGNIFICANT" if (p_w < 0.05 or p_s < 0.05) else "n.s."
    txt = f"Δ̄={mean_d:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}] · Wilcoxon p={p_w:.3f} · {sig} (n={len(a)})"
    return txt, {"mean_delta": round(mean_d, 4), "wilcoxon_p": round(p_w, 4), "sign_p": round(p_s, 4),
                 "ci": [round(ci["lo"], 4), round(ci["hi"], 4)], "n": len(a), "significant": sig == "SIGNIFICANT"}


def _derive_verdict(delta_by_embryo, stats, lb_note, eps=0.001):
    """Turn the measured per-embryo Δ + significance + LB note into a GO/NO-GO verdict (honest)."""
    if lb_note:                                            # CV moved but LB didn't → the honesty flag
        cv_up = (stats.get("mean_delta", 0) > eps) or any((v or 0) > eps for v in delta_by_embryo.values())
        if cv_up:
            return "CV-GO/LB-NEUTRAL"
    md = stats.get("mean_delta")
    if md is None and delta_by_embryo:
        vals = [v for v in delta_by_embryo.values() if v is not None]
        md = sum(vals) / len(vals) if vals else 0.0
    md = md or 0.0
    if md <= eps:
        return "NO-GO"
    return "GO" if stats.get("significant") else "WEAK-GO"


# ── the orchestration ─────────────────────────────────────────────────────────────────────────────
def run(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    if spec.get("mode") == "backfill":
        return backfill(worker)

    trail = []                                            # which agents actually fired (proof it chained)
    # ── 1. xai-diagnose: NAME the failing bucket / candidate lever (reuses xai_diagnose) ──
    if spec.get("diagnose_spec"):
        try:
            from . import xai_diagnose as XD
            dg = XD.diagnose(spec["diagnose_spec"])
            trail.append(f"xai-diagnose:{dg.get('verdict','')[:40]}")
        except Exception as e:  # noqa: BLE001
            trail.append(f"xai-diagnose:skipped({type(e).__name__})")

    # ── 2. run the lever (optional) — dispatch an EXISTING handler; else consume provided deltas ──
    if spec.get("lever_agent"):
        try:
            from . import HANDLERS
            h = HANDLERS.get(spec["lever_agent"])
            if h:
                h({"spec": spec.get("lever_spec", {})}, worker)
                trail.append(f"lever:{spec['lever_agent']}")
        except Exception as e:  # noqa: BLE001
            trail.append(f"lever:{spec.get('lever_agent')}:err({type(e).__name__})")

    # ── 3+4. official-score patched-metric delta (per embryo) + math-master paired significance ──
    delta_by_embryo, sig_txt, stats = {}, spec.get("significance", ""), {}
    if spec.get("before_json") and spec.get("after_json"):
        pb = _official_patched(spec["before_json"]); pa = _official_patched(spec["after_json"])
        for emb in ("44b6", "6bba"):
            if emb in pb and emb in pa and pb[emb] is not None and pa[emb] is not None:
                delta_by_embryo[emb] = round(pa[emb] - pb[emb], 4)
        trail.append("official-score:patched-metric")
    if spec.get("before_scores") and spec.get("after_scores"):
        sig_txt, stats = _significance(spec["before_scores"], spec["after_scores"])
        trail.append("math-master:paired")

    # explicit delta string (backfilled/already-measured) wins for display when arrays absent
    delta_str = spec.get("delta") or (
        " · ".join(f"{e} {d:+.4f}" for e, d in delta_by_embryo.items()) if delta_by_embryo else "")

    # ── 5. verdict (explicit override else derived from the measured evidence) ──
    verdict = spec.get("verdict") or _derive_verdict(delta_by_embryo, stats, spec.get("lb_note"))

    lever = spec.get("lever_name")
    mechanism = spec.get("mechanism", "")
    if not lever or not mechanism:
        return ("escalated", {"error": "need spec.lever_name + spec.mechanism", "trail": trail},
                "researcher", f"[{worker}] feasibility-gate: provide lever_name + mechanism (the WHY).")

    # ── 6. LEDGER: record the gate (single source of truth) ──
    gate = insights.record_gate(lever_name=lever, verdict=verdict, mechanism=mechanism,
                                delta=delta_str, significance=sig_txt or spec.get("significance", ""),
                                evidence=spec.get("evidence", ""), date=spec.get("date"),
                                extra={"trail": trail, "stats": stats} if (trail or stats) else None)
    # ── 7. INSIGHTS: surface on the Lever Feasibility Map (auto-refresh) ──
    insights.report_map(q, worker)
    emoji = insights._VERDICT_RANK.get(gate["verdict"], insights._VERDICT_RANK["UNKNOWN"])[1]
    msg = (f"[{worker}] **FEASIBILITY-GATE** {emoji} {gate['verdict']} · `{lever}`\n"
           f"chain: {' → '.join(trail) or '(deltas provided)'}\n"
           f"Δ {delta_str or '—'} · {sig_txt or gate['significance'] or '—'} · mechanism: {mechanism} "
           f"→ recorded on the Lever Feasibility Map (/insights).")
    return ("done", {"gate": gate, "trail": trail}, "all", msg)


def run_map(q, worker):
    """feasibility-map handler — render + return the current map (delegates to insights.report_map)."""
    return insights.report_map(q, worker)


# ── this session's gates (backfill) — each with its one-line mechanism ─────────────────────────────
# Derived from the ledger/verdicts already established this session (biohub_autonomous_run_20260714,
# biohub_pilkwang_pivot, biohub_insights_autorefresh, biohub_ledger_provenance_gate).
SESSION_GATES = [
    dict(lever_name="division-post-proc", verdict="NO-GO",
         delta="both embryos ≤0", significance="paired n.s.",
         mechanism="host/Ultrack already place the 2nd child; added forks are FP that cost edge-precision > div gain",
         evidence="EXP division-rescue"),
    dict(lever_name="ILP-weight", verdict="NO-GO",
         delta="no measured gain", significance="n.s.",
         mechanism="linking is saturated (edge head done); reweighting ILP costs does not lift adjJ on sparse GT",
         evidence="EXP fullconfig-search"),
    dict(lever_name="temporal-division", verdict="NO-GO",
         delta="0.709 vs 0.690 (probe sep, not metric)", significance="temporal≈single, gate NO-GO",
         mechanism="temporal multi-pool separability does NOT beat single-frame enough to justify a temporal head",
         evidence="EXP div-temporal-feas"),
    dict(lever_name="detector-signal", verdict="GO",
         delta="node-recall is the squared lever (adjJ≈R_node²·Q_link)", significance="XAI-confirmed",
         mechanism="detection/node-recall is THE lever — recall gains compound; edge head already saturated",
         evidence="EXP det-sweep / mh-ilp"),
    dict(lever_name="peak-retrain", verdict="NO-GO",
         delta="44b6 −0.011", significance="regression on the lever embryo",
         mechanism="retraining the peak detector REGRESSES the reliable embryo — dense external GT ≠ competition sparse GT",
         evidence="EXP peak-retrain"),
    dict(lever_name="gap_fill", verdict="GO",
         delta="+0.029 (dense subset)", significance="adj+score both up",
         mechanism="bridge track-end@t→start@t+k+1 with interpolated nodes under the 7µm gate recovers missed edges",
         evidence="EXP lever-hunt/gap_fill"),
    dict(lever_name="edge-consensus", verdict="CV-GO/LB-NEUTRAL",
         delta="CV +0.0026 / LB 0.888 (flat)", significance="CV up, LB unmoved",
         mechanism="multi-tracker consensus edges help canonical CV but LB is neutral — CV over-credits; trust Kaggle",
         evidence="EXP tracker-consensus"),
    dict(lever_name="adaptive-decode", verdict="NO-GO",
         delta="no metric gain", significance="n.s.",
         mechanism="stage-adaptive decode thresholds do not beat a single calibrated decode on sparse GT",
         evidence="EXP stage-dynamics"),
    dict(lever_name="edge-precision-prune", verdict="NO-GO",
         delta="net ≤0", significance="n.s.",
         mechanism="pruning low-prob edges trades recall for precision 1:1 on adjJ — no net lift (linking saturated)",
         evidence="EXP link-tune"),
]


def backfill(worker):
    """Record THIS SESSION's GO/NO-GO gates onto the Lever Feasibility Map (idempotent via ledger dedup)."""
    for g in SESSION_GATES:
        insights.record_gate(**g)
    res = insights.report_map({"spec": {}}, worker)
    gates = insights.feasibility_gates()
    n_go = sum(1 for x in gates if x["verdict"] == "GO")
    n_nogo = sum(1 for x in gates if x["verdict"] == "NO-GO")
    n_flag = sum(1 for x in gates if x["verdict"] == "CV-GO/LB-NEUTRAL")
    return ("done", {"backfilled": len(SESSION_GATES), "total_gates": len(gates),
                     "n_go": n_go, "n_nogo": n_nogo, "n_cv_go_lb_neutral": n_flag}, "all",
            f"[{worker}] **FEASIBILITY-MAP BACKFILL** · recorded {len(SESSION_GATES)} session gates → "
            f"{len(gates)} levers on the map ({n_go} GO / {n_nogo} NO-GO / {n_flag} CV-GO-LB-neutral). "
            f"On /insights (docs/INSIGHTS.md). {res[3].split('] ', 1)[-1] if len(res) > 3 else ''}")
