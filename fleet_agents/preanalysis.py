"""Pre-analysis agent — BEFORE an experiment, diagnose the current state and recommend the next lever.

Deterministic: reads the best scored run's component metrics + the EDA fingerprint, names the WEAKEST
bucket, and recommends which journey stage/lever to open next. (The final direction is Claude's; this
is the anatomy that informs it — the 'name the weakest link, set the target' step of Stage 1.)
"""
from __future__ import annotations

from . import ledger, metric


def next_lever(m: dict):
    """SHARED weakest-link → next-experiment rule (used by pre-analysis AND post-analysis, so the
    NEXT experiment is always derived from the LAST result's metrics — not a pre-baked list)."""
    nr, ep = m.get("mean_node_recall"), m.get("adj_edge_jaccard")
    dj, cr = m.get("division_jaccard"), m.get("mean_count_ratio")
    a44, a6 = m.get("adjJ_44b6"), m.get("adjJ_6bba")
    # embryo-disjoint FAILURE is the loudest signal: one embryo collapsing → generalization, not a sub-metric
    if None not in (a44, a6) and min(a44, a6) < 0.5 * max(a44, a6):
        worst = "44b6" if a44 < a6 else "6bba"
        return (f"embryo-disjoint FAILURE: adjJ_44b6={a44:.3f} vs adjJ_6bba={a6:.3f} — {worst} collapses. "
                f"Target GENERALIZATION to {worst} (stage-bridging augmentation), not a global sub-metric.")
    if dj is not None and dj < 0.05:
        return "Stage-6 DIVISION head (+0.1 term almost entirely unclaimed = biggest headroom, train-only)"
    if nr is not None and nr < 0.95:
        return "Stage-2 DETECTION recall (node recall below 0.95 — detection is the bottleneck)"
    if cr is not None and (cr > 1.15 or cr < 0.85):
        return "Stage-8 COUNT calibration (count_ratio off — recalibrate to estN, never exceed it)"
    return "Stage-5 LINKING / edge precision (node recall ~saturated → cut FP edges; gate ≈8.5µm)"


def diagnose(q, worker):
    runs = metric._scored_runs()
    if not runs:
        return ("escalated", {"reason": "no baseline"}, "researcher",
                f"[{worker}] PRE-ANALYSIS: no scored baseline yet — run Stage-1 baseline first, then I diagnose the weakest lever.")
    best = max(runs, key=lambda r: r["metrics"].get("official_score",
                                                    r["metrics"].get("adj_edge_jaccard", 0.0)))
    m = best["metrics"]
    nr, ep, dj = m.get("mean_node_recall"), m.get("adj_edge_jaccard"), m.get("division_jaccard")
    lever = next_lever(m)
    ledger.log("pre-analysis", kind="pre", run=best["name"],
               summary=f"node_rec={nr}, adj_edge_J={ep}, div_J={dj}, adjJ_44b6={m.get('adjJ_44b6')}, adjJ_6bba={m.get('adjJ_6bba')}",
               recommendation=lever)  # PRE phase: diagnosis → what the leader should run next
    # DIRECTED TO LEADER — this is a next-experiment recommendation the leader must act on, not chatter
    return ("done", {"run": best["name"], "node_recall": nr, "edge_J": ep, "div_J": dj, "next_lever": lever}, "leader",
            f"[{worker}] 🧭 NEXT EXPERIMENT (from best run {best['name'][:22]}: node_rec={nr}, adj_edge_J={ep}, div_J={dj}): "
            f"weakest link → {lever}. Leader: set ONE experiment here (one change), score full metric on the frozen CV.")
