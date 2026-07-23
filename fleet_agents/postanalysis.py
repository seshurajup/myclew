"""Post-analysis agent — AFTER an experiment, deliver the verdict deterministically.

Reads the latest scored run vs the previous best: computes the CV delta, checks TRANSFER to BOTH
embryo folds (adjJ_44b6 & adjJ_6bba both improve — reject train-only gains), and decides kept/rejected
against the LOEO noise floor (~0.017). Backfills the journal. This is the trainer/researcher 'is it
real?' judgement, in Python. Claude only needed if the verdict is ambiguous.
"""
from __future__ import annotations

from . import ledger, metric
from .preanalysis import next_lever

NOISE = 0.017  # LOEO bootstrap std — a change must beat prior CV by more than this


def verdict(q, worker):
    runs = metric._scored_runs()
    if len(runs) < 2:
        return ("escalated", {"reason": "need >=2 scored runs"}, "researcher",
                f"[{worker}] POST-ANALYSIS: need a baseline + one experiment scored to compute a delta. "
                f"{len(runs)} scored run(s) so far.")
    scored = sorted(runs, key=lambda r: r["metrics"].get("official_score",
                                                         r["metrics"].get("adj_edge_jaccard", 0.0)))
    latest = runs[0] if runs else None  # _scored_runs is DESC by start_time → runs[0] = newest
    prev_best = max((r for r in runs if r is not latest),
                    key=lambda r: r["metrics"].get("official_score", r["metrics"].get("adj_edge_jaccard", 0.0)),
                    default=None)
    lm, pm = latest["metrics"], prev_best["metrics"]
    ls = lm.get("official_score", lm.get("adj_edge_jaccard"))
    ps = pm.get("official_score", pm.get("adj_edge_jaccard"))
    delta = (ls - ps) if (ls is not None and ps is not None) else None
    # transfer: both embryo folds must not regress
    tr_44, tr_6 = lm.get("adjJ_44b6"), lm.get("adjJ_6bba")
    p44, p6 = pm.get("adjJ_44b6"), pm.get("adjJ_6bba")
    transfers = None
    if None not in (tr_44, tr_6, p44, p6):
        transfers = (tr_44 >= p44 - NOISE) and (tr_6 >= p6 - NOISE)
    kept = bool(delta is not None and delta > NOISE and (transfers is not False))
    reason = ("kept: beats prior CV by >noise" + (" and transfers to both embryos" if transfers else "")
              if kept else
              ("rejected: within noise" if (delta is None or delta <= NOISE) else "rejected: train-only gain (fails transfer)"))
    # THE LOOP: the verdict on THIS result drives the NEXT experiment (weakest link of the current best)
    nxt = next_lever(lm if kept else pm)   # if rejected, diagnose from the still-best run
    ledger.log("post-analysis", kind="post", run=latest["name"],
               summary=f"CV={ls} vs prev {ps} → Δ={delta}; transfer={transfers}; VERDICT={'KEPT' if kept else 'REJECTED'} ({reason})",
               recommendation=nxt)  # POST phase: verdict on this result → next experiment
    return ("done", {"latest": latest["name"], "cv": ls, "prev_best": ps, "delta": delta,
                     "transfers": transfers, "kept": kept, "next_lever": nxt}, "leader",
            f"[{worker}] 📋 VERDICT {latest['name'][:22]}: CV={ls} vs prev best {ps} → Δ={delta}. "
            f"Transfer(both embryos)={transfers}. → {'KEPT' if kept else 'REJECTED'} ({reason}).\n"
            f"🧭 NEXT EXPERIMENT (from this result): {nxt}. Leader: set ONE change here, no Kaggle.")
