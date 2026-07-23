"""grandmaster — the experimental TRICKS from top Kaggle journals (DrHB/icecube-journal, rna-stanford)
implemented as Python agents. These are the transferable methodology moves a grandmaster runs, each
logged in the DrHB journal style ("same as EXP_X but <one change>" + status marker):

  combine-winners : take the BEST value of each independent lever (best gap, best min_track_len, best
                    block-set) and STACK them into one recipe → score on golden-12 (DrHB combines the
                    best loss + best features + best pooling; we combine the best post-proc knobs).
  ablate-best     : one-variable-at-a-time around the current best (the core "same as X but one change"
                    trick) — generalised beyond combo-search's two axes.

Pure Python, no Claude. Each proposes a candidate and enqueues a real verify-cv golden-12 run.
"""
from __future__ import annotations
import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
COMBO_STATE = COMP / "config" / "_auto" / "combo_search_state.json"
CACHE = COMP / "config" / "_auto" / "verified_cv_cache.json"
GM_STATE = COMP / "config" / "_auto" / "grandmaster_state.json"

# the independent levers + their candidate menus (same knobs the notebooks tune)
LEVERS = {
    "BIOHUB_GAP_CLOSE_UM": ["5.5", "6.0", "6.8"],
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": ["4", "6", "8", "10", "12"],
}


def _load(p, default):
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return default


def _scored():
    """{sig: cv} of every combo we've actually measured (combo-search evaluated + verify cache)."""
    out = {}
    st = _load(COMBO_STATE, {})
    for sig, v in (st.get("evaluated") or {}).items():
        if isinstance(v.get("score"), (int, float)):
            out[sig] = v["score"]
    for sig, v in (_load(CACHE, {}) or {}).items():
        if isinstance(v.get("cv"), (int, float)):
            out[sig] = v["cv"]
    return out


def _marginal_best():
    """For each lever, the value with the highest MEAN measured golden-CV — the grandmaster 'best of each'."""
    scored = _scored()
    best = {}
    for lever, vals in LEVERS.items():
        by_val = {}
        for sig, cv in scored.items():
            for tok in sig.split(","):
                if tok.startswith(lever + "="):
                    by_val.setdefault(tok.split("=", 1)[1], []).append(cv)
        ranked = [(v, sum(cs) / len(cs)) for v, cs in by_val.items() if cs]
        best[lever] = max(ranked, key=lambda x: x[1])[0] if ranked else vals[0]
    return best, scored


def combine_winners(q, worker):
    """DrHB 'combine the winners' trick: stack the marginal-best value of every lever into ONE recipe."""
    best_vals, scored = _marginal_best()
    if not scored:
        return ("done", {"scored": 0}, "all",
                f"[{worker}] combine-winners: nothing measured yet — let combo-search run first.")
    env = {k: v for k, v in best_vals.items()}
    sig = ",".join(f"{k}={env[k]}" for k in sorted(env))

    gm = _load(GM_STATE, {"proposed": []})
    if sig in scored:
        return ("done", {"combo": sig, "cv": scored[sig], "already": True}, "all",
                f"[{worker}] combine-winners: the best-of-each recipe ({sig}) already scored {scored[sig]:.4f} — no new candidate.")
    if sig in gm["proposed"]:
        return ("done", {"combo": sig, "queued": True}, "all",
                f"[{worker}] combine-winners: best-of-each recipe already queued for scoring ({sig}).")

    # DrHB-style row + a real golden-12 run of the stacked recipe
    from . import ledger
    from researchpapers.fleet import board
    _sh = lambda k: k.replace("BIOHUB_", "").replace("OUTPUT_", "").lower()
    change = " + ".join(f"`{_sh(k)}`=`{v}`" for k, v in sorted(env.items()))
    parent = _load(COMBO_STATE, {}).get("best", {}).get("exp")
    row = ledger.record(change=f"gm-combine:{sig.replace('BIOHUB_', '')[:38]}",
                        description=f"combine winners — best of each lever: {change}",
                        script="scripts/score_postproc_golden12.py", cv=None, train_set="golden12",
                        parent=parent, stage=8, observation="grandmaster combine-winners trick")
    board.add("S", "verify-cv", f"grandmaster combine-winners recipe ({sig}) → golden-12",
              {"ref": f"gm-combine:{change}", "env": env, "sig": sig, "exp": row.get("exp")})
    gm["proposed"].append(sig)
    GM_STATE.parent.mkdir(parents=True, exist_ok=True)
    GM_STATE.write_text(json.dumps(gm, indent=2))
    return ("done", {"combo": sig, "best_vals": best_vals, "row": row.get("exp")}, "all",
            f"[{worker}] COMBINE-WINNERS: stacked the marginal-best of each lever → NEW recipe "
            f"({change}) as {row.get('exp')} and queued a real golden-12 run. DrHB 'combine the winners' trick.")


def ablate_best(q, worker):
    """The core 'same as X but ONE change' trick, generalised: from the current best recipe, propose the
    next unscored single-lever variation (one knob moved by one step) and queue it for golden-12."""
    st = _load(COMBO_STATE, {})
    best = st.get("best", {})
    benv = best.get("env")
    if not benv:
        return ("done", {}, "all", f"[{worker}] ablate-best: no best recipe yet (combo-search still warming up).")
    scored = _scored()
    for lever, vals in LEVERS.items():
        cur = benv.get(lever)
        for v in vals:                                  # one-step neighbours of the best, unscored first
            if v == cur:
                continue
            cand = dict(benv); cand[lever] = v
            sig = ",".join(f"{k}={cand[k]}" for k in sorted(cand))
            if sig in scored:
                continue
            from . import ledger
            from researchpapers.fleet import board
            _sh = lambda k: k.replace("BIOHUB_", "").replace("OUTPUT_", "").lower()
            row = ledger.record(change=f"gm-ablate:{sig.replace('BIOHUB_', '')[:38]}",
                                description=f"`{_sh(lever)}`=`{v}`", script="scripts/score_postproc_golden12.py",
                                cv=None, train_set="golden12", parent=best.get("exp"), stage=8,
                                observation="grandmaster one-variable ablation")
            board.add("S", "verify-cv", f"grandmaster ablation ({sig}) → golden-12",
                      {"ref": f"gm-ablate:{_sh(lever)}={v}", "env": cand, "sig": sig, "exp": row.get("exp")})
            return ("done", {"combo": sig, "changed": f"{_sh(lever)}={v}", "row": row.get("exp")}, "all",
                    f"[{worker}] ABLATE-BEST: same as `{best.get('exp')}` but `{_sh(lever)}`=`{v}` → queued golden-12. "
                    f"One-variable-at-a-time (DrHB lineage trick).")
    return ("done", {}, "all", f"[{worker}] ablate-best: all one-step neighbours of the best already scored.")
