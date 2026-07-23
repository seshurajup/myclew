"""orchestrator — the deterministic DECISION LOOP (runs the whole journey WITHOUT the Claude leader).

Each cycle, if the GPU is free and nothing is unscored, it:
  1. reads the current best scored run + the weakest link (preanalysis.next_lever),
  2. picks the NEXT untried experiment by a fixed, domain-grounded plan,
  3. generates its config (config_gen) and enqueues it (approved) — the fleet trains+scores+journals it.
It NEVER repeats a config already in the journal, and only escalates to the (optional) leader when it
hits a lever with no deterministic recipe — while still keeping the aug/param search running, so the
system makes progress even if leader+researcher are absent.

PLAN (self-driving):
  A. aug singles — walk the physically-valid menu (name+strength), ONE isolated aug each, on the screen.
  B. aug mixes   — once singles are done, compose the singles that BEAT the no-aug baseline (top-2, top-3).
  C. detection   — if node recall is the weak link, sweep det_neg_weight (recall-tilt), one value each.
  else           — escalate the named lever to the leader (e.g. division recovery), keep A/B/C running.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from researchpapers.fleet import board

from . import config_gen, ledger, preanalysis

COMP = Path(__file__).resolve().parent.parent
TRAIN_SERVICE = "http://127.0.0.1:7799"
SCREEN = "splits_screen_matched.json"
BASE = "config/aug_ablation/00_no_aug.yml"

# physically-valid isolated augs (name → augment-list entry), domain-grounded (aug-find menu)
AUG_MENU = {
    "flip_xy":    [{"name": "flip", "pz": 0.0, "py": 0.5, "px": 0.5}],
    "rot90_yx":   [{"name": "rot90_xy"}],
    "brightness": [{"name": "brightness", "p": 0.5, "shift_range": 0.08}],
    "contrast":   [{"name": "contrast", "p": 0.5, "range": 0.15}],
    "gamma":      [{"name": "gamma", "p": 0.5, "range": 0.2}],
    "bias_field": [{"name": "bias_field", "p": 0.3}],
    "blur":       [{"name": "blur", "p": 0.3}],
    "noise":      [{"name": "noise", "p": 0.5, "sigma": 0.02}],
    "crop_scale": [{"name": "crop_scale", "p": 0.5, "smin": 0.8, "smax": 1.25}],
}
DET_NEG_SWEEP = [0.05, 0.02, 0.2]   # recall-tilt values to try if detection is the weak link

# GENERIC secondary support models — small models that each claim an unclaimed metric term on top of the
# inference base. (agent-kind, description). Append new ones here for future terms; nothing else changes.
SECONDARY_MODELS = [
    ("div-model", "forest/logistic sister classifier for the division term"),
    ("deep-sister", "deep sister model via pretrained detector features"),
]


def _busy() -> bool:
    try:
        with urllib.request.urlopen(f"{TRAIN_SERVICE}/api/board", timeout=4) as r:
            q = json.loads(r.read()).get("queue", {})
        return int(q.get("running_count", 0)) + int(q.get("queued_count", 0)) > 0
    except Exception:  # noqa: BLE001
        return True


def _pending_scores() -> int:
    import sqlite3
    try:
        c = sqlite3.connect(board.DB, timeout=5)
        n = c.execute("SELECT count(*) FROM questions WHERE kind='score' "
                      "AND status IN ('open','holding','claimed')").fetchone()[0]
        c.close()
        return n
    except Exception:  # noqa: BLE001
        return 0


def _open_count(kind: str) -> int:
    """How many questions of `kind` are still open/claimed/holding (so we don't re-enqueue duplicates)."""
    import sqlite3
    try:
        c = sqlite3.connect(board.DB, timeout=5)
        n = c.execute("SELECT count(*) FROM questions WHERE kind=? "
                      "AND status IN ('open','holding','claimed')", (kind,)).fetchone()[0]
        c.close()
        return n
    except Exception:  # noqa: BLE001
        return 1  # on error, assume one exists → don't pile up


def _tried() -> dict:
    """method → cv for every experiment already in the journal (so we never repeat)."""
    return {e.get("change"): e.get("cv") for e in ledger.entries() if e.get("change")}


_STRENGTH_KEYS = ("shift_range", "range", "sigma", "smin", "smax")


def _scale_aug(augment, mult):
    """Scale an aug's strength param(s) by `mult` — the data-driven tuning move (push what helped)."""
    out = []
    for e in augment:
        e2 = dict(e)
        for k in _STRENGTH_KEYS:
            if isinstance(e2.get(k), (int, float)):
                e2[k] = round(e2[k] * mult, 4) if k != "smin" else round(1 - (1 - e2[k]) * mult, 4)
        out.append(e2)
    return out


def _enqueue(name, cfg, did):
    board.add("C", "aug-ablation", f"orchestrator: {did} ({cfg})",
              {"config": cfg, "description": did, "approved": True})
    return ("done", {"config": cfg, "name": name}, "all",
            f"[orchestrator] NEXT EXPERIMENT (no leader needed): {name} → {cfg}. {did}. Enqueued (approved).")


def drive(q, worker):
    """One deterministic decision → enqueue the next experiment, or hold/escalate.
    OPTIONAL spec: dry_run (bool) — report the busy/pending state WITHOUT enqueuing anything."""
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    if spec.get("dry_run"):
        return ("holding", {"dry_run": True, "busy": _busy(), "pending_scores": _pending_scores()}, "all",
                f"[{worker}] orchestrator [dry-run]: would decide the next experiment (no enqueue).")
    if _busy() or _pending_scores() > 0:
        return ("holding", {}, "all", f"[{worker}] orchestrator holding: GPU busy or scores pending (one at a time).")
    tried = _tried()
    # weakest link + the REAL metrics of the best scored run — the data the orchestrator reasons from
    lever, bm = "", {}
    try:
        from . import metric
        runs = metric._scored_runs()
        if runs:
            best = max(runs, key=lambda r: r["metrics"].get("official_score", r["metrics"].get("adj_edge_jaccard", 0.0)))
            bm = best["metrics"]
            lever = preanalysis.next_lever(bm)
    except Exception:  # noqa: BLE001
        pass
    bm_get = bm.get  # real-number accessor for the data-driven, metric-targeted moves below

    # A. untried aug singles
    for aug, spec in AUG_MENU.items():
        name = f"auto_{aug}"
        if name not in tried:
            cfg = config_gen.make(BASE, name, augment=spec, split=SCREEN,
                                  purpose=f"orchestrator single-aug ablation: {aug} (weakest link: {lever[:60]})")
            return _enqueue(name, cfg, f"single aug '{aug}' (isolated)")

    # B. mix the singles that beat the no-aug baseline (DATA-DRIVEN: only mix what actually helped)
    base_cv = tried.get("auto_no_aug") or tried.get("00_no_aug")
    winners = [a for a in AUG_MENU
               if isinstance(tried.get(f"auto_{a}"), (int, float)) and isinstance(base_cv, (int, float))
               and tried[f"auto_{a}"] > base_cv]
    winners.sort(key=lambda a: tried[f"auto_{a}"], reverse=True)
    if len(winners) >= 2:
        top = winners[:2]
        name = "auto_mix_" + "_".join(top)
        if name not in tried:
            mix = [e for a in top for e in AUG_MENU[a]]
            cfg = config_gen.make(BASE, name, augment=mix, split=SCREEN,
                                  purpose=f"orchestrator mix of winners {top}")
            return _enqueue(name, cfg, f"mix winners {top} (both beat baseline in the data)")

    # D. DATA-DRIVEN STRENGTH TUNE — the super-agent never sees this; it comes from the REAL result:
    #    the single aug that helped MOST gets pushed stronger (a rejected-at-1x aug may win at a different mag).
    if winners:
        best_aug = winners[0]
        for mult, tag in ((1.6, "up"), (0.5, "down")):
            name = f"auto_{best_aug}_{tag}"
            if name not in tried:
                scaled = _scale_aug(AUG_MENU[best_aug], mult)
                cfg = config_gen.make(BASE, name, augment=scaled, split=SCREEN,
                                      purpose=f"orchestrator strength-tune {best_aug} x{mult} (it was the top single in the data)")
                return _enqueue(name, cfg, f"strength-tune the top winner '{best_aug}' x{mult} — derived from the results")

    # E. METRIC-TARGETED from the best run's REAL numbers (data the super-agent doesn't watch):
    cr, nr = bm_get("mean_count_ratio"), bm_get("mean_node_recall")
    if nr is not None and nr < 0.95:                         # detection is the bottleneck → recall-tilt sweep
        for nw in DET_NEG_SWEEP:
            name = f"auto_detneg_{str(nw).replace('.', 'p')}"
            if name not in tried:
                cfg = config_gen.make(BASE, name, params={"det_neg_weight": nw}, split=SCREEN,
                                      purpose=f"orchestrator recall-tilt det_neg_weight={nw} (node_recall={nr:.3f} in the data)")
                return _enqueue(name, cfg, f"recall-tilt det_neg_weight={nw} (data: node_recall={nr:.3f})")

    # F. SECONDARY SUPPORT MODELS — GENERIC (not hardcoded to div_J): small models that each claim an
    #    UNCLAIMED metric term on top of the inference base. Today the term is division; future terms
    #    (count-calibration, edge-precision, …) plug in by appending to SECONDARY_MODELS — no rename needed.
    if "secondary_models" not in tried:
        for agent, desc in SECONDARY_MODELS:
            if agent not in tried:
                board.add("S", agent, f"orchestrator (secondary support model): {desc}", {})
        board.add("S", "pipeline-run", "orchestrator: apply the secondary models on the inference base → combined golden CV",
                  {"config": "config/exp/winning_inference_div.yml"})
        tried["secondary_models"] = "queued"
        return ("done", {"secondary_models": [a for a, _ in SECONDARY_MODELS], "term": lever}, "all",
                f"[{worker}] orchestrator: primary levers done → driving the SECONDARY SUPPORT MODELS "
                f"({', '.join(a for a, _ in SECONDARY_MODELS)}) + pipeline-run (apply on inference base → combined golden CV). "
                f"Generic slot — future support models plug in the same way. No leader needed.")
    # G. CPU-ONLY PROGRESS (no GPU, no Claude): once every GPU recipe is queued/done, the Python leader
    #    keeps making progress by gridding the PUBLIC-NOTEBOOK post-proc knobs on the local golden-12 CV.
    #    combo-search is self-perpetuating (persisted state); we only (re)enqueue it when none is open, so
    #    the board never piles up. This is what lets the system advance while the Claude leader is absent.
    if _open_count("combo-search") == 0:
        board.add("S", "combo-search",
                  "orchestrator: grid public-notebook post-proc combos on golden-12 (no GPU, no Claude)", {})
        return ("done", {"state": "driving combo-search (CPU golden-12 grid)"}, "all",
                f"[{worker}] orchestrator: GPU recipes queued → Python leader now driving COMBO-SEARCH "
                f"(golden-12 grid over public-notebook post-proc knobs; no training, no Claude leader). Progress continues.")

    # everything the deterministic recipes cover is running/queued → idle-safe (not an escalation storm)
    return ("done", {"state": "all deterministic recipes queued; combo-search running"}, "all",
            f"[{worker}] orchestrator: all levers (GPU recipes + combo-search) are queued/running. "
            f"Idle until results land or a genuinely new lever is registered.")
