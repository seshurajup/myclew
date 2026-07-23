"""The grandmaster JOURNEY — the ordered experiment progression, as data + a status agent.

Distilled from top-solution write-ups (RSNA breast/aneurysm/trauma 1st-place, Dieter's ASL,
faridrashidi/kaggle-solutions) + our own memory. Every GM solution follows the same skeleton:
CV frozen day-one → dumb baseline + anatomy → tune the dominant lever → aug ablation (+0.01 ledger,
one change at a time) → architecture → linking → division → ensemble → post-proc/submit; ensemble &
post-proc come LAST and are tuned on CV only. This module makes the journey explicit and followable.
"""
from __future__ import annotations

# (stage, name, goal, deterministic_owner, reasoning_at_decision_point)
STAGES = [
    (0, "CV harness & contract", "score every run by the FULL official metric on frozen embryo-disjoint folds",
     "cv-strategy + scorer: build LOEO folds, wire official_score, leak-assert",
     "confirm the split axis matches Kaggle test-gen (embryo-disjoint; golden-12 secondary)"),
    (1, "dumb baseline & anatomy", "one working end-to-end number + decompose WHERE score is lost",
     "baseline: run simplest pipeline, metric-decompose into R_node/R_edge/Q_link/count buckets",
     "name the weakest link, set the stage-2 target"),
    (2, "single-model tuning", "push the dominant lever (edge precision / linking; node recall ~saturated)",
     "single-model-tune: successive-halving knob sweeps, isolate + log CV deltas",
     "pick which knob family to open when the sweep plateaus"),
    (3, "augmentation ablation", "add regularizers that TRANSFER across the 2 embryos (+0.01 ledger)",
     "aug-ablation: one-at-a-time toggle of a PHYSICALLY-VALID menu, keep positive-transfer only",
     "propose the aug menu; the aug-validity-checker forbids frame-skip/ZY-rot/independent-jitter"),
    (4, "architecture probe", "does a different backbone/head beat the tuned model (data levers first)",
     "arch-probe: train candidates under the identical frozen config, CV compare",
     "shortlist candidates worth the GPU (StarDist3D / MONAI variants), kill dead ends"),
    (5, "linking / temporal", "improve edge assignment given good nodes (link gate ~8.5µm; complexity hurts)",
     "linking: ablate geometric vs motion-relink vs learned edge head on fixed detections",
     "decide learned-vs-rule (edge head saturated → likely skip); pick the operating point"),
    (6, "division layer", "capture the 0.1*division-Jaccard WITHOUT flooding FP edges (train-only lever)",
     "division: up-weight the rare class, checkpoint on division_jaccard, guard edge_J",
     "division needs real masks (synthetic → FP flood) — decide the approach, not a blind sweep"),
    (7, "ensemble", "diversity gain, not another single-model tweak (auto-reject regressive unions)",
     "ensemble: fold/seed averaging; union only if full-metric doesn't drop",
     "choose the diversity axis (seed vs backbone vs preprocessing)"),
    (8, "post-proc & submit gate", "squeeze last bit with CV-only knobs; decide whether to spend a slot",
     "post-proc: grid scaling/threshold/count-calibration on CV; strip comments; verify row count",
     "go/no-go: submit only if it beats the 0.885 public bar locally; trust Kaggle over LB"),
]

ANTI_OVERFIT_RULES = [
    "Group by the leak axis (leave-one-embryo-out), decided BEFORE modeling; golden-12 is leaky → secondary only.",
    "Freeze the CV harness day-one; never change the metric or splits mid-competition.",
    "Score the FULL official metric end-to-end, never a component's own loss/recall/F1.",
    "ONE change per experiment, logged as a CV delta (the +0.01 ledger).",
    "Reject any change that helps the train fold but not the held-out embryo (primary overfit signal, ~2 embryos).",
    "Detection tuning does NOT transfer 1:1 to LB (~0.005 optimism); only smoothing/generalizing changes do.",
    "Ensemble & post-proc tuned on CV only, never probed via LB submissions (conserve the daily-5).",
]


def status(q, worker):
    """Report the journey map + which stage we're on (inferred from the ledger), and the next action."""
    from . import ledger
    try:
        st = ledger.summary() or {}
    except Exception:  # noqa: BLE001 — an unreadable/empty ledger must not crash the status report
        st = {}
    done_stages = st.get("stages_touched", []) if isinstance(st, dict) else []
    cur = next((s for s in STAGES if s[0] not in done_stages), STAGES[-1])
    line = (f"[{worker}] JOURNEY: {len(STAGES)} stages, +0.01 CV-delta ledger has {st.get('n', 0)} "
            f"experiment(s) ({st.get('kept', 0)} kept). Current focus → Stage {cur[0]}: {cur[1]} — {cur[2]}. "
            f"Deterministic: {cur[3]}. Claude decides: {cur[4]}.")
    return ("done", {"current_stage": cur[0], "map": [(s[0], s[1]) for s in STAGES], "ledger": st}, "all", line)
