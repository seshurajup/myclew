"""Journey stage agents (S1 baseline, S2 tune, S5 linking, S6 division, S7 ensemble, S8 post-proc).

baseline + single-model-tune run existing configs/drivers deterministically; linking/division/ensemble/
post-proc report the stage plan + the reusable code and escalate the DESIGN decision to the researcher
(these need reasoning + a specific config). Each carries the grounded biohub insight for that stage.
"""
from __future__ import annotations

from pathlib import Path

from . import experiments

COMP = Path(__file__).resolve().parent.parent


def baseline(q, worker):
    """S1 — run the simplest full pipeline (existing detector config), log to the journal."""
    cfg = q["spec"].get("config", "config/loeo_detector.yml")
    qq = dict(q)
    qq["kind"] = "aug-ablation"  # reuse the run_config path (dry-run → runner → journal)
    qq["spec"] = {**q.get("spec", {}), "config": cfg}
    status, result, to, msg = experiments.run_config(qq, worker)
    return (status, result, to, msg.replace("aug-ablation", "STAGE-1 baseline"))


def tune(q, worker):
    return ("escalated", {"driver": "tools/researchpapers/baseline/successive_halving.py"}, "researcher",
            f"[{worker}] STAGE-2 single-model-tune: sweep the dominant lever (edge precision/linking; node "
            f"recall ~saturated at 0.99) via successive_halving.py::run_config on the mini CV, ONE knob at a time. "
            f"Researcher: pick the knob family (det-loss neg_weight / decode topk / count calibration).")


def linking(q, worker):
    return ("escalated", {"reuse": "model_scratch/eval_v3linker.py"}, "researcher",
            f"[{worker}] STAGE-5 linking: ablate geometric vs motion-relink on FIXED detections. Link gate ≈8.5µm "
            f"(NOT 7.0 — 6bba peaks at 8.5); complexity hurts (velocity-Kalman/gap-close score below baseline). "
            f"Reuse eval_v3linker.py. Researcher: learned-vs-rule (edge head saturated → likely skip).")


def division(q, worker):
    return ("escalated", {"anchor": "36-div rich split + golden-12 div_J"}, "researcher",
            f"[{worker}] STAGE-6 division (+0.1 term, TRAIN-ONLY — all hand-rules score 0): up-weight the rare "
            f"division class, checkpoint on division_jaccard, division-oversampled loader; validate on the 36-div "
            f"rich split or golden-12 div_J. Researcher: loss weight (5–20) + oversample ratio (synthetic → FP flood).")


def ensemble(q, worker):
    return ("escalated", {}, "researcher",
            f"[{worker}] STAGE-7 ensemble (LAST): fold/seed averaging; heat-map average BEFORE peak-finding "
            f"(point-union floods FP edges → precision down → WORSE; detectors 72% redundant). Auto-reject any "
            f"union that drops the full metric. Researcher: diversity axis (seed vs backbone vs preprocessing).")


def postproc(q, worker):
    return ("escalated", {"bar": 0.885, "kaggle_submit": "NEVER without human permission"}, "leader",
            f"[{worker}] STAGE-8 post-proc/submit-gate (CV-ONLY): grid count-calibration/threshold on CV "
            f"(NEVER exceed estN — under-predicting gives a score bonus; per-movie topk beats global threshold). "
            f"⛔ DO NOT submit to Kaggle — I only report the local-CV candidate. A Kaggle submission requires "
            f"explicit HUMAN permission; keep everything on local CV until the human takes control.")
