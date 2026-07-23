"""pipeline-completeness — proves the fleet is SELF-SUFFICIENT: for every competition MODALITY, does an agent
exist for every stage of the canonical end-to-end pipeline (onboard → understand → CV → features/model → the
grandmaster tricks → ensemble → calibrate → post → submit)? Any stage with no covering agent is a GAP that
would force ad-hoc code — i.e. a place where "start a new comp and finish with OUR agents only" would break.

This is orthogonal to coverage-audit (which checks every agent is IN a pack). Here we check every pipeline
STAGE is FILLED, per modality. Run it after absorbing new capabilities to confirm 100% lossless coverage and
to pinpoint exactly what to build next. Pure/registry-driven; a BaseAgent with its own data-wise test.
"""
from __future__ import annotations
from .base import BaseAgent

# canonical end-to-end stages every competition pipeline needs
STAGES = ["onboard", "understand", "cv", "features", "model", "gm_tricks", "ensemble", "calibrate", "post", "submit"]

# per-modality: stage -> list of fleet agents that CAN fill it (any-of). Absorbed GM tricks slot into gm_tricks.
PIPELINE = {
    "tabular": {
        "onboard": ["comp-onboard"], "understand": ["tab-profile", "eda-stats", "data-audit"],
        "cv": ["cv-build", "split-build", "adversarial-val"], "features": ["tab-fe", "feature-select", "target-transform"],
        "model": ["tab-train", "tab-nn-train", "tab-autobaseline", "automl-oof-factory"],
        "gm_tricks": ["train-tricks", "pseudo-label", "shift-adapt"], "ensemble": ["tab-stack", "blend-optimize", "ensemble", "oof-diversity-prune"],
        "calibrate": ["calibrate"], "post": ["post-optimize", "optimized-rounder"], "submit": ["submission-build", "submit-verify"],
    },
    "vision": {
        "onboard": ["comp-onboard"], "understand": ["eda-stats", "data-audit"], "cv": ["cv-build", "split-build"],
        "features": ["aug-find", "domain-match"], "model": ["detector-transfer", "nnunet-segmentation-runner", "arch-builder"],
        "gm_tricks": ["train-tricks", "gan-train", "pseudo-label"], "ensemble": ["wbf-fusion", "snapshot-average", "multi-tta", "ensemble"],
        "calibrate": ["calibrate"], "post": ["post-optimize"], "submit": ["submission-build", "submit-verify"],
    },
    "detection_tracking": {
        "onboard": ["comp-onboard"], "understand": ["eda-stats", "detect-quality"], "cv": ["cv-build", "split-build"],
        "features": ["aug-find", "flow-gt-build"], "model": ["detector-transfer", "detector-arch-search", "gnn-link-train", "mh-ilp"],
        "gm_tricks": ["train-tricks", "temporal-audit"], "ensemble": ["wbf-fusion", "snapshot-average", "detector-select", "tracker-consensus"],
        "calibrate": ["calibrate"], "post": ["post-optimize", "tracker-postproc"], "submit": ["submission-build", "submit-verify"],
    },
    "nlp_llm": {
        "onboard": ["comp-onboard"], "understand": ["eda-stats"], "cv": ["cv-build", "split-build"],
        "features": ["llm-retrieve-rerank", "template-retrieval-reranker"], "model": ["llm-finetune", "lora-train", "llm-infer"],
        "gm_tricks": ["ttt-transductive-finetune", "noisy-label-cleaner"], "ensemble": ["self-consistency-aggregator", "mbr-consensus-selector", "infer-cascade"],
        "calibrate": ["risk-abstain-gate"], "post": ["runtime-budget-router"], "submit": ["submission-build", "submit-verify"],
    },
    "timeseries_forecast": {
        "onboard": ["comp-onboard"], "understand": ["tab-profile", "eda-stats"], "cv": ["online-walk-forward-retrainer", "cv-build"],
        "features": ["tab-fe", "calendar-holiday-fe", "ts-decompose-forecaster"], "model": ["tab-train", "forecast-trend-extrapolator", "forecast-drivers-then-derive"],
        "gm_tricks": ["train-tricks", "label-lag-anchor-blend"], "ensemble": ["tab-stack", "distributional-metric-recalibrator"],
        "calibrate": ["calibrate", "distributional-metric-recalibrator"], "post": ["hierarchy-consistency-postproc"], "submit": ["submission-build", "submit-verify"],
    },
    "reasoning_code": {
        "onboard": ["comp-onboard"], "understand": ["eda-stats"], "cv": ["cv-build"], "features": ["program-synthesis-data-generator"],
        "model": ["program-search", "ttc", "code-repair-agent"], "gm_tricks": ["ttt-transductive-finetune"],
        "ensemble": ["self-consistency-aggregator", "ttc"], "calibrate": ["risk-abstain-gate"], "post": ["program-golf-search"],
        "submit": ["submission-build", "submit-verify"],
    },
    "agentic": {
        "onboard": ["comp-onboard"], "understand": ["agent-env"], "cv": ["cv-build"], "features": ["agent-env"],
        "model": ["agent-policy"], "gm_tricks": ["best-of-n-diversity-allocator"], "ensemble": ["agent-eval"],
        "calibrate": ["risk-abstain-gate"], "post": ["post-optimize"], "submit": ["submission-build", "submit-verify"],
    },
}


def audit(handlers):
    """For each modality, which STAGES are FILLED (≥1 registered agent) vs GAP (none). Returns per-modality
    {stage: [present agents]} + gaps + a self_sufficient flag (all stages filled)."""
    H = set(handlers or [])
    out = {}
    for mod, stages in PIPELINE.items():
        filled, gaps = {}, []
        for stage in STAGES:
            cands = stages.get(stage, [])
            present = [a for a in cands if a in H]
            filled[stage] = present
            if not present:
                gaps.append(stage)
        out[mod] = {"filled": filled, "gaps": gaps, "self_sufficient": not gaps,
                    "coverage": round((len(STAGES) - len(gaps)) / len(STAGES), 3)}
    return out


class PipelineCompleteness(BaseAgent):
    name = "pipeline-completeness"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        from . import HANDLERS
        rep = audit(list(HANDLERS))
        self.save_state({"completeness": rep})
        rows = "\n".join(
            f"| {mod} | {int(rep[mod]['coverage']*100)}% | {'✅ self-sufficient' if rep[mod]['self_sufficient'] else '❌ gaps: ' + ', '.join(rep[mod]['gaps'])} |"
            for mod in rep)
        n_ok = sum(1 for m in rep.values() if m["self_sufficient"])
        msg = (f"[{worker}] **PIPELINE-COMPLETENESS** · can a NEW comp finish with OUR agents only?\n"
               f"| modality | coverage | verdict |\n|:-|--:|:-|\n{rows}\n"
               f"→ {n_ok}/{len(rep)} modalities fully self-sufficient (every onboard→submit stage has an agent)")
        self.log(summary=f"pipeline-completeness: {n_ok}/{len(rep)} modalities self-sufficient",
                 detail="per-modality end-to-end stage coverage; gaps = where ad-hoc code would be forced",
                 kind="verdict", recommendation="fill any gap stage with a reusable agent → 100% lossless, comp finishes with our agents only")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"completeness": rep, "self_sufficient": n_ok, "total": len(rep)}, msg, to="leader")


_AGENT = PipelineCompleteness()


def run(q, worker):
    return _AGENT.run(q, worker)
