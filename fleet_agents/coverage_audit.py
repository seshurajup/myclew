"""coverage-audit — the live map of the whole fleet: every registered agent placed in its RESPECTIVE pack,
with the biohub-specific agents grouped as their own 3D+time reference pack (kept separate from the reusable
generic core and the new modality packs). Writes docs/agent_coverage.md. Any agent not covered by a pack is
surfaced as "UNCLASSIFIED" so nothing silently drifts.
"""
from __future__ import annotations
from .base import BaseAgent, COMP

# ---- BIOHUB — ONLY the truly biology/comp-locked agents (cell division, the biohub tracker/detector head,
# the biohub metric). The generic capabilities that used to live here are split into reusable packs below. ----
BIOHUB = {
    "tracker-train", "tracker-predict", "tracker-postproc", "tracker-select", "tracker-consensus",
    "center-train", "combined-train", "lora-train", "lora-validate",
    "division", "division-rescue", "div-model", "deep-sister", "stage-1-div", "stage-dynamics",
    "detect-quality", "ext-label-stats", "official-score", "official-conformance", "full-cv-baseline",
    "pattern-tune", "lever-hunt", "div-temporal-feas", "psf-deconv",
}
# ---- generic CORE (reusable across all comps — math/xai/governance/research/submission/orchestration) ----
CORE = {
    "math-master", "xai", "ledger", "insights", "feasibility-gate", "feasibility-map",
    "orchestrate", "campaign", "improve-loop", "journey-status",
    "kaggle-scout", "research-search", "deep-research", "lit-search", "prior-art", "paper-research",
    "paper-verify", "recipe-adopt", "trick-extractor", "trick-gate", "decision-audit", "git-track", "heal",
    "guard", "plan-ingest", "notes-sync", "notebook-sync", "lb-sync", "scoreboard", "beat-bar", "learn",
    "eda-stats", "data-audit", "adversarial-val", "split-build", "cv-lb-calibrate", "scorer", "score",
    "metrics-report", "baseline", "single-model-tune", "arch-search", "ensemble", "component-graft",
    "distill", "compress-select", "keyframe", "quantize", "post-proc", "submission-build", "submit-verify",
    "submit-guard", "nb-preflight", "smoke", "train-monitor", "gpu-best-practices", "perf-choice", "analysis",
    "pre-analysis", "post-analysis", "pipeline", "config-gen", "comp-onboard", "coverage-audit", "setup-env", "pipeline-completeness",
    # promoted OUT of biohub — generic, reusable everywhere:
    "aug-find", "aug-ablation", "arch-probe", "cv-build", "verify-cv", "reproduce-score", "pipeline-run",
    "combine-winners", "ablate-best", "block-synth",
    # GENERIC cross-comp ONNX tool — export/verify/cost/quantize for ANY comp shipping a model (offline budget)
    "onnx",
    # GENERIC context-management primitive — offload/truncate large tool/worker output, keep the thread compact
    "context-offload",
}
# ---- NEW reusable modality/technique packs ----
PACKS = {
    # split OUT of biohub → reusable for ANY detection/tracking/assignment/video competition:
    "Detection & Tracking": {"mh-ilp", "linking", "link-tune", "detector-arch-search", "detector-select",
                             "det-sweep", "flow-gt-build", "gnn-link-train", "gnn-probe", "saliency-detect",
                             "frozen-exploit", "temporal-audit", "detector-transfer",
                             "gaussian-heatmap-encoder", "volumetric-patch-inference", "heatmap-peak-decoder"},
    # split OUT of biohub → reusable architecture/config search for any comp:
    "Arch/Config search": {"arch-builder", "arch-catalog", "layer-grow", "combo-search", "config-ablate",
                           "fullconfig-search", "public-config", "best-config"},
    # split OUT of biohub → reusable external-data transfer/matching:
    "External-data transfer": {"ext-transfer", "sample-match", "box-sample", "domain-match"},
    "Tabular": {"tab-profile", "tab-fe", "tab-train", "tab-stack", "tab-autobaseline", "tab-nn-train",
                "synth-artifact-fe", "oof-diversity-prune", "feature-select", "residual-boost",
                "full-retrain-calibrator", "knn-feature", "ae-latent-view", "gp-symbolic-feature",
                "automl-oof-factory", "geospatial-fe", "shift-adapt"},
    "GM toolkit": {"pseudo-label", "blend-optimize", "post-optimize", "calibrate", "target-transform"},
    # winner-standard training-loop primitives (EMA/SWA/mixup/cutmix/label-smoothing/focal/SAM/ArcFace):
    "Training tricks": {"train-tricks", "sr-bf16-optimizer"},
    # 2026 research-frontier techniques the winners haven't published yet (reusable across all modalities):
    "2026 frontier": {"muon-optimizer", "conformal-predict", "schedule-free", "dora-adapt", "hardware-tune"},
    # low-bit QAT — ternary/int4 fake-quant + STE training (fits bigger models on 5090/T4); grounded in BitNet b1.58:
    "Compression/Quantization": {"lowbit-qat"},
    "Gap toolkit": {"subset-classifier-router", "analysis-by-synthesis-refiner", "checkpoint-merger",
                    "constrained-label-assignment", "lb-shift-prober"},
    "Forecast/Finance/Sports": {"ts-decompose-forecaster", "forecast-trend-extrapolator", "rating-systems",
                                "market-odds-blend", "outcome-sharpen", "portfolio-position-sizer",
                                "online-walk-forward-retrainer", "label-lag-anchor-blend",
                                "forecast-drivers-then-derive", "distributional-metric-recalibrator"},
    "Domain FE": {"fin-ta-feature-library", "imu-feature-engineer", "molecular-featurizer",
                  "invariance-feature-normalizer", "calendar-holiday-fe", "hierarchy-consistency-postproc",
                  "knn-label-transfer", "linear-constraint-projector", "temporal-segment-decoder",
                  "annotation-error-corrector", "quaternion-imu-features"},
    "Optimization": {"combinatorial-local-search", "population-diversity-manager", "batched-oracle-search-harness",
                     "geometric-packing-optimizer", "gpu-relaxation-solver", "best-of-n-diversity-allocator"},
    "Training heads/regularizers": {"deep-supervision", "sed-attention-pool", "awp-perturb",
                                    "masked-sequence-norm", "masked-sequence-pool", "class-balance-sampler"},
    "Inference tricks": {"wbf-fusion", "snapshot-average", "multi-tta"},
    # AUDIO modality pack — grounded in top-5 of birdclef-2024/2023/2021 + bengaliai-speech + freesound-2019
    # (docs/audio_pack_grounded.md). SED head (sed-attention-pool) + class-balance-sampler already cover the
    # non-audio-specific pieces; this pack is the genuinely-missing audio front-end/aug/TTA/backbone.
    "Audio": {"audio-melspec-fe", "audio-augment", "audio-crop-tta", "audio-backbone", "audio-train", "audio-infer"},
    # GRAPH/GNN modality pack — grounded in top-5 of predict-ai-model-runtime + stanford-covid-vaccine +
    # champs-scalar-coupling (docs/graph_pack_grounded.md). gnn-link-train/gnn-probe already cover the
    # LINK-PREDICTION specific pieces (biohub linker); this pack is the genuinely-missing GENERAL graph
    # message-passing / feature-extraction+PE / readout-pooling. Pure torch (no torch_geometric dep).
    "Graph": {"graph-message-passing", "graph-feature-extractor", "graph-readout"},
    # MULTIMODAL modality pack — grounded in top-5 of petfinder-pawpularity-score (image+tabular) +
    # shopee-product-matching (image+text) + ariel-data-challenge-2024/2025 (signal+metadata)
    # (docs/multimodal_pack_grounded.md). LATE/prediction-level fusion is already covered by
    # ensemble/blend-optimize/infer-cascade/tab-stack/checkpoint-merger; this pack is the genuinely-missing
    # FEATURE/MODEL-level fusion (combine image+text+tabular embeddings INSIDE one model) + modality-dropout.
    "Multimodal": {"multimodal-fusion", "modality-encoder-adapter", "modality-dropout"},
    # VIDEO modality pack — grounded in top-5 of deepfake-detection-challenge + dfl-bundesliga-data-shootout +
    # nfl-player-contact-detection + nfl-impact-detection + youtube8m-2019 (docs/video_pack_grounded.md).
    # temporal-segment-decoder (per-frame→segments), masked-sequence-pool (mask-aware pooling), multi-tta and the
    # Detection&Tracking ROI detectors already cover the non-video-specific pieces; this pack is the genuinely-
    # missing FRAME/CLIP sampling + learnable TEMPORAL aggregation (TSM/1D-conv/GRU/attention) + motion features.
    "Video": {"video-frame-sampler", "video-temporal-aggregator", "video-motion-features"},
    "Agentic": {"agent-env", "agent-policy", "agent-eval"},
    "Prompt-program": {"skill-build", "agent-author", "agent-package", "agent-config-eval", "prompt-optimize",
                       "dspy-prompt-optimize", "prompt-metric", "prompt-dataset", "style-fingerprint",
                       "harness-opt-gate"},
    "LLM": {"llm-finetune", "llm-infer", "llm-eval", "llm-retrieve-rerank", "tir-executor", "infer-cascade",
            "self-consistency-aggregator", "consensus-early-stop", "budget-aware-inference-scheduler",
            "sample-pool-simulator", "risk-abstain-gate", "llm-synthetic-drill-generator",
            "ttt-transductive-finetune", "trainable-trace-auditor", "vlm-pdf-corpus-miner", "mbr-consensus-selector",
            "noisy-label-cleaner", "runtime-budget-router", "template-retrieval-reranker",
            "mtp-speculative-decode", "kv-cache-longctx", "moe-inference-cost",
            "llama2-infer", "llmc-train", "coworker-backend"},
    "Reasoning/Code": {"program-search", "program-golf-search", "program-synthesis-data-generator", "ttc",
                       "code-repair-agent", "llm-judge-attacker", "expression-search", "code-compress-optimizer",
                       "sprt-spsa-tuner", "lb-formula-prober", "fast-sim"},
    # GRID-REASONING / network-golf (ARC-AGI-ONNX, neurogolf-2026) — deterministic ONNX-golf TOOLS distilled
    # from the mined winners (patterns.md idioms + emit/verify/official-cost + rewrite-first worker context).
    # No LLM in these; the live researcher/leader brains drive them via fleet_dispatch.
    "Grid-reasoning (ONNX-golf)": {"arc-idioms", "arc-onnx-golf", "arc-worker-context"},
    "Vision/3D-seg": {"sdf-regression-loss", "topology-aware-loss", "keypoint-match-verifier",
                      "grid-rectification-unwarp", "density-regression-head", "trajectory-forecaster",
                      "heteroscedastic-uncertainty-head", "geometric-spatial-augmentor", "nnunet-segmentation-runner",
                      "foundation-3d-matcher", "region-decompose-router", "dicom-metadata-estimator", "chess-search-engine",
                      "nnue-trainer", "gan-train"},
    "Meta/Mining": {"gm-writeup-mine", "github-solution-mine", "solution-adopt", "gm-repo-distill", "binary-size-compressor", "kaggle-modality", "sub-journal"},
    # cross-cutting SUBMISSION pipeline — packages a best model → self-contained offline notebook → push → submit
    # → read PUBLIC+PRIVATE LB. Reusable on every new best model, for any offline code (notebook) competition.
    "Submission": {"kaggle-submit"},
    # METRIC/VALIDATION SECURITY — adversarial probes of the SCORING itself: does the official metric have a
    # degeneracy a top LB score is riding? guards our own CV against a broken metric. Reusable for any comp.
    "Metric/Validation security": {"metric-probe"},
}


def audit(handlers):
    placed = {}
    for name in (handlers or []):
        if name in BIOHUB:
            placed.setdefault("BIOHUB (3D+time)", []).append(name)
        elif name in CORE:
            placed.setdefault("Generic CORE", []).append(name)
        else:
            for pack, members in PACKS.items():
                if name in members:
                    placed.setdefault(pack, []).append(name); break
            else:
                placed.setdefault("UNCLASSIFIED", []).append(name)
    return placed


def write_matrix(placed, total):
    order = ["Generic CORE", "BIOHUB (3D+time)", "Detection & Tracking", "Arch/Config search",
             "External-data transfer", "Tabular", "GM toolkit", "Training tricks", "Gap toolkit",
             "Forecast/Finance/Sports", "Domain FE", "Optimization", "2026 frontier", "Compression/Quantization", "Training heads/regularizers", "Inference tricks", "Audio", "Graph", "Multimodal", "Video", "Agentic", "Prompt-program",
             "LLM", "Reasoning/Code", "Grid-reasoning (ONNX-golf)", "Vision/3D-seg", "Meta/Mining", "Submission",
             "Metric/Validation security", "UNCLASSIFIED"]
    lines = ["# Agent Coverage Matrix — every agent in its respective pack",
             "", f"**{total} agents**, auto-classified by `coverage-audit`. BIOHUB agents are kept as their own "
             "3D+time reference pack, separate from the reusable generic core and the new modality packs.", ""]
    for pack in order:
        if pack not in placed:
            continue
        names = sorted(placed[pack])
        lines.append(f"## {pack} ({len(names)})")
        lines.append(", ".join(f"`{n}`" for n in names)); lines.append("")
    p = COMP / "docs" / "agent_coverage.md"; p.write_text("\n".join(lines))
    return str(p)


class CoverageAudit(BaseAgent):
    name = "coverage-audit"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        from . import HANDLERS  # late import to avoid cycle at module load
        placed = audit(list(HANDLERS))
        path = write_matrix(placed, len(HANDLERS))
        counts = {k: len(v) for k, v in placed.items()}
        unclassified = placed.get("UNCLASSIFIED", [])
        msg = (f"coverage-audit: {len(HANDLERS)} agents classified into {len(placed)} packs "
               f"(biohub={counts.get('BIOHUB (3D+time)',0)}, core={counts.get('Generic CORE',0)}); "
               f"{len(unclassified)} unclassified → {path}")
        self.log(msg, kind="finding", recommendation="every agent in its respective pack; keep UNCLASSIFIED at 0")
        return self.done({"counts": counts, "unclassified": unclassified, "matrix": path}, msg)


_AGENT = CoverageAudit()


def run(q, worker):
    return _AGENT.run(q, worker)
