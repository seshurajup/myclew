"""solution-adopt — turn a competition's MINED top-1..5 solutions into an EXECUTABLE adopted workflow: an
ordered list of reusable fleet-agent calls that reproduces what the winners actually did, driven by the
grounded catalog (docs/gm_techniques_grounded.md) + a per-comp technique inventory. ONE agent, reusable
across every comp — it composes the reusable technique-agents (tab-fe, pseudo-label, blend-optimize,
post-optimize, calibrate, target-transform, quantize, ttc, ...), it does not fork them.

Input: a CompConfig (modality×paradigm×task×metric) + optional technique inventory (from gm-writeup-mine +
extraction, or inferred from the metric). Output: the ordered workflow with the WHY for each step (which
top-solution lever it adopts). This is the generalization of recipe-adopt/combine-winners to the full
winning pipeline. The plan is what campaign/orchestrate then executes.
"""
from __future__ import annotations
from .base import BaseAgent
from . import comp_config as CC


def _tech(inv, *keys):
    """True if any keyword appears in the comp's technique inventory (case-insensitive)."""
    blob = " ".join(str(v).lower() for v in (inv or {}).values()) if isinstance(inv, dict) else str(inv or "").lower()
    return any(k in blob for k in keys)


def adopt(cfg: CompConfig, inv=None, with_mining=False):
    """Return the ordered adopted workflow: [{step, agent, params, why}]. Grounded in real 2025-26 winners.
    with_mining: prepend gm-writeup-mine + github-solution-mine so the plan re-grounds itself in fresh top
    solutions before routing (off by default → identical legacy plan)."""
    m = cfg.modality; par = cfg.paradigm; metric = cfg.metric; steps = []

    def add(agent, why, **params):
        steps.append({"step": len(steps) + 1, "agent": agent, "params": params, "why": why})

    add("comp-onboard", "fingerprint comp → CompConfig + route (universal front door)", slug=cfg.slug)
    if with_mining:
        add("gm-writeup-mine", "fetch top-N solution writeups to ground the plan in fresh SOTA", slug=cfg.slug)
        add("github-solution-mine", "harvest winners' published code for reusable modules")

    # ---------------- PREDICTIVE ----------------
    if par == "predictive":
        prof = {"tabular": "tab-profile", "sequence": "tab-profile", "image": "img-profile",
                "video": "vid-profile", "pointcloud": "pc-profile", "volume-time": "detect-quality",
                "text": "llm-profile"}.get(m, "eda-stats")
        add(prof, "fingerprint data (shape/leakage/drift/balance) before modeling — every top solution")
        add("adversarial-val", "train/test drift check → design a leak-free CV matched to the split")
        add("split-build", f"leak-free CV ({cfg.cv_scheme}) — grouped/time as the winners used")

        if m in ("tabular", "sequence"):
            # survival factorization / QWK-regress target engineering (equity, child-mind)
            if metric in ("stratified_concordance_index", "concordance_index"):
                add("target-transform", "equity golden trick: factorize survival into efs-classifier + time-regressor",
                    method="factorize_survival")
            if metric == "quadratic_weighted_kappa":
                add("target-transform", "child-mind lever: regress a continuous latent target then round", method="rank_gauss")
            add("tab-fe", "GM #1 tabular lever: leak-safe OOF pair/n-gram target-enc + digit + row-aggs + interactions")
            add("tab-train", "GBDT trio (LGBM/XGB/CatBoost) + NN diversity, GPU-auto, OOF+test")
            add("pseudo-label", "self-training on confident test rows (child-mind/s5e4/s5e11 lift)")
            add("blend-optimize", "best-of {hill-climb/Caruana/Nelder-Mead/Ridge} over OOF (s5e11/s5e4/equity)")
        elif m in ("image", "video", "pointcloud", "volume-time"):
            add("arch-search", "efficient backbone+head (small>big per czii/byu/rsna); component-graft pretrained")
            add("aug-find", "Mixup/CutMix/TTA/SpecAugment aug policy + EMA (all vision/audio winners)")
            add("img-train" if m == "image" else "tracker-train", "train w/ EMA, deep-supervision, recall-tilted loss for the metric")
            add("pseudo-label", "iterative self-training / Noisy-Student (birdclef 0.87→0.93)")
            if m in ("volume-time", "pointcloud"):
                add("post-optimize", "detection decode: NMS max-pool peak + quantile-threshold on max (byu/czii)", op="quantile_thr")
            add("blend-optimize", "ensemble diverse checkpoints/architectures (best-of-N)")
        elif m == "text":
            add("llm-finetune", "LoRA/QLoRA fine-tune + distill soft-labels from a bigger teacher (KL, T=5)")
            if _tech(inv, "retriev", "rerank", "map@"):
                add("llm-retrieve-rerank", "retrieve-then-rerank cascade (eedi) for ranking metrics")
            add("quantize", "offline 2×T4: W8A8/AWQ/GPTQ + vLLM (the decisive LLM constraint)", mode="w8a8")
            add("llm-infer", "single-token allowed-ids + multi-stage confidence cascade under the time budget")

        # metric-specific post-processing (universal)
        if metric == "quadratic_weighted_kappa":
            add("post-optimize", "optimize QWK rounding thresholds on OOF (child-mind 1st-place)", op="qwk_round")
        elif metric in ("rmse", "rmsle", "mae", "r2", "smape"):
            add("post-optimize", "clip predictions to guard test outliers (s5e4: unclipped→RMSE 177)", op="clip")
        elif metric in ("roc_auc", "logloss", "average_precision", "partial_auc"):
            add("calibrate", "Platt/isotonic/temperature calibration on OOF (equity/rsna)", method="isotonic")

    # ---------------- AGENTIC ----------------
    elif par == "agentic":
        add("agent-env", "onboard env: action space, reward, BUDGET (the real constraint)")
        add("lb-replay-mine", "mine top-team replays (Orbit Wars/Lux) — cheap, high-signal")
        add("imitation-learn", "behavior-clone the strongest replays (lux rank3/4 beat rules fast)")
        add("agent-selfplay", "self-play RL (IMPALA/PPO/xLSTM) + opponent pool + teacher distillation (lux winners)")
        add("agent-policy", "candidate tournament: beat a strong heuristic before fancy search")
        if cfg.task == "attack":
            add("sec-attack", "multi-step tool-attack / dense-exfil construction within budget (JED)")
            add("sec-replay", "replay-mine attack traces")
            add("sec-eval", "offline budget check (~replay limit) before submit")
        add("agent-eval", "OFFLINE budget/reward check before submit (budget decides the score)")

    # ---------------- REASONING ----------------
    elif par == "reasoning":
        add("reason-dsl", "DSL of grid transforms (ARC classical track)")
        add("program-search", "enumerative/neural-guided DAG program search over the DSL")
        add("ttc", "TEST-TIME-TRAINING / active inference (per-task LoRA) + augmentation-voting (AIRV) — the ARC lever")
        add("reason-eval", "exact-match on train demos; ensemble classical + LLM candidates")

    # universal tail
    add("submission-build", "write submission in the CompConfig schema")
    add("nb-preflight", "verify offline-install + paths locally before any Kaggle push")
    add("submit-verify", "prove the pipeline end-to-end on the real data")
    add("beat-bar", "gate: must clear the best public/GM solution before claiming quality")
    add("submit-guard", "HUMAN-gated submit (never auto-submit)")
    return steps


class SolutionAdopt(BaseAgent):
    name = "solution-adopt"
    thread = "R"
    kind = "config-gen"

    def run(self, q, worker):
        spec = self.spec(q)
        try:
            cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else CC.CompConfig(
                slug=spec.get("slug", "?"), modality=spec.get("modality", "tabular"),
                paradigm=spec.get("paradigm", "predictive"), task=spec.get("task", "unknown"),
                metric=spec.get("metric", "unknown"), cv_scheme=spec.get("cv_scheme", "kfold"))
        except Exception:  # noqa: BLE001 — malformed config → safe predictive/tabular default
            cfg = CC.CompConfig(slug=spec.get("slug", "?"), modality="tabular", paradigm="predictive",
                                task="unknown", metric="unknown", cv_scheme="kfold")
        wf = adopt(cfg, inv=spec.get("inventory"), with_mining=bool(spec.get("with_mining", False)))
        agents = [s["agent"] for s in wf]
        msg = (f"solution-adopt {cfg.slug}: adopted {len(wf)}-step winning workflow "
               f"[{cfg.modality}/{cfg.paradigm}/{cfg.metric}] → {' → '.join(agents)}")
        self.log(msg, kind="config-gen", recommendation="hand to campaign/orchestrate to execute; each step = a reusable agent")
        return self.done({"workflow": wf, "agents": agents, "n_steps": len(wf)}, msg)


_AGENT = SolutionAdopt()


def run(q, worker):
    return _AGENT.run(q, worker)
