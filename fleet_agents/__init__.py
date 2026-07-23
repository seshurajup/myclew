"""biohub competition adapters for the researchpapers fleet.

Thin DETERMINISTIC modules that wrap THIS competition's existing, config/yml-driven pipeline
(src.cv, src.metric, scripts/train_from_config.py + config/*.yml) and expose it to the reusable
researchpapers fleet framework. Swap this package per competition; the framework stays unchanged.

Interface the framework expects:
  NAME     : str
  SEED     : list of (thread, kind, question, spec_dict)
  HANDLERS : dict  kind -> fn(question, worker) -> (status, result, to, message)
"""
from . import (  # noqa: F401
    adversarial, augfinder, best_config, block_synth, combo_search, config_ablate, config_gen, deep_sister, div_model, cv, dryrun, eda_stats, experiments, ext_label_stats, flow_gt_builder, fullconfig_search, arch_builder, arch_search as _asearch, box_sample as _boxs, sample_match as _smatch, combined_train as _ctrain, perf_choice as _perf, layer_grow as _lgrow, data_audit as _daudit2, gpu_best_practices as _gpu, paper_research as _paper, gnn_link_train as _glt, gnn_probe, xai as _xai, grandmaster, trick_extractor as _trick, trick_gate as _tgate, decision_audit as _daudit, prior_art as _prior, guard, heal as _heal, scoreboard as _sb, insights, journey, tracker_consensus, pipeline as _pipeline, recipe_adopt as _radopt, det_sweep as _dsweep, cv_lb_calibrate as _cvlb, submit_guard as _sguard, beat_bar as _bbar, improve_loop as _iloop, campaign as _camp, saliency_detect as _sdet, link_tune as _ltune, ext_transfer as _extf, domain_match as _dmatch, gan_train as _gan, detector_transfer as _dtrans, setup_env as _setupenv, gm_repo_distill as _gmdistill, pipeline_completeness as _pcomp, pattern_tune as _ptune, math_master as _mm, detector_arch_search as _das, deep_research as _dres, temporal_audit as _taud, lit_search as _lit, mh_ilp as _milp, detector_select as _dsel, tracker_select as _tsel, compress_select as _psel,
    kaggle_scout, ledger, learner, metric, metrics_report, monitor, note, notebook_sync, orchestrator, plan_ingest,
    postanalysis, pipeline_run, preanalysis, public_config, reproduce_score, runner, score_step, submission_build, scorer, smoke, split_build, stage1_div, stages, verify_cv,
    nb_preflight as _nbp, submit_verify as _sv, research_search as _rs, lb_sync as _lbs, frozen_exploit as _fex,
    distill as _dist, component_graft as _cg, keyframe as _kf, quantize as _qz, lowbit_qat as _lbq, paper_verify as _pverify,
    lora_train as _lora, full_cv_baseline as _fcvb, lora_validate as _lval, detect_quality as _dq, stage_dynamics as _sdyn, official_score as _oscore, division_rescue as _drescue,
    tracker_train as _ttrain, center_train as _cdtrain, tracker_predict as _tpredict, tracker_postproc as _tpostp,
    official_conformance as _oconf, git_track as _gtrack, div_temporal_feas as _dtfeas,
    metric_probe as _mprobe, lever_hunt as _lhunt, psf_deconv as _psfd,
    feasibility_gate as _feas,
    comp_onboard as _onboard,
    tab_profile as _tabprof, tab_train as _tabtrain, tab_stack as _tabstack, tab_autobaseline as _tabauto,
    tab_fe as _tabfe,
    agent_env as _agenv, agent_policy as _agpol, agent_eval as _ageval,
    gm_writeup_mine as _gmwm,
    pseudo_label as _pseudo, blend_optimize as _blend, post_optimize as _postopt,
    calibrate as _calib, target_transform as _ttrans,
    solution_adopt as _soladopt,
    prompt_skill_build as _skillb, prompt_agent_author as _agauth, prompt_agent_eval as _ageval2,
    dspy_prompt_pack as _dspyp,
    gap_pack as _gap,
    github_solution_mine as _ghmine,
    domain_feature_pack as _domfe,
    tab_diversity_pack as _tabdiv,
    forecast_sports_pack as _fsp,
    optimization_pack as _opt,
    training_head_pack as _thp,
    inference_tricks_pack as _inf,
    robustness_pack as _rob,
    llm_inference_pack as _llmi,
    mtp_speculative_pack as _mtp, kv_cache_pack as _kvc, moe_inference_pack as _moe,
    moe_quantile_balance as _moeqb,
    sparsity_metrics as _spm,
    shap_emd as _shemd,
    geometric_features as _geo,
    flow_matching as _flow,
    llm_backend as _llmb,
    skill_optimizer as _skopt,
    mup_scaling as _mup,
    gpu_patterns as _gpp,
    deck_builder as _deck,
    video_builder as _vidb,
    task_spec as _tspec,
    turboquant as _tq,
    attention_residual as _attnres,
    latent_moe as _latmoe,
    rswa_attention as _rswa,
    embedding_retrieval as _emb,
    nvfp4_loader as _nvl,
    diffusion_sampler as _diff,
    ui_component as _uic,
    shopify_agent as _shop,
    lightning_tricks as _ltricks,
    santa_agent as _santa,
    finance_pack as _fin,
    reasoning_code_pack as _rcp,
    misc_domain_pack as _misc,
    final_pure_pack as _fpp,
    tab_nn_train as _tabnn,
    heavy_runnable_pack as _hrp,
    llm_pack as _llmp,
    heavy_runnable2_pack as _hr2,
    reasoning_exec_pack as _rep,
    scaffold_pack as _scaf,
    train_tricks_pack as _tricks,
    volumetric_detection_pack as _vdet,
    masked_sequence_pack as _mseq,
    imbalance_sampler_pack as _imbs,
    sr_bf16_optimizer as _srbf,
    quaternion_imu_features as _qimu,
    muon_optimizer as _muon,
    context_offload as _coff,
    harness_opt_gate as _hgate,
    conformal_prediction as _conf,
    schedule_free as _sfree,
    dora_adapter as _dora,
    hardware_tune as _hwtune,
    prompt_metric as _pmetric,
    prompt_dataset as _pdataset,
    style_fingerprint as _styfp,
    audio_pack as _audio,
    audio_train as _audiotrain,
    audio_infer as _audioinfer,
    graph_pack as _graph,
    multimodal_pack as _mmf,
    video_pack as _vid,
    coverage_audit as _covaudit,
    kaggle_modality as _kmod,
    kaggle_submit as _ksubmit,
    sub_journal as _subj,
    onnx_tools as _onnx,
    arc_onnx_golf as _arcgolf, arc_idioms as _arcidioms, arc_worker_context as _arcctx,
    hf_kernels as _hfkernels,
)

NAME = "biohub-cell-tracking"

# agents that ALREADY write their own richer journal entry (rows / decisions / progress) — don't double-log
_SELF_LOGGED = {"pre-analysis", "post-analysis", "aug-ablation", "arch-probe", "score",
                "train-monitor", "metrics-report", "smoke", "reason"}


def on_result(agent, status, result, to, msg):
    """Framework hook — called after EVERY agent runs so ALL agents contribute findings to the journal
    (deduped). Skips holding/routine and the agents that self-log a richer entry."""
    if status == "holding" or agent in _SELF_LOGGED:
        return
    summary = msg.split("] ", 1)[-1].strip() if "] " in msg else msg
    detail = ""
    if isinstance(result, dict):
        detail = "; ".join(f"{k}={v}" for k, v in list(result.items())[:14]
                           if not isinstance(v, (dict, list)))
    kind = "verdict" if status == "escalated" else "finding"
    ledger.log(agent, summary=summary[:220], detail=detail, kind=kind)

# Research questions the fleet seeds — each maps to EXISTING competition code/config.
SEED = [
    # THREAD A — understand the problem
    ("A", "cv-build", "Build the embryo-disjoint mini CV split (leave-one-embryo-out via src.cv)",
     {"k": 2, "mini_per_fold": 8}),
    ("A", "analysis", "Decompose the best scored run into failure buckets (node_recall/edge/division)", {}),
    # THREAD C — augmentations (existing config/aug_ablation/*.yml)
    ("C", "aug-ablation", "Augmentation ablation: noise (config/aug_ablation/noise.yml)",
     {"config": "config/aug_ablation/noise.yml"}),
    ("C", "aug-ablation", "Augmentation ablation: contrast", {"config": "config/aug_ablation/contrast.yml"}),
    ("C", "aug-ablation", "Augmentation ablation: gamma", {"config": "config/aug_ablation/gamma.yml"}),
    ("C", "aug-ablation", "Augmentation ablation: rot90", {"config": "config/aug_ablation/rot90.yml"}),
    # THREAD B — architecture / detector variants (existing model_scratch/config/*.yml)
    ("B", "arch-probe", "Detector variant: v2_recall (model_scratch/config/exp_det_v2_recall.yml)",
     {"config": "model_scratch/config/exp_det_v2_recall.yml"}),
    ("B", "arch-probe", "Detector variant: focal", {"config": "model_scratch/config/exp_det_focal.yml"}),
    ("B", "arch-probe", "Detector variant: v3_stdfocal", {"config": "model_scratch/config/exp_det_v3_stdfocal.yml"}),
    # trainer's reliability check, done in Python
    ("A", "guard", "Validate the latest training job's liveness (succeeded, finished, plausible)", {}),
    # live GPU/CPU + log watchdog over any running training (catches hangs like the 32-min dataloader deadlock)
    ("A", "train-monitor", "Watch the running training: GPU/CPU + log freshness; kill+escalate on a hang", {}),
    # the grandmaster journey + data-grounded aug menu + the experiment ledger
    ("S", "journey-status", "Where are we in the grandmaster journey (which stage next)?", {}),
    ("C", "aug-find", "Derive the physically-valid augmentation menu FROM THE DATA", {}),
    ("S", "ledger", "Report the experiment ledger — the full experiment history", {}),
    ("S", "notes-sync", "Ingest structured research notes (docs/research_notes/*.md) into the journal", {}),
    # the remaining journey-stage agents + pure-analysis agents + the learner
    ("A", "eda-stats", "Report the data fingerprint (density/stage/motion/divisions) from EDA outputs", {}),
    ("A", "adversarial-val", "Confirm the CV axis is embryo-disjoint (leak-safe)", {}),
    ("A", "scorer", "Report the CV trajectory (official_score/golden_cv) across runs", {}),
    ("S1", "baseline", "Run the dumb baseline (simplest full pipeline) and log it", {"config": "config/loeo_detector.yml"}),
    ("S2", "single-model-tune", "Tune the dominant lever (edge precision) via successive-halving", {}),
    ("S5", "linking", "Ablate linking (geometric vs motion-relink) on fixed detections", {}),
    ("S6", "division", "Division head (+0.1 term, train-only) — up-weight rare class", {}),
    ("S7", "ensemble", "Ensemble LAST (fold/seed avg; reject regressive unions)", {}),
    ("S8", "post-proc", "Post-proc + submit-gate (CV-only; beat 0.885 bar)", {}),
    ("S", "learn", "Learner ready — capture new findings as Pattern-B .py + .learning lessons", {}),
    ("S", "kaggle-scout", "Pull top public notebooks + leaderboard via the Kaggle CLI (don't miss)", {}),
    ("A", "pre-analysis", "Diagnose the current state → recommend the next lever (before experiments)", {}),
    ("A", "post-analysis", "Verdict after an experiment: delta, transfer, kept/rejected", {}),
    # SELF-DRIVING: these three let the fleet run the whole loop with NO Claude leader/researcher
    ("S", "orchestrate", "Pick + enqueue the next experiment from the weakest link (replaces the leader)", {}),
    ("S", "config-gen", "Generate a config from a spec (replaces the researcher's yml authoring)", {}),
    ("A", "split-build", "Build a VALIDATED, leak-checked CV split (replaces researcher split work)", {}),
    ("S", "insights", "Refresh docs/INSIGHTS.md — the super-agent handoff report (complete work + direction)", {}),
    ("S", "context-offload", "CONTEXT MGMT: offload a large tool/worker output to output/run_artifacts/ and return a compact summary + path (keeps the board thread/inbox small)", {"label": "demo", "text": "seed"}),
    ("S", "harness-opt-gate", "PROMPT/HARNESS SELF-OPT GATE: blind-holdout keep-if-improves — accept a prompt/skill/harness edit iff combined (train+holdout) pass-count strictly improves over baseline (optimizer sees only train; holdout is blind). Prompt-optimization side ONLY — NOT for ML experiments (those use lever-hunt/feasibility-gate/Wilcoxon)", {}),
    ("S", "notebook-sync", "DAILY: pull new top public notebooks + extract learnings (do not miss the floor)", {}),
    ("S", "pipeline-run", "Run a config/exp end-to-end: inference base + div-model + golden-12 score (fast)", {"config":"config/exp/winning_inference_div.yml"}),
    ("S", "public-config", "Write one config/exp/public/*.yml per public notebook (full coverage)", {}),
    ("S", "best-config", "Assemble the best inference config from public learnings (Part A, no training)", {}),
    ("S", "deep-sister", "Train the 3D-CNN sister detector on image patches (Part B, div_J)", {}),
    ("S", "div-model", "Train the small XGBoost sister/division classifier for div_J (Part B)", {}),
    ("S", "stage-1-div", "STAGE-1 div_J verdict: run div-model on 36-event predicted-node split", {"threshold": 0.9}),
    ("S", "combo-search", "Grid public-notebook post-proc combos on golden-12 (no train, no Claude) → keep best", {}),
    ("S", "fullconfig-search", "WIDE 8-axis search over the yaroslav-v4 full ILP config (0.8803 base) → beat public", {}),
    ("S", "config-ablate", "Leave-one-block-out ablation of the yaroslav-v4 config → the load-bearing map", {}),
    ("A", "ext-label-stats", "Inventory external Zebrahub dense labels (links/divisions/flow prior) — hengck23 recipe", {}),
    ("S", "flow-gt-build", "Build per-node flow (dz,dy,dx) + division GT from external tracks → affinity supervision", {}),
    ("A", "gpu-best-practices", "5090/Blackwell + precision best practices (acc×speed); free-wins + arch-search candidates", {}),
    ("A", "paper-research", "Mine recent architecture innovations (accuracy+SPEED); feed proven candidates to arch-search", {}),
    ("A", "sample-match", "Profile the author's crop sampling/labelling scheme; GATE external data against it", {}),
    ("A", "perf-choice", "Benchmark compute backends (per-node/vectorised/GPU) → recommend fastest; never re-make the slow choice", {}),
    ("A", "box-sample", "Density-match dense external embryos to competition crops (sample vertex boxes)", {}),
    ("S", "domain-match", "REUSABLE src→target domain matching (feature CORAL/OT/quantile + image spectrum/LCN/histmatch + LEARNED adversarial mapper); drives adv-AUC→0.5", {}),
    ("S", "gan-train", "REUSABLE GPU adversarial image trainer (translate/augment): residual Gen vs patch Disc + structure guard; domain adaptation, synthetic aug, style transfer", {}),
    ("S", "train-tricks", "REUSABLE GM training-tricks pack (torch/CUDA): EMA·SWA·mixup·cutmix·label-smoothing·focal(bin+multi)·SAM·sub-center-ArcFace — the winner-standard training-loop primitives (from 179 repos)", {}),
    ("S", "pipeline-completeness", "PROVE self-sufficiency: for every comp modality, does an agent fill every onboard→submit stage? flags gaps that force ad-hoc code", {}),
    ("S", "gm-repo-distill", "REUSABLE self-improving loop over GM winner GitHub repos: clone→scan techniques→check fleet coverage→delete→manifest; finds capability GAPS to build", {}),
    ("S", "setup-env", "REUSABLE dependency manager for lib-gated agents (mast3r/autogluon/rdkit/nnunet...); ABI-safe install (numpy 2.4.6 + torch cu128 preserved); dry-run default", {}),
    ("S", "detector-transfer", "REUSABLE strong 3D-UNet detector + MULTI-SEED per-embryo transfer eval (mean±std, significance) + self-training; kills 1-seed noise verdicts", {}),
    ("A", "data-audit", "MEASURE + CORRECT the training data scale/outliers before training (per-embryo normalise)", {}),
    ("A", "arch-builder", "DERIVE the model architecture from data analysis (radius/k/layers/head-weights, each justified)", {}),
    ("A", "xai", "XAI/interpretability: saliency·occlusion·probe·attention — SEE what the model learned (no assumptions)", {}),
    ("B", "combined-train", "Train on box-sampled external + competition directly → division AP on held-out golden", {}),
    ("B", "layer-grow", "Choose network DEPTH layer-by-layer, each layer proven by training + XAI-validated", {}),
    ("B", "arch-search", "PROVE arch-builder search space by TRAINING each candidate → data-best (answers 8-heads)", {}),
    ("A", "arch-catalog", "QUERYABLE grounded modern-technique catalog (MoE/encoder-free/KV-cache/component-graft + int8/QAT-ternary/FP4-hard-constraint + hardware-tune/trust-region-self-train/speculative/GM-tricks + LOEO gate). catalog()/propose(target_profile) returns applicable techniques + MEASURED constraints for a target (hardware/data-regime/bit-budget/context) — e.g. 'T4-offline → int8 not FP4 + component-graft + gate-on-LOEO'. Grounded in docs/lowbit_*.json + hardware_config.json + session lessons; read-only, no training", {}),
    ("B", "gnn-link-train", "TRAIN division+flow heads on clean GT (the div_J lever) — LOEO, config-driven", {}),
    ("B", "div-temporal-feas", "GO/NO-GO gate for a TEMPORAL division head: does temporal beat single-frame separability (frozen-detector multi-pool probe, stratified by embryo/stage/hard-invisible, XAI-justified). Mini-first.", {"mode": "mini"}),
    ("B", "psf-deconv", "GO/NO-GO gate for PSF DECONVOLUTION of the never-deconvolved light-sheet images (SiMView, docs/host_process): does a light anisotropic Richardson-Lucy deconv SEPARATE merged nuclei (GT close-pair local-maxima resolution RAW-vs-DECONV, per-embryo), help UNLEARNED-DoG detection PRECISION (mismatch-free), and FIT 2xT4/12h — without the learned-detector train/test mismatch tanking it. XAI knobs from voxel-scale/7µm-gate/z-xy anisotropy; math_master paired significance; hardware_tune precision. Mini-first.", {"mode": "mini"}),
    ("A", "gnn-probe", "Does neighbourhood context (GNN signal) beat pairwise geometry for linking? (cheap CPU probe)", {}),
    ("S", "prior-art", "Prior-art LINKING methods (CTC/ISBI): catalog + resolve (Viterbi/min-cost/GNN/Trackastra)", {"stage":"linking"}),
    ("S", "prior-art", "Prior-art DETECTION methods (CTC/ISBI): catalog + resolve (KIT-GE/StarDist/Cellpose/watershed)", {"stage":"detection"}),
    ("S", "trick-gate", "EVIDENCE GATE: adopt a trick ONLY if it proves out on golden-12 (never popularity)", {}),
    ("A", "decision-audit", "Enforce decide-only-from-data: flag any ledger choice kept/recommended without measured CV", {}),
    ("S", "trick-extractor", "Mine ALL top-solution tricks (pre/detect/link/div/post/metric/TTA/ensemble) → coverage matrix", {}),
    ("S", "tracker-consensus", "Run N trackers on in-domain detections, keep consensus links → pseudo-labels (hengck23)", {}),
    ("S", "block-synth", "Diff notebooks → compose NEW post-proc code-block recipes → golden-12 (no Claude)", {}),
    ("S", "combine-winners", "GM trick: stack the best of each lever into one recipe → golden-12", {}),
    ("S", "ablate-best", "GM trick: 'same as X but ONE change' — one-variable ablation from the best", {}),
    ("S", "scoreboard", "LIVE golden-CV leaderboard as ONE markdown-table message (updates in place)", {}),
    ("S", "heal", "SELF-HEAL: on a training failure, escalate to Claude with error+diagnosis to fix", {}),
    ("S", "plan-ingest", "Ingest the human's docs/human_plan.yml → run its experiments/splits (no leader needed)", {}),
    # 2026-07-12: submission-readiness + research + monitoring agents (now fleet-wired)
    ("B", "lora-train", "LoRA/rsLoRA/LoRA+ warm-started fine-tune of pilkwang's UNetNodeTransformer on external Zebrahub+comp → lift edge & division_jaccard (div-weight lever); early-stop on real metric; adapter+merged save",
     {"r": 16, "alpha": 32, "div_weight": 3.0, "strong_intensity_aug": True, "max_epochs": 6, "eval_every": 100}),
    # 2026-07-12: pilkwang-pipeline stage agents (generic names) — full training + inference inside our fleet
    ("B", "tracker-train", "Train the detector + edge-transformer (UNetNodeTransformer); warm-startable from pilkwang weights", {}),
    ("B", "center-train", "Train the full-frame 3D-UNet center-prior detector (--resume capable)", {}),
    ("B", "tracker-predict", "Detect + edge-predict + ILP link → per-dataset .geff predictions", {}),
    ("B", "tracker-postproc", "Post-proc geffs (fuse/motion-relink/gap-close/safe-div/linefit) → submission.csv (+optional GT score)", {}),
    ("S", "full-cv-baseline", "Predict+ILP+score the pilkwang BASE on all 199 datasets → the HONEST full-CV baseline every LoRA/detector run must beat (resumable)", {}),
    ("S", "lora-validate", "Load a trained LoRA adapter + score all 199 datasets → real both-embryo generalization delta vs the full-CV baseline (the only signal that adopts an adapter)", {}),
    ("S", "detect-quality", "Measure the detector's node RECALL + PRECISION on the EXTERNAL 100%-dense Zebrahub crops (per embryo) — the only honest place to see over-detection / false positives that sparse competition GT hides", {}),
    ("S", "stage-dynamics", "Profile per-developmental-STAGE cell motion + division rate from GT; math-master tests whether they vary enough by stage to justify STAGE-ADAPTIVE self-calibrating ILP priors (motion gate + division prior)", {}),
    ("S", "official-score", "Score predicted .geff graphs with the ORGANIZERS' byte-identical metric (edge + division, ±1-frame lineage-coverage) — the honest division_jaccard, NOT our src/metric.py out-degree proxy; per embryo", {}),
    ("B", "division-rescue", "Add geometry-consistent 2nd-child forks to predicted geffs using the proven-generic division signature (angle~137°/asym~1.6/sib~2.4×, rate-capped to biology, unclaimed-only) — the lever neither Ultrack nor the host uses; score after with official-score", {}),
    ("S", "lever-hunt", "Metric-driven MINI-EXPERIMENT loop: XAI decomposes node-recall (missed GT → gap-recoverable vs scattered + gap-length histogram) → picks GOAL+SUBSET, runs a pluggable lever (gap_fill: bridge track-end@t→start@t+k+1 with interpolated nodes under the 7µm gate, math_master-governed per-movie self-calibrated + density-adaptive knobs), VERIFIES before/after with the OFFICIAL edge/adj/division metric + paired significance, and reports SOLID CLUE (adj+score both up) or DEAD. Reusable pipeline stage gap_fill_postproc.gap_fill_graph wired behind CELLMOT_GAPFILL_MAXK", {"mode": "both", "lever": "gap_fill"}),
    ("S", "research-search", "Multi-source model/paper search (HF/arXiv/bioimage/zenodo/github/kaggle+discussions/europepmc/figshare) → BM25 per-comp index; shortlist fast one-pass candidates for detector-select",
     {"source": "all", "query": "3d nuclei cell tracking zebrafish light-sheet",
      "competition": "biohub-cell-tracking-during-development"}),
    ("S", "lb-sync", "Snapshot the official leaderboard → submission-recency/activity + per-comp PG (know the bar)",
     {"competition": "biohub-cell-tracking-during-development"}),
    ("S", "feasibility-gate", "ORCHESTRATION: chain xai-diagnose (name the failing bucket) → run the lever → math-master (per-embryo paired Δ + significance) → official-score (patched metric) → ledger (kept+verdict+mechanism) → insights; every GO/NO-GO gate becomes a durable Lever Feasibility Map insight so we never re-run a killed lever. mode='backfill' records this session's gates", {"mode": "backfill"}),
    ("S", "feasibility-map", "Render the LEVER FEASIBILITY MAP — the ranked GO/NO-GO table (verdict · patched-metric Δ · significance · one-line mechanism · evidence) derived from the ledger → shown on /insights (docs/INSIGHTS.md), auto-refreshed on ledger write", {}),
    ("V", "paper-verify", "PROVE the source papers (Zebrahub Cell 2024 + Ultrack PMC12615266) match our training data — per-claim MATCH/PARTIAL verdict table, cached", {}),
    ("V", "official-conformance", "PROVE conformance with the official baseline repo: metric-core byte-identical + submission schema == geffs_to_csv + 14 division sandbox cases pass", {}),
    ("V", "metric-probe", "REUSABLE adversarial METRIC-VULNERABILITY prober (any comp): given the official scorer + a prediction, search STRUCTURAL perturbations (off-volume forks / FP-free hub-unify / garbage edges / node-sparsify) that CHANGE the score WITHOUT improving correctness → ranked exploit report + inferred bug class (unmatched-not-penalized / global-reachability-credit / under-prediction-bonus). Understand LB unreliability, guard CV, report bugs — NOT to submit exploits", {}),
    ("V", "git-track", "COMMIT the code (parent + official_repo) → return the commit hash; stamps every ledger row/decision so each experiment maps to its exact code state", {"message": "experiment snapshot"}),
    ("S", "coverage-audit", "FLEET MAP: classify every agent into its RESPECTIVE pack (biohub=3D+time reference pack, reusable for any 3D comp) → docs/agent_coverage.md", {}),
    ("S", "comp-onboard", "FRONT DOOR: fingerprint ANY competition slug → CompConfig(modality×paradigm×task×metric×cv×submission) and route to its pack (tab/img/vid/pc/biohub/llm/agent/reason) — or emit an unknown-comp gap report; generalizes orchestrate to any comp", {"slug": ""}),
    ("S", "kaggle-modality", "META/MINING: ground each ACTIVE comp's DATA-MODALITY in Kaggle's OWN metadata (Meta Kaggle Competitions↔CompetitionTags↔Tags + category) → resolve birdclef=audio / deep-past=text / nemotron=text automatically; caches docs/kaggle_modality_map.json that comp-onboard + the :7788 dashboard read offline (no hardcoded table)", {"slugs": []}),
    ("A", "tab-profile", "TABULAR: fingerprint the data (shape/dtypes/cardinality/missing/target-balance/train-test-drift/leakage-sniff) from a CompConfig", {}),
    ("M", "tab-fe", "TABULAR GM FEATURE-ENGINEERING: leak-safe OOF target-encoding + frequency-encoding + row-aggregates + top interactions (the #1 tabular grandmaster lever)", {}),
    ("M", "tab-train", "TABULAR: CV-train installed backends (XGBoost/LightGBM/CatBoost/HistGBM, GPU-auto) → OOF+test preds, honest CV via metric-registry", {}),
    ("M", "tab-stack", "TABULAR: metric-optimal simplex hill-climb blend of tab-train OOFs → blended test pred (falls back to best single)", {}),
    ("M", "tab-autobaseline", "TABULAR TURNKEY: one call profile→CV-train-all→blend→submission.csv; competitive default with zero hand-tuning (playground-series)", {}),
    ("M", "agent-env", "AGENTIC: onboard the comp's environment — action space, reward, BUDGET (the real constraint); smoke rollout to confirm wiring (pokemon-tcg/autonomous-agent/ai-agent-security)", {}),
    ("M", "agent-policy", "AGENTIC: candidate-tournament policy search (beat a strong heuristic first) + optional evolutionary refinement; returns best policy by mean reward", {}),
    ("M", "agent-eval", "AGENTIC: OFFLINE score a policy — mean reward, BUDGET compliance, frac-of-optimal — before any submit (JED budget lesson: budget decides the score)", {}),
    ("R", "gm-writeup-mine", "GROUND THE FLEET: fetch top-N real solution writeups for finished comps (nvidia-kaggle bearer API) → docs/gm_writeups/; feeds trick-extractor to grow the grounded GM technique catalog (stays current with 2025/26 SOTA)", {"slugs": []}),
    ("S", "sub-journal", "META/MINING: keep the experiment JOURNAL COMPLETE — sync EVERY Kaggle submission (kaggle competitions submissions --csv) into the journal so no experiment is missing; parses public+private+status, records each not-already-present submission with a real docs/sub_<ref>.json provenance artifact + cv-from-description, enriches existing rows with both scores, idempotent (dedupes by Kaggle ref). Comp-agnostic (competition is a parameter)", {}),
    ("M", "pseudo-label", "GM TOOLKIT: self-training/pseudo-labeling — select confident test rows as extra training labels (optionally temperature-denoised); the #1 repeated 2025-26 lever (birdclef 0.87→0.93)", {}),
    ("M", "blend-optimize", "GM TOOLKIT: try ALL blenders (single/hill-climb/Caruana/Nelder-Mead/Ridge) on OOF, return the best + matching test blend — reusable across tab/img/llm ensembles", {}),
    ("M", "post-optimize", "GM TOOLKIT: metric-specific post-processing — QWK-rounding / prediction-clip / temperature-scale / quantile-threshold / rank-average", {}),
    ("M", "calibrate", "GM TOOLKIT: probability calibration (Platt/isotonic/temperature) on OOF with ECE before/after — honest gain", {}),
    ("M", "target-transform", "GM TOOLKIT: target engineering — sqrt/log1p/rank-gauss transforms (exact inverse) + survival target factorization (equity classifier+regressor golden trick)", {}),
    ("R", "solution-adopt", "ADOPT THE WINNING WORKFLOW: turn a comp's mined top-1..5 solutions into an ordered EXECUTABLE pipeline of reusable technique-agents (grounded in real 2025-26 winners) — one agent, any comp", {}),
    ("R", "skill-build", "PROMPT-PROGRAM: author the winning ADK SKILL — a deterministic leakage-safe AutoML floor (run_pipeline.py + check_submission.py + SKILL.md) for agent-authoring comps (autonomous-agent-prediction-beta)", {}),
    ("R", "agent-author", "PROMPT-PROGRAM: author the ADK agent bundle (agent.yaml + champion system.md + data-analyst sub-agent + skill) — the submission for agent-authoring comps", {}),
    ("R", "agent-package", "PROMPT-PROGRAM: validate the ADK bundle (required manifest + allowed model ids) and zip CONTENTS at root → submission.zip", {}),
    ("R", "agent-config-eval", "PROMPT-PROGRAM: OFFLINE gate — run the authored skill on a synthetic HIDDEN-LABEL smoke matrix → mean AUC; no change ships without hidden-label evidence (anti public-LB overfit)", {}),
    ("R", "prompt-optimize", "PROMPT-TUNING-AS-PROGRAMMING: evaluate prompt/skill VARIANTS on the hidden-label smoke matrix, keep the best by AUC (grounded: the deterministic floor is the real lever, prompt wording ties/regresses)", {}),
    ("S", "dspy-prompt-optimize", "PROMPT-PROGRAM OPTIMIZATION: DSPy Signature+Module (Predict/CoT/ReAct/PoT) driven by an OPTIMIZER (BootstrapFewShot/MIPROv2/COPRO/GEPA/BootstrapFinetune) against a trainset+metric; PLUS a from-scratch GEPA reflective-evolution loop (Pareto frontier + APEX Mixed-tier data selection, arXiv:2507.19457 + 2606.11459) that runs OFFLINE with a mock LM. Degrades cleanly when dspy/LM absent", {}),
    ("M", "subset-classifier-router", "GAP-SCAN: classify each item into a family → route to the specialist model/postproc (waveform/vesuvius/mitsui inference-time mixture-of-experts)", {}),
    ("M", "analysis-by-synthesis-refiner", "GAP-SCAN: test-time gradient refinement so forward(pred)≈observed with a KNOWN operator — inverse problems (waveform 28.8→7.6, ariel transit fit)", {}),
    ("M", "checkpoint-merger", "GAP-SCAN: weight-space model merging (linear/TIES) — beats prediction blending + cuts tokens (aimo mergekit)", {}),
    ("M", "constrained-label-assignment", "GAP-SCAN: Hungarian/joint-MLE decode under per-group COUNT constraints (cmi +0.02)", {}),
    ("M", "lb-shift-prober", "GAP-SCAN: fit an affine/offset correction from a probe grid (polymer: caught a °C/°F unit bug + constant offset)", {}),
    ("R", "github-solution-mine", "GROUND FROM CODE: harvest winners' GitHub repos linked in writeups (gh API, no full clone) → fetch key ML modules (train/model/loss/dataset/infer) → index for distillation into agents", {"limit": 0}),
    ("M", "fin-ta-feature-library", "DOMAIN FE: financial technical-analysis features (vol/momentum/RSI/z-score/Hurst) from a price series (mitsui/jane-street)", {}),
    ("M", "imu-feature-engineer", "DOMAIN FE: kinematic features from IMU/accelerometer streams (magnitude/jerk/gravity-removal/spectral energy) (cmi-detect-behavior)", {}),
    ("M", "online-walk-forward-retrainer", "TIMESERIES: incremental retrain over a time stream (refit every N steps on data-so-far) — survives non-stationarity (jane-street/mitsui)", {}),
    ("M", "synth-artifact-fe", "TABULAR: generator-fingerprint FE for synthetic-from-original comps (digit/snap/is-round/orig-freq) — recurring playground lever", {}),
    ("M", "oof-diversity-prune", "ENSEMBLE: OOF error-correlation matrix → prune near-twin models, keep decorrelated legs (the weak orthogonal tail drives the lift)", {}),
    ("M", "feature-select", "TABULAR: consensus feature importance (GBDT gain + permutation) → stable top-K subset (small-data overfit control)", {}),
    ("M", "residual-boost", "ENSEMBLE: fit a booster on the residuals of a baseline/generating-function; final = base + residual (cdeotte lever)", {}),
    ("M", "full-retrain-calibrator", "ENSEMBLE: 100%-train retrain iteration count iters×(1+1/(K-1)) + seed-averaging for rank/threshold metrics", {}),
    ("M", "ts-decompose-forecaster", "FORECAST: multiplicative decomposition into interpretable ratio factors (calendar/group) + reconstruction (s5e1 sticker-sales)", {}),
    ("M", "forecast-trend-extrapolator", "FORECAST: choose the future-horizon trend multiplier (const/linear/ReLU) beyond the training range — the out-of-range lever", {}),
    ("M", "rating-systems", "SPORTS: competitive ratings from a win/loss+margin game graph — Elo (MOV+carry), Colley, SRS (march-mania)", {}),
    ("M", "outcome-sharpen", "SPORTS: Brier/log-loss tail sharpening + expert overrides (deliberate anti-calibration EV gamble)", {}),
    ("M", "best-of-n-diversity-allocator", "BEST-OF-N: pick N DIVERSE candidates (not averaged) to maximize expected max (rna best-of-5 TM-score)", {}),
    ("M", "sr-bf16-optimizer", "TRAINING MEMORY: stochastic-rounding bf16 AdamW — unbiased sub-ULP accumulation lets bf16 optimizer states train like fp32 at half the memory (offload_adam)", {}),
    ("M", "quaternion-imu-features", "DOMAIN FE: orientation features from rotation quaternions — |q|=1 imputation, angular velocity/distance, SO(3) rotation augmentation (cmi 1st place)", {}),
    ("M", "temporal-segment-decoder", "SEGMENTATION: frame-probabilities → scored action segments (per-group threshold + min-duration) (MABe)", {}),
    ("S", "masked-sequence-norm", "SEQUENCE: MaskedBatchNorm-style per-channel z-score over ONLY valid timesteps of a padded batch — padding excluded from mean/var, no train/serve skew (CMI sensor 2nd)", {}),
    ("S", "masked-sequence-pool", "SEQUENCE: leakage-free mean/max/attention pooling of a padded variable-length sequence into one vector (padding contributes nothing) (CMI sensor 2nd)", {}),
    ("S", "class-balance-sampler", "IMBALANCE: tempered class-balanced sampling weights (count^power: 0=natural, -0.5=sqrt, -1=balanced) + Cui effective-number mode + deterministic resample (BirdCLEF 2nd)", {}),
    ("M", "deep-supervision", "TRAINING: multi-scale deep-supervision loss — seg head per decoder level, target adaptive_MAX_pooled to keep tiny objects, per-level weights (CZII CryoET 1st)", {}),
    ("M", "sed-attention-pool", "TRAINING: weakly-supervised attention pooling head — per-frame logits + softmax-over-time attention → clip prediction (MIL) + learnable GeM pool (BirdCLEF 2nd)", {}),
    ("M", "audio-melspec-fe", "AUDIO: waveform → log-mel spectrogram (pure torch.stft + Slaney mel filterbank, no librosa/torchaudio) — configurable n_mels/n_fft/hop/fmin/fmax/power→dB + per-instance norm + optional freq-channel; THE audio front-end (all 5 BirdCLEF/freesound winners)", {}),
    ("M", "audio-augment", "AUDIO: SpecAugment (time+freq masking) on the mel + waveform aug (gaussian/pink noise, gain, background-mix, OR-mixup-in-time) — DataLoader-safe, shape-preserving (BirdCLEF-2023 2nd / freesound 1st)", {}),
    ("M", "audio-crop-tta", "AUDIO: fixed-window crop training + multi-window sliding-crop TTA on long clips → aggregate(mean/max/min/temperature-mean) + neighbor-smooth [0.1,0.2,0.4,0.2,0.1]; the BirdCLEF long-clip standard generic multi-tta lacks", {}),
    ("M", "audio-backbone", "AUDIO: mel→CNN classifier wrapper (timm EfficientNet-b0 the BirdCLEF workhorse, else small pure-torch CNN); heavy PANNs/wav2vec2 iface escalates clean when weights absent", {}),
    ("M", "audio-train", "AUDIO: end-to-end multi-label audio-classification TRAINER — composes melspec-fe + specaug/waveform-aug/OR-mixup + class-balance-sampler + timm-EfficientNet-b0 + BCE + EMA; PRIMARY val=soundscape 5s windows (official macro-ROC-AUC, LB-proxy) + SECONDARY leak-safe author-grouped focal K-fold; small-first ladder (classes/per_class/seconds/epochs); saves Kaggle-ready ckpt (BirdCLEF)", {}),
    ("M", "audio-infer", "AUDIO: CPU sliding-window inference → Kaggle-ready submission.csv — loads an audio-train ckpt, splits each test soundscape into consecutive 5s windows, melspec+batch-predict, optional neighbor-smooth, writes row_id=stem_endsec + one prob col per species in EXACT sample_submission order; reports windows/sec → 90-min budget projection (the offline BirdCLEF notebook body)", {}),
    ("M", "kaggle-submit", "SUBMISSION: reusable offline-notebook (code-competition) submission pipeline — packages a best-model ckpt as a private Kaggle DATASET (create/version), GENERATES a fully SELF-CONTAINED CPU inference notebook (inlines mel front-end + model rebuild + sliding-window inference + submission writing, NO fleet import, discovers input paths by content, header-correct fallback when test empty), pushes the kernel (CPU/no-internet, dataset+competition attached), submits the kernel output, and reads back PUBLIC+PRIVATE LB. Repeats on every new best model. spec {ckpt, competition, dataset_slug, message, sample_submission, inference='audio'}", {}),
    ("M", "graph-message-passing", "GRAPH: general pure-torch MPNN forward — aggregator (mean/max/sum/attention) over an edge_index with optional edge features, N layers with residual+LayerNorm+SAGE root transform, directional+DropEdge options; node-classification / graph-regression heads. Implemented via index_add/scatter_reduce (NO torch_geometric dep). GraphSAGE was the repeated winner (predict-ai-model-runtime 1st/2nd/5th, champs-scalar-coupling GAT 4th)", {}),
    ("M", "graph-feature-extractor", "GRAPH: deterministic node features (in/out degree, local clustering coeff, k-hop reachable counts) + structural/positional encodings (Laplacian eigvec PE — Fiedler separates communities; random-walk return-probability PE) + edge & global graph descriptors from an edge_index. The FE+PE winners hand-craft (OpenVaccine distance-matrix PE, champs charges/angles, runtime degree/opcode)", {}),
    ("M", "graph-readout", "GRAPH: graph-level pooling (mean/sum/max/attention/Set2Set) mapping node embeddings + batch index → one fixed vector per graph (permutation-invariant, variable-size safe) — the graph-regression head (predict-ai-model-runtime 1st global-mean-pool, 2nd sum-reduce; Set2Set the canonical learnable molecular pool)", {}),
    ("M", "multimodal-fusion", "MULTIMODAL: FEATURE/MODEL-level fusion — dict of per-modality feature tensors → project each to a shared dim → fuse via a configurable strategy (concat / sum / mean / gated / FiLM / cross-attention / bilinear) → fused representation (+ optional regression/classification head). N modalities, variable input dims. Grounded in shopee 'NFNet-F0 + Indonesian-BERT concatenated at final feature layers', petfinder Swin-embedding+12-metadata→SVR/MLP head, shopee-2nd GAT-over-similarity attention. LATE/decision fusion stays in ensemble/blend-optimize/infer-cascade", {}),
    ("M", "modality-encoder-adapter", "MULTIMODAL: wrap heterogeneous per-modality inputs (image embedding / text embedding / tabular vector) into aligned, L2-normed, shared-dim embeddings with per-modality LayerNorm + projection + learnable modality-TYPE embeddings. Grounded in shopee F.normalize(cat([F.normalize(img), F.normalize(txt)])) shared-space trick + ariel per-planet normalization. The projection layer multimodal-fusion consumes", {}),
    ("M", "modality-dropout", "MULTIMODAL: training-time random modality masking (≥min_keep kept per sample) + inference missing-modality imputation via a learned per-modality NULL token → the model is robust when a modality is absent. Grounded in petfinder's KEY negative finding (image-only ≈ image+metadata) — the winners' robustness insurance so a multimodal net never DEPENDS on one modality", {}),
    ("M", "video-frame-sampler", "VIDEO: sample T frame INDICES from a length-n_frames clip — uniform (evenly-spaced, TSN/eval) / stride (start+stride·k, cyclic-wrap) / dense (non-uniform, densest around an event index — the NFL {-44..0..37} schedule) / random (segment-binned + jitter, TSN train-time); variable-length safe (cyclic pad short, subsample long), always in-range, + gather_frames helper. THE video-training dataloader foundation (deepfake 32-frame / DFL every-2nd / NFL dense-around-event)", {}),
    ("M", "video-temporal-aggregator", "VIDEO: per-frame embeddings [B,T,D] → clip vector [B,D'] via mean/max/attention(content-energy weighted, captures sparse-in-time signal)/tconv(1D conv over time)/tsm(Temporal-Shift-Module)/gru — the piece that turns an image backbone into a video model. TSM/1D-conv/GRU are the DFL camaro-3rd / NFL Dmytro / youtube8m winner heads; references masked-sequence-pool for the mask-aware mean/max/attention path", {}),
    ("M", "video-motion-features", "VIDEO: frame-difference (abs, 1st/2nd order) + temporal-gradient (central-diff = optical-flow proxy along time) + brightness-constancy flow-magnitude proxy + motion_channels stacker (append motion maps as extra CNN input channels). Non-zero where motion is, ~zero on a static clip; training-free (no RAFT weights). The deepfake/DFL/NFL motion cue (DFL ohkawa3 abs-diff channel, kmat prev/next diff, NFL-impact RAFT velocity)", {}),
    ("M", "awp-perturb", "TRAINING: Adversarial Weight Perturbation — per-param grad-normalised ascent + restore-before-step flat-minima regulariser, distinct from SAM (MABe 5th)", {}),
    ("M", "muon-optimizer", "2026 SOTA OPTIMIZER: Muon — momentum orthogonalized by 5-step Newton-Schulz quintic; equalizes step size across an ill-conditioned weight matrix's directions (drop-in torch optim; beats SGD/Adam on 2D weights)", {}),
    ("S", "conformal-predict", "2026 UNCERTAINTY: split conformal + APS/RAPS adaptive prediction SETS (classification) & residual intervals (regression) with finite-sample 1-alpha COVERAGE + Mondrian per-group; wraps ANY model's holdout scores", {}),
    ("M", "schedule-free", "2026 OPTIMIZER: Schedule-Free (Road-Less-Scheduled + ScheduleFree+) — no LR schedule/horizon, Polyak-average AS the iterate with gradients at an interpolation point; matches tuned cosine", {}),
    ("M", "dora-adapt", "2026 PEFT: DoRA — weight-decomposed low-rank adaptation (per-output MAGNITUDE vector + LoRA DIRECTION); identity at init, beats plain LoRA at equal rank when per-channel scaling matters", {}),
    ("S", "hardware-tune", "HARDWARE AUTO-TUNER: profiles the live GPU (RTX 5090 sm_120 / Kaggle T4), EMPIRICALLY benchmarks matmul dtype (fp32/tf32/fp16/bf16) + picks the fastest numerically-safe training config (amp dtype, tf32, torch.compile, channels_last, VRAM-scaled batch, optimizer) and WRITES docs/hardware_config.json → every train agent reads it via hardware_tune.load_config(); reusable per box", {}),
    ("S", "hf-kernels", "HF HUB KERNELS (all comps): discover + ARCH-CHECK pre-compiled Hugging Face `kernels-community` kernels (flash-attn/fp8-scaled_mm/activation/moe/quant) for THIS box — no local build. box_variant()→torch28-cxx11-cu128; check(repo) matches torch/cuda/platform; cuobjdump_check(repo) DECISIVELY dumps the fatbin for sm_120 SASS (arch is NOT in the folder name). MEASURED: kernels-community/quantization+activation ship sm_120 ✓; deep-gemm DEAD on sm_120 (sm_90/sm_100 only). Honest: variant-match≠runs (load-test), internet-only (not offline T4). Discovery widens via HF MCP; usability proven here", {}),
    ("S", "prompt-metric", "PROMPT-OPT METRIC SOURCE: turns a NAMED metric (exact/norm_exact/contains/token_f1/numeric/multiple_choice/regex_match/json_field/keyword_coverage) into score(pred,gold)∈[0,1] + feedback(pred,gold) — the callable dspy-prompt-optimize/GEPA need, reconstructed in-process from a JSON-safe name (metric fns can't cross the board). Pairs with prompt-dataset", {}),
    ("S", "prompt-dataset", "PROMPT-OPT DATASET SOURCE: builds a trainset of {input,gold} examples from spec['examples'] (inline) / spec['file'] (.jsonl/.json/.csv) / spec['synthetic'] (arithmetic|sentiment|multiple_choice) with train/val split; feeds dspy-prompt-optimize. to_dspy() → dspy.Example. The dataset half of the (dataset+metric+runner) a prompt optimiser needs", {}),
    ("S", "style-fingerprint", "AUTHOR STYLE FINGERPRINT (interpretable, reference-free): builds a per-attorney PROFILE from his patent corpus — distinctive signature n-grams (Dunning log-likelihood keyness × cross-doc frequency, per-1k rate), paragraph/sentence OPENER distributions + transition profile, canonical SECTION-header sequence, cross-patent BOILERPLATE blocks (shingle+MinHash Jaccard cluster), and micro conventions (reference numerals/hedging/passive/claim transitions). score(draft,profile) → per-layer [0,1] + composite + OPTIMIZER FEEDBACK naming concrete misses; discrimination_auc(profile,his,others) proves it identifies HIM; as_metric(profile) plugs into dspy-prompt-optimize/GEPA as a style reward. Complements LUAR/PatentScore embeddings with concrete fingerprints", {}),
    ("M", "combinatorial-local-search", "OPTIMIZATION: iterated local search over permutations (2-opt/swap + SA + double-bridge kick) vs a black-box objective (santa-2024)", {}),
    ("M", "population-diversity-manager", "OPTIMIZATION: genetic algorithm (order-crossover + mutation + elitist diversity selection) over permutations (santa)", {}),
    ("M", "batched-oracle-search-harness", "OPTIMIZATION: memoized black-box scorer + multi-start island parallelism (the GPU-oracle search infra)", {}),
    ("M", "gaussian-heatmap-encoder", "DETECTION: encode (x,y[,z][,class]) keypoints → N-D windowed-Gaussian heatmap training target (4σ, per-point sigma) — the soft-target regression codec (CZII CryoET / ECG)", {}),
    ("M", "volumetric-patch-inference", "DETECTION: minimal/fixed-overlap N-D patch tiling of a big volume + overlap-averaged stitch reconstruction — the sliding-window inference scheduler (CZII CryoET)", {}),
    ("M", "heatmap-peak-decoder", "DETECTION: decode an N-D heatmap → centroids via local-max+radius-NMS (peak) or connected-component centroid+voxel-count filter (blob), optional soft-argmax subpixel (ECG/CZII)", {}),
    ("S", "wbf-fusion", "INFERENCE: Weighted Boxes Fusion — cluster boxes(IoU)/points(distance) from N models/TTA and confidence-weighted-average coords+conf (Lyft 1st; point-variant fuses cell detectors)", {}),
    ("S", "snapshot-average", "INFERENCE: average logits/probs across N snapshot/seed outputs (prob/logit/rank modes + per-model weights); raw reusable ensemble reducer", {}),
    ("S", "multi-tta", "INFERENCE: multi-transform test-time augmentation — apply invertible flips/rot90/scales, predict, invert, fuse (mean/WBF); 2D/3D", {}),
    ("M", "shift-adapt", "ROBUSTNESS: adversarial train-vs-test discriminator → importance weights + shift-aligned holdout CV (s5e12)", {}),
    ("M", "geospatial-fe", "DOMAIN FE: grid-cell target-encoding + spatial-KNN class-fraction for lat/lon / RA-Dec (s6e6)", {}),
    ("M", "linear-constraint-projector", "POSTPROC: project predictions onto a known linear manifold Ax=b (mass-balance / Einthoven) — CSIRO/ECG", {}),
    ("M", "runtime-budget-router", "RUNTIME: send only highest-value items to the expensive model under a time/compute budget (cascades, T4)", {}),
    ("M", "mbr-consensus-selector", "DECODE: Minimum Bayes Risk — pick the candidate most similar to the pool (translation/structure generation)", {}),
    ("M", "noisy-label-cleaner", "DATA: resolve conflicting/duplicate (input,label) supervision → majority + soft target", {}),
    ("M", "knn-label-transfer", "RETRIEVAL: similarity-weighted neighbor label vote (homology/structure label transfer) — CAFA", {}),
    ("M", "self-consistency-aggregator", "LLM-INFER: vote-share + entropy-weighted aggregation of N sampled answers (AIMO self-consistency)", {}),
    ("M", "consensus-early-stop", "LLM-INFER: stop sampling once ≥k agree or the leader's margin is uncatchable (save compute)", {}),
    ("M", "risk-abstain-gate", "LLM-INFER: submit-vs-skip under an asymmetric penalty by expected value (Konwinski/penalized comps)", {}),
    ("M", "budget-aware-inference-scheduler", "LLM-INFER: per-problem time/token budget from global remaining × difficulty (AIMO)", {}),
    ("M", "sample-pool-simulator", "LLM-INFER: evaluate any (k, early-stop) config's accuracy in O(1) by subsampling a pre-generated pool", {}),
    ("M", "mtp-speculative-decode", "LLM-INFRA (Gemma-4 §2.6/Fig 1): MTP drafter speculative decoding — expected accepted tokens/verify + decode speedup vs draft/target cost ratio + optimal draft length γ", {"alpha": 0.8, "cost_ratio": 0.1}),
    ("M", "kv-cache-longctx", "LLM-INFRA (Gemma-4 §2 long-context): KV-cache bytes vs context length under local:global ratio + sliding-window + KV-sharing + values=keys → % reduction (the 37.5% global-KV lever)", {"seq_len": 32768, "n_layers": 48, "n_kv_heads": 8, "head_dim": 128, "dtype_bytes": 1}),
    ("M", "moe-inference-cost", "LLM-INFRA (Gemma-4 26B-A4B/Table 1): MoE active-vs-total params, per-token FLOPs, compute-saving ratio + resident-memory cost (26B total for ~4B active)", {"n_experts": 8, "active_experts": 1, "expert_params": 3.0e9, "shared_params": 2.0e9}),
    ("M", "moe-quantile-balance", "LLM-INFRA (Kimi-K3 Stable LatentMoE): quantile-normalize router-score columns before top-k → per-expert load balances by construction with NO aux-loss/learned bias (enables 16-of-896 sparsity)", {"n_tokens": 512, "n_experts": 16, "k": 2, "bias_scale": 1.5}),
    ("M", "sparsity-metrics", "TRAINING (sparsityLLM/rank.py): hidden-state sparsity (l0/top-k-energy/Gini/effective-rank) as a label-less difficulty signal — sparse=easy/in-dist, dense=hard/OOD; difficulty_score + curriculum_order for easy→hard sample ordering", {"n": 200, "dim": 512}),
    ("M", "shap-emd-distance", "XAI/METRIC (thiagorr162/shap-emd): model-aware Earth-Mover's distance on compositional/histogram features — ground cost = area between SHAP dependence curves, exact transport LP (no POT dep); for part-of-whole feature neighbor/duplicate metrics", {"n_features": 8, "n_ref": 10}),
    ("M", "geometric-features", "GEOMETRIC-DL (torchmd-net PhysNet): 3D point-cloud → rotation+translation-invariant edge features (cosine cutoff + exp-normal RBF + brute-force neighbor list); reusable equivariant-GNN featurizer for molecules/cells/particles", {"n_points": 40, "cutoff": 3.0, "num_rbf": 32}),
    ("M", "flow-matching", "GENERATIVE (ppflow OT-CFM): conditional flow-matching — straight-path velocity regression (u=x1-x0 MSE), Euler-sampled; simulation-free generation/augmentation for any continuous target (embeddings/coords/latents)", {"dim": 2, "train_steps": 800}),
    ("M", "llm-backend", "INFRA (omnigent llms/ pattern): multi-provider LLM client (Ollama/OpenRouter/vLLM/any OpenAI-compatible + Anthropic + local-HF + offline dummy) so the fleet is not Claude-only; env-configured, stdlib-only", {"model": "dummy/echo", "prompt": "ping"}),
    ("M", "mup-scaling", "SCALING (microsoft/ArchScale μP): maximal-update parametrization width rules — readout LR ×base/width + init 1/fan_in keep Δlogit O(1) as width grows, so small-model LR/init transfer to wide models without re-sweeping", {"widths": [64,128,256,512,1024], "base_width": 64}),
    ("M", "gpu-patterns", "GPU/KERNELS (srush/GPU-Puzzles): parallel primitives (scan/reduce/tiled-matmul reference) + roofline arithmetic-intensity model → compute- vs memory-bound classification and the right kernel/tiling lever", {"m": 4096}),
    ("S", "deck-builder", "REPORTING (hugohe3/ppt-master distilled): deterministic spec→.pptx (title/bullets/table) on python-pptx — turn an experiment ledger/CV summary into a shareable deck; no LLM/SVG", {}),
    ("S", "video-builder", "REPORTING (remotion alternative): Python-native frames→GIF/MP4 (overlay_points for tracking overlays) via imageio — no Node/Chromium; render tracking/training animations", {"n_frames": 12, "size": 64}),
    ("S", "task-spec", "ORCHESTRATION (GeminiLight/MindOS): spec-driven task contract — goal/current-state/data-flow(readers+writers lint)/plan/impact/edge-cases/acceptance checklist; gate() makes 'done' objective and binds agents to a shared goal", {}),
    ("M", "turboquant", "RETRIEVAL/QUANT (Google TurboQuant / turbovec): data-oblivious vector quantizer — random rotation makes coords Beta-marginal, one Lloyd-Max codebook fixed by DIM (no training/rebuild); 8x embedding compression w/ graceful recall for RAG/dedup/NN", {"n": 2000, "dim": 64, "bits": 4}),
    ("M", "attention-residual", "ARCH (Kimi-K3 Attention Residuals): residual that SELECTIVELY retrieves across depth (learned gate over all prior layer states) instead of a uniform running sum — deep layers re-read a specific earlier depth; generalizes x=x+f(x)", {"dim": 16, "depth": 6}),
    ("M", "latent-moe", "ARCH (Kimi-K3 Stable LatentMoE + Gated MLA): run experts and attention KV in a low-rank LATENT (down-proj→operate→up-proj) — experts cost ~d_latent not d_model, MLA KV-cache Nx smaller; enables 16-of-896 sparsity + long context", {"d_model": 128, "d_latent": 32}),
    ("M", "rswa-attention", "ATTENTION (Baidu Unlimited-OCR R-SWA, arXiv:2606.23050): Reference Sliding Window Attention — persistent reference/prefix block + local window → CONSTANT KV-cache for long-document/long-generation decoding (no linear cache growth)", {"n_ref": 64, "window": 128, "decode_len": 8000}),
    ("M", "embedding-retrieval", "RAG/RETRIEVAL (NVIDIA Nemotron-3-Embed-1B + minishlab/potion-code-16M): dense-embedding search/MMR/dedup — Nemotron 34-lang cross-lingual RAG (self-host, permissive), potion static CPU code-search; pairs w/ turboquant 8x + llm_backend", {}),
    ("M", "nvfp4-loader", "BLACKWELL/LLM (Unsloth Gemma-4 NVFP4): load+finetune planner for NVFP4 models on RTX 5090 (sm_120 native 4-bit) — VRAM accounting (4.5b/w + KV), fits-check for 12B/26B-A4B/31B, vLLM auto-NVFP4-kernel (not Marlin) + Unsloth FastModel recipe", {"model": "12b", "ctx_len": 8192}),
    ("M", "diffusion-sampler", "GENERATIVE (google/hackable_diffusion torch port): DDPM/DDIM Gaussian diffusion — noise schedule + eps/x0/v params + stochastic (DDPM) & deterministic few-step (DDIM) samplers; SDE view complementing flow-matching ODE view; no JAX", {"dim": 2, "T": 100}),
    ("S", "ui-component", "REPORTING/UI (facebook/astryx theme-as-vars): design-token → self-contained HTML dashboard (KPI cards + tables, light/dark, no Node) — render leaderboards/CV dashboards/metric reports", {"theme": "light"}),
    ("S", "shopify-agent", "COMMERCE (Shopify 2026 SDKs): order analytics (revenue/AOV/top/RFM) + product payloads + Admin API + embedded-APP scaffold (React Router+TS+Polaris+App Bridge, token-exchange auth, Shopify Functions)", {}),
    ("M", "lightning-tricks", "TRAINING (PyTorch Lightning): hardware-tuned Trainer kwargs (bf16-mixed on Ampere+/5090, 16-mixed on T4, grad-accum, FSDP/DDP) + best callbacks + 11-trick KB (compile/TF32/dataloader/QAT)", {"cap": [12,0]}),
    ("M", "santa-agent", "OPTIMIZATION (Kaggle Santa 2019-2025): combinatorial-opt toolkit — 2-opt/Or-opt TSP local search + simulated_annealing (the workhorse) + beam_search + per-year solution KB (ILP if clean else SA, neighborhood shape = lever)", {"n_cities": 30}),
    ("M", "skill-optimizer", "PROMPT-OPT (microsoft/SkillOpt): SGD-analogy skill/prompt optimization — reflect(LLM gradient)→rank-and-select→held-out gate, converges without weight training; reflect via llm_backend (Ollama/OpenRouter/Claude)", {"target": 7.0, "epochs": 40}),
    ("M", "portfolio-position-sizer", "FINANCE: alpha signal → risk-managed allocation (vol-targeting + tanh + leverage clip) — hull's winning lever", {}),
    ("M", "market-odds-blend", "SPORTS/FINANCE: moneyline → no-vig implied prob, tier-blend with the model prediction (march-mania)", {}),
    ("M", "forecast-drivers-then-derive", "TIMESERIES: forecast raw drivers then apply the known target formula (beats direct noisy-target; mitsui)", {}),
    ("M", "label-lag-anchor-blend", "TIMESERIES: blend model output with recently-revealed-label persistence anchor (survive regime shift; mitsui)", {}),
    ("M", "distributional-metric-recalibrator", "POSTPROC: per-group affine (scale+shift) correction of predictions for train/test shift (CSIRO/ariel)", {}),
    ("M", "expression-search", "CODE: brute-force the shortest arithmetic expression reproducing an int→int table (code-golf / tiny mappings)", {}),
    ("M", "code-compress-optimizer", "CODE: deflate/zlib byte minimizer + self-extracting stub for byte-limited artifacts (code-golf)", {}),
    ("M", "sprt-spsa-tuner", "TUNING: SPRT accept/reject of a change from win/loss matches + gradient-free SPSA parameter step (chess engine)", {}),
    ("M", "lb-formula-prober", "KAGGLE-META: reverse-engineer a hidden LINEAR scoring formula from probe (features,score) pairs by least squares", {}),
    ("M", "trainable-trace-auditor", "LLM: audit a reasoning trace for learnability (reference-integrity/operand-locality/hidden-compute) — nemotron lever", {}),
    ("M", "hierarchy-consistency-postproc", "POSTPROC: enforce ontology-DAG consistency (parent≥max children) on multi-label probs (CAFA GO-terms)", {}),
    ("M", "invariance-feature-normalizer", "DOMAIN FE: egocentric/frame-canonical coordinates so a model transfers across sources (MABe/NFL)", {}),
    ("M", "template-retrieval-reranker", "RETRIEVAL: retrieve candidates by similarity + rerank (RNA templates)", {}),
    ("M", "calendar-holiday-fe", "DOMAIN FE: day-of-week/month/is-weekend + cyclical + optional holidays from dates (s5e1 sales)", {}),
    ("M", "annotation-error-corrector", "DATA: flag likely-wrong GT via OOF-residual outliers for human review (BYU — biggest data-side lever)", {}),
    ("M", "binary-size-compressor", "DEPLOY: deflate an artifact (weights/binary) + report the byte win under a size cap (chess/code-golf)", {}),
    ("M", "knn-feature", "TABULAR: leak-safe OOF kNN target-mean + distance meta-features", {}),
    ("M", "geometric-spatial-augmentor", "AUG: coordinate augmentation (rotate/flip/dropout/jitter) — top anti-overfit lever for tracking (NFL/MABe)", {}),
    ("M", "infer-cascade", "LLM-INFER: multi-stage confidence cascade — unsure items escalate to the bigger model (fit the budget)", {}),
    ("M", "llm-synthetic-drill-generator", "LLM: fabricate synthetic supervised drills from templates+vocab to teach a skill (nemotron/deep-past)", {}),
    ("M", "heteroscedastic-uncertainty-head", "UNCERTAINTY: ensemble → (mu, sigma) for GaussianNLL / calibrated-uncertainty metrics (ariel/CSIRO)", {}),
    ("M", "tab-nn-train", "TABULAR NN: real torch neural-tabular trainer (residual-MLP, GPU-auto, OOF) — the ensemble-diversity member GBDT-only tab-train lacked", {}),
    ("M", "sdf-regression-loss", "3D-SEG: signed-distance-field target + boundary-weighted loss (sharper centers; BYU/vesuvius)", {}),
    ("M", "topology-aware-loss", "3D-SEG: topology-aware score (Dice + Betti-0 component agreement) for thin/connected structures (vesuvius/vessels)", {}),
    ("M", "ae-latent-view", "TABULAR: REAL torch autoencoder → latent diversity features", {}),
    ("M", "keypoint-match-verifier", "VISION: cv2 SIFT/ORB + RANSAC homography inliers (copy-move forgery / image matching)", {}),
    ("M", "grid-rectification-unwarp", "VISION: cv2 homography dewarp of a gridded document to canonical (ECG digitization)", {}),
    ("M", "tir-executor", "LLM: Tool-Integrated Reasoning — execute code blocks in a sandbox, splice stdout into the trace (AIMO/Konwinski)", {}),
    ("M", "llm-eval", "LLM: score generations vs truth via the CompConfig metric (+ exact-match fallback)", {}),
    ("M", "llm-infer", "LLM: constrained decoding — mask logits to allowed token ids (choice/Yes-No); full generate needs a model", {}),
    ("M", "llm-finetune", "LLM: LoRA/QLoRA SFT wiring (peft LoraConfig) — attach a base model + GPU to train", {}),
    ("M", "llm-retrieve-rerank", "LLM: embedding retrieval + rerank (retriever runnable; cross-encoder rerank needs a model)", {}),
    ("M", "density-regression-head", "VISION: weakly-sup density head (per-pixel density summed to a count) from image totals (CSIRO/counting)", {}),
    ("M", "trajectory-forecaster", "TRACKING: per-agent GRU forecaster of future position deltas (NFL player tracking)", {}),
    ("M", "gpu-relaxation-solver", "OPTIMIZATION: gradient-descent projection minimizing an overlap penalty (santa-2025 packing feasibility)", {}),
    ("M", "geometric-packing-optimizer", "OPTIMIZATION: pack N congruent circles into the smallest square (lattice seed + shrink) (santa-2025)", {}),
    ("M", "program-search", "REASONING: enumerate a grid-transform DSL to find a program matching io examples (ARC)", {}),
    ("M", "program-golf-search", "CODE: shortest correct program by byte length (code-golf)", {}),
    ("M", "program-synthesis-data-generator", "REASONING: sample DSL programs → (input,program,output) triples for solver training (ARC)", {}),
    ("M", "fast-sim", "RL: vectorized BATCHED environment stepper (many envs at once) for self-play throughput", {}),
    ("M", "code-repair-agent", "SWE: localize→patch→VERIFY (run F2P+P2P tests) accept/skip loop (Konwinski bug-fixing)", {}),
    ("M", "llm-judge-attacker", "SECURITY: craft inputs maximizing score divergence across a judge panel (LLM-judge red-team, defensive framing)", {}),
    ("M", "ttc", "REASONING: test-time compute — augmentation-inverse-vote (AIRV) over predictions (ARC lever)", {}),
    ("M", "region-decompose-router", "VISION: detect sub-regions (panels/lanes) + route each to a specialist (recodai forgery)", {}),
    ("M", "dicom-metadata-estimator", "MEDICAL: estimate missing acquisition metadata (orientation/spacing) from the image (RSNA)", {}),
    ("M", "molecular-featurizer", "BIO: RDKit fingerprints + descriptors from SMILES (needs rdkit) (CAFA/polymer)", {}),
    ("M", "gp-symbolic-feature", "TABULAR: gplearn symbolic-regression features (needs gplearn)", {}),
    ("M", "automl-oof-factory", "TABULAR: AutoGluon OOF factory + auto-stacker (needs autogluon)", {}),
    ("M", "nnunet-segmentation-runner", "3D-SEG: nnU-Net v2 ResEnc runner (needs nnunetv2) (BYU/RSNA/vesuvius)", {}),
    ("M", "foundation-3d-matcher", "VISION: foundation dense matcher MASt3R/DUSt3R (needs mast3r+weights) (image-matching)", {}),
    ("M", "chess-search-engine", "GAME: alpha-beta search + NNUE eval (needs python-chess + engine) (fide-chess)", {}),
    ("M", "nnue-trainer", "GAME: tiny quantization-aware eval-net trainer w/ data filtering (needs torch+data)", {}),
    ("M", "vlm-pdf-corpus-miner", "NLP: VLM PDF extraction of aligned pairs (needs a VLM model) (deep-past)", {}),
    ("M", "ttt-transductive-finetune", "LLM: LoRA-finetune on test few-shot exemplars at inference (needs a base model) (jigsaw)", {}),
    ("S", "frozen-exploit", "Exploit 6bba frozen frames: static-edge injection + amortized-TTA dedup plan (per-embryo Δ)", {}),
    ("S", "submit-verify", "PROVE the T4-feasible pipeline locally end-to-end → valid submission.csv + per-embryo score (gate before any push)", {}),
    ("S", "nb-preflight", "VERIFY the Kaggle submission notebook offline-install LOCALLY before any push (env gate, 30h/wk quota)", {}),
    # 2026-07-12: heavy-weight-reuse + T4-feasibility score levers (build-your-own missing agents)
    ("B", "distill", "USE the heavy weights: distill Cellpose/micro-SAM (0.97 recall) into a fast one-pass UNet student → T4-fast + high recall",
     {"teacher": "cellpose-SAM", "nframes": 4}),
    ("B", "component-graft", "REUSE the external pretrained backbone/blocks under our fast detection head (graft weights, not the whole slow model)",
     {"source": "cellpose-cpsam", "keep_upto": 8}),
    ("S", "keyframe", "Big detector on SPARSE keyframes + rule-based fill of intermediate frames → make a slow accurate model T4-feasible",
     {"interval": 5, "big_spf_t4": 51.7, "cheap_spf": 0.024}),
    ("S", "quantize", "INT8-W8A8 PTQ + ToMe token-merge on a heavy ViT detector → could UNLOCK the best model on 2×T4 (what depth-pruning couldn't)",
     {"base_spf_t4": 51.7, "tome_r": 0.5}),
    ("M", "lowbit-qat", "LOW-BIT QAT (pure torch): BitNet-b1.58 ternary {-1,0,+1} absmean quantizer + int4/int8 fake-quant + straight-through-estimator + QuantLinear + wrap_qat (keeps norms/embeds/head fp) → FINE-TUNE a detector/head under ternary/int4 weights so bigger models fit the 5090/T4 (the QAT lever we lacked — quantize/compress-select are PTQ+depth only). Grounded in BitNet b1.58 (2402.17764); Bonsai/Ternary-Bonsai is PTQ of Qwen3, not training",
     {"scheme": "ternary", "steps": 150}),
    # ---- GENERIC cross-comp ONNX tool (ANY comp that ships a model: offline-budget shrink/verify/cost) ----
    ("S", "onnx", "GENERIC ONNX tool (all comps): export torch/sklearn→ONNX, VERIFY via onnxruntime, MEASURE cost (params + charged activation memory bytes + latency) for offline-budget triage, and QUANTIZE (fp16/int8, composing with quantize/compress-select). arc-onnx-golf layers neurogolf scoring on top of this", {}),
    # ---- GRID-REASONING / network-golf (ARC-AGI-ONNX, neurogolf-2026): deterministic ONNX-golf TOOLS.
    # The live researcher/leader agents (researchpapers runtime) drive these via fleet_dispatch; the ARC-
    # solving + architecture-rewrite reasoning lives in the brains, not in these python tools. ----
    ("S", "arc-idioms", "Query the ARC-ONNX construction catalogue (patterns.md by score band): ONNX idioms + rule families + cost-saving rules + worked exemplars → candidate constructions for a task's rule family", {}),
    ("S", "arc-onnx-golf", "Emit a minimal ONNX for an identified transform, VERIFY (output>0 one-hot equality on train+test+arc-gen), and return the OFFICIAL cost (memory+params) + score = max(1,25-ln(cost)). Faithful to neurogolf_utils (one-hot [1,10,30,30], opset10, ban-list, ORT-profiler memory)", {}),
    ("S", "arc-worker-context", "Assemble the rewrite-first per-task worker CONTEXT + PROMPT (best-known ONNX+builder+cost+target+history+similar-tasks+idioms) for the live researcher agent — architecture-rewrites>>pruning, concrete target, cross-task transfer, attempt-log/MEMORY.md", {"demo": True}),
]

HANDLERS = {
    "cv-build": cv.handle,           # (was researcher/leader) build embryo-disjoint CV via src.cv
    "analysis": metric.decompose,    # (was researcher) decompose metric from MLflow
    "aug-ablation": experiments.run_config,  # (was researcher dry-run + trainer submit) END-TO-END Python
    "arch-probe": experiments.run_config,
    "score": score_step.score_after_train,  # CLOSES THE LOOP: predict+score trained weights → golden_cv → journal
    "metrics-report": metrics_report.report,  # LEADER-facing COMPLETE metrics table after train+score
    "smoke": smoke.smoke,            # PRE-FLIGHT: tiny real run under a hard timeout (catch hangs before full)
    "train-monitor": monitor.watch,  # LIVE watchdog: GPU/CPU + log freshness; kill+escalate on a hang

    "guard": guard.check,            # (was trainer) post-run reliability check
    "journey-status": journey.status,  # the ordered first→final progression (grandmaster pattern)
    "aug-find": augfinder.find,      # derive valid augs from the data physics (e60–e62 rules)
    "ledger": ledger.report,         # the running +0.01 CV-delta experiment journal
    "notes-sync": note.sync,         # parse structured research notes → journal (research → Python)
    # pure-analysis agents
    "eda-stats": eda_stats.report,
    "adversarial-val": adversarial.report,
    "scorer": scorer.report,
    # journey stage-runners (S1..S8)
    "baseline": stages.baseline,
    "single-model-tune": stages.tune,
    "linking": stages.linking,
    "division": stages.division,
    "ensemble": stages.ensemble,
    "post-proc": stages.postproc,
    # the learner + kaggle scout + pre/post analysis
    "learn": learner.learn,
    "kaggle-scout": kaggle_scout.scout,
    "pre-analysis": preanalysis.diagnose,
    "post-analysis": postanalysis.verdict,
    # self-driving trio — the fleet runs the whole journey without the Claude leader/researcher
    "orchestrate": orchestrator.drive,   # LEADER's job: pick + enqueue the next experiment (deterministic)
    "config-gen": config_gen.generate,   # RESEARCHER's job: author a one-change config
    "split-build": split_build.build,    # RESEARCHER's job: build a validated, leak-checked split
    "insights": insights.report,         # HANDOFF: refresh the super-agent markdown (complete work + direction)
    "notebook-sync": notebook_sync.sync,   # DAILY: pull public notebooks + mine learnings
    "plan-ingest": plan_ingest.ingest,
    "reproduce-score": reproduce_score.score,
    "pipeline-run": pipeline_run.run,   # config-driven end-to-end: inference base + div-model + score (fast)
    "public-config": public_config.generate,  # one config/exp/public/*.yml per notebook (full coverage)
    "best-config": best_config.build,   # Part A: assemble best inference options (no training)
    "div-model": div_model.train,
    "deep-sister": deep_sister.train,
    "stage-1-div": stage1_div.run,
    "combo-search": combo_search.search,   # AUTONOMOUS: grid public-notebook post-proc on golden-12 (no Claude)
    "fullconfig-search": fullconfig_search.search,  # WIDE 8-axis search over the yaroslav-v4 full config → beat public
    "config-ablate": config_ablate.report,          # leave-one-block-out ablation of the yaroslav-v4 config
    "ext-label-stats": ext_label_stats.report,       # AFFINITY: inventory external dense labels (hengck23 recipe)
    "flow-gt-build": flow_gt_builder.build,          # AFFINITY: per-node flow+division GT from external tracks
    "gpu-best-practices": _gpu.report,               # RESEARCH: 5090/Blackwell best practices → search candidates
    "paper-research": _paper.report,                 # RESEARCH: recent arch innovations by accuracy+speed → arch-search candidates
    "sample-match": _smatch.run,                     # GATE: external data must match the author's sampling profile
    "perf-choice": _perf.run,                        # PERF: benchmark backends → data-driven fastest (no per-node loops)
    "box-sample": _boxs.run,                          # DENSITY-MATCH external → competition crops (vertex boxes)
    "data-audit": _daudit2.report,                   # DATA: measure+correct GT scale/outliers before training
    "arch-builder": arch_builder.build,
    "div-temporal-feas": _dtfeas.run,                # GO/NO-GO gate: temporal-vs-single division separability (frozen multi-pool probe)
    "psf-deconv": _psfd.run,                          # GO/NO-GO gate: does PSF deconv separate merged nuclei + help DoG precision + fit 2xT4/12h (SiMView no-deconv)
    "combined-train": _ctrain.run,                   # PAYOFF: train external(boxed)+competition → golden transfer test
    "pipeline": _pipeline.run,                        # RUN an ordered agent chain deterministically (no Claude between steps)
    "recipe-adopt": _radopt.run,                      # GRAFT a reproduced public recipe onto ours, keep only measured-positive knobs
    "det-sweep": _dsweep.run,                          # detection det×pool sweep → max node-recall at CALIBRATED count
    "cv-lb-calibrate": _cvlb.run,                      # fit CV→LB map from journal anchors → never submit blind (grandmaster discipline)
    "submit-guard": _sguard.run,
    "beat-bar": _bbar.run,                             # WORKFLOW: calibrate→adopt→det-sweep→submit-guard with grandmaster branching
    "improve-loop": _iloop.run,                        # WORKFLOW: diagnose weakest lever → route to its agent → score → repeat until converged
    "campaign": _camp.run,                             # TOP WORKFLOW: marshals the WHOLE fleet in 7 phases (total coverage — no agent unused)
    "saliency-detect": _sdet.run,                      # XAI saliency → ADD-ONLY candidate nuclei (weakly-sup detection → node-recall boost)
    "link-tune": _ltune.run,                           # EDGE-PRECISION / LINKING lever (recall saturated) — sweep motion-relink/gap/mtl → canonical golden-12
    "ext-transfer": _extf.run,                         # TRAIN on box-sampled external → embryo-disjoint transfer to competition (the ext-training CV)
    "domain-match": _dmatch.run,                        # REUSABLE any-comp domain matching (feature+image+learned adversarial mapper) → adv-AUC→0.5
    "gan-train": _gan.run,                              # REUSABLE GPU adversarial image trainer (translate/augment) — domain adaptation, synthetic aug, style transfer
    "train-tricks": _tricks.run,                        # REUSABLE GM training-tricks pack — EMA/SWA/mixup/cutmix/label-smoothing/focal/SAM/ArcFace (torch/CUDA)
    "pipeline-completeness": _pcomp.run,
    "gm-repo-distill": _gmdistill.run,
    "setup-env": _setupenv.run,
    "detector-transfer": _dtrans.run,                   # REUSABLE strong 3D detector + multi-seed per-embryo transfer eval (mean±std) + self-training — noise-robust GO/NO-GO
    "pattern-tune": _ptune.run,                        # autonomously tune box-sample until boxed external matches competition on ALL columns
    "math-master": _mm.run,
    "detector-arch-search": _das.run,
    "deep-research": _dres.run,
    "temporal-audit": _taud.run,                          # comprehensive data-quality scan: frozen frames, global setup-jumps, long-jumpers, label+image anomalies
    "lit-search": _lit.run,                          # deterministic HF(models+datasets)+arXiv literature search, journaled
    "mh-ilp": _milp.run,                             # multi-hypothesis DoG + tracksdata ILP (threshold-free node-recall lever), XAI-guided
    "detector-select": _dsel.run,                    # CHOOSE pretrained detector (arch+weights, NO training) by per-embryo training-CV recall proof
    "tracker-select": _tsel.run,
    "compress-select": _psel.run,                    # CHOOSE model-compression (ShortGPT layer-drop by Block-Influence / LaCo merge / width) by per-embryo recall + time budget
    "layer-grow": _lgrow.run,                        # choose depth layer-by-layer with proof + XAI validation
    "arch-search": _asearch.run,                     # EXECUTOR: train each arch candidate → prove the best
    "arch-catalog": arch_builder.catalog_query,      # QUERYABLE grounded modern-technique catalog + propose(target) — int8-not-FP4/ternary-needs-QAT/trust-region/LOEO-gate
    "gnn-link-train": _glt.train,                    # EXECUTOR: train division+flow heads on clean GT
    "xai": _xai.report,                              # XAI: saliency/occlusion/probe/attention — see-not-assume               # META: derive the architecture from data (layers/radius/k justified)
    "gnn-probe": gnn_probe.report,                    # GNN: does neighbourhood context beat pairwise geometry?
    "prior-art": _prior.report,                      # PRIOR-ART: CTC/ISBI top-solution coverage matrix
    "trick-gate": _tgate.run,                        # EVIDENCE GATE: adopt a trick only if golden-12 proves it
    "decision-audit": _daudit.report,                # ENFORCE decide-only-from-data across the ledger
    "trick-extractor": _trick.report,               # KAGGLE TRICKS: full trick-coverage matrix from top solutions
    "tracker-consensus": tracker_consensus.run,       # AFFINITY: multi-tracker consensus → in-domain pseudo-labels
    "block-synth": block_synth.synth,      # AUTONOMOUS: diff notebooks → compose NEW post-proc code-block recipes
    "combine-winners": grandmaster.combine_winners,  # GM TRICK: stack the best of each lever into one recipe
    "ablate-best": grandmaster.ablate_best,          # GM TRICK: "same as X but ONE change" (one-variable ablation)
    "verify-cv": verify_cv.run,            # AUTONOMOUS: REAL golden-12 run of a notebook's params (no hardcoded CV)
    "scoreboard": _sb.report,
    "heal": _heal.heal,                    # SELF-HEALING: escalate training failures to Claude to fix              # LIVE: one markdown-table message = the golden-CV leaderboard, updated in place
    "submission-build": submission_build.build,
    # ---- 2026-07-12: the submission-readiness + research + monitoring agents (were unregistered) ----
    "nb-preflight": _nbp.run,        # VERIFY the Kaggle notebook offline-install LOCALLY before any push (30h/wk quota gate)
    "submit-verify": _sv.run,        # PROVE the T4-feasible pipeline end-to-end locally → valid submission.csv + per-embryo score
    "lora-train": _lora.run,         # LoRA warm-start fine-tune of pilkwang UNetNodeTransformer → lift edge+division (div-weight lever)
    "tracker-train": _ttrain.run,    # train UNet detector + edge transformer (warm-startable)
    "center-train": _cdtrain.run,    # train full-frame center-prior detector (--resume)
    "tracker-predict": _tpredict.run,  # detect + edge + ILP → per-dataset .geff
    "tracker-postproc": _tpostp.run,   # pilkwang post-proc (fuse/relink/gap/safe-div/smooth) + submission.csv
    "full-cv-baseline": _fcvb.run,     # predict+ILP+score BASE on all 199 → honest full-CV baseline (the bar)
    "lora-validate": _lval.run,        # score a trained LoRA adapter on all 199 → real generalization delta vs baseline
    "detect-quality": _dq.run,         # detector recall+PRECISION on external 100%-dense crops → expose over-detection
    "stage-dynamics": _sdyn.run,       # per-stage motion+division from GT → math-master significance → stage-adaptive ILP priors
    "official-score": _oscore.run,     # score geffs with the ORGANIZER metric (edge+division) — honest division_jaccard, not the proxy
    "lever-hunt": _lhunt.run,          # metric-driven mini-experiment loop: XAI subset+goal → lever → official-metric verify → SOLID/DEAD
    "feasibility-gate": _feas.run,     # ORCHESTRATION: chains xai-diagnose → lever → math-master → official-score → ledger → insights; each GO/NO-GO gate becomes a durable Lever Feasibility Map insight (never re-run a killed lever)
    "feasibility-map": _feas.run_map,  # render the Lever Feasibility Map (GO/NO-GO table + mechanisms) → /insights (docs/INSIGHTS.md)
    "division-rescue": _drescue.run,   # add geometry-consistent rate-capped 2nd-child forks → lift division_jaccard (the geometry lever)
    "research-search": _rs.run,      # REUSABLE multi-source (HF/arXiv/bioimage/zenodo/github/kaggle+discussions/europepmc/figshare) model+paper search, BM25 index
    "lb-sync": _lbs.run,             # SNAPSHOT the official leaderboard → submission-recency/activity analytics (per-comp PG)
    "paper-verify": _pverify.run,    # PROVE Zebrahub(Cell 2024)+Ultrack(PMC12615266) claims vs our training data → verdict table (cached)
    "official-conformance": _oconf.run,  # PROVE conformance w/ official baseline: metric-core identical + submission schema + division sandbox
    "git-track": _gtrack.run,            # COMMIT code (parent+official_repo) → hash; stamps ledger so experiments map to code state
    "metric-probe": _mprobe.run,         # REUSABLE adversarial metric-vulnerability prober: structural perturbations that move the score without improving correctness → ranked exploit report + bug class (LB-unreliability / CV-guard / bug-report, NOT to submit)
    "coverage-audit": _covaudit.run,   # FLEET MAP: every agent in its pack (biohub=reusable 3D+time pack)
    "comp-onboard": _onboard.run,        # FRONT DOOR: any slug → CompConfig + pack route (or unknown-comp gap report)
    "kaggle-modality": _kmod.run,        # META: ground each active comp's MODALITY in Kaggle's OWN tags/category (Meta Kaggle join) → cached map that comp-onboard + :7788 read offline
    "tab-profile": _tabprof.run,         # TABULAR: fingerprint data (drift/leakage/balance) from CompConfig
    "tab-fe": _tabfe.run,                # TABULAR GM FE: leak-safe OOF target/freq-enc + row-aggs + interactions
    "tab-train": _tabtrain.run,          # TABULAR: CV-train installed backends (GPU-auto) → OOF+test, honest CV
    "tab-stack": _tabstack.run,          # TABULAR: metric-optimal blend of tab-train OOFs
    "tab-autobaseline": _tabauto.run,    # TABULAR TURNKEY: profile→train→blend→submission in one call
    "agent-env": _agenv.run,             # AGENTIC: onboard env (actions/reward/budget) + smoke rollout
    "agent-policy": _agpol.run,          # AGENTIC: candidate-tournament policy search → best policy
    "agent-eval": _ageval.run,           # AGENTIC: offline policy score + budget compliance before submit
    "gm-writeup-mine": _gmwm.run,        # GROUND: fetch real top-solution writeups → docs/gm_writeups/ (repeatable sweep)
    "sub-journal": _subj.run,            # META/MINING: sync EVERY Kaggle submission into the journal (public+private, cv-from-desc, sub_<ref>.json provenance) — journal-completeness, idempotent by ref

    "pseudo-label": _pseudo.run,         # GM TOOLKIT: self-training pseudo-label selection (#1 repeated lever)
    "blend-optimize": _blend.run,        # GM TOOLKIT: best-of {hill-climb/Caruana/Nelder-Mead/Ridge} blend
    "post-optimize": _postopt.run,       # GM TOOLKIT: QWK-round/clip/temperature/quantile-thr/rank-average
    "calibrate": _calib.run,             # GM TOOLKIT: Platt/isotonic/temperature calibration + ECE
    "target-transform": _ttrans.run,     # GM TOOLKIT: rank-gauss/log/sqrt + survival factorization
    "solution-adopt": _soladopt.run,     # ADOPT: comp's top-1..5 solutions → executable workflow of reusable agents
    "skill-build": _skillb.run,          # PROMPT-PROGRAM: author deterministic AutoML floor SKILL (the winning lever)
    "agent-author": _agauth.run,         # PROMPT-PROGRAM: author ADK agent bundle (agent.yaml+system.md+skill)
    "agent-package": _agauth.run_package,  # PROMPT-PROGRAM: validate + zip-at-root → submission.zip
    "agent-config-eval": _ageval2.run,   # PROMPT-PROGRAM: offline hidden-label smoke matrix → mean AUC gate
    "prompt-optimize": _ageval2.run_optimize,  # PROMPT-TUNING: evaluate prompt/skill variants by hidden AUC
    "subset-classifier-router": _gap.run_router,       # GAP: family classifier → specialist routing (MoE)
    "analysis-by-synthesis-refiner": _gap.run_refiner, # GAP: forward-operator test-time refinement (inverse problems)
    "checkpoint-merger": _gap.run_merger,              # GAP: weight-space merge (linear/TIES)
    "constrained-label-assignment": _gap.run_assign,   # GAP: Hungarian decode under count constraints
    "lb-shift-prober": _gap.run_prober,                # GAP: affine/offset correction from a probe grid
    "github-solution-mine": _ghmine.run,               # GROUND FROM CODE: harvest winners' GitHub repos → key ML modules
    "fin-ta-feature-library": _domfe.run_fin,          # DOMAIN FE: financial TA features (mitsui/jane-street)
    "imu-feature-engineer": _domfe.run_imu,            # DOMAIN FE: IMU/sensor kinematic features (cmi)
    "sr-bf16-optimizer": _srbf.run,                    # TRAINING MEMORY: stochastic-rounding bf16 AdamW (half-memory states)
    "quaternion-imu-features": _qimu.run,              # DOMAIN FE: quaternion orientation features + SO(3) augment (cmi 1st)
    "online-walk-forward-retrainer": _domfe.run_online,  # TIMESERIES: incremental online retraining
    "synth-artifact-fe": _tabdiv.run_synth,            # TABULAR: generator-fingerprint FE (synthetic-from-original)
    "oof-diversity-prune": _tabdiv.run_prune,          # ENSEMBLE: prune near-twins, keep decorrelated OOFs
    "feature-select": _tabdiv.run_fselect,             # TABULAR: consensus-importance top-K selection
    "residual-boost": _tabdiv.run_resboost,            # ENSEMBLE: fit booster on baseline residuals
    "full-retrain-calibrator": _tabdiv.run_retrain,    # ENSEMBLE: 100%-train retrain iters + seed-avg
    "ts-decompose-forecaster": _fsp.run_ts,            # FORECAST: multiplicative ratio decomposition
    "forecast-trend-extrapolator": _fsp.run_trend,     # FORECAST: future-horizon trend multiplier
    "rating-systems": _fsp.run_rating,                 # SPORTS: Elo/Colley/SRS ratings
    "outcome-sharpen": _fsp.run_sharpen,               # SPORTS: Brier tail sharpening + overrides
    "best-of-n-diversity-allocator": _fsp.run_bestofn, # BEST-OF-N: diverse candidate allocation
    "temporal-segment-decoder": _fsp.run_segdecode,    # SEGMENTATION: frame-prob → action segments
    "masked-sequence-norm": _mseq.run_norm,            # SEQUENCE: masked BatchNorm z-score (padding excluded)
    "masked-sequence-pool": _mseq.run_pool,            # SEQUENCE: masked mean/max/attention pooling
    "class-balance-sampler": _imbs.run,                # IMBALANCE: tempered class-balanced sample weights
    "deep-supervision": _thp.run_deep_supervision,            # TRAINING: multi-scale max-pooled DS loss
    "sed-attention-pool": _thp.run_sed_attention,             # TRAINING: weakly-supervised attention pooling head
    "audio-melspec-fe": _audio.run_melspec,                   # AUDIO: waveform → log-mel spectrogram (pure torch.stft + mel filterbank)
    "audio-augment": _audio.run_augment,                     # AUDIO: SpecAugment + waveform aug (noise/gain/bg-mix/OR-mixup)
    "audio-crop-tta": _audio.run_crop_tta,                   # AUDIO: fixed-crop train + sliding-window TTA aggregation (long clips)
    "audio-backbone": _audio.run_backbone,                   # AUDIO: mel→CNN classifier (timm EfficientNet else small CNN)
    "audio-train": _audiotrain.run,                          # AUDIO: end-to-end multi-label trainer (soundscape LB-proxy CV + author-grouped focal CV, small-first ladder)
    "audio-infer": _audioinfer.run,                          # AUDIO: CPU sliding-window inference → Kaggle submission.csv (the offline notebook body)
    "kaggle-submit": _ksubmit.run,                           # SUBMISSION: package ckpt→dataset + self-contained offline notebook → push kernel → submit → read PUBLIC+PRIVATE LB (reusable code-comp submitter)
    "graph-message-passing": _graph.run_message_passing,     # GRAPH: general pure-torch MPNN (mean/max/sum/attention aggr, edge feats, residual+norm) via scatter/index_add
    "graph-feature-extractor": _graph.run_feature_extractor, # GRAPH: node FE (degree/clustering/k-hop) + Laplacian/random-walk positional encodings + edge/global feats
    "graph-readout": _graph.run_readout,                     # GRAPH: graph-level pooling (mean/sum/max/attention/Set2Set) node embeddings + batch → per-graph vector
    "multimodal-fusion": _mmf.run_fusion,                     # MULTIMODAL: FEATURE-level fusion (concat/sum/gated/FiLM/bilinear/cross-attention) of per-modality tensors → fused rep (+head)
    "modality-encoder-adapter": _mmf.run_encoder_adapter,     # MULTIMODAL: align image/text/tabular inputs → L2-normed shared-dim embeddings (per-modality LayerNorm+proj+type-emb)
    "modality-dropout": _mmf.run_modality_dropout,            # MULTIMODAL: train-time modality masking + inference missing-modality imputation via learned null token
    "video-frame-sampler": _vid.run_frame_sampler,            # VIDEO: sample T frame indices (uniform/stride/dense-around-event/random-jitter) + gather helper
    "video-temporal-aggregator": _vid.run_temporal_aggregator, # VIDEO: per-frame emb [B,T,D] → clip vec via mean/max/attention/tconv/TSM/GRU (image backbone → video model)
    "video-motion-features": _vid.run_motion_features,        # VIDEO: frame-diff + temporal-gradient + flow-magnitude proxy + motion-channel stacker (motion cue)
    "awp-perturb": _thp.run_awp,                              # TRAINING: adversarial weight perturbation regulariser
    "muon-optimizer": _muon.run,                              # 2026: Muon Newton-Schulz orthogonalized optimizer
    "conformal-predict": _conf.run,                           # 2026: conformal prediction sets/intervals (coverage)
    "schedule-free": _sfree.run,                              # 2026: Schedule-Free optimizer (no LR schedule)
    "dora-adapt": _dora.run,                                  # 2026: DoRA weight-decomposed low-rank adaptation
    "hardware-tune": _hwtune.run,                             # HARDWARE: profile GPU + benchmark dtype → best training config for this box
    "prompt-metric": _pmetric.run,                            # PROMPT-OPT: named metric → score+feedback callables
    "prompt-dataset": _pdataset.run,                          # PROMPT-OPT: build {input,gold} trainset from inline/file/synthetic
    "style-fingerprint": _styfp.run,                          # AUTHOR STYLE: interpretable per-attorney fingerprint (signature phrases/openers/structure/boilerplate/micro) → score+feedback+discrimination-AUC; as_metric plugs into dspy-prompt-optimize
    "dspy-prompt-optimize": _dspyp.run,                       # PROMPT-PROGRAM: DSPy optimizers + from-scratch GEPA+APEX reflective evolution (offline)
    "combinatorial-local-search": _opt.run_localsearch,       # OPTIMIZATION: ILS+SA over permutations
    "population-diversity-manager": _opt.run_population,       # OPTIMIZATION: GA with diversity selection
    "batched-oracle-search-harness": _opt.run_oracle,         # OPTIMIZATION: memoized oracle + island search
    "gaussian-heatmap-encoder": _vdet.run_encode,             # DETECTION: keypoints → N-D Gaussian heatmap target
    "volumetric-patch-inference": _vdet.run_patch,            # DETECTION: N-D patch tiling + overlap-avg stitch
    "heatmap-peak-decoder": _vdet.run_decode,                 # DETECTION: heatmap → centroids (peak-NMS / blob)
    "wbf-fusion": _inf.run_wbf,                               # INFERENCE: Weighted Boxes/Points Fusion
    "snapshot-average": _inf.run_snapshot,                    # INFERENCE: snapshot/seed logit-prob-rank averaging
    "multi-tta": _inf.run_tta,                                # INFERENCE: invertible multi-transform TTA + fuse

    "shift-adapt": _rob.run_shift,                     # ROBUSTNESS: shift importance weights + aligned CV
    "geospatial-fe": _rob.run_geo,                     # DOMAIN FE: grid-cell TE + spatial-KNN
    "linear-constraint-projector": _rob.run_project,   # POSTPROC: project onto Ax=b constraint manifold
    "runtime-budget-router": _rob.run_router,          # RUNTIME: budget-aware expensive/fast routing
    "mbr-consensus-selector": _rob.run_mbr,            # DECODE: minimum Bayes risk candidate selection
    "noisy-label-cleaner": _rob.run_cleanlabel,        # DATA: conflicting-label resolution + soft target
    "knn-label-transfer": _rob.run_transfer,           # RETRIEVAL: similarity-weighted label transfer
    "self-consistency-aggregator": _llmi.run_selfconsistency,  # LLM-INFER: vote-share answer aggregation
    "consensus-early-stop": _llmi.run_earlystop,       # LLM-INFER: stop sampling on consensus/uncatchable
    "risk-abstain-gate": _llmi.run_abstain,            # LLM-INFER: EV-based submit/skip under penalty
    "budget-aware-inference-scheduler": _llmi.run_scheduler,   # LLM-INFER: per-problem time budgeting
    "sample-pool-simulator": _llmi.run_poolsim,        # LLM-INFER: O(1) inference-config accuracy simulation
    "mtp-speculative-decode": _mtp.run_mtp,            # LLM-INFRA: Gemma-4 MTP drafter speculative-decode speedup (§2.6/Fig 1)
    "kv-cache-longctx": _kvc.run_kv,                   # LLM-INFRA: Gemma-4 long-context KV-cache profiler + 37.5% reduction lever (§2)
    "moe-inference-cost": _moe.run_moe,                # LLM-INFRA: Gemma-4 26B-A4B MoE active-vs-total param/FLOP/memory cost (Table 1)
    "moe-quantile-balance": _moeqb.run_qbalance,       # LLM-INFRA: Kimi-K3 Stable LatentMoE aux-free quantile load-balancer
    "sparsity-metrics": _spm.run_sparsity,             # TRAINING: sparsityLLM hidden-state sparsity difficulty/OOD signal + curriculum ordering
    "shap-emd-distance": _shemd.run_shapemd,           # XAI/METRIC: shap-emd model-aware optimal-transport distance on compositional features
    "geometric-features": _geo.run_geomfeat,           # GEOMETRIC-DL: torchmd-net PhysNet RBF/cutoff point-cloud→edge featurizer (equivariant)
    "flow-matching": _flow.run_flowmatch,              # GENERATIVE: ppflow OT conditional flow-matching objective + Euler sampler
    "llm-backend": _llmb.run_llmbackend,               # INFRA: omnigent-pattern multi-provider LLM client (Ollama/OpenRouter/Claude/local/dummy)
    "mup-scaling": _mup.run_mup,                       # SCALING: ArchScale μP width-scaling rules (hyperparameter transfer across widths)
    "gpu-patterns": _gpp.run_gpupatterns,              # GPU/KERNELS: GPU-Puzzles parallel primitives + roofline compute/memory-bound cost model
    "deck-builder": _deck.run_deck,                    # REPORTING: ppt-master-distilled spec->pptx deck builder
    "video-builder": _vidb.run_video,                   # REPORTING: frames->GIF/MP4 assembler (remotion alternative, imageio)
    "task-spec": _tspec.run_taskspec,                  # ORCHESTRATION: MindOS spec-driven acceptance-gated task contract
    "turboquant": _tq.run_turboquant,                  # RETRIEVAL/QUANT: TurboQuant data-oblivious embedding quantizer (8x, training-free)
    "attention-residual": _attnres.run_attnres,        # ARCH: Kimi-K3 AttnRes selective depth-retrieval residual
    "latent-moe": _latmoe.run_latentmoe,               # ARCH: Kimi-K3 latent-space MoE + MLA KV-cache compression
    "rswa-attention": _rswa.run_rswa,                  # ATTENTION: Unlimited-OCR R-SWA constant-KV-cache long-context decoding
    "embedding-retrieval": _emb.run_embedding,         # RAG/RETRIEVAL: Nemotron-3-Embed + potion-code embedding search/MMR/dedup
    "nvfp4-loader": _nvl.run_nvfp4loader,              # BLACKWELL/LLM: Gemma-4 NVFP4 load+finetune planner for RTX 5090
    "diffusion-sampler": _diff.run_diffusion,          # GENERATIVE: hackable_diffusion DDPM/DDIM torch port (complements flow-matching)
    "ui-component": _uic.run_ui,                       # REPORTING/UI: astryx-style token→HTML dashboard generator
    "shopify-agent": _shop.run_shopify,                # COMMERCE: Shopify analytics + Admin API + app scaffold (2026 SDKs)
    "lightning-tricks": _ltricks.run_lightning,            # TRAINING: PyTorch Lightning speed/config advisor (hardware-tuned)
    "santa-agent": _santa.run_santa,                   # OPTIMIZATION: Santa combinatorial-opt toolkit + 2019-2025 solution KB
    "skill-optimizer": _skopt.run_skillopt,            # PROMPT-OPT: SkillOpt SGD-analogy prompt/skill optimizer with held-out gate
    "portfolio-position-sizer": _fin.run_sizer,        # FINANCE: vol-targeted risk-managed allocation
    "market-odds-blend": _fin.run_odds,                # SPORTS/FINANCE: no-vig market blend
    "forecast-drivers-then-derive": _fin.run_drivers,  # TIMESERIES: forecast drivers → known formula
    "label-lag-anchor-blend": _fin.run_anchor,         # TIMESERIES: persistence-anchor blend
    "distributional-metric-recalibrator": _fin.run_recal,  # POSTPROC: per-group affine recalibration
    "expression-search": _rcp.run_expr,                # CODE: shortest arithmetic expression search
    "code-compress-optimizer": _rcp.run_compress,      # CODE: deflate byte minimizer + self-extract stub
    "sprt-spsa-tuner": _rcp.run_sprt,                  # TUNING: SPRT accept/reject + SPSA step
    "lb-formula-prober": _rcp.run_lbformula,           # KAGGLE-META: recover hidden linear scoring formula
    "trainable-trace-auditor": _rcp.run_traceaudit,    # LLM: reasoning-trace learnability audit
    "hierarchy-consistency-postproc": _misc.run_hier,  # POSTPROC: DAG parent≥child propagation
    "invariance-feature-normalizer": _misc.run_invar,  # DOMAIN FE: egocentric frame-canonical coords
    "template-retrieval-reranker": _misc.run_template,  # RETRIEVAL: retrieve + rerank templates
    "calendar-holiday-fe": _misc.run_calendar,         # DOMAIN FE: calendar/holiday features
    "annotation-error-corrector": _misc.run_annoterr,  # DATA: flag likely-wrong GT for review
    "binary-size-compressor": _misc.run_bincompress,   # DEPLOY: deflate artifact under size cap
    "knn-feature": _misc.run_knnfeat,                  # TABULAR: OOF kNN meta-features
    "geometric-spatial-augmentor": _fpp.run_augment,   # AUG: coordinate rotate/flip/dropout/jitter
    "infer-cascade": _fpp.run_cascade,                 # LLM-INFER: confidence cascade escalation
    "llm-synthetic-drill-generator": _fpp.run_drills,  # LLM: template synthetic drill generation
    "heteroscedastic-uncertainty-head": _fpp.run_hetero,  # UNCERTAINTY: ensemble → (mu,sigma)
    "tab-nn-train": _tabnn.run,                        # TABULAR NN: real torch neural-tabular trainer
    "sdf-regression-loss": _hrp.run_sdf,               # 3D-SEG: SDF target + boundary loss
    "topology-aware-loss": _hrp.run_topology,          # 3D-SEG: Dice + Betti-0 topology score
    "ae-latent-view": _hrp.run_ae,                     # TABULAR: torch autoencoder latents
    "keypoint-match-verifier": _hrp.run_keypoint,      # VISION: SIFT + RANSAC inliers
    "grid-rectification-unwarp": _hrp.run_rectify,     # VISION: cv2 homography dewarp
    "tir-executor": _llmp.run_tir,                     # LLM: sandbox code execution (TIR)
    "llm-eval": _llmp.run_eval,                        # LLM: metric eval of generations
    "llm-infer": _llmp.run_infer,                      # LLM: constrained-decode logits masking
    "llm-finetune": _llmp.run_finetune,                # LLM: LoRA SFT wiring
    "llm-retrieve-rerank": _llmp.run_rerank,           # LLM: embedding retrieval + rerank
    "density-regression-head": _hr2.run_density,       # VISION: density-sum counter
    "trajectory-forecaster": _hr2.run_trajectory,      # TRACKING: GRU trajectory forecaster
    "gpu-relaxation-solver": _hr2.run_relax,           # OPTIMIZATION: overlap-penalty relaxation
    "geometric-packing-optimizer": _hr2.run_pack,      # OPTIMIZATION: circle packing
    "program-search": _rep.run_progsearch,            # REASONING: DSL program search
    "program-golf-search": _rep.run_proggolf,         # CODE: shortest-program search
    "program-synthesis-data-generator": _rep.run_synthdata,  # REASONING: DSL data generation
    "fast-sim": _rep.run_fastsim,                     # RL: batched env stepper
    "code-repair-agent": _rep.run_coderepair,         # SWE: patch verify loop
    "llm-judge-attacker": _rep.run_judgeattack,       # SECURITY: judge-divergence attack
    "ttc": _rep.run_ttc,                              # REASONING: AIRV test-time vote
    "region-decompose-router": _scaf.run_region,      # VISION: panel/lane decomposition
    "dicom-metadata-estimator": _scaf.run_dicom,      # MEDICAL: metadata estimation
    "molecular-featurizer": _scaf.run_molecular,      # BIO: rdkit (guarded)
    "gp-symbolic-feature": _scaf.run_gp,              # TABULAR: gplearn (guarded)
    "automl-oof-factory": _scaf.run_automl,           # TABULAR: autogluon (guarded)
    "nnunet-segmentation-runner": _scaf.run_nnunet,   # 3D-SEG: nnunetv2 (guarded)
    "foundation-3d-matcher": _scaf.run_foundation,    # VISION: mast3r (guarded)
    "chess-search-engine": _scaf.run_chess,           # GAME: python-chess (guarded)
    "nnue-trainer": _scaf.run_nnue,                   # GAME: nnue trainer (guarded)
    "vlm-pdf-corpus-miner": _scaf.run_vlm,            # NLP: VLM (guarded)
    "ttt-transductive-finetune": _scaf.run_ttt,       # LLM: test-time finetune (guarded)
    "frozen-exploit": _fex.run,      # EXPLOIT frozen frames (6bba): static edges + amortized-TTA dedup plan (temporal_audit detects; this uses)
    # ---- 2026-07-12: the score-lever agents that were MISSING (heavy-weight reuse + T4 feasibility) ----
    "distill": _dist.run,            # USE heavy weights: Cellpose/micro-SAM teacher → fast one-pass UNet student (offline distill, no-train inference)
    "component-graft": _cg.run,      # REUSE external pretrained BACKBONE/blocks under our fast head (graft weights, not whole model) — task #27
    "keyframe": _kf.run,             # big detector on SPARSE keyframes + rule-based fill of intermediate frames (spend T4 budget where it matters)
    "quantize": _qz.run,            # INT8-W8A8 PTQ + ToMe token-merge (the 2 biggest T4-ViT wins) — could UNLOCK the best model on T4
    "lowbit-qat": _lbq.run,         # LOW-BIT QAT (pure torch): BitNet-b1.58 ternary + int4/int8 fake-quant + STE + QuantLinear/wrap_qat → fine-tune under quantized weights (the training lever quantize/compress-select lacked)
    # ---- GENERIC cross-comp ONNX tool + GRID-REASONING network-golf TOOLS (deterministic; no LLM) ----
    "hf-kernels": _hfkernels.run,          # GENERIC: discover + arch-check (cuobjdump sm_120) HF Hub kernels-community kernels for THIS box; no local build
    "onnx": _onnx.run,                     # GENERIC: export/verify/cost(params+memory+latency)/quantize ONNX for ANY comp (offline-budget triage, deployment)
    "arc-idioms": _arcidioms.run,          # query the patterns.md ONNX-golf construction catalogue (idioms/families/cost-rules/exemplars) by score band
    "arc-onnx-golf": _arcgolf.run,         # emit + (output>0) verify + OFFICIAL cost(memory+params)/score for an identified transform (faithful to neurogolf_utils)
    "arc-worker-context": _arcctx.run,     # assemble the rewrite-first per-task worker context+prompt (best/target/history/similar/idioms) for the live researcher agent
    # ---- 2026-07-19: CONTEXT MANAGEMENT (deepagents delta) — offload big outputs to disk, keep the thread compact ----
    "context-offload": _coff.run,          # GENERIC: write a LARGE tool/worker output to output/run_artifacts/<comp>/ and return a compact summary + path (read back with mode=read); the one deep-agent primitive the fleet lacked
    "harness-opt-gate": _hgate.run,        # PROMPT/HARNESS SELF-OPT: blind-holdout keep-if-combined-improves gate (deepagents better-harness) for the prompt-optimize triad ONLY — NOT ML experiments
}

# every agent extends BaseAgent (via FunctionAgent adapter) so none miss the main features
from . import base  # noqa: E402
AGENTS = base.build_agents(HANDLERS)   # {kind: BaseAgent} — spec/state/post/log/escalate/verify for all
_RAW_HANDLERS = dict(HANDLERS)
HANDLERS = {k: a.run for k, a in AGENTS.items()}   # fleet calls go through the test-quarantine GATE


def verify_all():
    """Run every agent's DATA-WISE verifier; returns {name: passed}."""
    return {name: ag.verify() for name, ag in AGENTS.items() if ag.has_test()}
