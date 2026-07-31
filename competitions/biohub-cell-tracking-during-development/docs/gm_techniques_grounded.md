# Grounded GM Techniques — mined from REAL 2025-26 Kaggle top-5 writeups

Source: 87 actual solution writeups (1st–5th place) across 19 finished competitions, fetched via the
nvidia-kaggle bearer API (`fetch_leaderboard_writeups`/`fetch_writeup`) into `docs/gm_writeups/<slug>/`,
distilled by extraction sub-agents. This is what the top solutions ACTUALLY did — not priors. Each item is
tagged with the comp(s) that prove it. These drive the agent design (metric-registry, math-master, packs).

Comps covered: playground-s5e4/s5e7/s5e11, child-mind-piu, equity-post-HCT, jane-street, rohlik-v2,
stanford-rna-3d-folding, czii-cryo-et, byu-flagellar-motors, isic-2024, rsna-2024-lumbar, birdclef-2025,
wsdm-chatbot-arena, eedi, drawing-with-llms, map-charting, arc-prize-2024, lux-ai-season-3.

## CROSS-CUTTING META-LEVERS (appear in the MAJORITY of comps — the real "GM quality")

1. **CV matched to the split, leak-free** — *every* comp. GroupKFold by patient/study/experiment/well
   (isic, rsna, czii, equity), time-holdout with a gap fold to mimic private non-stationarity (jane-street,
   rohlik), StratifiedKFold on target bins (child-mind, s5e11). Trust CV over LB where it correlates; where
   it doesn't (rna, czii, byu, birdclef >0.9), teams switch to LB — and *say so*. → `split-build`⟳, `cv-lb-calibrate`, `adversarial-val`.
2. **Pseudo-labeling / self-training** — child-mind, s5e4, s5e11, birdclef (0.87→0.93 via Noisy-Student!),
   byu, rsna, isic, wsdm, eedi, map-charting. THE single most repeated lever. Power/temperature-transform the
   soft labels to denoise. → new `pseudo-label` reusable agent + option in tab/img/llm packs.
3. **Blend-weight optimization beyond averaging** — Caruana greedy hill-climb (s5e11, equity), Nelder-Mead
   (rsna, s5e4), Ridge/linear stack (s5e11 — linear beats non-linear when models overfit), differential-
   evolution (s5e11), Optuna weights (equity, isic), multi-level stack with a NON-linear L2 that routes on a
   condition (s5e4 with-ELM vs without). → math-master `caruana_ensemble_selection`/`nelder_mead_weights`/`ridge_stack`; `tab-stack` L2.
4. **Metric-specific post-processing** — QWK OOF threshold/rounding optimization (child-mind), prediction
   CLIPPING to guard outliers (s5e4 unclipped→RMSE 177; jane-street clip[-5,5]), quantile-threshold on max
   for detection (byu, czii), temperature scaling of logits (rsna T=0.91, wsdm), rank-averaging (isic).
   → math-master `optimized_rounder`/`clip_guard`/`platt_scale`; a `post-proc`⟳ generalization.
5. **Loss design for imbalance / the metric** — weighted CE [1,2,4] (rsna severe), heavy positive-pixel
   weighting 256× (czii), TopK-worst-voxel BCE (byu), recall-tilted for F-beta (czii F4, byu F2), BCE>MSE for
   ranking targets (equity), zero-mean weighted-R² loss (jane-street), sqrt/Tweedie target for MAE (rohlik).
   Focal/label-smoothing frequently tried and REJECTED (rsna, birdclef-1st). → training-config knobs, not new agents.
6. **EMA + TTA + Mixup/CutMix** — every vision/audio comp (isic, rsna, birdclef, czii, byu). SpecAugment/Sumix
   for audio. → `aug-find`✓, add EMA + TTA to img/vid train wrappers.
7. **Small models + loss/representation design beat big models** — czii, byu, rsna, isic all say small
   backbones (convnext-small, effnet-b0/b3) + smart targets won. → bias `arch-search` toward efficient backbones.
8. **Reuse open-source / pretrained + best-of-N diversity** — rna-folding (DRfold2/Protenix/Boltz combined,
   diverse candidates for best-of-5 TM-score), wsdm/eedi/map (Qwen2.5 + distill from 72B teachers). Our biohub
   lesson generalized. → `component-graft`✓, `distill`✓, `recipe-adopt`✓.
9. **Target/count encoding of pair & n-gram column combos + digit-extraction** — the DOMINANT tabular FE
   lever (s5e11, s5e4, rohlik). Multi-bin then TE; descriptive stats OVER the TE columns. → `tab-fe`⟳ (add now).
10. **Target factorization** — split a hard target into sub-tasks: efs classifier + efs_time regressor
    (equity), regress hidden continuous then threshold (child-mind), efs-as-feature. → `tab` option.

## PER-MODALITY (grounded specifics)

### Tabular (child-mind, equity, s5e4/s5e7/s5e11)
Metrics: **QWK**, **stratified C-index**, **AUC**, **RMSE**. Models: LightGBM/XGBoost/CatBoost + TabM/RealMLP
NN for diversity + AutoGluon. FE: pair/n-gram target-enc, digit-extract, descriptive-stats-over-TE. Ensemble:
Ridge/Caruana/Nelder-Mead over a large diverse zoo; non-linear L2 stack. Post: QWK-rounding, clip. GPU: RAPIDS cuML.

### 3D / volumetric (rna-folding, czii-cryo-et, byu-motors)
Metrics: **TM-score (best-of-5)**, **F4/F2 @ distance**. Coarse heatmap/blob regression on 3D/2.5D-UNet
(downscale 8-32× exploiting distance tolerance) → CC3D centroids / NMS max-pool peak → quantile-threshold on
max. Recall-tilted loss. Reuse pretrained folders + diverse best-of-N. TTA, WBF, EMA, deep supervision.
(Directly relevant to biohub detection: NMS-maxpool peak + quantile-threshold + recall-tilted loss.)

### Image / audio (isic, rsna, birdclef)
Metrics: **partial-AUC@80%TPR**, **weighted-log-loss**, **macro-ROC-AUC**. ISIC: image-OOF as meta-features
into GBDT + rank-average (image→tabular fusion). RSNA: two-stage keypoint-crop→classify, per-level samples.
BirdCLEF: SED head on log-mel, iterative Noisy-Student self-training, Mixup/Sumix/SpecAugment, OpenVINO/ONNX.
Small backbones, EMA, TTA, patient/study-grouped CV.

### LLM (wsdm, eedi, map-charting, drawing-with-llms)
Metrics: **accuracy**, **MAP@25/@3**, VQA-composite. Patterns: Qwen2.5/3 backbones; retrieve-then-rerank
cascade (eedi); per-question candidate-restricted CHOICE classification (map-charting); distill 72B→14B
(soft-label KL, T=5) ; synthetic data + hard-negative mining; order-swap TTA; multi-seed weight merging;
R-Drop+AWP+EMA for label noise. **Offline 2×T4 is decisive**: AWQ/GPTQ/auto-round/**SmoothQuant-W8A8**
(LMDeploy), vLLM prefix-cache, single-token gen with `allowed_token_ids`, **multi-stage confidence cascade**
(small model → re-infer only uncertain with big), layer-wise disk-streaming to run 32B on one T4, **float16
not bf16 on T4 (2×)**. → `llm-finetune`(⟳lora-train), `llm-infer`(cascade), `quantize`(add W8A8/AWQ/GPTQ), `llm-eval`.

### Reasoning (arc-prize-2024)
Metric: **% solved exact-match**. THE lever: **test-time fine-tuning / active inference** — a per-task LoRA
(~300 steps, batch 1, 100 models/submission) jumped one model 11→33 solved. Data aug (rot/flip/color/
transpose) + **inference-time-augmentation voting (AIRV)**. Classical **DSL + DAG program search** (rank4)
still competitive; ensemble classical+LLM; verifier/selector heads; LB brute-force task-mapping. Fit 12h via
tiny per-task LoRAs + vLLM. → `reason-dsl`, `program-search`, `ttc`(=test-time-training+AIRV), `reason-eval`.

### Agentic (lux-ai-season-3)
Metric: **win-rate/TrueSkill**. Winners: large-scale **self-play RL** (IMPALA/V-trace, PPO, recurrent-PPO/
xLSTM) + teacher-student distillation + **opponent pools** + partial-obs state inference + symmetry aug + TTA
(rotation/flip averaging) + centralized critic. **Imitation learning (behavior cloning) from top-team replays
(rank3/4) beat weeks of rules almost immediately and is CHEAP.** Runtime: model-size caps, inference-aug
toggle if overtime, Rust env rewrite for sim speed. → `agent-policy`(add self-play+IL), `lb-replay-mine`(BC from replays), `agent-selfplay`, `agent-eval`(budget).

## INTEGRATION STATUS (what's folded into agents so far)
- ✅ metric-registry: added average_precision, partial_auc(ISIC), f2/f0.5/f4, smape, concordance_index, +
  fn=None for stratified_concordance_index/map_at_k/tm_score/wrmsse (verified 32/32).
- ⏳ next: math-master GM tools (optimized_rounder/caruana/nelder-mead/ridge/calibration/clip); tab-fe pair/
  n-gram TE + digit; tab-stack L2+optimizers+clip; pseudo-label agent; then llm/reason/sec packs grounded above.
- Reusable `gm-writeup-mine` agent wraps the fetch→extract loop so this catalog keeps growing.

---

# GAP-SCAN ADDITIONS (full 61-comp coverage — techniques the 19-comp sample missed)

Source: strict gap-scan over the newly-fetched comps (arc-2025, aimo-2, neurips-polymer, cmi-behavior,
mitsui, waveform-inversion, vesuvius, image-matching, make-data-count, ariel). Each = a NEW reusable agent
to build. ✅=built this pass · ⏳=backlog.

## New metrics (added to registry, verified 37/37)
- ✅ `gaussian_nll` (ariel — proper distributional scoring, needs mu+sigma) · `pass_at_k`/`maj_at_k` (aimo/arc self-consistency)
- ✅ registered fn=None: `spearman_sharpe` (mitsui = mean/std of daily rank-corr), `surface_dice`/betti (vesuvius topology), `weighted_mae`

## New cross-cutting agents (reusable across comps)
- ✅ `subset-classifier-router` — classify each test item into a family → route to specialist model/postproc/weights (waveform/vesuvius/mitsui mixture-of-experts)
- ✅ `analysis-by-synthesis-refiner` — test-time gradient refinement so forward(pred)≈observed, known operator (waveform 28.8→7.6; ariel transit fit)
- ✅ `checkpoint-merger` — weight-space merge (linear/TIES/DARE) beats prediction blend & cuts tokens (aimo mergekit)
- ✅ `constrained-label-assignment` — Hungarian/joint-MLE decode under per-group count constraints (cmi +0.02)
- ✅ `lb-shift-prober` — probe grid of constant offsets → fit affine correction (polymer: caught °C/°F unit bug)
- ✅ math-master `soft_spearman` — differentiable rank-correlation objective (mitsui 0.2·MSE+0.8·(1−ρ))

## Backlog (⏳ — build into the modality packs they belong to)
- LLM/reasoning: `budget-aware-inference-scheduler`+`consensus-early-stop`, `sample-pool-simulator`, `tir-executor` (tool-integrated reasoning), `speculative-decoding-accelerator`+`kv-cache-quant`, `program-synthesis-data-generator`, `augment-consistent-candidate-scorer`, `iterative-refinement-solver`+`output-shape-predictor` (ARC diffusion/TRM), `checkpoint-merger`(done)
- Timeseries/finance: `online-walk-forward-retrainer`+`label-lag-anchor`, `purged-embargo-cv`, `fin-ta-feature-library`, `forecast-drivers-then-derive`
- Uncertainty/inverse: `distributional-metric-recalibrator`+`heteroscedastic-uncertainty-head`, `bayesian-forward-model-inferencer`, `target-basis-denoiser` (label-space PCA), `structure-aware-snapper`
- 3D-seg: `topology-aware-loss`(clDice/persistence), `topology-postproc`(Betti repair), `sdf-regression-loss`, `nnunet-segmentation-runner`
- Vision: `foundation-3d-matcher`(MASt3R/VGGT)+`retrieval-shortlister`+`pair-scene-classifier`
- Signal/sensor: `imu-feature-engineer`+`sensor-frame-canonicalizer`+`orientation-aug`+`missingness-router`, `kalman-detrender`, `detector-calibrator`
- Chemistry: `molecular-featurizer`+`smiles-enumeration-aug`+`polymer-chain-extender`+`noisy-label-rescaler`+`molecular-dedup`
- Data/meta: `label-provenance-detective`, `metadata-linkage-featurizer`, `partial-label-cv-sanitizer`, `entity-marker-tagger`
- Training: `optimizer-sweep`(Muon), `self-error-head`
