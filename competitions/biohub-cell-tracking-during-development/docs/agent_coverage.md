# Agent Coverage Matrix — every agent in its respective pack

**322 agents**, auto-classified by `coverage-audit`. BIOHUB agents are kept as their own 3D+time reference pack, separate from the reusable generic core and the new modality packs.

## Generic CORE (78)
`ablate-best`, `adversarial-val`, `analysis`, `arch-probe`, `arch-search`, `aug-ablation`, `aug-find`, `baseline`, `beat-bar`, `block-synth`, `campaign`, `combine-winners`, `comp-onboard`, `component-graft`, `compress-select`, `config-gen`, `context-offload`, `coverage-audit`, `cv-build`, `cv-lb-calibrate`, `data-audit`, `decision-audit`, `deep-research`, `distill`, `eda-stats`, `ensemble`, `feasibility-gate`, `feasibility-map`, `git-track`, `gpu-best-practices`, `guard`, `heal`, `improve-loop`, `insights`, `journey-status`, `kaggle-scout`, `keyframe`, `lb-sync`, `learn`, `ledger`, `lit-search`, `math-master`, `metrics-report`, `nb-preflight`, `notebook-sync`, `notes-sync`, `onnx`, `orchestrate`, `paper-research`, `paper-verify`, `perf-choice`, `pipeline`, `pipeline-completeness`, `pipeline-run`, `plan-ingest`, `post-analysis`, `post-proc`, `pre-analysis`, `prior-art`, `quantize`, `recipe-adopt`, `reproduce-score`, `research-search`, `score`, `scoreboard`, `scorer`, `setup-env`, `single-model-tune`, `smoke`, `split-build`, `submission-build`, `submit-guard`, `submit-verify`, `train-monitor`, `trick-extractor`, `trick-gate`, `verify-cv`, `xai`

## BIOHUB (3D+time) (24)
`center-train`, `combined-train`, `deep-sister`, `detect-quality`, `div-model`, `div-temporal-feas`, `division`, `division-rescue`, `ext-label-stats`, `full-cv-baseline`, `lever-hunt`, `lora-train`, `lora-validate`, `official-conformance`, `official-score`, `pattern-tune`, `psf-deconv`, `stage-1-div`, `stage-dynamics`, `tracker-consensus`, `tracker-postproc`, `tracker-predict`, `tracker-select`, `tracker-train`

## Detection & Tracking (16)
`det-sweep`, `detector-arch-search`, `detector-select`, `detector-transfer`, `flow-gt-build`, `frozen-exploit`, `gaussian-heatmap-encoder`, `gnn-link-train`, `gnn-probe`, `heatmap-peak-decoder`, `link-tune`, `linking`, `mh-ilp`, `saliency-detect`, `temporal-audit`, `volumetric-patch-inference`

## Arch/Config search (8)
`arch-builder`, `arch-catalog`, `best-config`, `combo-search`, `config-ablate`, `fullconfig-search`, `layer-grow`, `public-config`

## External-data transfer (4)
`box-sample`, `domain-match`, `ext-transfer`, `sample-match`

## Tabular (17)
`ae-latent-view`, `automl-oof-factory`, `feature-select`, `full-retrain-calibrator`, `geospatial-fe`, `gp-symbolic-feature`, `knn-feature`, `oof-diversity-prune`, `residual-boost`, `shift-adapt`, `synth-artifact-fe`, `tab-autobaseline`, `tab-fe`, `tab-nn-train`, `tab-profile`, `tab-stack`, `tab-train`

## GM toolkit (5)
`blend-optimize`, `calibrate`, `post-optimize`, `pseudo-label`, `target-transform`

## Training tricks (2)
`sr-bf16-optimizer`, `train-tricks`

## Gap toolkit (5)
`analysis-by-synthesis-refiner`, `checkpoint-merger`, `constrained-label-assignment`, `lb-shift-prober`, `subset-classifier-router`

## Forecast/Finance/Sports (10)
`distributional-metric-recalibrator`, `forecast-drivers-then-derive`, `forecast-trend-extrapolator`, `label-lag-anchor-blend`, `market-odds-blend`, `online-walk-forward-retrainer`, `outcome-sharpen`, `portfolio-position-sizer`, `rating-systems`, `ts-decompose-forecaster`

## Domain FE (11)
`annotation-error-corrector`, `calendar-holiday-fe`, `fin-ta-feature-library`, `hierarchy-consistency-postproc`, `imu-feature-engineer`, `invariance-feature-normalizer`, `knn-label-transfer`, `linear-constraint-projector`, `molecular-featurizer`, `quaternion-imu-features`, `temporal-segment-decoder`

## Optimization (6)
`batched-oracle-search-harness`, `best-of-n-diversity-allocator`, `combinatorial-local-search`, `geometric-packing-optimizer`, `gpu-relaxation-solver`, `population-diversity-manager`

## 2026 frontier (5)
`conformal-predict`, `dora-adapt`, `hardware-tune`, `muon-optimizer`, `schedule-free`

## Compression/Quantization (1)
`lowbit-qat`

## Training heads/regularizers (6)
`awp-perturb`, `class-balance-sampler`, `deep-supervision`, `masked-sequence-norm`, `masked-sequence-pool`, `sed-attention-pool`

## Inference tricks (3)
`multi-tta`, `snapshot-average`, `wbf-fusion`

## Audio (6)
`audio-augment`, `audio-backbone`, `audio-crop-tta`, `audio-infer`, `audio-melspec-fe`, `audio-train`

## Graph (3)
`graph-feature-extractor`, `graph-message-passing`, `graph-readout`

## Multimodal (3)
`modality-dropout`, `modality-encoder-adapter`, `multimodal-fusion`

## Video (3)
`video-frame-sampler`, `video-motion-features`, `video-temporal-aggregator`

## Agentic (3)
`agent-env`, `agent-eval`, `agent-policy`

## Prompt-program (10)
`agent-author`, `agent-config-eval`, `agent-package`, `dspy-prompt-optimize`, `harness-opt-gate`, `prompt-dataset`, `prompt-metric`, `prompt-optimize`, `skill-build`, `style-fingerprint`

## LLM (25)
`budget-aware-inference-scheduler`, `consensus-early-stop`, `coworker-backend`, `infer-cascade`, `kv-cache-longctx`, `llama2-infer`, `llm-eval`, `llm-finetune`, `llm-infer`, `llm-retrieve-rerank`, `llm-synthetic-drill-generator`, `llmc-train`, `mbr-consensus-selector`, `moe-inference-cost`, `mtp-speculative-decode`, `noisy-label-cleaner`, `risk-abstain-gate`, `runtime-budget-router`, `sample-pool-simulator`, `self-consistency-aggregator`, `template-retrieval-reranker`, `tir-executor`, `trainable-trace-auditor`, `ttt-transductive-finetune`, `vlm-pdf-corpus-miner`

## Reasoning/Code (11)
`code-compress-optimizer`, `code-repair-agent`, `expression-search`, `fast-sim`, `lb-formula-prober`, `llm-judge-attacker`, `program-golf-search`, `program-search`, `program-synthesis-data-generator`, `sprt-spsa-tuner`, `ttc`

## Grid-reasoning (ONNX-golf) (3)
`arc-idioms`, `arc-onnx-golf`, `arc-worker-context`

## Vision/3D-seg (15)
`chess-search-engine`, `density-regression-head`, `dicom-metadata-estimator`, `foundation-3d-matcher`, `gan-train`, `geometric-spatial-augmentor`, `grid-rectification-unwarp`, `heteroscedastic-uncertainty-head`, `keypoint-match-verifier`, `nnue-trainer`, `nnunet-segmentation-runner`, `region-decompose-router`, `sdf-regression-loss`, `topology-aware-loss`, `trajectory-forecaster`

## Meta/Mining (7)
`binary-size-compressor`, `github-solution-mine`, `gm-repo-distill`, `gm-writeup-mine`, `kaggle-modality`, `solution-adopt`, `sub-journal`

## Submission (1)
`kaggle-submit`

## Metric/Validation security (1)
`metric-probe`

## UNCLASSIFIED (30)
`attention-residual`, `deck-builder`, `diffusion-sampler`, `embedding-retrieval`, `flow-matching`, `geometric-features`, `gpu-patterns`, `head-consistency`, `hf-kernels`, `latent-moe`, `lightning-tricks`, `llm-backend`, `llm-tool-train`, `local-pilot`, `moe-quantile-balance`, `mup-scaling`, `nvfp4-loader`, `paper-learn`, `paper-md`, `rswa-attention`, `santa-agent`, `shap-emd-distance`, `shopify-agent`, `shorts-builder`, `skill-optimizer`, `sparsity-metrics`, `task-spec`, `turboquant`, `ui-component`, `video-builder`
