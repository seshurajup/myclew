# Competition Workflows → Agent Inventory (reusable core + missing agents)

**Purpose (per user steer 2026-07-15):** design the *detailed end-to-end workflow* for every competition
type FIRST. Each workflow step is tagged with the agent that runs it — an existing handler `like-this`,
or `[NEW]` for one we must build. **Reusable core** = any step that appears in ≥2 workflows; it MUST be a
single competition-agnostic agent reading `CompConfig`, never duplicated per modality. **Missing agents** =
the `[NEW]` set, deduplicated across all workflows. We build agents in detail only AFTER this mapping, so
the inventory is *derived from real work*, not guessed.

Legend: `existing-handler` · **[NEW]** to build · _(reuse X)_ = generalize an existing biohub agent.

---

## 0. UNIVERSAL SPINE (every competition passes through this — the reusable backbone)

```
comp-onboard[NEW] → tab/img/vid/pc/llm/agent/reason PROFILE → cv-build → <train/solve> →
                    scorer → math-master(paired_delta_report) → xai(hurt on regress) →
                    ledger → submission-build → nb-preflight → submit-verify → submit-guard(human gate)
```
Every workflow below is a specialization of this spine. The spine agents are the **must-be-reusable**
core: `comp-onboard`, `cv-build`, `scorer`, `math-master`, `xai`, `ledger`, `submission-build`,
`nb-preflight`, `submit-verify`, `submit-guard`, `git-track`, `insights`, `orchestrate`/`campaign`.
Today several are biohub-coupled in their I/O (scorer/official-score/split-build/eda-stats) → generalize
via `CompConfig` dispatch, do NOT fork.

---

## 1. TABULAR — `playground-series-s6e7` (predictive, flagship turnkey)

| # | Step | Agent |
|---|------|-------|
| 1 | Pull comp desc/files/metric/sample-sub | `kaggle-scout` → **comp-onboard[NEW]** infers CompConfig(modality=tabular, task=classif/regress, metric) |
| 2 | Profile: dtypes, cardinality, missing, target balance, leakage scan | **tab-profile[NEW]** (calls `eda-stats` _(generalize)_ + `data-audit` _(generalize gotchas)_) |
| 3 | Adversarial train/test drift | `adversarial-val` _(reuse; already generic C2ST)_ |
| 4 | Build leak-safe CV (Stratified/Group/Time per config) | `split-build` _(generalize: currently embryo-LOEO only → add strat/group/kfold/timeseries)_ + `cv-build` |
| 5 | Feature engineering (encode, impute, interactions, target-enc w/ fold-safety) | **tab-fe[NEW]** |
| 6 | Train N backends (LightGBM/XGBoost/CatBoost/HistGBM) GPU | **tab-train[NEW]** (wraps GPU LGBM/XGB/Cat; calls `single-model-tune` for HPO) |
| 7 | HPO / early-stop per backend | `single-model-tune` _(generalize off biohub configs)_ |
| 8 | Blend / stack out-of-fold | **tab-stack[NEW]** (calls `ensemble` _(reuse)_) |
| 9 | One-call baseline (steps 2-8 defaulted) | **tab-autobaseline[NEW]** (turnkey top-quartile) |
| 10 | Score OOF via comp metric | `scorer` _(generalize metric-registry)_ + `official-score` _(generalize)_ |
| 11 | Keep/reject decision | `math-master paired_delta_report` + `xai(hurt)` + `ledger` |
| 12 | Build submission.csv (id + target cols per sample-sub) | `submission-build` _(generalize schema from CompConfig)_ |
| 13 | Offline-safe notebook + verify + gate | `nb-preflight`, `submit-verify`, `submit-guard` |

**Missing:** comp-onboard, tab-profile, tab-fe, tab-train, tab-stack, tab-autobaseline. **Generalize:** eda-stats, data-audit, split-build, single-model-tune, scorer, official-score, submission-build.

---

## 2. TABULAR-SEQUENCE + DOMAIN — `rogii-wellbore-geology-prediction`

Same spine as tabular, plus:

| # | Step | Agent |
|---|------|-------|
| A | Sequence/depth-ordered CV (no future leakage across a well) | `split-build` w/ `cv_scheme=grouped-sequence` _(generalize)_ |
| B | Domain feature hook (geology/well physics: rock props, depth trends) | **domain-features[NEW]** (pluggable; biohub `stage-dynamics`/`ext-label-stats` are the biology instance → generalize the *hook*, keep domain module swappable) |
| C | Sequence features (rolling/lag windows per well) | **seq-features[NEW]** (or `tab-fe` mode=sequence) |
| D | Sequence model option (1D-CNN/GRU/transformer over depth) | **seq-train[NEW]** (or `tab-train` backend=sequence) |

**Missing:** domain-features, seq-features/seq-train (may collapse into tab-fe/tab-train with a `sequence` mode — decide at build time by fixture).

---

## 3. IMAGE — generic vision (classification / detection / segmentation)

| # | Step | Agent |
|---|------|-------|
| 1 | Onboard + profile (image sizes, channels, class balance, mask/box presence) | comp-onboard[NEW] → **img-profile[NEW]** |
| 2 | CV (stratified/group-by-source) | `split-build` _(generalize)_ |
| 3 | Backbone/head select (timm/HF; task-appropriate head) | `detector-arch-search`+`arch-search`+`component-graft` _(reuse — already generic backbone/head search)_ via **img-train[NEW]** wrapper |
| 4 | Augmentation policy | `aug-find`/`aug-ablation` _(reuse — generic)_ |
| 5 | Train (GPU, AMP) | **img-train[NEW]** (calls `combined-train`/`layer-grow` machinery _(reuse)_) |
| 6 | TTA / ensemble | `ensemble`, `tracker-consensus`→ **img-ensemble** (reuse `ensemble`) |
| 7 | Score (mAP/Dice/acc per metric-registry) | `scorer`+`official-score` _(generalize)_ |
| 8 | Compress for offline T4 (INT8/ToMe/distill/keyframe) | `compress-select`,`quantize`,`distill`,`keyframe`,`component-graft` _(reuse — already built, generic)_ |
| 9 | Submission + gates | `submission-build`,`nb-preflight`,`submit-verify` |

**Missing:** img-profile, img-train (thin wrappers). **Reuse-as-is:** arch-search, detector-arch-search, aug-find, aug-ablation, ensemble, compress-select, quantize, distill, keyframe, component-graft, saliency-detect.

---

## 4. VIDEO — generic (action/track/temporal)

Image spine + temporal:

| # | Step | Agent |
|---|------|-------|
| A | Temporal profiling (fps, clip len, frozen/dup frames) | `temporal-audit`+`frozen-exploit` _(reuse — generic temporal)_ → **vid-profile[NEW]** |
| B | Frame sampling / keyframe budget for T4 | `keyframe` _(reuse)_ |
| C | Temporal model (3D-CNN/video-transformer/optical-flow) | **vid-train[NEW]** (reuse flow infra `flow-gt-build`) |
| D | Temporal linking/smoothing | `linking`,`mh-ilp`,`tracker-postproc` _(reuse if tracking task)_ |

**Missing:** vid-profile, vid-train.

---

## 5. 3D / POINT-CLOUD — generic volumetric (no time)

| # | Step | Agent |
|---|------|-------|
| 1 | Profile (voxel dims, sparsity, per-axis resolution) | **pc-profile[NEW]** |
| 2 | 3D model (3D-UNet/sparse-conv/PointNet) | **pc-train[NEW]** (reuse `arch-builder`/`center-train` 3D machinery) |
| 3 | Score / submission | `scorer`,`submission-build` _(generalize)_ |

**Missing:** pc-profile, pc-train. (biohub's `center-train`/`tracker-train` are the 3D+time instance → factor the pure-3D parts out.)

---

## 6. 3D + TIME — `biohub-cell-tracking` (REFERENCE PACK — already built)

The existing 38 biohub agents ARE this workflow: `ext-label-stats`→`box-sample`→`center-train`/`tracker-train`/
`lora-train`→`tracker-predict`→`mh-ilp`/`linking`/`division`→`tracker-postproc`→`official-score`→`full-cv-baseline`→
`submission-build`. **Action:** refactor to consume `CompConfig` so it becomes the *reference implementation* of the
3D+time pack, and so its generic seams (arch-search, ensemble, compress-select, aug-find, math-master, xai, ledger)
are visibly shared with packs 1-5.

---

## 7. LLM — text classification / generation / fine-tune

| # | Step | Agent |
|---|------|-------|
| 1 | Onboard + profile (token lengths, label space, prompt vs completion) | comp-onboard[NEW] → **llm-profile[NEW]** |
| 2 | CV (stratified by label/source) | `split-build` _(generalize)_ |
| 3 | Prompt/zero-shot baseline (offline model on T4) | **llm-infer[NEW]** |
| 4 | Fine-tune (LoRA/QLoRA) | **llm-finetune[NEW]** _(generalize `lora-train`/`lora-validate` — already LoRA machinery, just biohub-headed)_ |
| 5 | Eval (task metric + calibration) | **llm-eval[NEW]** (calls `scorer`+`math-master` calibration) |
| 6 | Compress/quantize for offline | `quantize`,`compress-select`,`distill` _(reuse)_ |
| 7 | Submission + gates | `submission-build`,`nb-preflight`,`submit-verify` |

**Missing:** llm-profile, llm-infer, llm-finetune (generalize lora-train), llm-eval.

---

## 8. AGENTIC — `autonomous-agent-prediction-beta`, `pokemon-tcg-ai-battle`

| # | Step | Agent |
|---|------|-------|
| 1 | Onboard: identify env API, action space, reward/score, replay budget | comp-onboard[NEW] → **agent-env[NEW]** (wraps the comp's env/simulator) |
| 2 | Baseline heuristic policy | **agent-policy[NEW]** |
| 3 | Search/optimize policy (self-play / evolutionary / bandit) | **agent-search[NEW]** + **agent-selfplay[NEW]** |
| 4 | Mine leaderboard replays for opponent/env patterns | **lb-replay-mine[NEW]** _(reuse Orbit Wars replay-mining — proven)_ |
| 5 | Offline eval vs budget BEFORE submit | **agent-eval[NEW]** |
| 6 | Submission (agent code/notebook) + gate | `submission-build`,`nb-preflight`,`submit-verify`,`submit-guard` |

**Missing:** agent-env, agent-policy, agent-search, agent-selfplay, lb-replay-mine, agent-eval.

---

## 9. AGENTIC-SECURITY — `ai-agent-security-multi-step-tool-attacks` (our JED experience)

Agentic spine + security specialization (reuse [[aas_dense_exfil_strategy]] [[aas_local_cv_methods]]):

| # | Step | Agent |
|---|------|-------|
| A | Construct multi-step tool-attack / exfil sequence within budget | **sec-attack[NEW]** (dense-exfil pattern S=0.09×N_eff; stacking-dead lesson baked in) |
| B | Replay-mine attack traces | **sec-replay[NEW]** _(specialize lb-replay-mine)_ |
| C | Offline budget check (~336s replay limit is the real constraint) | **sec-eval[NEW]** |
| D | Dual-use guardrail — defensive/eval framing only | governance in `decision-audit`/`trick-gate` _(reuse)_ |

**Missing:** sec-attack, sec-replay, sec-eval.

---

## 10. REASONING — `arc-prize-2026-arc-agi-3` (program synthesis)

| # | Step | Agent |
|---|------|-------|
| 1 | Onboard: grid/task format, train/test demos, exact-match metric | comp-onboard[NEW] → **reason-profile[NEW]** |
| 2 | DSL of grid transforms | **reason-dsl[NEW]** |
| 3 | Program search (enumerate/neural-guided over DSL) | **program-search[NEW]** |
| 4 | Test-time compute / hypothesis-and-verify | **ttc[NEW]** |
| 5 | Exact-match eval on train demos | **reason-eval[NEW]** |
| 6 | Submission + gate | `submission-build`,`nb-preflight`,`submit-verify` |

**Missing:** reason-profile, reason-dsl, program-search, ttc, reason-eval.

---

## 11. NOVEL / UNKNOWN — `neurogolf-2026` (the cold-start test)

No pre-coded pack. Workflow:
```
comp-onboard[NEW] pulls desc+data+metric → fingerprints modality×paradigm×task →
  matches nearest pack (by CompConfig similarity) → IF match ≥ threshold: run that pack →
  ELSE emit "unknown-comp report": closest pack + the ONE new capability to build + a proposed CompConfig.
```
This is the proof the fleet generalizes. `comp-onboard` must degrade gracefully to a *report*, never crash.

---

## DERIVED INVENTORY (dedup across all 11 workflows)

### A. Reusable core — build/generalize ONCE, shared everywhere (the moat)
- **Contract:** `comp_config.py`[NEW] (CompConfig + 5 interfaces + metric-registry + paradigm field)
- **Front door:** `comp-onboard`[NEW]
- **CV:** `cv-build`✓, `split-build`⟳(add strat/group/kfold/timeseries/sequence), `verify-cv`✓, `cv-lb-calibrate`✓
- **Profiling:** `eda-stats`⟳, `data-audit`⟳, `adversarial-val`✓, `kaggle-scout`✓, `temporal-audit`✓, `frozen-exploit`✓
- **Scoring:** `scorer`⟳(metric-registry), `official-score`⟳, `reproduce-score`✓, `metrics-report`✓
- **Decision gate:** `math-master`✓, `xai`✓, `ledger`✓, `insights`✓, `decision-audit`✓, `trick-gate`✓
- **Tune/search:** `single-model-tune`⟳, `arch-search`✓, `detector-arch-search`✓, `component-graft`✓, `combo-search`✓
- **Aug:** `aug-find`✓, `aug-ablation`✓, `sample-match`✓, `ext-transfer`✓
- **Ensemble:** `ensemble`✓, `tracker-consensus`✓, `combine-winners`✓
- **Compress/deploy (offline T4):** `compress-select`✓, `quantize`✓, `distill`✓, `keyframe`✓, `layer-grow`✓
- **Submission:** `submission-build`⟳(schema from CompConfig), `nb-preflight`✓, `submit-verify`✓, `submit-guard`✓, `official-conformance`✓, `notebook-sync`✓
- **Research:** `research-search`✓, `paper-research`✓, `deep-research`✓, `lit-search`✓, `prior-art`✓, `recipe-adopt`✓, `trick-extractor`✓, `block-synth`✓, `paper-verify`✓
- **Orchestration:** `orchestrate`✓, `campaign`✓, `improve-loop`✓, `beat-bar`✓, `plan-ingest`✓, `git-track`✓, `heal`✓, `guard`✓
- **LB:** `lb-sync`✓, `scoreboard`✓
- **Coverage:** `coverage-audit`[NEW]

✓ = already generic, use as-is · ⟳ = generalize (biohub-coupled I/O today) · [NEW] = build

### B. Missing agents to BUILD (deduped, in build order)
1. `comp_config.py` contract (foundation)
2. `comp-onboard` (front door; verified on ≥3 real slugs)
3. Generalize the ⟳ set: scorer, official-score, split-build, eda-stats, data-audit, single-model-tune, submission-build
4. **Tabular:** tab-profile, tab-fe, tab-train, tab-stack, tab-autobaseline
5. **Agentic:** agent-env, agent-policy, agent-search, agent-selfplay, lb-replay-mine, agent-eval
6. **Security:** sec-attack, sec-replay, sec-eval
7. **LLM:** llm-profile, llm-infer, llm-finetune (⟳lora-train), llm-eval
8. **Reasoning:** reason-profile, reason-dsl, program-search, ttc, reason-eval
9. **Image:** img-profile, img-train
10. **Video:** vid-profile, vid-train
11. **3D:** pc-profile, pc-train
12. **Domain/seq:** domain-features, seq-features/seq-train (may fold into tab-fe/tab-train)
13. `coverage-audit`

### C. Reusability invariants (enforced at build time)
- No `tab-*`/`img-*`/`llm-*` agent re-implements CV, ensemble, scoring, tuning, submission, compression, research — it CALLS the core.
- Every profile/train/score agent takes `CompConfig`; zero hard-coded comp names or paths.
- Each pack ships a synthetic fixture under `test_fleet_agents/fixtures/`; pack is "done" only when `profile→…→submission` runs GREEN on its fixture.
- Every kept number passes `math-master paired_delta_report` with an eval-set tag; every regression passes `xai(hurt)`.
