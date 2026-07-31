# Multi-Modal Kaggle Agent Fleet — Detailed Development Plan

**Goal:** evolve our 113-agent fleet (today: a biohub 3D+time cell-tracking machine with a strong generic
backbone) into a **modality-general Kaggle competition fleet** that handles **tabular, image, video, 3D,
3D+timeline, and LLM** competitions — with the same discipline: decide-from-data, math_master gate, XAI
diagnosis, honest ledger, offline-verified submission.

**Non-negotiables (carried from the biohub work):**
1. Every agent = `BaseAgent` subclass + module-level `run(q,worker)` + a **data-wise self-test** that runs real logic.
2. **Extend before build.** Reuse the generic core; never fork a parallel stack.
3. Every scoring number goes through the **official metric** (no proxies), every before/after through **`paired_delta_report`** (math_master), every regression through **xai `hurt`**.
4. **Verification one agent at a time** — build → data-test GREEN → register → only then next.
5. **Modality is data, not code forks** — a single fleet, parameterised by a competition contract.

---

## 1. Current-state audit (the 113 agents, classified)

### 1A. GENERIC CORE — already competition-agnostic or trivially so (~75)
| group | agents |
|---|---|
| **Math/stats** | `math-master` (103 fns: distances, effect-size, paired tests, FDR, calibration, power) |
| **XAI** | `xai` (saliency · mechanistic · feature · `data` audit · `division` mechanism · `hurt` root-cause) |
| **Scoring/CV** | `official-score`,`scorer`,`score`,`metric`,`metrics-report`,`verify-cv`,`cv`,`cv-build`,`cv-lb-calibrate`,`cv-contract`,`split-build`,`adversarial-val` |
| **Data/EDA** | `eda-stats`,`data-audit`,`ext-label-stats`,`sample-match`,`box-sample`,`pattern-tune`,`adversarial` |
| **Research** | `research-search`,`deep-research`,`lit-search`,`paper-research`,`prior-art`,`kaggle-scout`,`lb-sync`,`notebook-sync`,`notes-sync`,`paper-verify` |
| **Search/Tune** | `config-gen`,`config-ablate`,`fullconfig-search`,`combo-search`,`best-config`,`arch-search`,`arch-builder`,`single-model-tune`,`ablate-best`,`combine-winners`,`block-synth`,`det-sweep`(pattern) |
| **Model-ops** | `distill`,`quantize`,`compress-select`,`component-graft`,`layer-grow`,`perf-choice`,`gpu-best-practices` |
| **Submission** | `nb-preflight`,`submit-verify`,`submit-guard`,`submission-build`,`notebook-sync`,`public-config` |
| **Governance/infra** | `git-track`,`reproduce-score`,`official-conformance`,`ledger`,`insights`,`journey`,`scoreboard`,`note`,`monitor`,`guard`,`heal`,`train-monitor`,`learn`,`decision-audit`,`metric-anatomy` |
| **Meta/trick** | `trick-extractor`,`trick-gate`,`recipe-adopt`,`pre-analysis`,`post-analysis` |
| **Workflow** | `campaign`,`improve-loop`,`beat-bar`,`orchestrate`,`pipeline`,`pipeline-run`,`plan-ingest`,`runner`,`dryrun`,`smoke`,`stages`,`grandmaster`,`journey-status` |
| **Training (generic-ish)** | `lora-train`,`lora-validate` (PEFT — already the LLM training core) |

### 1B. BIOHUB 3D+TIME ADAPTER — the reference modality pack (~38)
Detection: `tracker-predict`,`tracker-train`,`center-train`,`detect-quality`,`detector-select`,`detector-arch-search`,`mh-ilp`.
Linking: `tracker-postproc`,`tracker-select`,`tracker-consensus`,`link-tune`,`linking`.
Division: `division-rescue`,`div-model`,`stage-1-div`,`deep-sister`,`stage-dynamics`,`division`.
Flow/affinity: `gnn-link-train`,`gnn-probe`,`flow-gt-build`,`ext-transfer`,`combined-train`.
Misc: `keyframe`,`frozen-exploit`,`saliency-detect`,`temporal-audit`,`official-score`(geff I/O),`arch-probe`,`aug-*`,`baseline`,`ensemble`.

**Verdict:** the backbone is real and strong. The task is (a) **harden a modality contract** so the core is provably comp-agnostic, and (b) **build 5 more modality packs** (tabular, image, video, 3D, LLM) that plug into it, with the biohub pack refactored as the reference 3D+time implementation.

---

## 2. Target architecture — 3 layers

```
Layer 2  MODALITY PACKS      tabular | image | video | 3d | 3d+time(biohub) | llm
                              each implements the CONTRACTS below
─────────────────────────────────────────────────────────────────────────────
Layer 1  GENERIC CORE        math · xai · scoring · research · search · model-ops
                              submission · governance · workflow   (the ~75)
─────────────────────────────────────────────────────────────────────────────
Layer 0  CONTRACTS           CompConfig · Dataset · Model · Scorer · Submission
                              (competition-agnostic interfaces, one small module)
```

### 2A. The CompConfig contract (Layer 0) — NEW `fleet_agents/comp_config.py`
One declarative object per competition (like our biohub `src/config`, generalised):
```
CompConfig = {
  slug, modality ∈ {tabular,image,video,3d,3d_time,llm},
  task ∈ {classification,regression,segmentation,detection,tracking,ranking,generation,...},
  metric: {name, direction, official_scorer_ref},        # e.g. auc/rmse/map/dice/jaccard/f1/bleu
  data: {train, test, target_col|label_dir, id_col, group_col|time_col},
  cv: {scheme ∈ {kfold,stratified,group,time,adversarial,leave-one-<key>-out}, folds, seed},
  submission: {template, id_col, pred_col(s), offline_env_extras},   # incl POLARS_PREFER_PKG-class gotchas
  compute: {gpu, time_budget_h, offline},
}
```
Every generic agent reads `CompConfig` instead of hard-coded biohub paths. Modality packs are chosen by `modality`.

### 2B. The five interfaces every modality pack implements
1. **`profile`** — dataset fingerprint (shapes, dtypes, class balance, leakage risks, drift train↔test via `math-master adversarial_auc`).
2. **`cv`** — build a leak-safe split honoring `cv.scheme`.
3. **`repr`** — features (tabular), augment/backbone (image/video/3d), tokenize/format (llm).
4. **`train`** / **`predict`** — model fit + inference.
5. **`score`** — via the official metric; **`postproc`** — task-specific cleanup; **`submission`** — offline-verified file.

The **generic core provides**: the CV *contract* (`split-build`,`adversarial-val`,`cv-lb-calibrate`), the *search* (`config-*`,`arch-*`,`combo-search`), the *ensembling* (`ensemble`,`combine-winners`,`trick-*`), the *submission gate* (`nb-preflight`,`submit-verify`), the *governance* (`ledger`,`git-track`,`decision-audit`), the *diagnosis* (`xai data/hurt`,`math-master`). Packs supply only the modality-specific `repr/train/predict/postproc`.

---

## 3. Per-modality build plan (Layer 2)

### 3.1 TABULAR (highest ROI — most Kaggle comps)  → new pack `tab-*`
- **`tab-profile`** — dtypes, cardinality, missingness, target leakage, train↔test drift (adversarial-AUC per column via math-master), class balance. *(extends `eda-stats`+`data-audit`)*
- **`tab-cv`** — stratified/group/time/adversarial fold builder honoring `CompConfig.cv`. *(extends `split-build`)*
- **`tab-fe`** — feature engineering: target/count/frequency encoding (leak-safe fold-wise), interactions, aggregations, datetime, text-tfidf hooks; every feature gated by permutation importance + drift.
- **`tab-train`** — one trainer, N backends: **LightGBM / XGBoost / CatBoost / GBM (sklearn HistGBM) / a tabular-MLP**; GPU where available; early-stop on `CompConfig.metric`; OOF + test preds saved.
- **`tab-stack`** — OOF-level blending/stacking/rank-average with the `ensemble`/`combine-winners` reject-regressive rule.
- **`tab-pseudo`** — confident-test pseudo-labeling with a held-out gate.
- **`tab-select`** — pick the winning (features × backend × cv) by *paired_delta_report*, per-fold.

### 3.2 IMAGE (classification/segmentation/detection) → new pack `img-*`
- **`img-profile`** — resolution/channels/class balance/duplicate & leakage scan, train↔test drift.
- **`img-aug`** — derive a valid augmentation menu from data (extends biohub `aug-find`); TTA menu.
- **`img-train`** — timm/torchvision backbones (classification), UNet/segmentation-models (seg), a detection head; mixed precision; OOF + TTA. *(reuses `arch-search`,`gpu-best-practices`,`quantize`,`distill`)*
- **`img-tta`** — flip/scale/crop TTA + logit averaging.
- **`img-ensemble`** — seed/fold/backbone averaging.

### 3.3 VIDEO → new pack `vid-*`
- **`vid-frame-sample`** (uniform/keyframe — reuse `keyframe`), **`vid-train`** (3D-CNN / video-transformer / per-frame-then-temporal-agg), **`vid-tta`** (temporal + spatial), **`vid-temporal-agg`** (mean/attention pooling).

### 3.4 3D (point cloud / voxel / mesh) → new pack `pc-*`
- **`pc-voxelize`/`pc-sample`**, **`pc-train`** (sparse-conv / PointNet++ / 3D-UNet — reuse biohub `TemporalUNet3D` patterns), **`pc-aug`** (rotation/jitter/scale, anisotropy-aware like our SCALE trick).

### 3.5 3D + TIMELINE (biohub) → **refactor existing pack to the contract**
Wrap the existing `tracker-*`,`division-*`,`flow-*` agents behind the `profile/cv/repr/train/predict/score/postproc` interface so it becomes the **reference implementation** other packs mirror. Zero new capability — pure conformance.

### 3.6 LLM → new pack `llm-*` (build on `lora-train`)
- **`llm-data`** — format prompts/instructions/RAG chunks; leak & contamination scan.
- **`llm-finetune`** — generalize `lora-train` beyond the biohub UNetNodeTransformer to any HF causal/seq2seq model (QLoRA/rsLoRA/DoRA already wired); eval-gate on the comp metric.
- **`llm-infer`** — batched offline inference (vLLM/transformers), submission formatting.
- **`llm-eval`** — task metric (accuracy/F1/BLEU/ROUGE/exact-match/pass@k) + calibration (ECE from math-master) + `xai` token-attribution.
- **`llm-prompt`** — prompt/few-shot search gated by CV (the `config-*`/`combo-search` pattern on prompts).

---

## 4. Verification discipline (unchanged, enforced)
- Each new agent ships with `test_fleet_agents/<name>_test.py` that runs REAL logic on a tiny fixture and asserts a concrete output; preflight **quarantines** RED agents (they escalate instead of running).
- Every modality pack ships a **synthetic mini-competition fixture** (e.g. a 200-row tabular toy with a known signal, a 20-image toy) so `profile→cv→train→score→submission` is proven end-to-end offline.
- Modality×capability **coverage matrix** kept in `docs/agent_coverage.md`, auto-updated.

## 5. Build order (take-best-choice, no waiting)
1. **Contracts** — `comp_config.py` + the 5-interface base classes. *(unblocks everything)*
2. **Generalize the core seams** — make `official-score`/`scorer`/`split-build`/`nb-preflight` read `CompConfig` (they're 80% there).
3. **Tabular pack** (highest ROI) — full `tab-*`, each data-tested on a synthetic fixture.
4. **LLM pack** — generalize `lora-train`→`llm-finetune` + `llm-eval/infer/prompt`.
5. **Image pack** — `img-*`.
6. **Video + 3D packs** — `vid-*`,`pc-*`.
7. **Biohub refactor** — conform the 3D+time pack to the contract (reference).
8. **Coverage matrix + a cross-modality `campaign` template** that runs profile→cv→train→ensemble→submit for any modality.

## 6. Coverage matrix (target)
| capability | tabular | image | video | 3d | 3d+time | llm |
|---|---|---|---|---|---|---|
| profile/EDA | tab-profile | img-profile | vid* | pc* | eda-stats✓ | llm-data |
| cv-split | tab-cv | (img)split-build | ✓ | ✓ | split-build✓ | (group) |
| repr/features | tab-fe | img-aug | vid-sample | pc-voxelize | flow-gt✓ | llm-data |
| train | tab-train | img-train | vid-train | pc-train | tracker-train✓ | llm-finetune |
| predict | tab-train | img-train | vid-train | pc-train | tracker-predict✓ | llm-infer |
| score | official-score(generic) | ✓ | ✓ | ✓ | official-score✓ | llm-eval |
| ensemble | tab-stack | img-ensemble | vid-tta | ✓ | tracker-consensus✓ | (self-consistency) |
| submit-verify | nb-preflight✓ | ✓ | ✓ | ✓ | submit-verify✓ | ✓ |

Legend: ✓ = exists today (generic or biohub); others = to build.

---

# CRITIQUE PASS 1 (critic mode — grounded in the existing agents + real competition types)

**Fatal gap in v1: it is PREDICTIVE-ONLY.** The examples given (`pokemon-tcg-ai-battle`,
`autonomous-agent-prediction-beta`, `arc-prize-2026-arc-agi-3`, `rogii-wellbore-geology`) prove the taxonomy
must split on **PARADIGM**, not just data shape. And we ALREADY have agentic-competition experience the plan
ignored — Orbit Wars and the JED AI-Agent-Security work (LB-replay mining, in-proc TTA wrappers, dense-exfil).

### C1.1 — Add a PARADIGM axis, orthogonal to data-modality
| paradigm | you output… | LB measures… | examples |
|---|---|---|---|
| **Predictive** | a label/target per row/pixel/token | error vs hidden GT | tabular, image, video, 3d, 3d+time, most LLM |
| **Agentic** | a **policy/agent** that acts in an env | performance vs opponents/environment | pokemon-tcg, autonomous-agent, Orbit Wars, JED |
| **Reasoning** | a **program/solution** per novel task | exact solve rate on unseen puzzles | ARC-AGI-3 |

A competition is (data-modality × paradigm). `rogii-wellbore-geology` = predictive × (sequence/tabular, domain).
`pokemon-tcg` = agentic × (game-state). `arc-agi-3` = reasoning × (grid). The generic core + contracts stay;
we add **two new pack families** (agentic, reasoning) and a **domain-features hook** for specialized predictive comps.

### C1.2 — NEW pack: AGENTIC  `agent-*`  (reuse Orbit Wars / JED patterns — do NOT reinvent)
- **`agent-env`** — wrap the competition's environment/SDK/simulator into a uniform `reset/step/observe/act` contract; verify with a random-policy rollout.
- **`agent-policy`** — a heuristic baseline policy first (rules from env analysis), then a learned/search policy; the "simple ML breaks tuned heuristics" lesson applies.
- **`agent-search`** — decision-time search (MCTS / beam / minimax) over the env model where legal.
- **`agent-selfplay`** — self-play / league training loop with opponent pools (RL where the env allows).
- **`lb-replay-mine`** — **we already built this for Orbit Wars** — pull opponent/LB replays, mine winning patterns, upload as a dataset. Generalize it.
- **`agent-eval`** — offline arena: score a candidate policy vs a fixed opponent pool + the replay-derived bots, with `math-master paired_delta_report` over matches (win-rate CI, not a single game).
- **Governance carry-over:** [top-10 LB is the bar], [only submit when top-10 capable], [local arena ≠ Kaggle arena — trust Kaggle], [mine LB replays], [TTA/policy wrappers in-proc only].

### C1.3 — NEW pack: REASONING / PROGRAM-SYNTHESIS  `reason-*`  (ARC-AGI class)
- **`reason-dsl`** — a primitive/DSL library for the task family (grid ops for ARC), extensible.
- **`program-search`** — enumerative + **LLM-guided** program synthesis (propose → execute → verify on the task's train pairs → keep programs that fit all examples).
- **`ttc`** — test-time-compute loop: sample many candidate solutions, execute, self-verify/vote, best-of-N (this is where compute buys score in ARC-class comps).
- **`reason-eval`** — exact-solve rate on held-out tasks + a difficulty decomposition (which task types fail) via `xai`.
- Reuses `llm-infer` (candidate generation) + `math-master` (voting/agreement) + the `improve-loop` (attack the weakest task-type).

### C1.4 — Predictive gaps v1 under-specified
- **Domain/sequence comps** (rogii wellbore = depth-sequence of geological measurements): add a **`domain-features` hook** to the tabular/`repr` interface for physics/domain feature libraries, and a **sequence sub-type** (1D-CNN/LSTM/Transformer over ordered rows) — not every "tabular" comp is i.i.d. rows.
- **NLP-as-tabular / multimodal**: text columns, image+tabular fusion — the `repr` interface must compose, not be single-modality.
- **Metric zoo is under-modeled:** v1 named a few metrics; the Scorer contract must ship a **metric registry** (auc, logloss, rmse, rmsle, mae, map@k, dice, iou, f1-macro, quadratic-weighted-kappa, spearman, ndcg, bleu/rouge/exact-match, win-rate) each with direction + an official-scorer hook, because **the metric drives the CV, the loss, and the ensemble rule.**

### C1.5 — Corrections to my own audit
- I over-counted "generic": several so-called generic agents are **biohub-coupled in their I/O** (`official-score` reads `.geff` graphs; `split-build`/`adversarial-val` assume embryo keys; `eda-stats` reads density/stage). These need a **thin modality-dispatch** inside them, not a claim of "already generic."
- `lora-train` is coupled to `UNetNodeTransformer` warm-resume — the LLM pack must generalize the model-load path, not just re-point data.
- The **coverage matrix must become a live, tested artifact** (`docs/agent_coverage.md`) generated by a `coverage-audit` agent that actually imports each pack and runs its fixture — otherwise it rots (exactly the "empty train_set field" rot we just hit in the ledger).

### C1.6 — Revised build order (paradigm-aware)
1. **Contracts** (`comp_config.py` + 5 interfaces + **Scorer metric-registry** + **paradigm field**).
2. **Generalize the core seams** with modality/paradigm dispatch (`official-score`, `split-build`, `nb-preflight`).
3. **Tabular pack** (predictive, highest ROI) incl. sequence sub-type + domain-features hook.
4. **Agentic pack** (reuse Orbit Wars/JED) — high value, we have proven patterns.
5. **LLM pack** + **Reasoning pack** (they share `llm-infer`/`ttc`).
6. **Image**, then **Video/3D**.
7. **Biohub refactor** to the contract (reference 3D+time).
8. **`coverage-audit`** agent + live matrix.

---

# CRITIQUE PASS 2 (critic mode — cold-start, security sub-paradigm, turnkey tabular, and de-risking)

New examples (`ai-agent-security-multi-step-tool-attacks`, `neurogolf-2026`, `playground-series-s6e7`) expose
three more holes. The plan so far assumes **I know the competition's shape before coding** — but the fleet
must handle a **brand-new comp it has never seen** (neurogolf), and must reuse **our own JED security work**.

### C2.1 — THE MISSING FRONT DOOR: a `comp-onboard` agent (build this FIRST, before any pack)
Nothing routes an *unknown* competition into the architecture. Add:
- **`comp-onboard`** — input: a competition slug. Actions (all via existing agents): `kaggle-scout`+`kaggle competitions files/pages` → pull description, data manifest, metric, sample submission; `eda-stats`/`tab-profile` → fingerprint the data; then **INFER the `CompConfig`** (data-modality × paradigm × task × metric × cv-scheme) and **route to the matching pack** — or, if none matches (neurogolf-class), emit an **"unknown-comp" report** naming the closest pack + the specific new capability to build. This is the generalization of `journey`/`orchestrate` to *any* comp. Without it, every new comp needs me hand-holding — defeating the point.
- Verification: run it on 3+ real slugs (a tabular playground, biohub, an agent comp) and assert it produces a valid `CompConfig` + correct route.

### C2.2 — SECURITY sub-paradigm under Agentic (reuse JED, don't reinvent)
`ai-agent-security-multi-step-tool-attacks` = **agentic × adversarial**. We have the [[aas_dense_exfil_strategy]]
and [[aas_local_cv_methods]] experience (S=0.09×N_eff, replay-budget the limit, replay>SDK>deprecated). Add
under the agentic pack:
- **`sec-attack`** — construct multi-step tool-attack / exfil sequences within the scoring budget (dense-exfil pattern, stacking-dead lesson).
- **`sec-replay`** — the replay-mining path (our proven `lb-replay-mine` specialized for attack traces).
- **`sec-eval`** — score a candidate strategy against the budget model **offline** before submit (the ~336s replay-budget limit is the real constraint, not cleverness).
- Governance: dual-use guardrail — these agents operate only for the **competition's defensive/eval framing**; the same guardrails as the platform's security rules.

### C2.3 — TABULAR must be TURNKEY (playground series is high-frequency, low-margin)
Playground comps reward **speed + a strong default**, not novelty. Requirement: `comp-onboard`→`tab-*`→submission
must run **end-to-end with zero hand-tuning** and land a competitive baseline (LightGBM/CatBoost blend + leak-safe
CV) in one campaign. Add **`tab-autobaseline`** = the one-call "profile→cv→3-backend-train→blend→submission" that
gives a top-quartile playground result by default; everything else (fe, pseudo, stacking) is refinement on top.

### C2.4 — De-risking the build itself (grounded in tonight's failures)
- **Fixtures before features.** Each pack ships a **synthetic mini-comp** (tabular toy, image toy, 1-env agent toy, 1-task ARC toy) checked into `test_fleet_agents/fixtures/`. No pack agent is "done" until its fixture runs `profile→…→submission` GREEN. This is the antidote to "GREENs locally, dies on the real comp" (the polars gotcha, the division-blind golden eval).
- **Every score through the paired gate.** Reuse `math-master.paired_delta_report` as the *only* keep/reject rule — the ledger's empty `train_set` rot and the "0.94 subset over-credit" trap must be structurally impossible: the Scorer contract **requires** an eval-set tag + a paired-delta verdict on every recorded number.
- **Coverage as a test, not a doc.** `coverage-audit` agent imports each pack, runs its fixture, and regenerates `docs/agent_coverage.md`; a pack with a RED fixture is quarantined (same governance as data-wise tests).
- **Don't over-fork.** ~75 "generic" agents are the moat — resist duplicating them per modality. A new `tab-train` wraps sklearn/LGBM; it does NOT re-implement `arch-search`, `ensemble`, `ledger`, `submit-verify` — it *calls* them.

### C2.5 — Final taxonomy (data-modality × paradigm) + routing
| | Predictive | Agentic | Reasoning |
|---|---|---|---|
| **tabular/seq** | playground-s6e7, rogii-wellbore → `tab-*` (+seq, +domain hook) | — | — |
| **image/video/3d** | `img-*`/`vid-*`/`pc-*` | — | — |
| **3d+time** | biohub → reference pack | — | — |
| **text/LLM** | LLM classification → `llm-*` | `agent-*` (autonomous-agent, pokemon-tcg) + `sec-*` (ai-agent-security) | `reason-*` (arc-agi-3) |
| **novel/unknown** | | `comp-onboard` routes or reports the gap (neurogolf) | |

### C2.6 — FINAL build order (this is what I execute now, one agent at a time, verified)
1. **`comp_config.py`** (contract: modality × paradigm × task × metric-registry × cv × submission).  ← FOUNDATION
2. **`comp-onboard`** agent (the front door; verified on ≥3 real slugs).
3. **Scorer generalization**: `official-score`/`scorer` read `CompConfig.metric` via the metric-registry; require eval-set tag + paired-delta on every record.
4. **Tabular pack**: `tab-profile`→`tab-cv`→`tab-train`(LGBM/XGB/CatBoost/HistGBM)→`tab-stack`→`tab-autobaseline`; fixture-tested.
5. **Agentic pack**: `agent-env`→`agent-policy`→`agent-eval`→`lb-replay-mine`→`sec-*`; toy-env-tested.
6. **LLM + Reasoning**: `llm-finetune`(generalize `lora-train`)/`llm-infer`/`llm-eval` → `reason-dsl`/`program-search`/`ttc`.
7. **Image → Video/3D** packs.
8. **Biohub refactor** to the contract (reference 3D+time) + **`coverage-audit`** live matrix.

**Execution rule (per user): take the best choice, do not wait, verify each agent before the next.**
