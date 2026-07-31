# Fleet Python Agents — the workers the leader/researcher can dispatch to

There are **69** deterministic Python worker agents. Dispatch a job to one by enqueuing a
question with its `kind`. From the runtime shell (PYTHONPATH=tools/researchpapers):

```bash
python -m researchpapers.fleet_dispatch <kind> "<question text>" '<optional spec JSON>'
```

The matching worker claims it, runs it, scores via the official scorer, and posts a finding/verdict.
Every recorded CV needs scorer-JSON proof (provenance gate). Agents (kind — purpose):

- `ablate-best` — grandmaster — the experimental TRICKS from top Kaggle journals (DrHB/icecube-journal, rna-stanford) implemented as Python agents
- `adversarial-val` — Adversarial-validation agent — confirm the CV axis matches the hidden test (embryo-disjoint)
- `analysis` — Metric-decomposition adapter — deterministic, from real MLflow component metrics
- `arch-builder` — arch-builder — DERIVE the model architecture from data analysis, never hardcode it
- `arch-probe` — Experiment adapter — run an EXISTING competition config via the config-driven trainer
- `arch-search` — arch-search — PROVE arch-builder's search space by starting SIMPLE and growing one axis at a time
- `aug-ablation` — Experiment adapter — run an EXISTING competition config via the config-driven trainer
- `aug-find` — Aug-finder agent — DERIVE the valid augmentation menu FROM THE DATA (not a fixed list)
- `baseline` — Journey stage agents (S1 baseline, S2 tune, S5 linking, S6 division, S7 ensemble, S8 post-proc)
- `best-config` — best-config — assemble the BEST inference config from the public learnings + CV↔LB anchors (Part A)
- `block-synth` — block-synth — mine the DISTINCT post-proc code blocks across public notebooks and compose NEW ones
- `box-sample` — box-sample — make the DENSE external embryos look like the SPARSE competition crops by sampling sub-boxes with a matching CELL COUNT (the user's insig
- `combine-winners` — grandmaster — the experimental TRICKS from top Kaggle journals (DrHB/icecube-journal, rna-stanford) implemented as Python agents
- `combined-train` — combined-train — train the division/affinity model on BOX-SAMPLED external + competition data directly, then evaluate on golden-12
- `combo-search` — combo-search — autonomous golden-12 combination search over PUBLIC-NOTEBOOK post-proc knobs
- `config-ablate` — config-ablate — leave-one-BLOCK-out ablation of the yaroslav-v4 full config (0.8803 base)
- `config-gen` — config-gen — deterministic YAML config author (takes this off the researcher)
- `cv-build` — CV-build adapter — REUSES the competition's own src.cv (embryo-disjoint) via its venv
- `data-audit` — data-audit — MEASURE the training data's scale/quality and CORRECT it, before anything trains on it
- `decision-audit` — decision-audit — enforce "decide only from data" across the whole ledger
- `deep-sister` — deep-sister (Part B) — deep sister model
- `div-model` — div-model — a SECONDARY SUPPORT MODEL (generic): small classifier that claims an unclaimed metric term
- `division` — Journey stage agents (S1 baseline, S2 tune, S5 linking, S6 division, S7 ensemble, S8 post-proc)
- `eda-stats` — EDA-stats agent — surface the data fingerprint from the competition's precomputed EDA outputs
- `ensemble` — Journey stage agents (S1 baseline, S2 tune, S5 linking, S6 division, S7 ensemble, S8 post-proc)
- `ext-label-stats` — ext-label-stats — inventory the EXTERNAL dense lineage labels and prove they're usable supervision
- `flow-gt-build` — flow-gt-build — turn the external dense lineage tracks into per-node FLOW + DIVISION supervision
- `fullconfig-search` — fullconfig-search — WIDE autonomous golden-12 search over the FULL post-proc config
- `gnn-link-train` — gnn-link-train — TRAIN the division + flow heads on the clean external GT (the div_J lever)
- `gnn-probe` — gnn-probe — does a GRAPH neural net help, or is pairwise geometry already enough? The problem is a spatiotemporal graph (cells=nodes, candidate links=
- `gpu-best-practices` — gpu-best-practices — from our precision + 5090/Blackwell research: catalogue every GPU best practice (compile, kernels, memory layout, precision) with
- `guard` — Guard agent — the trainer's 'is this result reliable?' check, done deterministically
- `heal` — heal — the SELF-HEALING bridge
- `insights` — insights — the fleet's FINAL-INSIGHTS markdown for the super-leader / super-researcher handoff
- `journey-status` — The grandmaster JOURNEY — the ordered experiment progression, as data + a status agent
- `kaggle-scout` — Kaggle-scout agent — pull top public NOTEBOOKS (+ leaderboard) via the Kaggle CLI, so we don't miss
- `layer-grow` — layer-grow — CHOOSE the network depth layer-by-layer, each layer justified by PROOF
- `learn` — Learner agent — capture NEW knowledge as a Pattern-B lesson (.py pure code + .learning) and refresh
- `ledger` — Experiment ledger — grandmaster experiment-journal style (DrHB/icecube-journal + DrHB/rna-stanford)
- `linking` — Journey stage agents (S1 baseline, S2 tune, S5 linking, S6 division, S7 ensemble, S8 post-proc)
- `metrics-report` — metrics-report — the LEADER-facing COMPLETE metrics table for a finished+scored run
- `notebook-sync` — notebook-sync — DAILY: pull new/updated top public Kaggle notebooks and EXTRACT learnings
- `notes-sync` — Note-parser agent — ingest STRUCTURED research notes into the journal (no LLM)
- `orchestrate` — orchestrator — the deterministic DECISION LOOP (runs the whole journey WITHOUT the Claude leader)
- `paper-research` — paper-research — mine RECENT architecture innovations from papers, weighing ACCURACY *and* SPEED
- `perf-choice` — perf-choice — benchmark the compute BACKENDS for a hot operation and recommend the fastest, so no agent ever ships the slow choice again (the per-node
- `pipeline-run` — pipeline-run — config-driven end-to-end: inference base + secondary support models → COMBINED golden CV
- `plan-ingest` — plan-ingest — the HUMAN's plan file → executed by the Python fleet (works with NO super-leader)
- `post-analysis` — Post-analysis agent — AFTER an experiment, deliver the verdict deterministically
- `post-proc` — Journey stage agents (S1 baseline, S2 tune, S5 linking, S6 division, S7 ensemble, S8 post-proc)
- `pre-analysis` — Pre-analysis agent — BEFORE an experiment, diagnose the current state and recommend the next lever
- `prior-art` — prior-art — cover PREVIOUS-YEAR / external top solutions (not just this comp's notebooks)
- `public-config` — public-config — one config/exp yml per PUBLIC notebook (full coverage of all 73+)
- `reproduce-score` — reproduce-score — golden-12 score a public notebook's pipeline (Python, no leader)
- `sample-match` — sample-match — understand HOW the author sampled/labelled the competition crops and GATE our external data against that scheme (the user's insight: ma
- `score` — Score step — the competition's PREDICT+SCORE, run AFTER training to produce the golden/official CV
- `scoreboard` — scoreboard — ONE live message on /runtime that is a MARKDOWN TABLE of the golden-CV leaderboard, updated in place as scores land (no new message each 
- `scorer` — Scorer agent — report the CV trajectory (official_score / golden_cv) across runs (deterministic)
- `single-model-tune` — Journey stage agents (S1 baseline, S2 tune, S5 linking, S6 division, S7 ensemble, S8 post-proc)
- `smoke` — smoke — pre-flight: run a TINY real end-to-end of a config (1 epoch, a few iters, real data) under a HARD timeout, so a broken/hanging config is caugh
- `split-build` — split-build — deterministic, VALIDATED CV-split builder (takes this off the researcher)
- `stage-1-div` — stage-1-div — STAGE-1 div_J verdict: run div-model on 36-event predicted-node split
- `submission-build` — submission-build — assemble the Kaggle submission from the best predictions (HUMAN submits)
- `tracker-consensus` — tracker-consensus — hengck23's core recipe: run MANY trackers, keep the links they AGREE on
- `train-monitor` — train-monitor — LIVE watchdog over a running training (GPU/CPU + log freshness)
- `trick-extractor` — trick-extractor — mine EVERY top-solution trick from the downloaded Kaggle notebooks, by area
- `trick-gate` — trick-gate — the EVIDENCE GATE: a trick is adopted ONLY if it proves out on golden-12, never on popularity
- `verify-cv` — verify-cv — compute the REAL golden-12 CV for a public learned-graph notebook (no hardcoding)
- `xai` — xai — the full Explainable-AI / mechanistic-interpretability suite, as ONE reusable engine
