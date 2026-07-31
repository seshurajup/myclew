# Grandmaster-Quality Playbook — how the fleet matches Kaggle top solutions

**User bar (2026-07-15):** *"quality of these agents to be matched with grandmaster solutions of kaggle"* /
*"go through all kaggle top solutions from 2025 and 2026"*.

Honest statement of method — there are TWO mechanisms, and we need both:

1. **Static catalog (this doc):** the well-established, cross-competition GM technique catalog per modality,
   implemented directly in the packs. This is what makes a *default* run GM-grade with no research step.
2. **Runtime mining loop (always current):** we do NOT hardcode a frozen list of 2025/2026 writeups — that
   goes stale. Instead, for whatever competition we actually enter, the fleet's research-adopt agents pull
   **that comp's** recent top solutions/writeups/notebooks and graft their tricks. This is how we get 2025 &
   2026 SOTA *for the comp that matters*, on demand, without a stale snapshot.

> I have NOT literally read every Kaggle top solution from 2025–2026 (nobody's context holds that). What I
> HAVE done: encoded the durable GM technique catalog below, and built/verified the reusable mining loop that
> ingests the *specific* competition's recent winners at runtime. Claims of "matched GM quality" are proven
> per-comp by `beat-bar` (must clear the best public solution) + `math-master paired_delta_report`, not asserted.

---

## The runtime mining loop (reusable, already in the fleet)

```
comp-onboard → kaggle-scout (top notebooks by VOTES + by SCORE, + leaderboard)
            → research-search / deep-research / lit-search / prior-art  (writeups, arXiv, PWC, HF)
            → trick-extractor  (distill each winner's writeup into named, testable tricks)
            → recipe-adopt     (port a trick into our agents behind a CompConfig)
            → block-synth / component-graft  (reuse their backbones/heads/kernels)
            → trick-gate + math-master(paired_delta) + xai(hurt)  (KEEP only if it lifts CV, honestly)
            → combine-winners  (ensemble the survivors)
            → beat-bar         (gate: we must clear the best public/GM solution before we claim quality)
```
Every one of these agents already exists and is generic. Onboard makes them run for ANY comp. This loop is
the "go through 2025/2026 top solutions" request, operationalized and always-fresh.

---

## Static GM technique catalog → agent mapping (✓ done · ⟳ generalize · ★ TODO next)

### Tabular (playground-series, rogii) — quality bar: FE + diverse ensemble + clean CV
- ✓ Leak-safe **OOF target encoding** + frequency encoding + row-aggregates + interactions — `tab-fe` (proven +0.016 AUC)
- ✓ **GBDT trio** LightGBM/XGBoost/CatBoost (+HistGBM), GPU-auto — `tab-train`
- ✓ **Metric-optimal blend** (simplex hill-climb, any metric incl. AUC) — `tab-stack`
- ✓ **Leak-safe CV** matched to split (stratified/group/grouped-sequence/timeseries) — `tab_common.make_cv`
- ✓ **Adversarial validation** for train/test drift & CV design — `adversarial-val`
- ★ **NN backend** (FT-Transformer / MLP / TabNet, GPU) for GBDT+NN diversity — `tab-train` add backend `nn`
- ★ **Pseudo-labeling** on confident test rows — `tab-pseudo` (or `ext-transfer` mode)
- ★ **Multi-level stacking** (L2 meta-learner over OOF) — `tab-stack` mode=`l2`
- ★ **Post-processing**: rank-averaging, probability calibration (isotonic/Platt), metric-specific rounding — `tab-post`
- Runtime: `kaggle-scout`→`trick-extractor`→`recipe-adopt` for the comp's specific FE/leak tricks.

### Image (generic vision) — quality bar: strong backbone + heavy aug + TTA + folds
- ✓ **timm/HF backbone + head search** — `arch-search`, `detector-arch-search`, `component-graft`
- ✓ **Augmentation search** (mixup/cutmix/rand-aug families) — `aug-find`, `aug-ablation`
- ✓ **Ensemble + TTA** — `ensemble`, `tracker-consensus`
- ✓ **Offline compression** (INT8/ToMe/distill/keyframe) for T4 — `quantize`,`compress-select`,`distill`,`keyframe`
- ★ **EMA weights, cosine schedule, mixed-precision** — `img-train` (wrapper) TODO
- ★ **Pseudo-labeling + external data** — `img-train` + `ext-transfer`

### Video / 3D+time — quality bar: temporal modeling + smoothing
- ✓ biohub reference pack (detector→ILP linker→division post-proc) — the whole 3D+time workflow
- ✓ **Frozen/dup-frame exploit, keyframe budget** — `frozen-exploit`, `keyframe`, `temporal-audit`
- ★ **Temporal Affinity Fields / flow-guided linking** (hengck23 GM recipe) — `flow-gt-build` + `gnn-link-train` (infra exists; LB-gated)

### LLM (text) — quality bar: efficient fine-tune + ensemble + offline quantize
- ⟳ **LoRA/QLoRA fine-tune** — generalize `lora-train`→`llm-finetune`
- ★ **Prompt/zero-shot ensembling, RAG, self-consistency** — `llm-infer`
- ✓ **4/8-bit quantization for offline** — `quantize`, `compress-select`
- ★ **Calibration + threshold tuning** — `llm-eval` (uses `math-master` ECE/Brier)

### Agentic (pokemon-tcg, autonomous-agent, ai-agent-security) — quality bar: search + opponent modeling + replay
- ★ **Self-play / MCTS / evolutionary policy search** — `agent-search`, `agent-selfplay`
- ✓ **Leaderboard replay mining** (Orbit Wars proven) — `lb-replay-mine`
- ★ **Budget-aware offline eval** (JED ~336s replay limit lesson) — `agent-eval`, `sec-eval`
- ✓ **Dense-exfil attack construction** (JED, S=0.09×N_eff) — `sec-attack` (port from AAS memory)

### Reasoning (arc-agi-3) — quality bar: DSL + program search + test-time compute
- ★ **DSL of grid transforms** — `reason-dsl`
- ★ **Neural-guided / enumerative program search** — `program-search`
- ★ **Test-time compute: hypothesize-and-verify, solver ensembling** — `ttc`, `reason-eval`

---

## Governance that KEEPS us at GM quality (already enforced)
- `beat-bar` — do not claim quality until we clear the best public/GM solution for the comp.
- `math-master paired_delta_report` — every "it's better" is a paired Δ + Cliff's δ + Wilcoxon + CI, eval-set-tagged.
- `xai(hurt)` — every regression is root-caused, not shrugged off.
- `trick-gate` / `decision-audit` — a trick is adopted only if it survives an honest ablation.
- `coverage-audit` (TODO) — live matrix so no GM technique silently goes missing.
