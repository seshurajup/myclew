# Results-Writeback Contract — MLflow → EXP-<id>.md

> **Design note (trainer), DESIGN ONLY — nothing runs on GPU/queue until the human's go.**
> Division (leader): **researcher** owns the `EXP-<id>.md` template + field schema + `exp_journal.py` (fills the `Result` table) and leads; **trainer** defines this writeback contract — which MLflow params/tags/metrics map to which `Result` columns, and the key that lets a scored run find its `EXP-<id>.md`. This doc is for us to agree on; researcher edits freely.

## 1. Source of truth (what my scoring path already emits to MLflow)
`baseline/score_v1.py` (and future `score_v2+`) logs to experiment **`kaggle-biohub-cell-tracking`**, one fresh run per score (pops `MLFLOW_RUN_ID`), `log_system_metrics=True`:
- **metrics** (`mlflow.log_metrics`): `official_score` (= adj_edge_jaccard + 0.1·division_jaccard — THE golden-12 official), `adj_edge_jaccard`, `division_jaccard`, `micro_adjJ`, `golden_cv`, `mean_node_recall`, `mean_count_ratio`, `div_tp_total`, `adjJ_44b6`, `adjJ_6bba`, `n`.
- **tags**: `phase=golden12_score`; `config_path` (standing rule).
- **params**: `config_file` (yaml basename), `geff_dir`.
- **run_name**: the method (e.g. `baseline_v1_v1_2_hr_baseaug`) or the `--run-name` passed; ref run = `pilkwang_baseline_score_validate`.

Training runs (`src.baseline.train` → official trainer) log their own per-method run with `config_file`/`config_path` + system metrics (per the MLflow discipline standing rule); screening rungs additionally carry prune telemetry (see §4).

## 2. Column mapping (EXP-<id>.md `Result` table)
Table header today: `| run | status | golden-12 | recall | pruned | dur |`

| Result column | MLflow source | Notes |
|---|---|---|
| `run` | run_name | key display; ref run + each arm appear as rows |
| `status` | MLflow run `status` | FINISHED / FAILED / KILLED (KILLED = pruned/aborted) |
| `golden-12` | metric **`official_score`** where `tag.fidelity=golden12` | the FINAL rung, full golden-12 |
| `mini-official` | metric **`official_score`** where `tag.fidelity=mini` | screening rungs — a REAL (mini) official adjJ, not a proxy (see §4); fidelity-to-golden-12 noted |
| `recall` | metric `mean_node_recall` | logged on both mini and golden-12 score runs |
| `pruned` | prune tag (see §4) | `-` if not pruned; else e.g. `rung1@0.90` |
| `dur` | run end − start (min) | from MLflow timestamps |

> **Fidelity is a first-class dimension, not a caveat.** Every score run carries a `fidelity` tag (`mini` \| `golden12`). Represent it as EITHER a distinct `mini-official` column (above) OR a single `official` column plus a `fidelity` marker per row — **researcher's call (Q3)**. Both a mini row and a golden-12 row are REAL official adjJ on the SAME metric (`predict → pilk_post → src.metric`), differing only in the eval split.

Extra lineage available for the journal frontmatter / detail rows (leader's requested set — all already logged, no new work): `micro_adjJ`, `golden_cv`, `mean_node_recall`→recall, `mean_count_ratio`→count_ratio, `adjJ_44b6`/`adjJ_6bba` per-embryo, `config_file`/`config_path` lineage, and the run URL `http://localhost:5000/#/experiments/<exp_num>/runs/<run_id>`.

## 3. Keying — how a scored run finds its EXP-<id>.md  ⟵ **needs researcher decision**
`config_file` alone is ambiguous (the pilkwang ref run has no arm config; one EXP can span multiple configs; screening vs golden-12 rows share a config). **Proposal (trainer):** every train + score run logs an explicit MLflow tag **`exp_id`** (e.g. `EXP-001`), sourced from the experiment config or a tiny `config→exp_id` map; `exp_journal.py` then keys the writeback on `tag.exp_id` and appends/updates rows under that EXP's `<!-- AUTO -->` marker. Fallback if you prefer no new tag: key on `param.config_file` ↔ EXP frontmatter `config:`, and special-case the ref run. **Please pick one and say where the `exp_id` value originates so I wire the tag at score time.**

## 4. Screening rungs & prune — needs reconciliation
- **Screening now ranks on MINI-OFFICIAL adjJ** (leader, methodology change): the bracket rungs run the REAL metric (`predict → pilk_post → src.metric`) on the matched mini-val, NOT the acc·recall proxy (proven BLIND: flat 0.944–0.956 while official spans 0.61–0.82). So a screening row carries a **faithful mini-official adjJ number** (`official_score` with `tag.fidelity=mini`), recorded AS a real score with its fidelity-to-golden-12 noted — **do NOT label it "proxy-only."** The final rung is the full golden-12 (`tag.fidelity=golden12`). Only mark a row `proxy-only` if it genuinely used acc·recall (legacy/fallback), which should now be rare.
  - Contract requirement: the screening scorer must log `official_score` (+ the same lineage metrics) on the mini split and set `tag.fidelity=mini` and `tag.eval_split=<mini-set name>`. Confirm the mini scorer path/name with researcher so my keying matches.
- `pruned`: the `BIOHUB_PRUNE_RUNGS` self-kill — **what does it log?** (proposed: tag `pruned=<rung>@<bar>` + run status KILLED). Confirm so `pruned`/`status` fill correctly.
- Caveat retained: resolution (1,2,2)-vs-(1,4,4) still stays a FULL converged A/B (late-convergence), not bracket-pruned — the mini-official is faithful for early-showing ideas, but a mini rung would still under-serve a late-converging finer detector.

## 5. Idempotency & timing
- Writeback is **append-or-update by run_id** under the `<!-- AUTO: exp_journal.py fills below -->` marker (re-scoring a run updates its row, not duplicates). Human-written sections (Why/Prediction) are never touched.
- Trigger: after a score run finishes (natural fit for trainer's post-job consolidation lane) — same step where I already write `docs/baseline_v*_exp_result.md`. No GPU; pure MLflow read + markdown edit.

## 6. Open questions for researcher (own the schema)
1. Keying: `exp_id` tag (recommended) vs `config_file`? Where does `exp_id` originate?
2. `golden-12` column = `official_score` (confirmed OK?) — and do you want a second `micro`/`golden_cv` column, per leader's field list?
3. **Mini-official screening** (updated per leader): screening rows carry a REAL mini-official adjJ, not a proxy. (a) Represent as a distinct `mini-official` column or a single `official` column + `fidelity` marker? (b) Confirm the mini scorer logs `official_score` with `tag.fidelity=mini` + `tag.eval_split=<mini-set>` — what's the mini scorer path/name and mini-set id so I key it? (c) Any legacy acc·recall rows are the ONLY ones labeled `proxy-only`.
4. Prune telemetry: exact tag/param the prune wire logs.
5. Does `exp_journal.py` read MLflow directly, or a score-emitted JSON sidecar? If a sidecar is easier for you, I can have the scorer drop `output/.../score.json` with these exact keys — your call.

_Once you confirm §3–§6, this contract is frozen and the writeback "just works" when training resumes._
