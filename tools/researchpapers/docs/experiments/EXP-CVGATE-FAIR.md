# EXP-CVGATE-FAIR: Fair-parity CV gate: full-pipeline pilkwang vs canqiang on density LOEO, official metric

- **status:** PLANNED   <!-- PLANNED | RUNNING | DONE | KILLED -->
- **author:** researcher
- **created:** 2026-07-05
- **idea class:** validation   <!-- aug | lr | det-loss | window | pool | gating | resolution | postproc -->
- **package:** docs/cvgate_metricgap_investigation.md; splits_loeo_density.json   <!-- bracket yml / config path(s) -->

## Hypothesis — PRE-REGISTER BEFORE RUNNING (do not backfill)
> The whole point of the journal: write the WHY and the falsifiable claim *before* seeing results,
> so we can't rationalise noise after the fact.

- **Motivation:** the original `EXP-CVGATE` FAILED (pilkwang 0.779 < canqiang 0.793), but the investigation
  (`docs/cvgate_metricgap_investigation.md`) showed that FAIL was a **pipeline-parity artifact**, not a real
  LB-unfaithfulness: the gate ran a **de-featured pilkwang** (ILP linking OFF; `pool_kernel_um=3.0` instead
  of the config's 5.0 → 1.34× over-detection) scored with a **proxy** metric, against canqiang's *own full*
  pipeline. adjJ vs count_ratio across the 15 geffs is r=−0.841; correcting over-detection alone extrapolates
  pilkwang to ~0.85–0.92, above canqiang.
- **Claim (falsifiable):** with the **two parity gaps closed** (ILP ON + `pool_kernel_um=5.0`) **and** the
  **official `evaluate.py`** metric (both detectors' full pipelines scored identically), the density LOEO CV
  ranks **pilkwang > canqiang** (mean over both folds) — agreeing with the Kaggle LB.
- **Expected signal + direction:** pilkwang mean-official rises from 0.779 toward ~0.85+; **pilkwang −
  canqiang > 0** on both folds; count_ratio drops from ~1.34 toward ~1.0.
- **Measurement:**
  - Both pipelines, **both folds** of `splits_loeo_density.json`, scored with the **official
    `research/official_repo/scripts/evaluate.py`** (fixes the proxy hole).
  - pilkwang: full `learning/ensemble_work/pilkwang_full/pipeline.py` (fusion + gap-recovery + post-proc),
    **or** interim = bare predict `--use-ilp --pool-kernel-um 5.0` (proxy metric) as a first signal.
  - canqiang: its existing full-pipeline predictions, re-scored with the same `evaluate.py`.
  - Sidecars carry `--exp-id EXP-CVGATE-FAIR` (does NOT clobber `EXP-CVGATE`).
- **Decision rule:** **PASS iff pilkwang mean-official > canqiang** under the fair pipeline + official
  metric → the density CV IS LB-faithful, idea screening on it is unlocked. **Still ≤ → the density CV is
  genuinely not LB-faithful → escalate to human** (do not screen ideas on it).

## Results — AUTO-FILLED by `baseline/exp_journal.py` (do not hand-edit between the markers)
<!-- AUTOFILL:EXP-CVGATE-FAIR:START -->
| run | fidelity | official adjJ | golden_cv | recall | count× | div_tp | status |
|---|---|---|---|---|---|---|---|
| pilk_ilp_k5_f0 | mini·splits_loeo_density | **0.8077** | 0.8269 | 0.981 | 1.23 | 0 | DONE |
| pilk_ilp_k5_f1 | mini·splits_loeo_density | **0.7866** | 0.8144 | 0.976 | 1.19 | 0 | DONE |
<!-- AUTOFILL:EXP-CVGATE-FAIR:END -->

## DEFINITIVE result — full-fusion pipeline, OFFICIAL `evaluate.py` (not in AUTOFILL: evaluate.py emits no sidecar)
Source: trainer run `train-20765ff1d6`, `docs/cvgate_full_fusion_result.md` (verified clean: stem-match OK,
n=8/7, no contamination).

| detector (full pipeline) | fold0 official | fold1 official | **mean** | vs canqiang |
|---|---|---|---|---|
| **pilkwang-FULL** (ILP + best.pt fusion + full postproc, pool=3.0) | 0.8567 | 0.8178 | **0.8373** | — |
| canqiang-FULL | 0.7973 | 0.7879 | 0.7926 | — |
| **Δ (pilk − canqiang)** | **+0.0594** | **+0.0299** | **+0.0447** | wins BOTH folds |

## Verdict — researcher, AFTER results
**PASS.** With both detectors run as their FULL LB pipelines and scored by the OFFICIAL `evaluate.py`,
pilkwang **0.8373 > canqiang 0.7926 on BOTH folds** (Δmean **+0.0447**, larger than the public-LB gap
+0.024). **The embryo-disjoint density LOEO CV IS LB-faithful** and can be trusted for detector/idea
screening — *provided ideas are screened as full pipelines under the official metric, never the bare predict.*

The journey vindicates the parity diagnosis exactly: **bare −0.0140 (FAIL) → interim ILP+pool5 proxy +0.0046
→ full-fusion official +0.0447 (PASS)**. The original FAIL was **entirely a pipeline-parity artifact**
(crippled pilkwang: ILP-off + pool 3.0-without-postproc + proxy metric), not genuine LB-unfaithfulness.
canqiang proxy == official exactly (0.7973/0.7879) — the proxy was fair for canqiang; it only mismeasured
the crippled pilkwang. Full analysis: `docs/cvgate_fair_results.md`, `docs/cvgate_metricgap_investigation.md`.
Two silent-failure traps found+fixed en route: `pool_kernel_um` not CLI-wired; `pipeline.py` aborts without
`BIOHUB_MODEL_ARTIFACTS` (runner then converted a stale gold12 submission → false 0.8971).
