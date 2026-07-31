# EXP-CVGATE: CV-faithfulness gate: pilkwang vs canqiang on embryo-disjoint density CV

- **status:** PLANNED   <!-- PLANNED | RUNNING | DONE | KILLED -->
- **author:** researcher
- **created:** 2026-07-05
- **idea class:** validation   <!-- aug | lr | det-loss | window | pool | gating | resolution | postproc -->
- **package:** eda/loeo_density_cv_gate.md; splits_loeo_density.json   <!-- bracket yml / config path(s) -->

## Hypothesis — PRE-REGISTER BEFORE RUNNING (do not backfill)
> The whole point of the journal: write the WHY and the falsifiable claim *before* seeing results,
> so we can't rationalise noise after the fact.

- **Motivation:** This is not an *idea* experiment — it is the **methodology gate** that decides whether
  we are even allowed to screen ideas offline. golden-12 gave the WRONG ranking of the two public models:
  canqiang official **0.903 > pilkwang 0.870**, yet the Kaggle LB says the opposite (pilkwang LB **0.890 >
  canqiang 0.866**). golden-12 over-credits denser detectors (density-blind on sparse GT, see
  `baseline/score_v1.py` count_ratio note), so ranking ideas on it is unsafe. The recipe's mandated CV is
  **embryo-disjoint** (Kaggle train/test share no embryo). `splits_loeo_density.json` is that CV: 2 LOEO
  folds, each holding out a whole embryo (verified zero embryo overlap, `eda/loeo_density_cv_gate.md`).
- **Claim (falsifiable):** On the embryo-disjoint density CV, scoring both public pipelines with the full
  `predict → pilk_post → src.metric` official metric ranks **pilkwang > canqiang** (mean official across
  both folds), i.e. it agrees with the Kaggle LB where golden-12 disagreed.
- **Expected signal + direction:** mean-of-2-folds official adjJ: **pilkwang − canqiang > 0**. Magnitude
  unknown a priori (density CV is a different distribution than golden-12); the *sign* is the whole test.
  A near-tie (|Δ| within fold-to-fold spread) is a **soft fail** — not faithful enough to prune on.
- **Measurement:**
  - Both pipelines, **both folds** of `splits_loeo_density.json` (15 datasets total; golden-12 predictions
    do NOT count — only 2/15 overlap, `eda/loeo_density_cv_gate.md §2`).
  - pilkwang = `edge_predictor_best.pth` **(1,4,4) genuine support_pack weights** (NOT the (1,2,2)
    official_repo retrain — that would test *our* model, not pilkwang's; `§3`). Runner:
    `predict_unet_transformer.py` → `score_v1.py --split-file splits_loeo_density.json --exp-id EXP-CVGATE`.
  - canqiang = `DeepCenterUNet3D` `best.pt`, distinct predict path:
    `baseline/run_canqiang_loeodens.py --fold F --exp-id EXP-CVGATE`.
  - Both emit `output/scores/*.json` sidecars (exp_id=EXP-CVGATE) → this table auto-fills.
- **Decision rule:** **PASS iff pilkwang mean-official > canqiang mean-official** across the 2 folds →
  the density CV is LB-faithful, and idea screening on it (the successive-halving bracket) is unlocked.
  **FAIL/near-tie → STOP**: do not screen ideas on this CV; escalate to the human (LB submissions are the
  only faithful judge). Per human directive, **no idea experiments are designed or launched until this
  gate PASSES.**

## Results — AUTO-FILLED by `baseline/exp_journal.py` (do not hand-edit between the markers)
<!-- AUTOFILL:EXP-CVGATE:START -->
| run | fidelity | official adjJ | golden_cv | recall | count× | div_tp | status |
|---|---|---|---|---|---|---|---|
| canqiang_loeodens_f0 | mini·splits_loeo_density | **0.7973** | — | — | 0.72 | 0 | DONE |
| canqiang_loeodens_f1 | mini·splits_loeo_density | **0.7879** | — | — | 1.05 | 0 | DONE |
| pilk_loeodens_f0 | mini·splits_loeo_density | **0.7882** | 0.8071 | 0.989 | 1.35 | 0 | DONE |
| pilk_loeodens_f1 | mini·splits_loeo_density | **0.7690** | 0.7992 | 0.991 | 1.32 | 0 | DONE |
<!-- AUTOFILL:EXP-CVGATE:END -->

## Verdict — researcher, AFTER results
{accept / reject vs the pre-registered decision rule · observed effect size · why · next step}
