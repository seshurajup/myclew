# Diabetes Prediction Challenge: Final Community Write-up

My name is Asif Waliuddin. I am a late-in-life Type 2 diabetic, and this project is personal. Kaggle calls this a beginner competition, but for me it is a chance to build a reliable, reproducible ML pipeline that others can learn from and build on.

This document is meant to be shared with the community: it summarizes the approach, what worked, what did not, and how to reproduce our best runs. Code and models are included in this repo.

---

## TL;DR

- **Best public score:** 0.69768  
  Achieved with cross-bundle stacking on 12 bundles (base predictions).
- **Best internal OOF:** 0.729185  
  Achieved by blending the best stacked model with the best 10-seed bagging (90/10).
- **Key lesson:** internal OOF gains did not always translate to public leaderboard gains.

## Links

- **Kaggle profile:** https://www.kaggle.com/asifwaliuddin
- **GitHub repo:** https://github.com/awaliuddin/Diabetes-Prediction-Challenge
- **Submission log:** [SUBMISSIONS_LOG.md](SUBMISSIONS_LOG.md)
- **Run dossiers:** [reports/runs/](reports/runs/)
- **Models (bundles):** [models/final/](models/final/)
- **Artifacts (summaries, Optuna):** [models/artifacts/](models/artifacts/)
- **Notebooks:** [notebooks/](notebooks/)

---

## Why This Project Matters

This is not just a leaderboard chase for me. I want to understand every piece of the system and make it reproducible for others. The goal is a project that teaches solid ML habits: clean experiments, versioned artifacts, and documented decisions.

---

## What We Built

We built an end-to-end pipeline with:

- **Clean configuration** (`config/config.yaml`)
- **Feature engineering** in `src/data/feature_engineer.py`
- **Cross-validation** (StratifiedKFold)
- **Multi-model training** (XGBoost, LightGBM, CatBoost)
- **Multi-seed bagging**
- **Cross-bundle stacking**
- **Run dossiers and experiment artifacts**

Everything is reproducible using the scripts in `scripts/` and the commands in `RUNBOOK.md`.

---

## Evaluation

Kaggle evaluates predictions using ROC-AUC. We use consistent OOF ROC-AUC in cross-validation to compare experiments internally and reduce the risk of overfitting to the public leaderboard.

---

## Feature Engineering Summary

We tested several feature sets:

- **enhanced (baseline):** ratios, clinical thresholds, domain-inspired interactions.
- **enhanced_v4 / v5:** added competitor-inspired ratios and AI-derived thresholds.
- **enhanced_v6:** added target/frequency encoding and tree-based thresholds.

Result: **enhanced_v6 underfit LightGBM badly** (LGBM ~0.70 OOF), even after GPU tuning. We reverted to the baseline feature set for competitive performance.

---

## Ensembling Strategy

We combined multiple ensembling techniques:

- **Weighted model ensembles** (typical weights around XGB 0.15 to 0.25, LGB 0.60 to 0.65, CAT 0.10 to 0.20)
- **Multi-seed bagging**
- **Cross-bundle stacking** (stacked base predictions from multiple run bundles)
- **Rank/logit feature augmentation** for stacker inputs
- **Blending** (stacked + bagged)

Stacking across 12 bundles produced the best public score.

---

## Best Results (Public Leaderboard)

| Method | Submission | Public Score |
| --- | --- | --- |
| Stacked base predictions (12 bundles) | `submissions/submission_stacked_bundles_base_20251224_191713_20251224_191921.csv` | **0.69768** |
| Blend stacker + bagging (85/15) | `submissions/submission_blend_stack85_bag15_20251227_101355.csv` | 0.69762 |
| Blend stacker + bagging (90/10) | `submissions/submission_blend_stack90_bag10_20251227_100114.csv` | 0.69761 |
| Bagging (7 seeds + optimized weights) | `submissions/submission_bagging_20251220_164820_20251220_164928.csv` | 0.69753 |

Full submission history is tracked in `SUBMISSIONS_LOG.md`.

---

## What Did Not Work (Important)

- **enhanced_v6 feature set** (target/frequency encoding) caused LightGBM to underfit.
- **Class weighting + native categorical** caused LightGBM collapse (AUC ~0.50).
- Some internal OOF improvements did **not** improve public scores.

We logged failures to keep the project honest and reproducible.

---

## Reproducibility (Core Commands)

### Train (default config)
```bash
make train
```

### Train multiple seeds (live logs)
```bash
make train-seeds-live TRAIN_SEEDS="11 23 37 41 53"
```

### Optimize ensemble weights
```bash
python scripts/ensemble_optimize.py --bundle models/final/bundle_<bundle>.pkl --grid-step 0.05 --stack --write-bundle
```

### Bag latest bundles
```bash
python scripts/bagging.py --latest 5 --ensemble weighted --weights 0.25,0.65,0.10
```

### Stack multiple bundles
```bash
python scripts/stack_bundles.py --bundles "<bundle1>,<bundle2>,..." --feature-mode base --stacker-folds 5
```

---

## Code and Models (Included)

We are sharing:

- **Code:** everything in `src/` and `scripts/`
- **Models:** `models/final/` (bundles) and `models/artifacts/` (summaries, Optuna outputs)
- **Predictions:** `submissions/` (predictions and submission CSVs)
- **Documentation:** `RUNBOOK.md`, `Tactical_Plan.md`, `SUBMISSIONS_LOG.md`

Note: Kaggle data is not included; download with `make download`.

---

## Final Reflection

This project grew into a real, production-style ML workflow. It is far more than a Kaggle submission: it is a reproducible system, a learning artifact, and (for me) a personal commitment to understanding the craft of ML.

If you are new to this space, I hope this repo helps you. If you are experienced, I welcome your feedback and improvements.

---

## Call to Action

- Open issues for ideas or bugs.
- Share improvements to features or ensembling.
- Help us push beyond the current ceiling.

Thank you for reading and supporting this work.