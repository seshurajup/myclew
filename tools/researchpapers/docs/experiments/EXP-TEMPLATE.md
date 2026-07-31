# {ID}: {TITLE}

- **status:** PLANNED   <!-- PLANNED | RUNNING | DONE | KILLED -->
- **author:** researcher
- **created:** {DATE}
- **idea class:** {CLASS}   <!-- aug | lr | det-loss | window | pool | gating | resolution | postproc -->
- **package:** {PACKAGE}   <!-- bracket yml / config path(s) -->

## Hypothesis — PRE-REGISTER BEFORE RUNNING (do not backfill)
> The whole point of the journal: write the WHY and the falsifiable claim *before* seeing results,
> so we can't rationalise noise after the fact.

- **Motivation:** {why this idea — grounded in a prior result / EDA fact / paper, with the reference}
- **Claim (falsifiable):** {the one specific thing this experiment tests}
- **Expected signal + direction:** {which metric moves, which way, rough magnitude — e.g. "mini-official
  adjJ +0.01..0.03 vs the baseaug arm; node_recall unchanged"}
- **Measurement:**
  - Screen: **mini-official** adjJ on `splits_screen_matched.json` (golden-12-matched, leak-free) —
    faithful cheap proxy (NOT acc·recall, which we proved blind).
  - Final judge: **golden-12 official** adjJ on `splits_ft.json` for the survivor only.
  - Density-changing? {yes/no} → if yes, gains are **NEEDS-LB** (human submits; never us).
- **Decision rule:** {keep iff … ; e.g. "mini-official beats the incumbent by ≥ the fold-to-fold noise
  AND golden-12 confirms > incumbent; else reject"}.

## Results — AUTO-FILLED by `baseline/exp_journal.py` (do not hand-edit between the markers)
<!-- AUTOFILL:{ID}:START -->
_(no scored runs yet — run `python baseline/exp_journal.py fill --id {ID}` after scoring)_
<!-- AUTOFILL:{ID}:END -->

## Verdict — researcher, AFTER results
{accept / reject vs the pre-registered decision rule · observed effect size · why · next step}
