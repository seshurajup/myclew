# Screening mini-set + MINI-OFFICIAL gate (methodology validity)

Authoritative spec for the fast-experiment screening substrate. Data analysis: `eda/screen_miniset_analysis.md`.
CPU-only work (GPU on hold). **This doc gates the whole successive-halving methodology.**

## 1. The matched mini-set — `splits_screen_matched.json`
Built to MIRROR the golden-12 / Kaggle CV test so a mini screen predicts golden-12/LB.
- **Leak-free:** excludes all 12 golden-12 embryos; val0/val1/train ∩ golden-12 = 0; embryo-disjoint ⇒
  **zero shared annotated cells**.
- **Group-matched:** each val fold = 6×44b6 / 6×6bba (golden-12 is 6/6).
- **Density-matched:** val density quantiles ≈ golden-12 (q50 303/314 vs 304; q75 595/606 vs 604; full
  48–820 spread). golden-12 is DENSER than the pool (med 304 vs 182), so a pool-representative set would
  mis-predict — this set is built to match, not to represent the pool.
- 2 folds (variance), train=24 density-spanning (38–1015).

**Supersedes** `splits_screen_mini.json`, which was **100% leaky** (all 12 golden-12 embryos in its
folds) and single-group per val (44b6-only / 6bba-only). Do not use the old one.

## 2. THE VALIDITY FIX — rank/prune on MINI-OFFICIAL, not acc·recall
**Problem (proven on baseline_v1):** the trainer's per-epoch `best_score = acc·recall` is NEARLY FLAT
across arms (v1_1 0.9557 · v1_2 0.9557 · v1_3 0.9595) while the golden-12 **official** adjJ spans
**0.6086 → 0.8249**. The undertraining/quality signal lives ONLY in the official metric (post-proc
recall 0.70 vs 0.92), not in acc·recall. So a bracket that ranks/prunes on acc·recall is **BLIND** —
it promotes/kills on noise. `successive_halving.learning`'s own ep5 table shows it (v1_2 0.944 vs
v1_3 0.949 — indistinguishable, yet official differs by 0.18).

**Fix (wired):** the driver ranks each rung on the **official metric computed on the matched mini-val**
(`predict → pilk_post → src.metric`), NOT acc·recall. Because the mini-val mirrors golden-12, this
mini-official tracks golden-12 cheaply (12 embryos). Implementation:
- `baseline/successive_halving.py` — `run_config` trains (namespaced ckpt, **NO acc·recall in-run
  prune**), then `mini_official()` predicts on the rung's val + scores official; rungs keep top-half on
  mini-official with a **conservative bar = worst survivor's mini-official**.
- `baseline/score_v1.py --split-file <matched> --fold N` — scores official on the mini-val (any embryo
  set), prints `MINI_OFFICIAL_SCORE=…`, tags `fidelity=mini`, `eval_split=splits_screen_matched`.
- Legacy acc·recall path is **removed** from ranking (only kept as a debug print, never a decision).

## 3. GO-GATE (requires GPU — run first when the hold lifts)
Before trusting ANY screen, prove mini-official tracks golden-12 on KNOWN arms:
```
# predict v1's 3 checkpoints on the matched mini-val + score mini-official
for m in baseline_v1_v1_1_ctrl_1x4x4 baseline_v1_v1_2_hr_baseaug baseline_v1_v1_3_hr_richaug; do
  # predict on splits_screen_matched fold 0, then:
  score_v1.py --method $m --split-file learning/ensemble_work/finetune/splits_screen_matched.json --fold 0
done
```
**PASS criterion:** mini-official RANKS the arms like golden-12 did (v1_1 0.8249 > v1_3 0.7919 > v1_2
0.6086) — i.e. rank-agreement (Spearman) high AND the winner preserved. If mini-official does NOT
reproduce that ordering, the mini-set/gate is not faithful → do NOT screen on it; raise mini val size
or revisit the match. **This validation is GPU-gated and currently blocked by the training hold.**

## 4. What the bracket may / may NOT screen
- ✅ Screen ideas whose benefit shows EARLY and moves mini-official: aug variants, LR, det-loss weight,
  det-threshold/gating, window/pool — early perf is a fair proxy.
- ❌ Do NOT bracket-screen RESOLUTION ((1,2,2) vs (1,4,4)): the finer detector converges LATE (v1:
  recall 0.70→0.92 by ~ep15); a 5-epoch rung under-serves it → run resolution as a FULL converged A/B
  (that is EXP-001 / baseline_v2), golden-12 judged.

## 5. Runner / files
- bracket runner (train_service-ready, FD-safe, no MLFLOW_RUN_ID): `baseline/run_screen.sh <bracket.yml> [--fold N] [--budget-hours H] [--dry-run]`.
- driver: `baseline/successive_halving.py` (mini-official gate). Dry-run GREEN (CPU).
- split: `learning/ensemble_work/finetune/splits_screen_matched.json`.
