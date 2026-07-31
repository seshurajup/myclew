# CV-GATE Result — density LOEO CV (splits_loeo_density.json)

**Conclusion (one line): GATE FAILS.** The embryo-disjoint density LOEO CV ranks **canqiang > pilkwang** on *both* folds and *both* aggregations — it does **not** rank pilkwang > canqiang, so it **inverts the leaderboard** exactly like golden-12 did. This CV is **NOT LB-faithful**. Per the gate rule, we **STOP**: no idea-bracket experiments on an unvalidated CV.

- **task_id:** CVGATE  |  **train_task_id:** `train-0c047082fd` (train_service :7799, status=`succeeded`)
- **script:** `baseline/run_cvgate_loeodens.sh` (trainer wrapper around researcher's exact 04:55 commands; no researcher code modified)
- **workdir:** `tools/researchpapers`  |  **log:** `baseline/train_log.txt`
- **weights:** GENUINE pilkwang **(1,4,4)** support_pack `research/pilkwang_support_pack/weights/unet_transformer/split_0/edge_predictor_best.pth` (Jul-3 pristine) — NOT the official_repo (1,2,2) retrain
- **metric:** official_score = adj_edge_jaccard + 0.1·div_jaccard (fresh predict → post → src.metric); MLflow exp 16, system-metrics ON, fidelity=mini / eval_split=splits_loeo_density
- **fresh predict** on both folds for both pipelines (golden-12 preds excluded: only 2/15 dataset overlap)

## Key result table

| Pipeline | LB ref | fold0 (44b6, n=8) | fold1 (6bba, n=7) | macro-mean | embryo-weighted | short judgment |
|---|---|---|---|---|---|---|
| pilkwang (1,4,4) | 0.890 | 0.7882 | 0.7690 | 0.7786 | 0.7792 | lower on BOTH folds |
| canqiang | 0.866 | **0.7973** | **0.7879** | **0.7926** | **0.7929** | higher on BOTH folds |
| Δ (pilk − canq) | +0.024 | −0.0091 | −0.0189 | −0.0140 | −0.0137 | wrong sign vs LB |

Figure: `docs/cvgate_loeo_density_pilkwang_vs_canqiang.png` (grouped bars, per-fold + aggregate). Takeaway: canqiang's bar is above pilkwang's in every group — the density-CV ordering is the reverse of the LB ordering.

## Main-line judgment

- **LB truth:** pilkwang (LB 0.890) > canqiang (LB 0.866).
- **golden-12 CV:** INVERTED it (canqiang 0.903 > pilkwang 0.870) → known LB-unfaithful.
- **density LOEO CV (this run):** canqiang 0.793 > pilkwang 0.779 → **also inverted** (Δ≈−0.014, consistent sign across both folds and both aggregations — not a noise flip).
- ⇒ The density LOEO split does **not** fix the ranking pathology; it reproduces it. Structural precondition (embryo-disjoint LOEO, verified by researcher) was met, but the metric ordering still disagrees with the LB. **Gate = FAIL.**

## Probe / sidecar analysis

| Object | node_recall | count_ratio | div_tp | note |
|---|---|---|---|---|
| pilkwang f0 | 0.989 | 1.353 | 0 | high recall, over-counts nodes ~1.35× |
| pilkwang f1 | 0.991 | 1.322 | 0 | same pattern, over-counts ~1.32× |
| canqiang f0 | n/a | 0.724 | 0 | under-counts on 44b6 yet still scores higher |
| canqiang f1 | n/a | 1.051 | 0 | near-unity count, best single cell |

- `div_tp_total = 0` for **both pipelines, all four runs** → no true-positive division events; the +0.1·div_jaccard term is 0 on both sides, so the ranking is driven purely by adj_edge_jaccard (fair, apples-to-apples).
- **pilkwang over-detects by ~32–35%** (count_ratio 1.32 / 1.35) with near-perfect node recall (0.99); canqiang's counts are better calibrated (0.72 / 1.05). The adj-edge metric here rewards canqiang's calibration over pilkwang's recall+over-detection, opposite to the LB — a likely mechanism for the inversion.

## Next-step suggestions (for leader — design only, gate is FAILED)

1. **Do NOT unlock idea brackets on this CV** — it is not LB-faithful (rule triggered: STOP).
2. Investigate *why* both CVs invert: candidate causes — (a) the official mini-metric (adj_edge_J + 0.1·div_J on this dataset subset) is not the LB metric, (b) count-calibration vs recall trade-off is scored differently than the LB, (c) density-CV test embryos are not LB-representative.
3. Consider a CV whose *metric/aggregation* matches the LB more closely, or validate against a held-out set with a known LB delta, before trusting any CV for idea screening.

## Evidence

- Sidecars: `output/scores/pilk_loeodens_f0.json`, `pilk_loeodens_f1.json`, `canqiang_loeodens_f0.json`, `canqiang_loeodens_f1.json`
- Predictions: `research/official_repo/predictions/seshu/pilk_loeodens/split_{0,1}` (pilkwang); canqiang via `baseline/run_canqiang_loeodens.py`
- MLflow: experiment 16, runs `pilk_loeodens_f{0,1}`, `canqiang_loeodens_f{0,1}`
- Split: `learning/ensemble_work/finetune/splits_loeo_density.json` (2-fold LOEO, embryo-disjoint: f0 6bba→44b6, f1 44b6→6bba)
