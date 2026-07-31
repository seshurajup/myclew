# Ensemble Submission Notebook — Build Plan (no training)

**Goal:** beat pilkwang **0.885** using two *pretrained* public models + rule-based
post-processing. No training. Judge = **Kaggle LB** (not our golden CV — it inverts here).

**Core idea:** ensemble the two detectors at the **HEATMAP** level (consensus, not point-union),
then run pilkwang's exact learned-graph + rule-based post-processing on top.

---

## Inputs (Kaggle datasets attached to the notebook — all already downloaded locally)
| input | what | role |
|---|---|---|
| `pilkwang/biohub-tracking-support-pack-50ep-v1` | TemporalUNet3D detector + SimpleNodeTransformer edge model + repo | model A + linker + post-proc |
| `canqiang/zebracelltrace-ff-checkpoint` (`best.pt`) | DeepCenterUNet3D detector | model B |
| competition test data | hidden test zarrs | inference |
| offline dep wheels (in support pack) | tracksdata, pyscipopt, geff… | no-internet install |

---

## Pipeline (4 stages)

### Stage 1 — DETECTION = heatmap ensemble (the "2 models, no training" step)
Per test dataset, per frame:
1. **Model A (pilkwang)**: TemporalUNet3D → sigmoid heatmap `Hₐ`, downsample (1,4,4), quantile-norm, 4-way flip TTA (X/Y/XY, no Z).
2. **Model B (canqiang)**: DeepCenterUNet3D → sigmoid heatmap `H_b`, its own norm (50/99.5 pct), pool 4.
3. **Align** both heatmaps to a common (1,4,4) voxel grid (resample B if needed).
4. **Ensemble (consensus):** `H = wₐ·Hₐ + w_b·H_b`  (start `wₐ=w_b=0.5`).
   - Averaging *sharpens* peaks where both agree (better 7µm localization → higher edge-J)
   - and *suppresses* peaks only one model fires (kills FP, unlike point-union).
5. **Peak-find** on `H`: 3D max-pool NMS (kernel from 5µm), threshold `τ`.
   - Knobs to try across submissions: `wₐ/w_b` (0.5/0.5, 0.6/0.4, 0.7/0.3), `τ`.

### Stage 2 — LINKING = pilkwang's learned graph (unchanged)
6. Edge scoring: pilkwang **SimpleNodeTransformer** on the ensembled detections (softmax edges, thr 0.5).
7. Solve: **ILP** (tracksdata ILPSolver: edge_w=−1·prob, appearance/disappearance 0.1, division 1.0).

### Stage 3 — RULE-BASED POST-PROCESSING (pilkwang chain, verified optimal in our sweep)
Applied in order (all pilkwang defaults — our sweep proved these are at the local optimum, so keep them):
8. drop edges > **14µm**
9. **motion-relink** (2-pass Hungarian, tight 6 / relaxed 10µm, const-velocity λ=0.5)
10. single-parent repair
11. **1-frame gap-close** (12µm, synthetic midpoint + intensity refine ≤3.2µm)
12. **safe divisions** (parent≤4.7 / sister≤7.2 / existing-child≤7.8µm, capped) — *net-positive, keep*
13. **line-fit smoothing** (w=0.8, ±2, deg 1)
   - gap2 recovery OFF (default). short-track filter OFF.
   - *These are the "rule" layer — the ensemble is the only thing we changed vs the 0.885 base.*

### Stage 4 — OUTPUT
14. Write `submission.csv`: node rows (`row_type=node`, t,z,y,x rounded) + edge rows (source_id,target_id).

---

## Where the ensemble + rules meet (the "post-processing way")
```
 zarr frame
   ├─ pilkwang UNet ─► Hₐ ─┐
   │                        ├─ AVERAGE ─► peak-find ─► detections
   └─ canqiang UNet ─► H_b ─┘                              │
                                                           ▼
                          pilkwang edge-transformer + ILP  (learned graph)
                                                           │
                                                           ▼
                    RULE POST-PROC: relink → gap-close → safe-div → smooth
                                                           │
                                                           ▼
                                                    submission.csv
```
Only **Stage 1 (heatmap average)** differs from pilkwang 0.885. Stages 2–4 are its proven pipeline.

---

## Validity checks before submitting (NOT a score gate)
- runs end-to-end on 2–3 test datasets without error
- `submission.csv` well-formed, **rows > 0**, both node+edge rows present
- predicted density plausible (not empty, not exploded)
- offline: no internet calls; deps install from bundled wheels
- GPU: fits Kaggle T4 (both models small); force T4 accelerator

## Submission strategy (LB is the judge)
- **Sub 1:** ensemble `wₐ=w_b=0.5`, τ = pilkwang's 0.99-equivalent on `H`. → read LB vs 0.885.
- If ≥0.885: sweep `wₐ/w_b` and `τ` across the 5 daily slots to climb.
- If <0.885: the heatmap ensemble doesn't help either → 0.885 needs training (documented).

## Risks / open questions
- **Heatmap alignment & scale**: A and B are different architectures; their sigmoid outputs may not be
  directly comparable. Mitigation: per-heatmap min-max/quantile normalize before averaging; or rank-normalize.
- **canqiang resolution**: confirm its effective grid matches (1,4,4) after pool 4; resample if not.
- **Runtime**: 2 detectors × TTA × full test → must fit Kaggle's time limit; cache/skip TTA on B if slow.
- **Can't pre-validate the gain locally** (golden CV inverts) → first real signal is Sub 1's LB.
