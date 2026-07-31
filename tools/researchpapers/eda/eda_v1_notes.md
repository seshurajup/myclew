# EDA v1 — Biohub Cell-Tracking During Development

Researcher EDA pass grounding the `baseline_v1` design. All numbers are from the recipe docs,
the official metric definition, and the **already-recorded** golden-12 reproduction under
`learning/ensemble_work/` (no fabrication — see source files cited inline).

## 1. Task & data structure
- **Goal:** predict a 3-D+time tracking graph of every cell nucleus in a developing zebrafish
  embryo (light-sheet). Nodes = cells per frame; edges = same cell across frames (+ divisions).
- **Data root:** `/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/input/biohub-cell-tracking-during-development/train/`
  — 199 embryos × 2 files = 398 (`{stem}.zarr` image + `{stem}.geff` GT graph).
- **Two groups:** `44b6` (71 embryos, biologically late / dense) and `6bba` (128, early / sparse).
- **Image shape:** `(T, Z, Y, X)` with `Z=64, Y=256, X=256`.
- **Voxel size µm `(z,y,x) = (1.625, 0.40625, 0.40625)`** → a `(1,4,4)` downsample is isotropic
  (0.40625×4 ≈ 1.625). A `(1,2,2)` downsample is **anisotropic-but-finer in XY** (≈0.8125 µm XY
  vs 1.625 µm Z) → better localization, which is the whole point of the (1,2,2) lever.
- **GEFF node** = one *annotated* cell at one frame (props `t,z,y,x`); edges `source_id→target_id`.

## 2. Labels are SPARSE — the #1 CV trap
- Only **~4%** of real cells are annotated. True per-embryo count = geff attr
  `estimated_number_of_nodes` (estN). True density ranges **38..1015 cells/frame**.
- Consequence (IMPROVE_PLAYBOOK Rule 1): golden-12 **over-credits density-adding changes**
  (extra predictions land among unlabeled GT → scored "ignored", not false-positive).
  - Density-**preserving** changes (linking, post-proc, topology) → **trust golden-CV**.
  - Density-**changing** changes (new detector, threshold, fusion) → golden-CV is **blind**;
    judge with a **recall proxy + predicted-count-vs-density-cap**, confirm on **LB**.
- A finer-resolution detector (v1) changes predicted density → treat its detection gains as
  **needs-LB**, but its **edge_precision / adj_edge** gains are partly CV-judgeable.

## 3. Evaluation metric & CV
- **Official = adjusted edge-Jaccard + 0.1 · division-Jaccard.** Node match = centroid distance
  within **7 µm**. Metric code: `research/official_repo/scripts/evaluate.py`.
- **golden-12** = pilkwang's leak-free `split_0` held-out fold (12 test embryos, 0 train),
  `learning/ensemble_work/finetune/splits_ft.json` / `golden12_splits.json`.
- golden-12 test set (6× `44b6`, 6× `6bba`) spans the full density range (see table below).

## 4. Reproduced baseline (Rule 0 — CONFIRMED, no recompute)
Source: `learning/ensemble_work/score_pilkwang.log` + `pilkwang_full_scores.csv`.

| pipeline stage | micro adjJ (golden-12) | golden_cv() | 44b6 | 6bba |
|---|---|---|---|---|
| RAW det + ILP | 0.8527 | 0.8789 | 0.8096 | 0.8635 |
| **FULL post-proc** | **0.8708** | **0.8940** | 0.8467 | 0.8768 |

This matches the recipe's stated baseline **0.8700 (↔ LB 0.885)**. Per-embryo GT node counts
(`t_true`) confirm the wide density spread the metric must handle:

| embryo | group | t_true (GT nodes) | full adjJ |
|---|---|---|---|
| 6bba_062c8d37 | 6bba | 6,030 | 0.965 |
| 6bba_085bf656 | 6bba | 8,463 | 0.963 |
| 44b6_0c582fdc | 44b6 | 27,958 | 0.901 |
| 44b6_12dfb391 | 44b6 | 58,672 | 0.847 |
| 6bba_05db0fb1 | 6bba | 69,800 | 0.801 |
| 6bba_07e24132 | 6bba | 21,485 | **0.622** ← worst |

**Division term is dead without training:** `div_tp = 0` on **all 12** embryos in both raw and
full CSVs. The +0.1·div-Jaccard term contributes ~0 today → the whole current score is essentially
the adj edge-Jaccard. Divisions only become worth chasing once localization is finer (recipe).

## 5. Why (1,2,2) is the lever (localization, not density)
- pilkwang trains his detector at `(1,4,4)` (isotropic ≈1.625 µm). Centroids can only be as precise
  as the grid. The 7 µm match tolerance means coarse localization still matches, but **edge
  precision** (which node connects to which across frames) degrades when neighbors blur together
  at high density — visible above: `44b6_12dfb391` / `6bba_05db0fb1` (densest) score lowest.
- `(1,2,2)` halves the XY grid spacing (≈0.81 µm XY) → sharper peaks, fewer merged detections in
  dense regions → higher edge_precision → higher `adj_edge`. **No public notebook does this.**
- Cost: `(1,2,2)` = **4× the voxels** of `(1,4,4)` per frame (Z unchanged, Y & X ×2 each). Memory
  and epoch time rise ~4× → batch size must drop from his default 16 (see design doc §memory).

## 6. Augmentation state (finding)
- pilkwang uses **only brightness + 8-way flip** (`augmentations.py`).
- The aug-ablation the recipe cites as "running" is **stalled/dead**: `config/aug_ablation/ablation.log`
  contains only a `TRAINING base` header, `driver.log` is empty. No `RESULTS.txt` exists. → we
  cannot yet trust any "which augs help" claim; if augs enter v1 they must be A/B'd on held-out recall.

## 7. Submission / hard rules
- **DO NOT SUBMIT TO KAGGLE.** Prepare submission-ready artifacts and report; human submits manually.
- Log every run to local **MLflow** (`http://localhost:5000`, experiment `kaggle-biohub-cell-tracking`),
  in addition to the researchpapers train board.

## 8. Open decision (routed to leader)
Two training stacks exist. baseline_v1 must pick one:
- **A (recommended):** pilkwang `research/pilkwang_support_pack/repo/scripts/train_unet_transformer.py`
  at `--downsample 1,2,2` — the proven 0.8700 pipeline, strided-zarr, **no new cache required**.
- **B:** `model_scratch/cellmot/` (fast `.npz` cache + gpu_aug + `config/*.yml`) — richer augs but a
  less-mature from-scratch detector; would need a `ds1x2x2` cache build (~40 GB).
Recipe move #2 says wire (1,2,2) *from pilkwang's script* → default is **A**.
