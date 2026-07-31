# CV-Validation Gate — Embryo-Disjoint Density CV (`splits_loeo_density.json`)

**Conclusion (wiring/EDA phase, CPU-only — no GPU used):**
The density CV is **structurally valid** (embryo-disjoint LOEO, 2 folds, zero overlap), the
predict→pilk_post→official pipeline is **wired for both public models on both folds**, and the
**minimal CPU dry-run is GREEN**. Two things the human/leader must note before the real predict:
1. **golden-12 predictions do NOT count** — golden-12 and the density-CV test sets overlap in only
   **2 of 15** datasets, so essentially every dataset needs a fresh predict.
2. **pilkwang weights must be the (1,4,4) genuine public model**, not the local (1,2,2) retrain that
   currently sits at `research/official_repo/weights/unet_transformer/split_0` (see §3).

The actual GPU predict+score (2 pipelines × 2 folds) is the **trainer's** lane. This doc hands over
the exact runnable commands.

---

## 1. Embryo-disjointness — VERIFIED (both folds)

`learning/ensemble_work/finetune/splits_loeo_density.json` is a 2-element list (leave-one-embryo-out).
Extracting `embryo = stem.split('_')[0]`:

| Fold | Train embryo (n) | Test embryo (n) | Shared embryo? |
|------|------------------|-----------------|----------------|
| 0    | `6bba` (8)       | `44b6` (8)      | **none** ✓ |
| 1    | `44b6` (10)      | `6bba` (7)      | **none** ✓ |

Every fold holds out a whole embryo — no embryo appears on both sides. This matches the Kaggle
test contract (train/test share no embryo). Note `6bba_05db0fb1` appears in fold0-**train** and
fold1-**test** — that is fine: each fold is independently disjoint; datasets are only reused across
*different* folds.

Test dataset lists (the 15 datasets the gate must score):
- **fold0 test (44b6, 8):** `0b24845f, 3bb3690f, 587a1e22, 66f9292d, 8f9ecab4, a2bb48bb, c8e2a523, d754aa59`
- **fold1 test (6bba, 7):** `05db0fb1, 283bf9f1, 74686d6a, 7d3058ae, b329af44, ebdf3b34, ebff6e76`

## 2. Reuse vs. gap — golden-12 predictions are NOT reusable here

Both detectors apply **fixed, fold-independent weights per dataset**, so any existing per-dataset
score whose dataset is in a density fold's test set *could* be reused. But every existing prediction
set (canqiang_scores.csv, `predictions/seshu/*`) covers only the **golden-12** datasets, and:

- **golden-12 ∩ density-CV-test = 2 datasets** (`44b6_0b24845f`, `6bba_05db0fb1`) out of 15.
- `learning/ensemble_work/canqiang_scores.csv` covers the 12 golden datasets → **only 2** are in the
  density test sets; **13 canqiang datasets are a gap**.
- No `predictions/seshu/*` dir covers more than 2/15 density datasets.

⇒ **Reuse is negligible.** Trainer should run a fresh predict for **both pipelines on both folds**
(15 datasets each). golden-12 numbers are the wrong test for this gate.

## 3. ⚠️ pilkwang weights — use the (1,4,4) GENUINE model, not the (1,2,2) local retrain

Two `unet_transformer/split_0` weight dirs exist and are **different models**:

| Path | `config.json downsample` | mtime | What it is |
|------|--------------------------|-------|-----------|
| `research/pilkwang_support_pack/weights/unet_transformer/split_0/` | **[1,4,4]** | Jul-03 02:47 | **Genuine pilkwang LB-0.890 detector** (pristine) |
| `research/official_repo/weights/unet_transformer/split_0/` | **[1,2,2]** | Jul-05 04:34 | **Local (1,2,2) high-res retrain** (our v2 lever) |

The gate tests whether the CV ranks **pilkwang's public pipeline** above canqiang's. Using the (1,2,2)
local retrain would test *our* model, not pilkwang's, and invalidate the LB-faithfulness check.
**Recommended weights = `research/pilkwang_support_pack/weights/unet_transformer/split_0/edge_predictor_best.pth`**
(the dry-run confirmed this dir's `config.json` = `[1,4,4]` and the model reconstructs). `load_model`
reads `config.json` from the weights' own directory, so the resolution travels with `--weights`.

## 4. Exact predict→pilk_post→official commands (hand to trainer)

`PY=/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/research/cellmot_venv/bin/python`
`ROOT=/home/seshu/kaggle/2026/biohub-cell-tracking-during-development`

### pilkwang (unet_transformer) — predict then score, per fold
```bash
cd "$ROOT/research/official_repo"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true            # ALWAYS-ON system metrics (recipe)
for F in 0 1; do
  PYTHONPATH="src:scripts" "$PY" scripts/predict_unet_transformer.py \
    --method pilk_loeodens \
    --data-dir "$ROOT/input/biohub-cell-tracking-during-development/train" \
    --splits  "$ROOT/learning/ensemble_work/finetune/splits_loeo_density.json" \
    --split   "$F" \
    --weights "$ROOT/research/pilkwang_support_pack/weights/unet_transformer/split_0/edge_predictor_best.pth"
  # -> writes geffs to research/official_repo/predictions/seshu/pilk_loeodens/split_$F/
  "$PY" "$ROOT/tools/researchpapers/baseline/score_v1.py" \
    --geff-dir "$ROOT/research/official_repo/predictions/seshu/pilk_loeodens/split_$F" \
    --split-file "$ROOT/learning/ensemble_work/finetune/splits_loeo_density.json" \
    --fold "$F" --run-name "pilk_loeodens_f$F" --exp-id EXP-CVGATE
done
```
`score_v1.py` applies the full pilkwang post-proc + official metric, prints `official_score`
(adj_edge + 0.1·div), writes an `output/scores/pilk_loeodens_f$F.json` sidecar, and logs MLflow
(`config_file` param, `config_path`/`fidelity=mini`/`eval_split=splits_loeo_density` tags, system metrics).

> `--method pilk_loeodens` is namespaced so predict does **not** clobber existing
> `predictions/seshu/unet_transformer/split_0`.

### canqiang (DeepCenterUNet3D) — self-contained predict+score, per fold
canqiang's public predict path is a DIFFERENT architecture; `predict_unet_transformer.py` does NOT
cover it. Use the gate runner (adapted from `learning/ensemble_work/run_canqiang.py`, which is
hard-coded to golden-12):
```bash
cd "$ROOT/tools/researchpapers"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
for F in 0 1; do
  "$PY" baseline/run_canqiang_loeodens.py --fold "$F" --exp-id EXP-CVGATE
done
# -> official_score per fold, output/scores/canqiang_loeodens_f$F.json sidecar, MLflow run
```

### The gate decision
Aggregate each pipeline's `official_score` across the two folds (or report per-fold + mean). **The CV
PASSES iff pilkwang > canqiang.** (golden-12 got this INVERTED: canqiang 0.903 > pilkwang 0.870.)
If the density CV does **not** rank pilkwang > canqiang, **STOP** — do not screen ideas on it.

## 5. Minimal dry-run — GREEN (CPU, no GPU)

`baseline/dryrun_gate_loeodens.py` validates both pipelines' wiring without any GPU inference:
splits load, both folds' `.zarr`+`.geff` resolve, pilkwang weights+config load and the model
reconstructs on CPU (window=2, downsample=(1,4,4), 2.08M params), the namespaced output dir is
writable, and both canqiang folds pass their own `--dry-run` (DeepCenterUNet3D loads on CPU,
datasets resolve). Result: **DRY-RUN GREEN — wiring validated for both pipelines.**

```
$PY baseline/dryrun_gate_loeodens.py --folds 0,1
```

## Files
- `eda/loeo_density_cv_gate.md` — this doc
- `baseline/run_canqiang_loeodens.py` — canqiang density-CV predict+score runner (--fold/--dry-run/MLflow)
- `baseline/dryrun_gate_loeodens.py` — CPU dry-run harness for both pipelines
- scorer (pilkwang path): `baseline/score_v1.py --geff-dir … --split-file … --fold …` (already present)
