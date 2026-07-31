# Complete Training System — Full Learning Guide

**Status:** Production-ready, 130+ configs, fully reproducible  
**Last Updated:** 2026-07-07  
**Scope:** Everything you need to train, tune, and replicate the biohub model

---

## 📚 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture (cellmot)](#architecture-cellmot)
3. [Config System (130 configs)](#config-system-130-configs)
4. [Running Experiments](#running-experiments)
5. [Training Discipline](#training-discipline)
6. [Reproducibility](#reproducibility)
7. [Common Recipes](#common-recipes)

---

## System Overview

### What We Have

**Self-contained training pipeline:**
- ✓ Own trainer (model_scratch/train_v0.py)
- ✓ Own architecture (cellmot package)
- ✓ Own loss functions
- ✓ Own metric (golden-CV)
- ✓ 130+ experiment configs
- ✓ NOT dependent on external weights

**Three entry points:**
1. **model_scratch/train_v0.py** — Research trainer (full control, from-scratch training)
2. **scripts/train_from_config.py** — Config-driven runner (standard path for most runs)
3. **baseline/run_experiments_v*.sh** — Batch runners (multiple configs at scale)

### Why From-Scratch?

From the README (model_scratch/README.md):
> **Success is ALWAYS the full official score, end-to-end** — never a part-by-part proxy.
> Whatever we train, "is this better?" is decided by the **complete official metric on golden-CV**:
> `adjusted edge-Jaccard + 0.1·division-Jaccard`

This means:
- Every checkpoint is selected by **golden-CV**, not training loss
- Early stopping monitors **golden-CV**, not loss
- Comparisons are **end-to-end through the linker**, not detector recall alone
- A component improvement is kept **only if full golden-CV improves**

---

## Architecture (cellmot)

### Package Structure

```
model_scratch/cellmot/
├── backbone.py       (95 lines)   — UNet3D architecture
├── blocks.py        (380 lines)   — Convolution blocks (residual, dilated)
├── heads.py         (60 lines)    — Detection, edge, division head predictors
├── loss.py         (131 lines)    — Loss functions (centernet_focal, balanced BCE)
├── data.py         (363 lines)    — Dataset pipeline, augmentation, sampling
├── gpu_aug.py      (144 lines)    — GPU-side augmentation (contrast, noise, etc.)
├── adapters.py      (92 lines)    — Architecture adapters (torch → MONAI bridges)
└── config.py        (60 lines)    — Config loading and validation
```

### UNet3D Architecture

**From base.yml:**
```yaml
backbone:
  type: unet3d
  impl: scratch              # Our documented UNet3D
  stem_channels: 32
  stem_kernel: [1, 3, 3]     # Keep Z early (L01 §4)
  channels_per_stage: [32, 64, 128, 256]
  blocks_per_stage: 2
  block: residual            # L01 §6 gradient flow
  downsample_strides: [[1, 2, 2], [1, 2, 2], [2, 2, 2]]  # Z early, then isotropic
  padding_mode: reflect      # No dark halo (L01 §3)
  norm: {type: group, num_groups: 8}  # GroupNorm for batch 1-2 (L01 §7)
  activation: gelu           # Transformer-ready (L01 §7)
  dilated_bottleneck: {enabled: true, dilation: 2}  # +RF without res loss
```

**Key Design Decisions:**
- **Z early:** First two stages do XY downsampling only ([1,2,2], [1,2,2])
  - Reason: Z is anisotropic (1.625 µm) vs XY (0.40625 µm) — keep Z full early for detail
  - Then isotropic [2,2,2] in bottleneck (enough context)
- **GroupNorm:** Batch size is 1-2 per GPU, so GroupNorm (num_groups=8) instead of BatchNorm
- **Reflect padding:** Avoids dark borders that confuse the detector
- **Dilated bottleneck:** Increases receptive field (≥49 µm needed for context)
- **GELU activation:** Modern transformer-ready (not just ReLU)

### Detection Head

**Gaussian blob target:**
```yaml
heads:
  detection:
    enabled: true
    out_channels: 1
    target: gaussian_blob      # Dense heatmap, not points
    blob_sigma_um: 3.0         # Blur radius (µm)
    decode: threshold|topk|set # Three decoding modes
```

**Decoding Modes:**
1. **threshold** (v0, brittle)
   - Keep local maxima above a fixed cut
   - Pro: Simple
   - Con: Magic threshold, not learnable, varies by dataset
   
2. **topk** (v2, CenterNet-style, ADOPTED)
   - Keep top-K local maxima (K ≈ estimated node count / time)
   - Pro: Self-calibrating, no magic threshold
   - Con: Requires accurate node count estimate
   
3. **set** (future, DETR-style)
   - Hungarian matching, threshold-free, end-to-end learnable
   - Pro: Metric-aligned (scorer also uses Hungarian)
   - Con: More complex, slower training

**Current:** topk (v2_recall and v3 use this)

### Loss Function

**centernet_focal:**
```yaml
loss:
  detection:
    type: centernet_focal
    lambda_neg: 0.5            # Recall-tilted: half the background penalty
    full_supervision: true     # Single-voxel GT positives (not smooth blobs)
```

**Why class-balanced + recall-tilted?**
- Nuclei are RARE (0.01% of voxels)
- Standard BCE would ignore most negatives
- Focal loss down-weights hard negatives (far from peaks)
- Recall tilt (lambda_neg=0.1 or 0.5) = lower background penalty = higher recall
- **Pilkwang's lesson:** recall-tilted BCE beats precision-focused focal for this problem

---

## Config System (130 configs)

### Config Hierarchy

**Base → Experiment Override:**
```
base.yml (all knobs)
  └─ exp_det_v2_recall.yml (only changes: loss + decode mode)
```

**Structure:**
```yaml
experiment:
  name: det_v2_recall
  notes: "Pilkwang recall recipe: class-balanced recall-tilted BCE + topk decode"

# Only change what's different from base.yml
model:
  heads:
    detection:
      decode: topk
      topk_per_frame: auto

loss:
  detection:
    type: centernet_focal
    lambda_neg: 0.1            # More recall
```

**Philosophy:** One variable per config file, so deltas are isolatable.

### All 130 Configs

#### 📂 Augmentation Ablations (35)
**Location:** `config/aug_ablation/`

Single-augmentation tests:
- `00_no_aug.yml` — Baseline (no augmentation)
- `10_crop_scale.yml` — Random crop + scale
- `20_flip_xy.yml` — Flip in XY plane
- `21_rot90_yx.yml` — 90° rotation
- `30_brightness.yml`, `31_contrast.yml`, `32_gamma.yml` — Color transforms
- `33_bias_field.yml` — Intensity bias (MRI-like artifact)
- `34_blur.yml`, `35_noise.yml` — Degradation
- `loeo_*_*.yml` — Leave-one-embryo-out (LOEO) variants

**Usage:**
```bash
# Test single augmentation
python scripts/train_from_config.py config/aug_ablation/20_flip_xy.yml

# Full ablation suite
for f in config/aug_ablation/*.yml; do
  python scripts/train_from_config.py $f
done
```

#### ⚡ Auto-Generated (15)
**Location:** `config/_auto/`

Fleet orchestrator creates these automatically:
- `auto_flip_xy.yml`, `auto_rot90_yx.yml`, `auto_noise.yml`, etc.
- `auto_mix_flip_xy_bias_field.yml` — Combines winning augs
- `best_inference.yml` — Best inference config

**Why:** Orchestrator sweeps augmentation space and auto-generates promising mixes.

#### 🏆 Public Notebooks (60)
**Location:** `config/exp/public/`

Replicate top LB notebooks:
- `pilkwang.yml` — LB 0.885 reference (unet_transformer + ILP)
- `yusuketogashi.yml` — LB 0.893 baseline
- `beicicc.yml` — Yusuke variants (v10, exp024-034)
- `drkongvis.yml` — Motion tuning (v13-v27)
- `vmerckle.yml` — Linking research
- 40+ more

**Purpose:** Mine public notebooks for parameters (thresholds, motion distances, etc.)

**How to use:**
```bash
# Replicate pilkwang
python scripts/train_from_config.py config/exp/public/pilkwang.yml

# Replicate beicicc exp034 (safe-div-precision)
python scripts/train_from_config.py config/exp/public/beicicc.yml
```

#### 🧠 Detector Variants (5)
**Location:** `model_scratch/config/`

Architecture experiments:
- `exp_det_v0.yml` — Baseline (threshold decode)
- `exp_det_focal.yml` — Focal loss, precision-tilted
- `exp_det_v2_recall.yml` — Recall-tilted (ADOPTED)
- `exp_det_v3_stdfocal.yml` — Standard focal
- `exp_det_v3_scaled.yml` — Architecture tweak

**Example (exp_det_v2_recall.yml):**
```yaml
experiment:
  name: det_v2_recall
  notes: "Pilkwang recall recipe"

loss:
  detection:
    lambda_neg: 0.1            # <- The key change

train:
  epochs: 25
  checkpoint_metric: golden_cv_official  # Not loss!
```

#### 📊 LOEO/LOSO (10)
**Location:** `config/`

Leave-one-embryo-out / Leave-one-stage-out:
- `loeo_detector.yml` — Train on one embryo, eval on other (CV split)
- `loeo_detector_aug.yml` — Same + augmentation
- `loso_detector_aug.yml` — Leave-one-stage-out (density-stratified)

**Why:** Prevents overfitting to the 2 embryos in training

#### 🎯 Main Experiments (4)
- `exp1.yml` — Initial baseline
- `exp2_labels.yml` — Pseudolabeling experiment
- `exp2_student.yml` — Student model distillation
- `config/exp/winning_inference_div.yml` — Best inference config

---

## Running Experiments

### Method 1: Config-Driven Launcher

**Simplest path:**
```bash
python scripts/train_from_config.py config/loeo_detector.yml
```

**What it does:**
1. Loads YAML config
2. Resolves data paths (ROOT / path)
3. Sets environment (MLFLOW_TRACKING_URI, CUDA_VISIBLE_DEVICES)
4. Runs trainer subprocess (inherits env)
5. Logs to MLflow (port 5000)

**With extra flags:**
```bash
python scripts/train_from_config.py config/_auto/auto_flip_xy.yml \
  --epochs 30 --batch-size 4 --patience 10
```

### Method 2: Direct Trainer

**Full control:**
```bash
# Build label cache (one-time)
python model_scratch/train_v0.py --build-cache \
  --subset-per-embryo 15 --train-frames 20

# Train from cache
python model_scratch/train_v0.py \
  --config model_scratch/config/exp_det_v2_recall.yml \
  --epochs 50 --eval-every 2 --patience 5 \
  --out model_scratch/results/det_v2_recall

# Dry-run (GPU-safe, no torch import)
python model_scratch/train_v0.py --dry-run
```

**Key flags:**
- `--dry-run` — Validate schema + paths without GPU
- `--build-cache` — One-time label preparation
- `--hold-embryo 44b6` — Embryo-disjoint CV (train on other, eval on 44b6)
- `--peak-cache` — CenterNet targets (single voxel, not smooth)
- `--swa` — Stochastic weight averaging (generalizes better)
- `--epochs 50` — Override config
- `--eval-every 2` — Checkpoint frequency
- `--patience 5` — Early stopping

### Method 3: Batch Runner

**Scale to multiple configs:**
```bash
#!/bin/bash
# baseline/run_experiments_v2.sh

configs=(
  config/loeo_detector.yml
  config/_auto/auto_flip_xy.yml
  config/aug_ablation/20_flip_xy.yml
)

for cfg in "${configs[@]}"; do
  echo "Running $cfg..."
  python scripts/train_from_config.py "$cfg"
done
```

---

## Training Discipline

### Golden-CV Selection

**NOT loss-based, NOT validation recall, but FULL PIPELINE:**

```python
# scripts/train_from_config.py
checkpoint_metric = cfg["train"]["checkpoint_metric"]  # = "golden_cv_official"

# Every epoch:
if epoch % eval_every == 0:
  predictions = model.predict(golden_cv_datasets)
  edges, divisions = link_and_divide(predictions)
  score = official_metric(edges, divisions)  # The TRUE metric
  
  if score > best_score:
    save_checkpoint()  # Save weights
    best_score = score
```

**Why?**
- Detector might overfit to part-by-part metrics
- Official metric catches distribution shift better
- CV↔LB correlation proven (golden-12 correlates with Kaggle LB)

### Early Stopping

**Monitor golden-CV, not loss:**
```yaml
train:
  patience: 5                    # Stop after 5 epochs no improvement
  checkpoint_metric: golden_cv_official  # Watch this
```

**Example:**
- Epoch 1: golden-CV = 0.850 (save)
- Epoch 2: golden-CV = 0.852 (save, best so far)
- Epoch 3: golden-CV = 0.851 (no save)
- Epochs 4-5: golden-CV = 0.850, 0.849 (no save)
- Epoch 6: Stop! Patience exhausted.

### Reproducibility

**Deterministic seeding:**
```yaml
experiment:
  seed: 42  # Fixed across all runs

# Then:
torch.manual_seed(seed)
np.random.seed(seed)
```

**Why 42?** Arbitrary convention; the point is it's **fixed**.

---

## Common Recipes

### Recipe 1: Quick Test (10 min)

**Validate pipeline without full training:**
```bash
python model_scratch/train_v0.py --dry-run
# ✓ Checks YAML, data paths, model build, no GPU
```

### Recipe 2: Single Augmentation (2 hours)

**Test one augmentation on LOEO CV:**
```bash
python scripts/train_from_config.py config/aug_ablation/20_flip_xy.yml
# ✓ Trains on embryo A, evals on B (golden-CV)
# ✓ Golden-CV logged to docs/experiment_ledger.jsonl
```

### Recipe 3: Replicate Public Notebook (4 hours)

**Adopt pilkwang's parameters:**
```bash
python scripts/train_from_config.py config/exp/public/pilkwang.yml
# ✓ Uses their det_threshold, motion thresholds, aug, etc.
# ✓ Our UNet + their hyperparams = comparable baseline
```

### Recipe 4: Ablation Study (24 hours)

**Sweep all 35 augmentations in parallel (fleet):**
```bash
# Fleet orchestrator does this automatically:
# - Config generator creates 35 configs
# - 5 workers process them
# - Results logged to experiment_ledger.jsonl
# - Journal shows which augs won
```

### Recipe 5: Detector Tuning (6 hours)

**Compare three loss functions (v0 vs focal vs recall):**
```bash
# Parallel runs (can use 3 GPUs or time-share):
python model_scratch/train_v0.py --config model_scratch/config/exp_det_v0.yml &
python model_scratch/train_v0.py --config model_scratch/config/exp_det_focal.yml &
python model_scratch/train_v0.py --config model_scratch/config/exp_det_v2_recall.yml &
wait

# Compare golden-CVs:
grep "det_v0\|det_focal\|det_v2" docs/experiment_ledger.jsonl | jq '.cv'
```

### Recipe 6: Embryo-Disjoint CV (4 hours)

**Prove no leakage:**
```bash
python model_scratch/train_v0.py \
  --config model_scratch/config/exp_det_v2_recall.yml \
  --hold-embryo 44b6 \
  --out model_scratch/results/det_v2_hold_44b6
# ✓ Trains on 6bba only, evals on 44b6 (strict split)
# ✓ Golden-CV should still be strong (no leak)
```

---

## Reproducibility Checklist

Before shipping a result:

- ✓ **Seed fixed** (experiment.seed in YAML)
- ✓ **Data path resolved** (absolute or relative to ROOT)
- ✓ **CV split used** (embryo-disjoint, not random)
- ✓ **Metric is golden-CV** (not loss, not recall alone)
- ✓ **Config versioned** (in config/*.yml, not hardcoded)
- ✓ **Weights saved** (checkpoint on golden-CV, not loss)
- ✓ **Dry-run passes** (no torch import, schema valid)
- ✓ **Result logged** (to experiment_ledger.jsonl, MLflow)
- ✓ **Run documented** (config notes explain changes from base)

---

## Complete File Reference

```
TRAINING CODE:
  model_scratch/train_v0.py         (349 lines) — Main trainer
  scripts/train_from_config.py      (3.9K)     — Config launcher
  model_scratch/cellmot/            (88K pkg)  — Architecture + data

CONFIGS (130 total):
  config/aug_ablation/              (35)       — Single augmentations
  config/_auto/                     (15)       — Auto-generated
  config/exp/public/                (60)       — Public notebooks
  model_scratch/config/exp_det*.yml (5)        — Detector variants
  config/loeo*.yml, loso*.yml       (10)       — Embryo-disjoint CV
  config/exp/*.yml                  (4)        — Main experiments

INFRASTRUCTURE:
  src/metric.py                     (175 lines) — Official golden-CV scorer
  src/detect.py                     (125 lines) — Detector + decoding
  src/link.py                       (138 lines) — Linker (rule-based ILP)
  src/pipeline.py                   (50 lines)  — End-to-end runner
  
DOCUMENTATION:
  model_scratch/README.md                      — From-scratch philosophy
  model_scratch/config/base.yml                — All knobs documented
  model_scratch/lessons/                       — Research notes per lesson
```

---

## Next Steps

1. **Run a quick test:**
   ```bash
   python model_scratch/train_v0.py --dry-run
   ```

2. **Train on LOEO:**
   ```bash
   python scripts/train_from_config.py config/loeo_detector.yml
   ```

3. **Check results:**
   ```bash
   tail docs/experiment_ledger.jsonl | jq '.cv'
   ```

4. **Replicate pilkwang:**
   ```bash
   python scripts/train_from_config.py config/exp/public/pilkwang.yml
   ```

---

**Status:** ✅ Complete, reproducible, production-ready  
**Quality:** Golden-CV metric end-to-end, embryo-disjoint CV, no leakage  
**Configs:** 130+ ready to run  
**Independence:** No external weight dependency — fully replicable from scratch  

