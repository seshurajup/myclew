# Complete coverage map — every part of our system, as a lesson

The ordered lessons below cover **everything** — data, the full architecture, all training code,
augmentation, the metric, post-processing, and our whole research journey. Each lesson pairs a
**working `.py`** (real code) with a **`.learning`** file; its **outputs come from running the code**
(shapes via `shape_trace.py`, tables/plots as real artifacts). Nothing hand-faked.

Status: ✅ done · 🔧 in the completion pass (per-line comments / building) · ▫ planned

## Part A — Data & timing  (understand the data first)
- ✅ da01 — How many cells per frame? (real table: 44b6 vs 6bba; annotated vs true estN)
- ✅ da02 — Following time / the timeline (real heatmap; cells grow; start-timing)
- ✅ da03 — Density → developmental stage (real stage plot; S0–S4; embryo↔stage confound)

## Part B — The detector architecture  (temporal_unet.py, line by line)
- 🔧 pt01 — nn.Module & the conv block (Conv3d/BatchNorm3d/ReLU/Sequential) + diagram
- 🔧 pt02 — The U-Net shape (MaxPool3d/Upsample/torch.cat/ModuleList) + U-Net diagram
- 🔧 pt03 — Temporal attention (MultiheadAttention/LayerNorm; attend across the frame window)
- 🔧 pt04 — Detector forward → peaks (head Conv3d(32,1,1) → heatmap → NMS → coords) + metric diagram

## Part C — Linking & loss
- 🔧 pt05 — The edge model (simple_node_transformer.py: Linear/GELU/Dropout/cross-attention) + graph diagram
- 🔧 pt06 — The loss & the division weight (compute_loss; the `weight[div_rows]=1.0` lever) + division diagram

## Part D — Training  (train_unet_transformer.py)
- 🔧 pt07 — Dataset & DataLoader (FrameWindowDataset W=2, downsample→isotropic, normalise, workers)
- 🔧 pt08 — The training loop (AdamW, backward/step/grad-clip, checkpoint, seeds, reproducibility)

## Part E — Augmentation  (augmentations.py — why each, shown on real frames)
- ✅ aug01 — Flips (real flip_augment + real before/after on a frame; the Z train-vs-TTA twist)
- ✅ aug02 — Brightness jitter (real before/after; justified by the 7→1800 intensity spread)

## Part F — Scoring & post-processing
- 🔧 me01 — The official metric (adj edge-Jaccard + 0.1·division, 7 µm match, pred_valid) + match diagram
- 🔧 pp01 — Post-processing (motion-relink, gap-close, safe-divisions, line-fit smoothing)

## Part G — Our research journey  (what we actually did this session)
- 🔧 rs01 — Reproducing pilkwang 0.885 locally (golden-12 = leak-free split_0; 0.870 ↔ 0.885)
- 🔧 rs02 — Why local CV misleads (sparse labels; canqiang 0.903-local / 0.866-LB inversion)
- 🔧 rs03 — The division lever (oracle +0.049; geometry/edge-prob/image all 0 TP; why)
- 🔧 rs04 — Fine-tuning for divisions (smoking gun: div weight 1.0; up-weight + div-checkpoint + oversample)
- 🔧 rs05 — lucifer's 0.888 = gap2 (no training; 2-frame bridge → +node recall)
- 🔧 rs06 — Why the ensemble is hard (72% redundant; recall maxed; heatmap-average only safe fusion)
- 🔧 rs07 — Our next approach ((1,2,2) higher-res detector + division-aware head; validate → submit)

## Cross-cutting rules (every lesson)
1. Real code only, FULL code (no `...`), **inline `# comment` on every line**.
2. Outputs come from **running the working code** (no hallucination) — notebook-style.
3. Every concept has a concrete example from THIS competition (real numbers).
4. Four threads: [PyTorch] · [Data] · [Craft] (bestfitting) · [Domain] (cell-biology/microscopy).
5. Diagrams are ours (matplotlib, real shapes) — never copyrighted figures.
6. Light code theme; note | code | shapes 3-panel; 20% / 65% / 15% widths.
