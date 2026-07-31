# Learning PyTorch through the Biohub Cell-Tracking Competition

A step-by-step path from **PyTorch basics → this competition's real code**. Every
module is grounded in the actual data/code we have (no toy abstractions), and every
module produces a `.md` lesson you can read + a runnable `.py` you can execute.

**Rule of the curriculum:** one module at a time, basics first. Each ends with a `.md`.
We use the competition's own code as the textbook — pilkwang's pipeline (`temporal_unet.py`,
`simple_node_transformer.py`, `train_unet_transformer.py`, `metrics.py`) "uses every line
of code well", so it's an ideal real-world PyTorch reference.

**CODE RULE (absolute) — every code snippet is from THIS competition, never a toy.**
No abstract/random examples ever: no `torch.tensor([[1,2,3]])`, no `x = torch.randn(2,3)`,
no generic "cat/dog" analogies. Every line of code either **is** the real pipeline code
(`temporal_unet.py`, `simple_node_transformer.py`, `train_unet_transformer.py`) or **operates on
real competition data** (a zarr frame of a named embryo, real cell coordinates, the real metric).
Concepts are explained *through* this domain — a nucleus blob, a division event, the 7 µm match —
not in the abstract.

**CORE PRINCIPLE — every design choice is justified BY THE DATA.** We never just state
"use flip augmentation" or "use this padding". For each choice we run a small test on the
real volumes and *show the number that forces the decision*. Examples we'll prove from data:
- **why downsample `(1,4,4)`** → voxel z=1.625, xy=0.40625; ×4 xy = 1.625 → makes the grid **isotropic**.
- **why flip X/Y but NOT Z** → z is physically anisotropic (light-sheet PSF), fewer z-planes → a z-flip isn't a plausible image.
- **why brightness jitter** → intensity varies embryo-to-embryo/stage-to-stage (we'll plot it).
- **why `padding=1` on 3×3 conv** → to keep the volume the same size through the U-Net (we'll show the shapes).
- **why up-weight the division loss** → divisions are ~8-in-thousands rare (we'll count them).
Each such "why" ends up in the module's `.md`, shown from the data.

**SECOND PRINCIPLE — write like a grandmaster (bestfitting-style).** The goal isn't just
"make it run", it's **craft** — the habits that separate top Kaggle Grandmasters (like
bestfitting / Shubin Dai) from everyone else. Every module *models* these, and Part 6
teaches them explicitly:
- **Reproducibility first** — seed everything (`torch`, `numpy`, `random`, cuDNN); same code → same number, always.
- **Config-driven, no magic numbers** — hyper-params live in a config (dataclass/YAML), not scattered in the code. One place to change, one place to log.
- **Modular `src/`** — clean separation: `data` / `model` / `train` / `infer` / `metric`. Small functions, real docstrings, readable names. (pilkwang's repo is a good example; so is our `cellmot/`.)
- **A CV you can trust** — match the split to how the test is made; trust local CV over public LB; validate the CV against known scores (we did: golden-12 = leak-free fold).
- **One change at a time, measured** — never change five things at once; change one, score it, keep it only if it helps. (Exactly how we swept the rules this session.)
- **Ensembling & TTA as first-class** — averaging diverse models/augmentations is how the last few points are won.
- **Read others' code and understand *every line*** — you can only improve what you fully understand (this whole curriculum is that habit).

**How to run any module:** `research/cellmot_venv/bin/python learning/<script>.py`

---

## Part 0 — Data EDA  ✅ DONE
Understanding the data before any model (you already did these).
- `01_cells_per_frame.py` + table — how many cells per frame
- `02_timeline_heatmap.py` + image — the time dimension per embryo
- `03_true_density_stage.py` + plot — real density → developmental stage
- Summary: `LEARNING.md`

**THIRD PRINCIPLE — teach it as a domain expert, fully.** Each construct is explained
not only as PyTorch, but with the *biology/microscopy reason* it exists: e.g. why a **3-D**
conv (nuclei span several z-planes in a light-sheet volume), why nuclei look like compact
bright blobs, why divisions are hard, why the imaging is anisotropic. The code choices only
make sense through the domain.

## Part 1 — PyTorch basics = the constructs OUR pipeline actually uses  ← WE START HERE
NOT abstract tensor tutorials — every PyTorch piece is pulled straight from
`temporal_unet.py` / `simple_node_transformer.py` / `train_unet_transformer.py` and
explained where it appears, with all four threads (PyTorch + data + craft + domain).
- **pt01 — `nn.Module` & the conv block**: `nn.Sequential`, `nn.Conv3d`, `nn.BatchNorm3d`, `nn.ReLU` — the real `_conv_block(...)`; why 3-D conv, why `kernel=3, padding=1`, why `bias=False`
- **pt02 — The U-Net shape**: `nn.MaxPool3d`, `nn.Upsample`/`F.interpolate`, `torch.cat`, `nn.ModuleList` — encoder/decoder/skip; why down-then-up recovers cell positions
- **pt03 — Temporal attention**: `nn.MultiheadAttention`, `nn.LayerNorm`, `nn.Identity` — the `_TemporalAttention` block; why attend across time (cells persist frame-to-frame)
- **pt04 — The edge model MLP**: `nn.Linear`, `nn.GELU`, `nn.Dropout` — `simple_node_transformer.py`; scoring cell→cell links
- **pt05 — Losses**: `softmax`, BCE, focal, and the **division weight** line — the real `compute_loss`; why divisions need up-weighting (rare, from the data)
- **pt06 — `Dataset` & `DataLoader`**: the real `FrameWindowDataset` — how frames+coords become batches; seeding & workers
- **pt07 — Optimizer & training loop**: `AdamW`, `loss.backward()`, `opt.step()`, grad-clip, `autocast` (mixed precision) — the real training loop
- **pt08 — Checkpoints & warm-start**: `state_dict`, `torch.save/load`, freezing the detector — exactly what our fine-tune does

## Part 2 — The competition's DETECTOR (3D U-Net + attention)
Read pilkwang's `temporal_unet.py` line by line, with runnable pieces.
- **pt07 — 3D convolutions**: Conv3d, kernels, padding, on a real cell volume
- **pt08 — The U-Net**: encoder/decoder/skip-connections, why it segments
- **pt09 — Temporal attention**: the transformer block over time (multi-head attention)
- **pt10 — Detection head & peak-finding**: heatmap → sigmoid → maxpool NMS → cell coords
- **pt11 — Detection loss**: class-balanced BCE, why `neg_weight` matters (recall lever)

## Part 3 — LINKING (the learned graph)
- **pt12 — The edge transformer** (`simple_node_transformer.py`): scoring cell→cell links
- **pt13 — The edge loss & the division weight**: the `weight[div_rows]=1.0` line (our smoking gun)
- **pt14 — Assignment**: Hungarian / ILP — turning edge scores into tracks

## Part 4 — SCORING & POST-PROCESSING
- **pt15 — The official metric**: adjusted edge-Jaccard + 0.1·division (`metrics.py`), the 7µm match
- **pt16 — Post-processing**: motion-relink, gap-close, divisions, smoothing

## Part 5 — Competition-specific research (what we studied)
- **pt17 — Augmentation**: flips/brightness — why, and how they're applied in training
- **pt18 — Dataset timing & external data**: stages, Zebrahub, leave-one-stage-out CV
- **pt19 — Cross-validation done right**: the golden-12 = leak-free fold, official-metric CV

---

## Progress
- [x] Part 0 — EDA
- [ ] Part 1 — PyTorch basics ← next: **pt01 Tensors**
- [ ] Part 2 — Detector
- [ ] Part 3 — Linking
- [ ] Part 4 — Scoring & post-proc
- [ ] Part 5 — Research topics
