# SpatialDINO → zebrafish nuclei detection — MINI-FIRST transfer test

**Date:** 2026-07-20 · **Verdict: NO-GO (weak partial transfer, not a fatal content gap)**

Decisive question: do SpatialDINO's 3D-ViT features TRANSFER to zebrafish nuclei
detection, or is the organelle→nuclei content gap fatal? Answered honestly on real
data, LOEO (train 6bba, held-out 44b6), frozen-backbone probe. Score upside is capped
regardless (biohub is recall-saturated ~0.909); this is a foundation/efficiency track.

## TL;DR
- Backbone loads and runs perfectly (0 missing/unexpected keys, 21.5M params, forward
  OK at 4096 and 8192 tokens).
- **Features are NOT random** — a probe on frozen patch tokens recalls founder-lineage
  centers ~6–9× above chance on the held-out embryo. So the content gap is **not total**.
- **But transfer is WEAK and not useful.** The learned probe barely beats a **trivial
  patch-brightness baseline** (MLP +~0.10 recall; linear ≈ tied), collapses ~2× from
  in-distribution to held-out (domain gap), and a light detection head **cannot localize
  to the 7 µm gate at all** (recall ~0 even in-distribution) because patch resolution
  (8 vox) is coarser than the gate in z. This is **2–4× worse** than the incumbent
  pilkwang UNet detector (node-recall **0.995**).
- **fp8 gave no speedup** here (0.98×) — eager `torch._scaled_mm` needs `torch.compile`
  to win (per `docs/hardware_config.json`); not realized in this setup.

## Setup (what ran, honest)
- Backbone: `research/spatialdino/backbone.pth`, vits8 (patch 8³, embed 384, depth 12,
  6 heads, pos_embed=none). Built from `spatialdino_repo` `Encoder`; xformers avoided via
  `XFORMERS_DISABLED=1` (pure-torch attention fallback), plus in-process stubs for
  `torch_pca`/`omegaconf`/`SwiGLU` and `importlib.metadata.version`. **No pip installs;
  torch 2.8.0+cu128 / numpy / cv2 ABI untouched.**
- Data: images `(T=100, Z=64, Y=256, X=256)` uint16; GT `.geff` = **sparse** founder-
  lineage centers only (~6/frame, e.g. 52 nodes across 100 frames), voxel coords.
  Scale (z=1.625, y=x=0.40625 µm), 7 µm match gate, Hungarian per-frame.
- One frame = one volume → **8192 patch tokens** (the ≥8192 regime requested).
- Backbone **frozen** (bf16 accurate features) — the cleanest, most-interpretable
  transfer test (standard DINO linear-probe protocol); fine-tuning would confound
  "features transfer" with "adaptation" and is the correct *next* step only if the
  probe shows promise. It did not.
- Precision: features read in bf16 for the verdict (accurate); fp8 measured separately
  for the efficiency question, reusing the production `fleet_agents/fp8_cell_detector.py`
  `Fp8Linear` (real `_scaled_mm`).

## Result 1 — dense detection head (frozen backbone + light 3D conv head)
Head: tokens `[1,384,8,32,32]` → trilinear-up ×8 + Conv3d → heatmap `[1,1,64,256,256]`;
σ=2-vox Gaussian targets, weighted-MSE (pos_weight 80) for sparse GT; 300 steps, 121 ms/step.
- Training loss **0.307 → 0.0067** (fits). But held-out 44b6 detection **recall@7µm = 0.000**
  at every threshold, AND **recall ≈ 0 on the TRAINING embryo too** (3/36, 0/38, 0/43).
- Diagnosis: the head collapses to a **fixed spatial prior** (constant 48 peaks regardless
  of input; nearest-peak median 23–50 µm). Root cause is a **resolution mismatch** — the
  patch grid is 8 vox (z: 8×1.625 = 13 µm > the 7 µm gate), so a σ=2-vox / sub-7µm target
  is **unreachable** from frozen patch features with a light up-head. This is a probe-design
  limit, not a clean content verdict → forced the patch-level test below.

## Result 2 — patch-level probe (DECISIVE, at native feature resolution)
Label a patch positive if it contains a GT center; train probe on 6bba, eval "is the
founder center's patch ranked in the top-K of 8192?" on held-out 44b6. This removes
sub-patch localization and directly tests feature CONTENT.

Held-out 44b6 **recall@K** (GT-center-patches = 173):

| K (of 8192) | chance | mean-intensity | max-intensity | SpatialDINO linear | SpatialDINO MLP |
|---:|---:|---:|---:|---:|---:|
| 50  | 0.006 | 0.069 | 0.017 | 0.087 | 0.069 |
| 100 | 0.012 | 0.121 | 0.064 | 0.133 | 0.145 |
| 200 | 0.024 | 0.220 | 0.121 | 0.266 | 0.283 |
| 400 | 0.049 | 0.347 | 0.266 | 0.358 | 0.457 |
| 800 | 0.098 | 0.520 | 0.393 | 0.566 | 0.607 |

In-distribution 6bba (same probe): MLP @400 **0.752**, @800 **0.845** → **~2× drop** to
held-out (0.457 / 0.607): a real cross-embryo domain gap.

Reading it:
- **Above chance** (MLP @400 = 0.457 vs 0.049 → 9×). Features encode something real.
- **Barely above trivial brightness.** Nuclei are bright; mean-intensity alone gets @400
  0.347, @800 0.520. SpatialDINO-MLP adds only **+0.10 / +0.09**; the linear probe is
  **~tied with intensity**. So most of the "signal" is brightness any method captures; the
  pretrained features contribute a small-but-real margin beyond it.
- Absolute level is **useless as a detector**: even the best held-out operating point
  (recall 0.61 while keeping the top 10% of all patches) is far below the incumbent.

## Baseline (same held-out, same metric)
pilkwang UNet detector on 44b6, node-recall within 7 µm = **0.9948** (tp 2482 / n_gt 2495;
precision ~0.006 is meaningless — it predicts ALL ~15k–87k nuclei/volume vs the sparse
founder-lineage GT; recall is the clean signal, and it saturates).

## fp8 (efficiency track)
Backbone forward @ 8192 tokens: bf16 **104 ms** vs fp8 **107 ms** (48 Linears swapped to
`Fp8Linear`, forward-only) = **0.98× (no win)**. Consistent with `hardware_config.json`:
eager `_scaled_mm` needs `torch.compile` to beat bf16 (per-tensor quant/dequant overhead
eats the GEMM gain otherwise). The SpatialDINO Encoder (pure-torch attention fallback +
dynamic shapes) was not compiled here, so the fp8 speedup was **not realized**. No fp8
conv (head stays bf16 by policy).

## VERDICT — NO-GO (blunt)
SpatialDINO features do **partially transfer** — they beat chance ~6–9× and modestly beat
a brightness baseline, so the organelle→nuclei content gap is **not fatal/total**. But
**frozen-feature transfer is far too weak to be worth GPU-days**:
1. Can't localize to the 7 µm gate (recall ~0 even in-distribution) at 8-vox patch resolution.
2. At patch resolution, only ~0.46 recall@400 held-out — barely over trivial intensity
   (~0.35) — and ~2× worse than in-distribution (domain gap).
3. 2–4× worse than the existing UNet detector's **0.995** node-recall.
Do **not** spend GPU-days scaling this as a detector for THIS competition.

**Honest caveat (restated):** even a GO would be score-capped — the metric is recall-
saturated (~0.909 ceiling) and the incumbent detector already saturates node recall, so
there is no LB headroom. This was a foundation/efficiency probe, not an LB play.

**If ever pursued anyway (not recommended now):** the only paths with a chance are the
heavy ones the frozen probe deliberately skipped — (a) the repo's **feature upsampler**
(FeatUp/JAFAR-style, `upsample_factor=3`) to recover sub-patch detail before the head;
(b) **unfreezing + full fine-tune** (the +0.10-over-intensity margin suggests adaptation
could help); (c) **continued SSL on zebrafish** to close the domain gap. All are GPU-days
for capped payoff.

## Repro
- `scratchpad/sdino_detector/build_backbone.py` — load + forward (dry-run wiring).
- `scratchpad/sdino_detector/geff_utils.py` — geff reader + recall matcher (+ pilkwang baseline).
- `scratchpad/sdino_detector/run_transfer.py` — dense-head train + held-out recall + fp8 bench.
- `scratchpad/sdino_detector/diag.py` — in-dist vs held-out collapse diagnostic.
- `scratchpad/sdino_detector/patch_probe.py` — decisive patch-level probe (linear + MLP).
- `scratchpad/sdino_detector/intensity_baseline.py` — brightness control.
