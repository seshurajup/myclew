# fp8 cell-detector — capability proof (measured, RTX 5090 sm_120)

**What this proves:** a 3D cell-center heatmap detector can be built matmul-dominated and trained
**end-to-end in fp8** (E4M3 forward + E5M2-gradient backward) on the 5090 via native
`torch._scaled_mm` — no torchao / transformer-engine (ABI risk; deliberately not installed). It does
**not** prove fp8 is faster here, nor that it beats the production UNet3D pipeline.

Files:
- `fleet_agents/fp8_cell_detector.py` — the model (`Fp8CellDetector`, `Fp8Linear`, `Fp8LinearFn`).
- `fleet_agents/fp8_cell_detector_verify.py` — the measured verification harness.
- `fleet_agents/arch_builder.py` — `MODERN_CATALOG` entry `fp8-transformer-detector`.

Reproduce: `OMP_NUM_THREADS=1 STEPS=300 CELLMOT_FP8_BACKWARD=1 research/cellmot_venv/bin/python fleet_agents/fp8_cell_detector_verify.py`

## Architecture (why fp8 can train it)
UNet3D (conv3d) has **no fp8 conv3d kernel** → stuck at bf16. This detector is Linear/attention-dominated:
- Conv3d patch-embed stem, patch (8,16,16) — the ONLY conv, **9.0% of params**.
- 6 transformer encoder blocks (MHSA + MLP), embed dim 256, 8 heads, learnable 3D pos-embed.
- Linear head → un-patchify to a per-voxel cell-center logit volume (same peak-detect contract as UNet).

**Param split:** conv 524,544 (9.0%) · linear 5,258,752 (90.7%) · other 14,848.
`hardware_tune.select_train_precision(model)` → **`fp8`** (`arch=transformer`, conv<linear). ✔

## The 5 measured numbers (5090, vol 16×64×64, bs 4, 300 steps)

| # | metric | result |
|---|--------|--------|
| 1 | fp8 end-to-end train step RUNS (fwd+loss+bwd+step) | **YES**, no error |
| 2 | CONVERGENCE (BCE, fixed synthetic blobs) | fp8 **0.727 → 0.046**; bf16 0.727 → 0.059 — both converge, neither diverges, fp8 ≈ bf16 |
| 3 | SPEED (s/iter fp8 vs bf16) | fp8 0.00865 vs bf16 0.00349 → **0.40× (fp8 is 2.5× SLOWER)** |
| 4 | fp8 COMPUTE FRACTION (fwd MACs) | **89.4%** on fp8 (attn qkv/proj + MLP + head); 10.6% bf16 (conv stem 8.9%, attn score/context 1.7%); softmax/LayerNorm bf16 |
| 5 | PEAK VRAM fp8 vs bf16 | 0.274 GB vs 0.274 GB → **Δ 0.0 GB (fp8 frees nothing here)** |

**Training mode achieved: TRUE fp8 fwd+bwd** (all three GEMMs — forward, grad_input, grad_weight — in
fp8; gradient tensor in E5M2). `CELLMOT_FP8_BACKWARD=0` gives fp8-forward-only (bf16 backward), also
verified to converge. The bf16 path (`set_fp8(False)`) is the always-working reference.

## Why fp8 is slower here (the honest ceiling)
The 5090 fp8 tensor cores are real: **raw `torch._scaled_mm` = 2.13× vs bf16** at 4096³ (matches the
box profile 2.09×). But our `Fp8Linear` does per-op absmax quantize + a column-major `.contiguous()`
copy per operand, unfused, in eager mode — memory-bound overhead that gives the entire GEMM win back:

| GEMM 4096³ | time | speedup |
|---|---|---|
| bf16 `a@b` | 0.765 ms | 1.00× |
| raw `_scaled_mm` (pre-quantized) | 0.359 ms | **2.13×** |
| `_scaled_mm` + quantize + transpose (one operand) | 0.711 ms | 1.08× |

A `Linear` quantizes **two** operands across **three** GEMMs (fwd + 2 bwd), so the overhead compounds.
At the cell-detector's small patch-token matmuls (M=B·T=128, K=256) the GEMMs are tiny and overhead
dominates → 0.40×. VRAM is unchanged because bf16 activations are cached for backward (we cast to fp8
transiently, not persistently). A real speed win requires **fused quantize epilogues + cached
column-major fp8 weights** (torchao / TransformerEngine) — exactly what we don't install.

## Scaling to real biohub data
Patch the real zarr volumes into (D,H,W) tiles matching `vol`/`patch` (stride-tile with overlap;
16×64×64 is a starting tile, raise to fill VRAM). Target = Gaussian blob (σ≈2 vox) at each
founder-lineage GT center (sparse labels — mask/weight unlabeled regions, don't treat absence as
negative). Same forward → sigmoid → peak-detect → link as the current detector. Precision auto-selected
by `select_train_precision(model)`.

**HONEST caveat:** this is a capability foundation, not a score. Matching pilkwang's ~0.90 is a
separate **large from-scratch training effort with uncertain payoff** — biohub is recall-saturated
(node recall is the lever, and the edge head is already saturated), and a fresh transformer detector
would have to re-earn detection recall the UNet already has. Pursue this only if (a) you want fp8
throughput on the 5090 for large-scale from-scratch training AND (b) you first wire fused fp8
(torchao) so the 2.13× GEMM win actually survives — otherwise bf16 is strictly faster today.

## Verdict
fp8 **CAN** train a cell detector end-to-end on the 5090 (proven: runs, converges ≈bf16, 89% of MACs
in fp8, true fp8 backward). The **0.40×** eager number is NOT fp8's fault — it's (1) unfused per-GEMM
quantize + redundant layout copies, (2) no `torch.compile`, and (3) matmuls too small (M=128, K=256).
Corrected on this box: quant-once eager = **1.25×**, cached-W = 1.35×, and **`torch.compile` fused =
1.84×** for a large (4096³) full training step; raw GEMM is 2.13× (2.92× MXFP8). See
`docs/fp8_5090_compat_matrix.md` for the full box-verified matrix and decision rule.

**Bottom line:** for THIS small detector, **bf16 is correct** (its matmuls are below the fp8 break-even
size). fp8 only pays off for large from-scratch transformer training, and only under `torch.compile`.
The precision policy should gate fp8 on matmul SIZE + compile, not just conv-vs-linear param split.
Value delivered = a reusable precision-selectable matmul-dominated detector + an honest, corrected map
of where fp8 actually wins on this box.
