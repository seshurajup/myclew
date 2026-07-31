# Affinity Flow-Net — adoption into our training (grounded in the literature)

**Papers:** ELEPHANT / incremental deep learning (eLife 2022, elifesciences.org/articles/69380);
Whole-Embryo Lineages from Sparse Annotations (bioRxiv 2021.07.28.454016); both show **optical flow
trained on validated links reduces false-positive/false-negative links** — hengck23's exact hint.

**Proven cheap version (DONE, +0.0016):** empirical motion prior — real link displacement p99≈3.25µm,
tightened the linker gate 5.5→4.0µm → golden-12 0.8719→**0.8735**. This is the linker-gate form.

**Full learned form (to add to training):**
1. **Head:** add a 3-channel flow head to the TemporalUNet3D decoder → per-voxel `(dz,dy,dx)` predicted
   displacement to the next frame. (cellmot/heads.py: a Conv3d(feat→3); config `heads.flow.enabled`.)
2. **GT:** at each GT node, target flow = displacement to its linked child (from .geff edges), rendered
   into the cache next to `tgt`/`points` (data.py: add `flow` array; supervise only at GT voxels/mask).
3. **Loss:** masked endpoint error (`||pred - gt||` at GT nodes) + small smoothness; weight ~0.3 vs det.
4. **Use at inference:** warp each detection by predicted flow → candidate position; the linker prefers
   links consistent with the predicted flow (bias/gate on `||obs_disp - pred_flow||`, gated at p99).

**Metric link:** improves EDGE-Jaccard (our dominant term after node recall) by cutting FP/FN links —
the same mechanism the cheap prior used, but per-voxel and learned (handles the rare fast-z tail).
**Status:** base detector training now RUNS (fixed). Flow head = next code addition (heads + data + loss).
