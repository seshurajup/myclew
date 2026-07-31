# Plan-B — Windowed (multi-frame) temporal head on OUR edge-transformer (design, 2026-07-10)

**Lever:** the honest bottleneck = **EDGE_J / linking** (headroom check: count-penalty NOT binding ~1%,
edge_J 0.72–0.74 vs perfect-linker ceiling ~0.90, **+0.16 headroom**, perfect-linker 2-fold ~0.889 ≈ public
LB target; `[[honest-loeo-linker-bound-headroom]]`). Trackastra is dead for our point pipeline (appearance
wall). So we build the temporal context INTO our own linker. **No mask/appearance dependency.**

## Current linker (what we're changing)
`SimpleNodeTransformer.forward(feat_t, feat_t1, coords_t, coords_t1, …)` is **strictly PAIRWISE**:
- 4 `CrossAttentionBlock`s do bi-directional cross-attention **between the two frames of the pair only**
  (nodes at t attend to t+1 and vice versa), then a `pair_mlp` scores each (i,j) from `[q_i, k_j, rel_pos]`.
- Each edge (t→t+1) sees **only frames t and t+1** — NO context from t−1 or t+2.
- `window_size` (config, currently **2**) only controls how many consecutive pairs land in a training
  window; the trainer loops `for i in range(W-1): predict_edges(frame[i], frame[i+1])` — each pair is still
  predicted independently with 2-frame context. **⇒ raising window_size alone gives NO temporal context.**

## The change: a temporal-context stage (opt-in)
Insert ONE temporal stage BEFORE the existing pairwise scoring, so each frame's node features are enriched
with motion/trajectory context from neighbouring frames in the window, then score pairs on the enriched
features:
1. **Temporal cross-attention (NEW, opt-in):** given the W-frame window's per-frame node features
   `[h_0…h_{W-1}]` (+ coords, masks), each frame's nodes attend to nodes in the **adjacent frames**
   (t−1, t+1 within the window) via K `CrossAttentionBlock`s (reuse the existing block; add a relative-time
   embedding). Output = temporally-aware node features `h'_t` carrying local trajectory/velocity signal.
2. **Existing pairwise scoring (unchanged):** the current bi-directional cross-attn + `pair_mlp` runs on
   `h'` instead of raw `h`. Same head, same ILP downstream.

Why this lifts edge_J: the +0.16 deficit is (a) **FP edges** — spurious links off the ~13% over-detected
false nodes, and (b) **wrong links** among true nodes at crossings/high density. Motion continuity from
t−1/t+2 gives a constant-velocity prior that (a) rejects physically-implausible (velocity-discontinuous)
links off false nodes and (b) disambiguates crossing trajectories — exactly the pairwise linker's blind
spot. Public 0.897 solutions out-LINK, not out-detect, us (headroom check) → this is the mechanism.

## Data change? NO.
The dataloader ALREADY emits W-frame windows (stored `(W, max_nodes, …)`; `window_size` config). Plan-B =
**architecture + train-loop-wiring + predict-path change**, NOT a data change. Config deltas: `window_size:
2→4` (4-frame temporal window) + a new opt-in flag (`edge_temporal: true`, default OFF → existing runs
byte-identical). The trainer computes `frame_det[*]` per window frame already (unet_feat + coords + mask) —
the temporal stage consumes exactly those before the per-pair loop.

## Exact A/B (one variable)
- **Treatment:** `window_size=4` + `edge_temporal=true` (temporal stage ON).
- **Control:** our current pairwise linker (`window_size=2`, temporal OFF) = the 150it baseline (fold0
  0.7152 / fold1 0.7322, floor mean **0.7237**).
- Everything else IDENTICAL: same detector, 150it/12ep fast-screen grid, same E50 aug, same SCIP ILP,
  same leak-clean val-holdout selection, same canonical scorer.
- **Screen→confirm** (locked discipline): fast-screen fold0 @150it; if it lifts edge_J/adj over 0.7152,
  CONFIRM full honest 2-fold vs 0.7237; promote only on 2-fold MEAN > 0.7237.

## GPU wall-time estimate
Current 150it/12ep ≈ ~40 min/fold (≈1.75 it/s). Temporal stage adds K attention blocks over the window +
`window_size 2→4` doubles pairs/window → ~1.5–2× per-iter → **~60–80 min/fold**, **~2–2.5 h for the 2-fold
screen**. Memory: temporal attn is over per-frame node sets (N up to ~60k dense) — reuse the existing
`pair_chunk_size` + `key_padding_mask` masking; grad-checkpoint the temporal blocks like the pairwise ones.

## Dev risk (flagged)
1. **Shared-code change** to `train_unet_transformer.py` + `simple_node_transformer.py` + the predict path.
   MITIGATE: opt-in `edge_temporal` flag, default OFF → every existing/queued run is byte-identical
   (same guarantee as the masked-loss env-gate).
2. **Train/predict PARITY** — the temporal stage must be wired into BOTH `train_unet_transformer.py`
   (training loop) AND `predict_unet_transformer.py` (inference), else train/test mismatch silently tanks
   it. This is the #1 correctness risk; validate with a 1-iter train + a predict smoke.
3. **Memory** on dense 44b6 (N~60k): temporal attention is O(N·N_adj) per frame — chunk + mask like the
   pairwise path; grad-checkpoint. Risk of OOM at bs8 — may need bs reduction on the dense fold.
4. **Window truncation** at video start/end (fewer than W neighbours) — pad + mask (the dataloader already
   pads; extend the mask to the temporal stage).
5. **Fallback:** with `window_size=2`/temporal OFF the model must be bit-identical to today (regression
   guard).

## Deliverables / plan
1. This design (done). 2. Implement `TemporalContextStage` (opt-in) in `simple_node_transformer.py`; wire
into the train loop + predict path behind `edge_temporal`; config `config/loeo_windlink_f0.yml`
(window_size=4, edge_temporal=true, 150it) + `_f1` + evals. 3. CPU unit-test the temporal module (shape/mask/
grad) + config dryrun-green + a 1-iter GPU smoke (train + predict parity) in the GPU gap. 4. Hand leader to
route. Promote gate: 2-fold MEAN > 0.7237.

## Provenance
Current linker: `research/official_repo/src/tracking_cellmot/models/simple_node_transformer.py`
(`SimpleNodeTransformer.forward` — pairwise). Train loop: `train_unet_transformer.py:876-894` (per-pair
`predict_edges`). Headroom: [[honest-loeo-linker-bound-headroom]]. Trackastra-dead context:
[[thread2-postproc-negative-trackastra]], `docs/trackastra_linker_design.md`.
