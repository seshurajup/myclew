# Trackastra learned-linker swap — design (2026-07-10)

**Lever:** LINKING / edge-precision — the honest LOEO bottleneck (detector over-produces everywhere;
`[[gap-decomposition-detector-is-lever]]` 2026-07-10 correction). Keep our detector + SCIP ILP, swap the
edge-affinity source from our pairwise edge-transformer → Trackastra (windowed transformer linker).
**Status:** designed + de-risked; Stage-1 is routable, Stage-2 gated. Trackastra 0.5.3 installed; prior
integration code exists (`eda/thread2/exp1_trackastra_link.py`).

## What the integration map found (decision-critical)
1. **3D is supported** — the `ctc` pretrained model is native 3D (`coord_dim:3`). 3D is NOT the confound.
   (`general_2d` is 2D-only — do not use.)
2. **Trackastra REQUIRES instance label MASKS**, not points. `get_features` → `WRFeatures.from_mask_img`
   runs `regionprops_table` (shape+intensity per label region). Our detector emits POINTS → must synthesize
   a label volume. **This is the primary confound:** the prior exp#1 painted *uniform* synthetic ellipsoids,
   so every region has identical shape features → the transformer's appearance signal is dead → it lost
   (0.6465 vs 0.8708). Real-mask variants (Cellpose/watershed) were only tried for divisions, inconclusive.
3. **No fine-tune in the pip wheel** — no LightningModule / Trainer / `trackastra.training`. Fine-tuning
   needs the GitHub repo (`scripts/train.py` + the LightningModule) + `trackastra[training]` (lightning,
   kornia) + CTC-format data (label-mask TIFFs + `man_track.txt`).
4. **The clean ILP hook exists:** `model._predict(imgs, masks)` returns `predictions["weights"]` =
   `(((node_i,node_j), affinity), …)` — feed these as edge costs into OUR SCIP ILP instead of Trackastra's
   own `track_greedy/track_ilp`. `exp1_trackastra_link.py::track_scaled` already re-implements this hook
   (with a coord-scaling knob for the anisotropy confound).

## The honest EV caveat (surface before investing)
Trackastra's value proposition = **appearance-based** linking (regionprops of real masks — size/shape/
intensity, esp. for divisions). Our pipeline has **point detections** → uniform synthetic masks → that
appearance signal is unavailable. So Trackastra's edge over our linker collapses to its **windowed temporal
attention** (4-frame window vs our PAIRWISE t→t+1 edge-transformer). The real, narrower hypothesis:

> **Trackastra's multi-frame windowed attention improves edge-precision over our pairwise linker, even
> without appearance features.**

If that windowed-temporal gain is small, Trackastra ≈ our linker (both position/motion-only) and the high
fine-tune effort is unjustified. So we GATE the expensive path behind a cheap test.

## STAGED plan (de-risked)

### STAGE-1 RESULT (2026-07-10) = NULL → Trackastra DEAD for our point pipeline
The de-confound sweep was **already run** (`tools/researchpapers/output/thread2_exp1_trackastra/`), so no
new GPU needed: voxel 0.6465 → **iso_z 0.6876** (anisotropy fix recovers +0.041, dense-tail 0.552) → um
0.5167. Even the BEST de-confound (iso_z) is **−0.183 below the anchor 0.8708** (our/pilkwang linker) on the
FAVORABLE golden-12 (leaky/easy) — honest LOEO would be worse. The residual after fixing anisotropy is the
**appearance-mask wall**: Trackastra keys on regionprops shape/intensity; our point detections → uniform
synthetic blobs → zero shape signal, which iso_z can't fix and a solver (greedy→ILP) / eval-set change can't
close (0.18 is huge). **⇒ Stage-2 fine-tune DROPPED** (fine-tuning on the same appearance-less synthetic
masks can't add the missing shape signal). **Linking lever → Plan-B** (windowed head on our OWN
edge-transformer — no mask/appearance dep), **GATED on the human's 156 LB** (if LB ~0.90, polish the
submission, don't build it). Confirm honest-LOEO edge_J HEADROOM before building Plan-B.

### Stage 1 — de-confounded PRETRAINED gate (SUPERSEDED by the result above; kept for provenance)
Isolate whether Trackastra has ANY signal on our detections once the two known confounds are removed:
- **Anisotropy fix:** feed centroids in **isotropic/µm** coords (`iso_z`/`um` knob already in
  `exp1_trackastra_link.py::track_scaled` + `exp1b_compare.py`), not raw voxels (z compressed ~4×).
- **Mask parity:** paint the SAME synthetic blobs at train- and test-time (we only run inference here).
- Take `predictions["weights"]` → feed our SCIP ILP → score honest 2-fold canonical vs floor **0.7237**.
- **A/B:** Trackastra-affinities+ILP  vs  our edge-transformer+ILP, SAME detections, SAME ILP.
- **Decision gate:** if de-confounded pretrained Trackastra ≥ our linker (or clearly closes the gap on
  edge_J) → windowed attention has signal → proceed to Stage 2. If it still loses badly → the loss was
  appearance (masks we can't provide) → fine-tune on appearance-less synthetic masks won't recover it →
  **Trackastra is likely dead for our point pipeline → pivot** (windowed extension of our OWN linker, or
  a different lever). Cost: GPU inference only, no training. **Routable now** (pending Stage-1 config).

### Stage 2 — FINE-TUNE (expensive, GATED on Stage 1 passing)
Only if Stage 1 shows signal. Fine-tune the 3D `ctc` model on OUR data so it learns our density/motion:
1. `pip install "trackastra[training]"` + clone `weigertlab/trackastra` for the training driver.
2. Build CTC-format data: label-mask TIFF stacks (synthetic blobs matching inference — train/test appearance
   PARITY, the §confound fix) + `man_track.txt` lineage from our GT `.geff` edges.
3. Warm-start `from_folder(models/trackastra/ctc)`, fine-tune `ndim=3, window=4, div_upweight≈3`.
4. Save `from_folder`-loadable; inference via the Stage-1 ILP hook. A/B 2-fold vs floor + vs our linker.
   Cost: repo integration + GPU training (2 folds). Effort: HIGH (new training infra).

## Promote gate
Same discipline: fast screen → CONFIRM on FULL honest 2-fold canonical; promote only if 2-fold MEAN >
0.7237 (the 150it floor). Single-fold wins don't count (convergence taught us that).

## Relation to other levers
- **[B] TTA:** detection flip-TTA is ALREADY default-on (spent); only edge/linker-TTA remains and is
  largely subsumed by a better linker — deprioritized below this.
- **Windowed extension of our OWN edge-transformer** = the fallback if Trackastra dies at Stage 1 (gets the
  windowed-temporal benefit without the mask requirement). Worth noting as the plan-B linker lever.

## Provenance
Integration map (subagent, 2026-07-10): trackastra 0.5.3 at kaggle_vision env; API `model/model_api.py`,
`model/predict.py:216` (weights), `data/wrfeat.py:200` (mask requirement), `data/data.py:115` (CTCData).
Prior code: `eda/thread2/exp1_trackastra_link.py` (the ILP-affinity hook + anisotropy knob),
`exp1b_compare.py`, `experiments/divisions/trackastra_link.py`. Confound history: [[thread2-postproc-negative-trackastra]].
