# thread-2 exp#1b — coord-scale anisotropy de-confound (golden-12 official)

> Design: exp#1's LOSS (`docs/thread2_exp1_trackastra_result.md`) was density-dependent; anisotropy
> (scale=(1.625,0.406,0.406), z compressed ~4×) was the prime suspect. Job `train-a51cfc7ddc`
> (train_service :7799, `succeeded`, exit 0, ~13 min GPU / ~4.5 min per config, `research/cellmot_venv`).
> Same exp#1 harness (nodes FIXED, greedy), 3 coord-scale configs in one job, official re-score vs **0.8708**.
> Outputs: `output/thread2_exp1_trackastra/exp1_trackastra_greedy_{voxel,iso_z,um}_{percell.csv,summary.json}`
> + `exp1b_compare.{csv,png}`. Chart copied to `docs/thread2_exp1b_coordscale_compare.png`.
> **Status: SECONDARY golden-12 evidence** — the primary track is now the 9-stage GRANDMASTER JOURNEY on the LOEO fold.

## Conclusion (one line)
**Anisotropy WAS part of exp#1's collapse — `iso_z` (z×4) is the best config both globally (0.6876 vs voxel 0.6465, +0.041) and on the dense tail (0.5518 vs voxel 0.4885, +0.063) — but the de-confound does NOT rescue the dense tail toward the 0.8708 anchor (iso_z still Δ=−0.183 global).** Full physical-µm (`um`) over-corrects and is *worst* (0.5167). So anisotropy is a real, partial confound, not the whole story: even correctly-scaled, off-the-shelf pretrained Trackastra ctc stays far below pilkwang's geometric linker on our point-blob substrate, and divisions never recover (div_J=0.0 in all 3).

## Results table (golden-12, official = adj edge-Jaccard + 0.1·div-Jaccard, nodes FIXED)
| config | coord scaling | global adj_edge_J | dense-tail adjJ | div_J | div_fp | Δ vs anchor |
|---|---|---|---|---|---|---|
| pilkwang anchor | geometric linker | **0.8708** | — | 0.0 | ~30 | 0 |
| voxel (=exp#1) | raw voxel (z compressed ~4×) | 0.6465 | 0.4885 | 0.0 | 377 | −0.2243 |
| **iso_z** ✅ | **z×4 (restore z scale)** | **0.6876** | **0.5518** | 0.0 | 366 | **−0.1832** |
| um | physical µm (z×1, xy×0.25) | 0.5167 | — | 0.0 | 231 | −0.3541 |

- `voxel` reproduces exp#1 exactly (0.6465) → harness faithful.
- `iso_z` is best on **both** aggregations → the primary hypothesis (compressed-z hurts the graph-transformer's positional bias) is directionally CONFIRMED.
- `um` worst → the fix is *restoring z*, not going all the way to physical µm (which shrinks xy separation and over-merges).

## Per-embryo: where iso_z helps (ordered by node count, dense → sparse)
| embryo | n_nodes | voxel | iso_z | um | iso_z − voxel |
|---|---|---|---|---|---|
| 44b6_144b256d | 78,784 | 0.516 | **0.757** | 0.430 | **+0.242** |
| 6bba_05db0fb1 | 75,619 | 0.432 | 0.499 | 0.082 | +0.067 |
| 44b6_12dfb391 | 59,169 | 0.589 | 0.629 | 0.243 | +0.040 |
| 44b6_0b24845f | 53,614 | 0.629 | 0.591 | 0.421 | **−0.038** |
| 6bba_07e24132 | 50,363 | 0.443 | 0.504 | 0.252 | +0.060 |
| 44b6_0c582fdc | 32,885 | 0.496 | 0.625 | 0.076 | +0.129 |
| 44b6_0113de3b | 25,445 | 0.713 | 0.769 | 0.655 | +0.056 |
| 44b6_0db75fae | 19,052 | 0.320 | 0.398 | 0.452 | +0.078 |
| 6bba_085bf656 | 9,047 | 0.833 | 0.845 | 0.840 | +0.011 |
| 6bba_05b6850b | 7,548 | 0.690 | 0.724 | 0.634 | +0.034 |
| 6bba_062c8d37 | 7,210 | 0.874 | 0.883 | 0.823 | +0.009 |
| 6bba_07477033 | 5,383 | 0.818 | 0.810 | 0.822 | −0.008 |

iso_z helps 10/12 embryos; the biggest gains are on the densest (144b256d +0.242, 0c582fdc +0.129) — exactly the tail that collapsed in exp#1. Two regressions are small. But even the best densest embryos sit at 0.50–0.63, nowhere near anchor.

## Verdict (interpretation locks)
1. **De-confound is REAL but PARTIAL.** iso_z recovers +0.041 global / +0.063 dense-tail — anisotropy was genuinely hurting. But iso_z at 0.6876 is still −0.183 below anchor and the dense tail is still 0.55. Correct scaling does not make off-the-shelf Trackastra competitive with pilkwang's geometric linker here.
2. **Residual gap is NOT anisotropy.** After fixing z, the densest embryos still score 0.43–0.63. The remaining loss is the deeper domain mismatch (pretrained on real instance masks; we feed synthetic point-blobs) + genuine density difficulty, not coordinate units.
3. **Divisions unrescued.** div_J=0.0 in all 3; FP forks 377→366 (iso_z barely moves it); um cuts forks to 231 only by degrading everything. Anisotropy correction does nothing for the division hallucination.
4. **Sweet spot = iso_z, not um.** Restore z, keep xy in voxel range.

## exp#4 gate (leader's call)
exp#4 (fine-tune 3D Trackastra) was gated on **"iso_z rescues the dense tail toward anchor."** It does NOT — it lifts the dense tail from 0.49→0.55, still 0.32 below anchor. Per the pre-registered rule (`[[thread2-postproc-negative-trackastra]]`): a partial de-confound that leaves a large gap argues that off-the-shelf pretrained learned linking does not fit our point-substrate, so **fine-tune is a bigger bet than the evidence yet justifies** — but iso_z is now the correct input scaling for *any* future Trackastra work (fine-tune included, so it doesn't inherit the anisotropy confound). Recommend: **do NOT auto-promote exp#4**; if a learned linker is pursued later it starts from iso_z. Meanwhile the GRANDMASTER JOURNEY (LOEO, Stage 1 dumb baseline next) is the primary track.

## Artifacts
- Results: `output/thread2_exp1_trackastra/exp1_trackastra_greedy_{voxel,iso_z,um}_summary.json` + `_percell.csv`;
  `exp1b_compare.csv` (per-embryo 3-way).
- Chart: `docs/thread2_exp1b_coordscale_compare.png` (per-embryo 3-way, harness-generated).
- Runner: `eda/thread2/run_exp1b_sweep.sh` → coord_scale sweep + `eda/thread2/exp1b_compare.py`. Job `train-a51cfc7ddc`, log `eda/thread2/train_log.txt`.
