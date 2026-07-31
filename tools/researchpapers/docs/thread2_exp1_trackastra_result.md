# thread-2 exp#1 — Trackastra ctc learned linker (golden-12 official)

> Design/queue: `docs/research/thread2_architecture.md` (exp#1, ledger L5–L7). Job `train-960812e622`
> (train_service :7799, status=`succeeded`, exit 0, ~4.5 min GPU on RTX 5090 via `research/cellmot_venv`,
> cuda=True). Nodes stay FIXED (pilkwang detections); only the linker is swapped. Official re-score vs the
> **0.8708** pilkwang anchor. Outputs: `output/thread2_exp1_trackastra/exp1_trackastra_greedy_{percell.csv,summary.json}`.

## Conclusion (one line)
**Off-the-shelf pretrained Trackastra (ctc, coord_dim=3, greedy) is a LOSS on our substrate: adj_edge_jaccard 0.6465 vs 0.8708 anchor (Δ=−0.2243), and divisions collapse (0 TP / 377 FP forks → division_jaccard 0.0).** BUT this is the *weak/confounded* probe researcher flagged — pretrained ctc expects real instance masks and got synthetic point-blobs + voxel (non-isotropic) coords + intensity only. The loss is **expected-ish and does NOT prove "learned linking is worse"**; it exhausts the *zero-training* learned-linker option and points at **exp#4 (fine-tune a 3D Trackastra on our tracks)** as the real test — pending leader's call on whether the confounded signal justifies the GPU.

## Results table (golden-12, official = adj edge-Jaccard + 0.1·div-Jaccard)
| arm | linker | adj_edge_J | div_J | score | div tp/fp/fn | Δ vs anchor |
|---|---|---|---|---|---|---|
| **pilkwang anchor** | geometric (ILP, FIXED nodes) | **0.8708** | 0.0* | 0.8708 | 0 / ~30 / 8 | 0 (baseline) |
| **exp#1 Trackastra ctc** | learned pretrained, greedy | **0.6465** | **0.0** | **0.6465** | **0 / 377 / 8** | **−0.2243** |

\* pilkwang div_tp is also 0 (thread-1 L3), but with ~30 FP forks; Trackastra emits **377** FP forks — ~12× more spurious divisions. On divisions the learned linker is strictly worse here.

## Per-embryo (sorted by adj_jaccard) — the loss is DENSITY/SCALE-dependent
| embryo | n_pilk_nodes | count ratio (t_pred/t_true) | adj_jaccard | edge tp/fp/fn |
|---|---|---|---|---|
| 6bba_062c8d37 | 7,210 | 1.20× | **0.874** | 862/69/36 |
| 6bba_085bf656 | 9,047 | 1.07× | 0.833 | 1091/132/77 |
| 6bba_07477033 | 5,383 | 1.11× | 0.818 | 554/68/48 |
| 44b6_0113de3b | 25,445 | 0.99× | 0.713 | 42/9/8 |
| 6bba_05b6850b | 7,548 | 1.19× | 0.690 | 714/170/131 |
| 44b6_0b24845f | 53,614 | 1.64× | 0.629 | 41/12/8 |
| 44b6_12dfb391 | 59,169 | 1.01× | 0.589 | 584/217/189 |
| 44b6_144b256d | 78,784 | 1.20× | 0.516 | 90/52/29 |
| 44b6_0c582fdc | 32,885 | 1.18× | 0.496 | 50/29/20 |
| 6bba_07e24132 | 50,363 | **2.34×** | 0.443 | 250/143/95 |
| 6bba_05db0fb1 | 75,619 | 1.08× | 0.432 | 706/437/477 |
| 44b6_0db75fae | 19,052 | 1.24× | **0.320** | 85/108/66 |

**Clear pattern:** the ~anchor-level embryos are the SMALL/sparse ones (5–9k nodes → adjJ 0.82–0.87); the big collapses are the LARGE/dense ones (50–79k nodes → 0.32–0.59). Pretrained Trackastra generalizes to sparse fields but breaks on our dense embryos — corroborates thread-1's "linking degrades with density" (Q_link vs estN corr −0.51), and here compounds with the out-of-distribution point-blob input.

## Verdict (interpretation locks applied — do NOT over-read)
1. **exp#1 = LOSS, −0.2243.** Pretrained Trackastra off-the-shelf underperforms pilkwang's geometric linker on the FIXED-detection substrate. The zero-training learned-linker lever is spent.
2. **Confound is real and load-bearing (per researcher's caveat):** ctc was trained on real instance masks; we feed pilkwang centroids as synthetic point-blobs, voxel coords (anisotropy absent from the model's positional bias), real intensity only. The loss is **not** evidence that learned linking is inferior — it's evidence that the *pretrained-domain-mismatched* model is inferior here.
3. **Divisions got worse, not better:** the division-aware model produced 377 FP forks (vs pilkwang ~30) with 0 TP — it hallucinates divisions from dense spurious edges. Division upside is NOT harvested by dropping in a pretrained linker.
4. **Density is the axis:** loss concentrates entirely on the large/dense embryos; sparse embryos hold near anchor. Any learned-linker follow-up must be judged on the dense tail.

## Next (from the data + queue)
- **exp#4 (GATED follow-up): fine-tune a 3D Trackastra on our tracks** with correct anisotropy/coords — the only clean test of "learned vs geometric linking" on this substrate. exp#1's job was to gate this; the confounded loss means exp#4 is now the open question, but it's a **heavier GPU bet** — leader decides whether the confounded signal justifies it, or whether to first de-confound cheaply (e.g. isotropic-scaled coords / blob-radius sweep) before committing to fine-tune.
- **exp#2 (CPU, independent): merged/duplicate-node audit** on golden-12 — sizes whether over-prediction is merged detections; relevant because the dense embryos (where exp#1 fails) are exactly the over-predicted ones (07e24132 @ 2.34×).
- **exp#3 (CPU, independent): div-rich 6+6 split div_jaccard re-measure** — both exp#1 and pilkwang score 0 div_tp on golden-12's thin 8 divisions; the real division ceiling still needs the div-rich split before any division bet is sized.

## Artifacts
- Results: `output/thread2_exp1_trackastra/exp1_trackastra_greedy_percell.csv` + `_summary.json`
  (adj_edge_jaccard=0.6465390, division_jaccard=0.0, score=0.6465390, anchor=0.8708, delta=−0.2242610, div 0/377/8).
- Runner: `eda/thread2/run_exp1_trackastra.sh` → `eda/thread2/exp1_trackastra_link.py --mode greedy`
  (absolute `research/cellmot_venv` python, ctc pretrained coord_dim=3, no retrain).
- Job log: `eda/thread2/train_log.txt` (startup + per-embryo progress; job `train-960812e622`).
- Figure: `docs/thread2_exp1_trackastra_percell.png` — per-embryo adjJ vs anchor, ordered by node count (density-dependence of the loss).
```

## Note on the trend PNG
The standing "top-3-per-baseline-version cross-version trend PNG" is a `baseline_v*` deliverable; this is a
thread-2 research probe with no baseline-version lineage, so the meaningful figure here is the per-embryo
density plot above, not a cross-version line plot. The baseline trend PNG resumes when a `baseline_v*`
version next completes.
