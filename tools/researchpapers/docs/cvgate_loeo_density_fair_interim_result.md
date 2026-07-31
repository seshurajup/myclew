# CV-GATE FAIR (interim) Result — pilkwang ILP + pool_kernel_um=5.0

**Conclusion (one line):** Once pilkwang is run **fairly** (ILP linking ON + `pool_kernel_um=5.0`), the density LOEO CV **FLIPS to pilkwang > canqiang aggregated** (macro 0.7972 > 0.7926, Δ+0.0046) — i.e. the original CVGATE FAIL was substantially a **pipeline-parity artifact**, not a true CV inversion. **Caveat:** the margin is razor-thin (fold1 is a tie, Δ−0.0012) and far below the LB gap (+0.024), and this still uses the **proxy** metric — so treat this as *direction-confirming, not conclusive*; the fully-fair run (full fusion + official `evaluate.py`) is still needed.

- **task_id:** CVGATE  |  **train_task_id:** `train-218678ad24` (succeeded, exit 0)
- **exp_id:** EXP-CVGATE-FAIR (journal `docs/experiments/EXP-CVGATE-FAIR.md`, 2 runs auto-filled) — original EXP-CVGATE untouched
- **script:** `baseline/run_pilk_interim_fair.sh` (researcher-handed, dry-run GREEN)
- **fix applied (pilkwang-side only):** `--use-ilp` (ILP global linking, was greedy) + `--pool-kernel-um 5.0` (was hardcoded 3.0); startup log confirmed `pool_kernel_um=5.0`; ILP solve runs under `suppress_output()` (L562) — engagement confirmed by the count_ratio drop, not stdout
- **metric:** proxy `score_v1.py` (same scorer as original gate → apples-to-apples); canqiang reused (its full post-proc was already run in EXP-CVGATE)

## Key result table

| Pipeline | fold0 (44b6, n8) | fold1 (6bba, n7) | macro | embryo-wtd | count_ratio f0/f1 |
|---|---|---|---|---|---|
| pilkwang **BARE** (orig gate) | 0.7882 | 0.7690 | 0.7786 | 0.7792 | 1.35 / 1.32 |
| pilkwang **ILP+k5 (FAIR)** | **0.8077** | 0.7866 | **0.7972** | **0.7979** | 1.23 / 1.19 |
| canqiang | 0.7973 | 0.7879 | 0.7926 | 0.7929 | 0.72 / 1.05 |
| **Δ (fair pilk − canqiang)** | **+0.0104** | **−0.0012** | **+0.0046** | **+0.0050** | — |

- Fair-fix lift over bare pilkwang: **+0.0195 (f0), +0.0176 (f1)**.
- Ranking: **f0 flips** to pilkwang>canqiang (+0.0104); **f1 essentially tied** (−0.0012); **aggregate flips** (macro +0.0046, micro +0.0050).

Figure: `docs/cvgate_fair_interim_pilk_vs_canqiang.png` (bare vs fair vs canqiang, per-fold + macro).

## Main-line judgment

- **Direction confirmed:** the parity fix moves the aggregate ordering from canqiang-leads to **pilkwang-leads**, matching the LB direction. The original FAIL was largely a de-featured-pilkwang artifact (ILP-off + wrong pool kernel over-detecting ~1.35×).
- **But thin & incomplete:** aggregate margin +0.005 ≪ LB gap +0.024; fold1 is a statistical tie; pilkwang still over-detects ~1.2× (not down to 1.0), and `div_tp=0` both sides (division non-discriminating). So the CV is *not yet demonstrated* to reproduce the LB magnitude or hold per-fold.
- The proxy metric is still in play; the thread-A metric hole (proxy vs official `evaluate.py`) is unaddressed here.

## Probe / sidecar analysis

| Object | official | count_ratio | recall | div_tp | note |
|---|---|---|---|---|---|
| pilk_ilp_k5 f0 | 0.8077 | 1.230 | 0.981 | 0 | over-detect cut 1.35→1.23; +0.0195 vs bare |
| pilk_ilp_k5 f1 | 0.7866 | 1.190 | — | 0 | over-detect cut 1.32→1.19; +0.0176 vs bare; ~ties canqiang |
| canqiang f0/f1 | 0.7973/0.7879 | 0.72/1.05 | — | 0 | unchanged (reused) |

## Next-step suggestions

1. **Fully-fair run (leader-requested, still pending):** run pilkwang's full `pilkwang_full/pipeline.py` (fusion + gap-recovery) on both folds and rescore **both** detectors with official `research/official_repo/scripts/evaluate.py` — with researcher's new `[ILP]` observability print. That tests whether the margin widens toward the LB's +0.024 and whether f1 flips too.
2. Decision rule stands: fully-fair still ≤ or ~tie → escalate to human that the density CV isn't reliably LB-faithful; fully-fair clearly pilkwang>canqiang with a healthy margin → CV salvageable, unlock idea brackets.
3. Do not unlock idea brackets on this interim alone — margin too thin, metric still proxy.

## Evidence

- Sidecars: `output/scores/pilk_ilp_k5_f{0,1}.json` (fair), `pilk_loeodens_f{0,1}.json` (bare), `canqiang_loeodens_f{0,1}.json`
- Predictions: `research/official_repo/predictions/seshu/pilk_ilp_k5/split_{0,1}`
- MLflow: exp 16, runs `pilk_ilp_k5_f{0,1}` (system-metrics ON)
- Journal: `docs/experiments/EXP-CVGATE-FAIR.md`
