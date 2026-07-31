# CV-GATE — DEFINITIVE Result (full-fusion pipeline + official metric)

**Conclusion (one line): GATE PASSES — the embryo-disjoint density LOEO CV IS LB-faithful.** With pilkwang's genuine **full LB pipeline** (UNet+transformer+ILP+best.pt fusion+full postproc) and both detectors scored by the **official `evaluate.py`**, pilkwang ranks **> canqiang on BOTH folds** (mean 0.8373 vs 0.7926, **Δ+0.0447** — larger than the public-LB gap +0.024). The two prior FAILs were **pipeline+metric artifacts**, not real CV inversions.

- **task_id:** CVGATE  |  **train_task_id:** `train-20765ff1d6` (succeeded, exit 0, wall-time 758s ≈ 6.3 min/fold)
- **exp_id:** EXP-CVGATE-FAIR  |  **script:** `baseline/run_pilk_full_loeodens.sh` (fixed: `set -euo pipefail`, `BIOHUB_MODEL_ARTIFACTS`=pack_v2, stale-clear + fresh-submission + stem-match hard-fails)
- **metric:** OFFICIAL `research/official_repo/scripts/evaluate.py`, both detectors — NO Kaggle

## Liveness verification (independently applied — all PASS)

| Check | Result |
|---|---|
| missing ~0 / fold coverage complete | ✅ 63/121 "missing" = global-registry noise, **identical to validated canqiang_full**; `[OK]` no-contamination both folds |
| n matches fold + embryo prefix | ✅ f0 n=8 (44b6), f1 n=7 (6bba) |
| fresh geffs from this job | ✅ split_0 mtime 12:09, split_1 12:15 (job 12:04–12:17) |
| wall-time plausible | ✅ 758s (~6.3 min/fold), not sub-minute |
| real compute | ✅ per-dataset `[ILP] N→M edges` for all 8+7 datasets |
| clean exit | ✅ exit 0 under `set -euo pipefail` |

## Key result table — per fold + aggregate (official metric)

| Detector (full pipeline) | fold0 (44b6, n8) | fold1 (6bba, n7) | mean (macro) | embryo-wtd |
|---|---|---|---|---|
| **pilkwang FULL** | **0.8567** | **0.8178** | **0.8373** | **0.8385** |
| canqiang FULL | 0.7973 | 0.7879 | 0.7926 | 0.7929 |
| **Δ (pilk − canq)** | **+0.0594** | **+0.0299** | **+0.0447** | **+0.0456** |

pilkwang wins **both** folds; aggregate margin **+0.0447 > LB gap +0.024**.

## The full journey — Δ(pilkwang − canqiang) by stage

| Stage | pipeline / metric | Δ (pilk − canq) | ranking |
|---|---|---|---|
| bare (orig `EXP-CVGATE`) | de-featured predict / proxy | **−0.0140** | canqiang wins → **FAIL** |
| interim (`pilk_ilp_k5`) | `--use-ilp` + `pool_kernel_um=5.0` / proxy | **+0.0046** | pilk wins, razor-thin (f1 tie) |
| **full fusion** | full LB pipeline / **official** | **+0.0447** | **pilk wins both folds → PASS** |

Figure: `docs/cvgate_journey_bare_interim_full.png` (bare vs interim vs full, pilk vs canqiang, both folds).

## Main-line judgment

- **Gate PASSES decisively.** Fixing both mis-specifications — the de-featured pilkwang predict (ILP-off + `pool_kernel_um=3.0` → ~1.35× over-detection) **and** the proxy metric — flips the ordering to pilkwang > canqiang on both folds, with a margin exceeding the LB gap.
- **Canqiang cross-check:** its proxy and official scores are identical (0.7973/0.7879) → the proxy was fair *for canqiang*; it only mismeasured the crippled bare-pilkwang. This is why both prior CVs "inverted".
- `div_tp=0` both sides (division non-discriminating; ranking is adj-edge-Jaccard) — does not affect the verdict.
- Two silent-failure traps found+fixed en route: `pool_kernel_um` was not CLI-wired; `pipeline.py` aborts without `BIOHUB_MODEL_ARTIFACTS` (first re-run then converted a stale gold12 submission → false 0.8971, caught by the liveness guard).

## Next-step suggestions

1. **CV validated → idea screening on `splits_loeo_density.json` is unlocked** — but ideas MUST be screened as *full pipelines* under the *official* `evaluate.py`, never the bare de-featured predict (which inverts).
2. Proceed to the successive-halving idea brackets per the original plan.

## Evidence

- Job `train-20765ff1d6`, log `baseline/train_log.txt`
- geffs: `research/official_repo/predictions/seshu/pilk_full_loeodens/split_{0,1}` (8 + 7, fresh)
- evaluate.py stdout: pilk_full 0.8567/0.8178 ; canqiang_full 0.7973/0.7879
- Journal: `docs/experiments/EXP-CVGATE-FAIR.md` (DEFINITIVE section + PASS verdict)
- Related: `docs/cvgate_loeo_density_result.md` (orig FAIL), `docs/cvgate_loeo_density_fair_interim_result.md` (interim), `docs/cvgate_full_fusion_FAILED_run.md` (aborted run), `docs/cvgate_metricgap_investigation.md`
