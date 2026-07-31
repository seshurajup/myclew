# CVGATE-FAIR — results (EXP-CVGATE-FAIR)

Decomposing the original CVGATE FAIL into **{metric fix, pipeline fix}**. task_id=CVGATE.

## (B) Metric-isolation — OFFICIAL `evaluate.py` on the EXISTING geffs (CPU, done)

Scored the **bare** `pilk_loeodens` geffs (the original de-featured gate output: ILP-off, pool 3.0) and the
converted **`canqiang_full`** geffs with the official `research/official_repo/scripts/evaluate.py`
(`CELLMOT_DATA_DIR` = train dir). This changes ONLY the metric (both detectors scored identically, as-is
geffs); it does NOT fix pilkwang's pipeline.

| detector (pipeline as scored) | fold0 official | fold1 official | **mean** | node_recall | div TP/FP/FN |
|---|---|---|---|---|---|
| pilkwang **BARE** (ILP-off, pool 3.0) | 0.7326 | 0.7306 | **0.7316** | 0.99 | f0 0/82/3 · f1 2/207/8 |
| canqiang (full) | 0.7973 | 0.7879 | **0.7926** | 0.98 | 0 TP (0/0/3, 0/0/10) |
| **Δ (pilk − canqiang)** | −0.065 | −0.057 | **−0.061** | | |

**Conclusion (B): the metric change alone does NOT flip the ranking — canqiang still wins (−0.061).**
- **The proxy metric is faithful (thread A CLOSED).** canqiang's official numbers (0.7973 / 0.7879) are
  **identical** to its proxy `score_v1` sidecars → `src/metric.py` ≈ official `evaluate.py` for ranking.
  Both zero the division term. So the metric is NOT the driver of the FAIL.
- pilkwang-bare scores *lower* under `evaluate.py` (0.732) than under the proxy (0.779) because the proxy
  additionally applied `pilk_post`; `evaluate.py` scores the raw geff. Either way it trails canqiang.
- **pilkwang-bare has huge division FP (82 / 207)** — the greedy linker invents spurious forks; **ILP
  linking would remove these**. Direct corroboration of the ILP-off diagnosis. The gap is the PIPELINE.

⇒ The FAIL is not a metric artifact. It hinges on the **pipeline fix**, which the interim (A) tests.

## (A) Pipeline-isolation — interim bare + `--use-ilp --pool-kernel-um 5.0` (GPU, done)

Trainer ran `baseline/run_pilk_interim_fair.sh` → `pilk_ilp_k5` geffs + sidecars (exp_id EXP-CVGATE-FAIR).
**Result depends on the metric** — the crux of the whole investigation:

| detector | metric | fold0 | fold1 | **mean** | vs canqiang 0.7926 |
|---|---|---|---|---|---|
| pilkwang bare (ILP-off, pool 3.0) | **PROXY** (score_v1 + pilk_post) | 0.7882 | 0.7690 | 0.7786 | −0.014 |
| **pilk_ilp_k5** (ILP+pool5) | **PROXY** (score_v1 + pilk_post) | 0.8077 | 0.7866 | **0.7972** | **+0.0046 FLIP** |
| pilkwang bare | **OFFICIAL** (evaluate.py, raw geff) | 0.7326 | 0.7306 | 0.7316 | −0.061 |
| **pilk_ilp_k5** (ILP+pool5) | **OFFICIAL** (evaluate.py, raw geff) | 0.7788 | 0.7640 | **0.7714** | **−0.021 (no flip)** |
| canqiang (full) | proxy ≡ official | 0.7973 | 0.7879 | 0.7926 | — |

- **ILP + pool5 lifts pilkwang on BOTH metrics** (bare→interim: proxy +0.019, official +0.040). Direction of
  the thread-C over-detection prediction validated (count_ratio 1.35→1.23/1.19 on the two folds).
- **The flip is metric-dependent:** it holds on the **proxy** (+0.0046, thin, fold0-driven; fold1 a tie) but
  **not** on the **official** metric (−0.021). The entire proxy↔official gap for pilkwang is **`pilk_post`**
  — the proxy applies `filter_output_graph` at score-time; the saved geffs `evaluate.py` reads do NOT include
  it. canqiang's geffs DO carry its full post-proc → the official comparison is still not apples-to-apples on
  pilkwang's side (missing `pilk_post` + `best.pt` fusion).
- **ILP-VERIFY (confirms ILP executed, not just pool_kernel):** official-metric **division-FP collapsed from
  bare 82 (f0) / 207 (f1) → 1 (f0) / 0 (f1)** — the greedy linker's spurious forks are gone, exactly what the
  ILP global solver removes. ILP definitively ran.

**Conclusion (A): pipeline fix is real and large, but the flip only holds once pilkwang's post-proc is
included — which is a LEGITIMATE LB-pipeline stage not yet baked into the official-scored geffs. This is
precisely what the fully-fair `pipeline.py` run resolves** (bakes `pilk_post` + `best.pt` fusion into the
geffs, then scores with `evaluate.py` like-for-like).

## Verdict + decision (leader's rule)
Aggregate proxy FLIP but thin/split, and official no-flip pending post-proc ⇒ **CLOSE** → **greenlight the
MEDIUM pilkwang-full GPU run** (`pilkwang_full/pipeline.py` on both density folds + official `evaluate.py`).
It is the decisive, like-for-like test: full pilkwang (post-proc + fusion, ≈ +0.007 from `best.pt`) vs
canqiang-full, both official. Harness is READY (canqiang geffs converted, `evaluate.py` validated, `[ILP]`
log wired) — only pilkwang-full's GPU predict + CSV→geff remain.

## (D) Fully-fair run — WIRED + dry-run GREEN (GPU handoff)

Runner: **`baseline/run_pilk_full_loeodens.sh`** (reproduces the gold12 recipe from
`pilkwang_full/full12.log`, `BIOHUB_TEST_DIR` retargeted to the density folds). Per fold: run
`pilkwang_full/pipeline.py` (ILP + `best.pt` fusion + full postproc; `BIOHUB_ALLOW_ARTIFACT_FALLBACK=1`) →
rename `submission.csv` → convert via `baseline/pilk_submission_to_geff.py` → geffs at
`predictions/seshu/pilk_full_loeodens/split_{0,1}`; then official `evaluate.py` on pilk_full_loeodens AND
canqiang_full. Preserves the gold12 `submission.csv` (backed up).

**Runner-bug fix (first real run aborted):** the initial CPU-only dry-run stubbed the predict and missed
that `pipeline.py::find_artifacts_root()` needs **`BIOHUB_MODEL_ARTIFACTS`** (a path env absent from the
config dump; the Kaggle fallback paths don't exist here). Fixes: export
`BIOHUB_MODEL_ARTIFACTS=…/pilkwang_support_pack_v2` (verified `has_model_artifact`: repo+weights+best.pt);
`set -euo pipefail`; clean stale `submission_loeodens_f*.csv` + `split_${F}` before each fold and **assert a
FRESH submission whose stems == the fold test set** (hard-fail on contamination); SMOKE uses a 1-dataset
TEST_DIR (not `BIOHUB_SLICE`, which trips pipeline's all-stems assert L2163). `[ILP]` line placed in the
DURABLE `pack_v2/repo/scripts/predict` source (pipeline rebuilds `tracking_repo` from the artifact each run).

**GENUINELY-EXERCISED smoke (real GPU, `SMOKE=1 FOLDS=0`, exit 0):** ARTIFACTS resolved (no abort),
`device=cuda`, `[ILP] 44b6_0b24845f: 34406 candidate → 30907 solution edges`, best.pt fusion loaded, FRESH
submission, stem-match `[OK]`, convert → 1 geff, official `evaluate.py` score. Full-pipeline pilkwang on
44b6_0b24845f = official adjJ **0.9210** vs canqiang 0.7973 (and vs BARE 0.646 on this exact embryo) — a
strong preview of a decisive flip (the 2-fold MEAN is the verdict). Converter + `evaluate.py` also proven on
the gold12 `submission.csv` (44b6_0113de3b full-pipeline **0.9395** vs bare 0.73).

**Residual gap (this IS the correct LB-faithful test):** pilkwang (ILP + `best.pt` fusion + heavy postproc)
vs canqiang (own detector + linker) stay different *algorithms* — the gate equalizes split + GT + official
metric + scorer = a fair *whole-pipeline* comparison (what the LB measures). Note: the genuine pipeline runs
detection at `pool_kernel_um=3.0` and controls density via postproc (NOT the pool=5.0 interim lever) —
faithful to what reproduced LB-0.890.
