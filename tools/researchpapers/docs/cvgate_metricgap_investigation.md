# CV-Gate Metric-Gap Investigation (LB ↔ local)

**task_id=CVGATE · researcher · 2026-07-05 · CPU-only, no GPU, no Kaggle, gate result unchanged.**

## RECOMMENDATION (read first)

The LB↔local gap is best explained by **hypothesis (B): wrong provenance / apples-to-oranges pipeline.**
The gate scored a **de-featured pilkwang** — the pack's LB pipeline is "UNet + node-transformer + **ILP**",
but our predict ran with **ILP linking OFF** and the detector's **`pool_kernel_um` at the hardcoded 3.0
instead of the config's 5.0** (trainer's support_pack inventory). **(A) proxy-metric** and **(C)
over-detection** are real but are downstream *symptoms* of (B): pool_kernel 3.0 (not 5.0) is a concrete
mechanism for the over-count (C), and ILP-off is entangled with `div_tp=0` (A) because ILP is the stage
that resolves division structure. Neither flips the ranking through a code bug.

**Single next action (GPU — flag to trainer):** a **fair re-predict** of pilkwang on both folds of
`splits_loeo_density.json` with the **two gaps closed** — `--use-ilp` **ON** and **`pool_kernel_um=5.0`**
— scored with the **official** `research/official_repo/scripts/evaluate.py`, vs canqiang's full pipeline.
(Full-frame `best.pt` fusion is a smaller separate lever, ≈ +0.005; not required for a fair single-detector
comparison. No ensemble needed — pilkwang is single-split by design.) If pilkwang still ≤ canqiang under
the real pipeline + real metric, the density CV itself is not LB-faithful → escalate to human. If it flips
to pilkwang > canqiang, the gate measured a de-featured pipeline and the CV may be salvageable.

Until that runs, **neither local CV (golden-12 nor density-LOEO) is trustworthy for detector choice.**

---

## Thread A — METRIC AUDIT

### A1. What the true competition metric is
- `recipe/.../overview.md`: **official = adjusted edge-Jaccard + 0.1·division-Jaccard**, node match =
  centroid distance within **7 µm**.
- `recipe/.../data.md` line 9 + `start_prompt.md` line 17: the **official scorer is
  `research/official_repo/scripts/evaluate.py`**, which computes via
  `tracking_cellmot.metrics.evaluate` / `per_sample_metrics` / `summarise` (traccuracy-style;
  micro-averaged edge & division Jaccard, `node_recall`, `max_distance`=7 µm).

### A2. What we actually scored the gate with — a PROXY, not the official scorer
The gate ran `baseline/score_v1.py`, which uses **`src/metric.py`** — a hand-reimplemented replica whose
own docstring states: *"The OFFICIAL aggregation weights are not public … treat `score` as a **proxy** and
CALIBRATE component weights against real LB feedback."* So the gate ranking was produced by the proxy, not
by `evaluate.py`. **This is a metric-provenance hole.**

- **Formula match:** the proxy's `official_score` = `adj_edge_jaccard + 0.1·division_jaccard`,
  weight-averaged by `w = edge_tp+fp+fn`, 7 µm one-to-one matching — structurally consistent with the
  public description. The **adjustment** `adj = max(0, jac·(1 − 0.1·(t_pred − t_true)/t_true))` and the
  aggregation are *reconstructed*, not the traccuracy library. Divergence risk is real but is expected to
  be **small for the RANKING** because both detectors are scored by the identical code.

### A3. Why `div_tp = 0` on all 15 datasets — GENUINE, not a broken matcher
GT divisions **do exist** but are extremely sparse (sparse ~4% labels → almost no annotated forks):

| embryo | gt_nodes | gt_edges | GT divisions (out-deg ≥ 2) |
|---|---|---|---|
| 44b6_d754aa59 | 72 | 70 | 1 |
| 44b6_587a1e22 | 381 | 371 | 1 |
| 44b6_3bb3690f | 400 | 396 | 0 |
| 44b6_c8e2a523 | 168 | 164 | 1 |
| 44b6_66f9292d | 200 | 198 | 0 |
| 44b6_a2bb48bb | 416 | 410 | 0 |
| 44b6_8f9ecab4 | 374 | 370 | 0 |
| 44b6_0b24845f | 51 | 49 | 0 |
| 6bba_7d3058ae | 1200 | 1173 | 2 |
| 6bba_283bf9f1 | 1342 | 1288 | 0 |
| 6bba_74686d6a | 586 | 576 | 2 |
| 6bba_ebff6e76 | 1318 | 1264 | 0 |
| 6bba_ebdf3b34 | 847 | 803 | 2 |
| 6bba_b329af44 | 1316 | 1268 | 1 |
| 6bba_05db0fb1 | 1229 | 1183 | 3 |
| **TOTAL** | ~11.1k | ~10.8k | **13** |

- Only **13 GT divisions across ~11k GT nodes** — both detectors miss all 13 → `div_tp=0`.
- This is a **known genuine property**, confirmed by `start_prompt.md` line 23:
  *"Divisions (+0.1× term): DEAD without training — geometric / edge-prob / image / fine-tune all got
  **0 division TP at (1,4,4) resolution**."* The official `evaluate.py` gets the same 0. Both public
  detectors are (1,4,4) → 0 div TP is expected.
- **Entangled with ILP-off (trainer's finding, thread B):** ILP is the linking stage that resolves
  parent/child *division* structure (`division_weight` is used only inside the ILP solver); our gate ran
  **greedy-only** linking, so predicted divisions were heuristic. So `div_tp=0` reflects **both** the
  sparse-(1,4,4) reality **and** the disabled ILP — it is **not** a scorer bug either way.
- **Effect on the ranking:** division_jaccard = 0 for **both** detectors → the +0.1·divJ term contributes
  **+0.0 to each** → it **cannot** explain pilkwang < canqiang. **The gate ranking is 100%
  adj_edge_jaccard** (⇒ thread C).

**Verdict A:** the metric is a *proxy* and should be re-confirmed with the official `evaluate.py`, but the
`div_tp=0` is genuine (sparse labels + ILP-off) and **non-discriminating**. The ranking hinges entirely on
adj_edge_jaccard.

---

## Thread B — PROVENANCE AUDIT

> Source: trainer's read-only support_pack provenance inventory (leader-authorized), verified in code by
> leader (`predict_unet_transformer.py` use_ilp L76, pool_kernel_um L70, ILP solver block L555; all ILP
> deps import in cellmot_venv).

### B1. What produced pilkwang's public LB-0.890
- Pack `README.md`: LB pipeline = **"UNet + node-transformer + ILP submission notebooks."** The genuine
  LB run is an **ILP-based** pipeline. Deps bundle the ILP stack (tracksdata, pyscipopt, ilpy, rustworkx).
- Weights ship **only `unet_transformer/split_0/`** (no split_1..N anywhere) → pilkwang is **single-split
  by design; NO multi-split ensemble** (the `pilk_loeodens/split_1` dir is trainer's fold-1 *output*, not
  a model). config.json: downsample [1,4,4], **`pool_kernel_um=5.0`**, window 2.
- `start_prompt.md`: full-frame `best.pt` **fusion** is a *separate smaller* lever (+0.0068 → LB 0.890);
  the full notebook is extracted at `learning/ensemble_work/pilkwang_full/`.

### B2. What the gate actually ran — a DE-FEATURED pilkwang (two concrete gaps)
`predict_unet_transformer.py --method pilk_loeodens --split F --weights <pack (1,4,4)>` → single split_0
model → our `score_v1.py` (proxy metric). Knobs that **did** match (CLI defaults): 4-way flip det-TTA
(`det_tta=True`), `--det-threshold 0.99`, max-pool NMS, downsample (1,4,4). **Two gaps vs the LB pipeline:**

1. **ILP LINKING OFF — PRIMARY.** `--use-ilp` is `action='store_true'`, default **False**; the gate command
   never passed it → `cfg.use_ilp=False` → the ILP solver block (L555) was **SKIPPED**, falling back to the
   greedy `max_parents=1 / max_children=2` linker. The LB pipeline is explicitly **"+ILP"** (global,
   flow-consistent linking). This is the core apples-to-oranges gap.
2. **`pool_kernel_um = 3.0`, not the config's 5.0.** The predict script reads `config.json` for model
   **architecture only** (downsample/layers/window); `PredictConfig.pool_kernel_um` stayed at the hardcoded
   default **3.0** (run log confirmed 3.0). A smaller max-pool kernel = **less peak suppression = more
   detections** → a concrete mechanism for the observed ~1.34× over-detection (thread C).

### B3. div_tp=0 is entangled with gap 1 (ties back to thread A)
ILP is the stage that resolves parent/child **division** structure (`division_weight` is used only inside
`td.solvers.ILPSolver`). With ILP off, predicted divisions were heuristic → contributes to `div_tp=0`
alongside the sparse-(1,4,4) reality. So `div_tp=0` is **not a scorer bug** — it partly reflects the
disabled ILP.

**Verdict B:** the gate scored a **de-featured pilkwang** — single-split, **ILP-off**, `pool_kernel_um=3.0`
(≠5.0) — against canqiang's **own full post-proc** pipeline. Trainer confirmed the parity fix is
**pilkwang-side only** (canqiang was already a complete pipeline). This is single-predict-vs-full-pipeline,
**not** the pipeline the LB (0.890) saw.

---

## Thread C — POSTPROC / OVER-DETECTION SENSITIVITY

### C1. The over-detection
pilkwang **count_ratio ≈ 1.35 (fold0) / 1.32 (fold1)** at **node_recall 0.99**, at the *genuine* settings
(0.99 threshold + TTA + NMS). So the ~33 % over-count is **intrinsic to the bare detector** on these
embryos, not a threshold misconfiguration. (canqiang: count_ratio 0.72 / 1.05 — near or below true count.)

### C2. Two channels by which over-detection depresses adj_edge_jaccard
1. **Explicit adjustment term** `adj = jac·(1 − 0.1·Δ)`, `Δ = (t_pred − t_true)/t_true`. At Δ=0.35 the
   factor is `1 − 0.035 = 0.965` → only **−3.5 %** directly. **Small.**
2. **Dominant channel:** every spurious node spawns spurious predicted **edges** → `edge_fp ↑` → **raw**
   jaccard falls. This is the large effect and it scales super-linearly with count_ratio (see C3).

### C3. CPU-only estimate — adjJ vs count_ratio across the 15 embryos
Re-scored the **existing** pilkwang gate geffs per-embryo (no re-predict). adj_edge_jaccard vs count_ratio
(`t_pred / estimated_number_of_nodes`), all 15 density-fold test datasets:

| count_ratio | adjJ | | count_ratio | adjJ |
|---|---|---|---|---|
| 0.97 | 0.873 | | 1.24 | 0.926 |
| 1.13 | 0.857 | | 1.26 | 0.817 |
| 1.14 | 0.856 | | 1.30 | 0.857 |
| 1.17 | 0.868 | | 1.32 | 0.773 |
| 1.19 | 0.803 | | 1.37 | 0.883 |
| 1.22 | 0.793 | | 1.56 | 0.619 |
| 1.23 | (—) | | 1.85 | 0.517 |
| — | — | | 1.92 | 0.646 |

- **Linear fit (n=15): adjJ ≈ 1.279 − 0.364·count_ratio, Pearson r = −0.841** (strong negative).
- Current pilkwang mean count_ratio **1.34 → fit predicts adjJ ≈ 0.788** (matches the gate's 0.777–0.788).
- **Extrapolated adjJ at count_ratio = 1.0 ≈ 0.915.** Even conservatively, the **low-over-detection subset
  (count_ratio < 1.3, n=8) already averages adjJ = 0.849** vs the high subset (≥1.3, n=7) at **0.727**.
- Both estimates sit **well above canqiang's 0.79** → if pilkwang's over-detection were brought toward
  1.0, the local ranking very plausibly **flips to pilkwang > canqiang**. (Linear extrapolation past the
  data is an over-estimate; the < 1.3 subset mean 0.849 is the more defensible floor — still > canqiang.)
- **Mechanism link (thread B):** `pool_kernel_um=3.0` instead of the config's 5.0 = a smaller NMS kernel =
  less peak suppression = more detections — a concrete, correctable cause of the 1.34× over-count.

### C4. Can we test "tighten threshold → count_ratio≈1.0 → does the ranking flip?" CPU-only?
**No.** The predicted geffs store **no per-node detection score/confidence** — only `edges/props/edge_prob`
and `edge_dist`; node props are just `t,z,y,x` (verified on
`predictions/seshu/pilk_loeodens/split_0/44b6_0b24845f.geff`; `src/io.read_geff` and
`score_pilkwang.geff_to_dicts` both drop any node score). The detector sigmoid prob lives only transiently
inside the predict script and is discarded at detection time. **Re-thresholding requires re-running the
detector head → a GPU re-predict** (or, better, running the fusion pipeline that controls density).
**FLAG for trainer.**

**Verdict C:** over-detection is the mechanical driver of pilkwang's low local adj_edge_jaccard, but a
*definitive* "would tightening flip it" test needs a GPU re-predict; the CPU regression (C3) gives the
best offline estimate.

---

## Hypothesis ranking

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **B** | **Wrong provenance / pipeline parity** — gate ran a de-featured pilkwang: **ILP-off** + `pool_kernel_um=3.0`(≠5.0) | **PRIMARY** | `--use-ilp` defaulted False → greedy linker; pool kernel hardcoded 3.0; LB pipeline is "+ILP" |
| **C** | Over-detection (count_ratio 1.34) depresses adj_edge_jaccard | **SECONDARY, downstream of B** | r=−0.841; extrapolated adjJ@1.0≈0.915, <1.3 subset 0.849 — both > canqiang 0.79; pool_kernel 3.0 is the cause |
| **A** | Broken/mismatched metric | **TERTIARY** — real but non-discriminating | proxy ≠ official `evaluate.py`; div=0 for **both** (partly ILP-off) & adjEdge structurally matches |

**Conclusion: the CVGATE FAIL is a pipeline-parity ARTIFACT (ILP-off + wrong pool kernel + proxy metric),
NOT demonstrated genuine LB-unfaithfulness of the density CV.** Both CPU threads point the same way: over
the existing geffs, correcting the over-detection alone already lifts pilkwang above canqiang (thread C),
and the over-detection has a concrete, correctable cause (thread B, gap 2). The density CV has **not** been
fairly tested yet.

## Next action — the FAIR re-run (GPU, trainer)

The fair re-run must fix **both** mis-specifications (pipeline **and** metric). Scored under a NEW journal
id **`EXP-CVGATE-FAIR`** (does NOT clobber the original `EXP-CVGATE` sidecars).

- **Interim / minimal signal (wiring DONE, CPU-dry-run GREEN):** the two predict gaps are now closable via
  CLI — `--use-ilp` (existing) + **`--pool-kernel-um 5.0`** (added to
  `research/official_repo/scripts/predict_unet_transformer.py`; default preserved at 3.0, flows into
  `PredictConfig.pool_kernel_um` which detection consumes at L335; validated: flag exposed, config
  constructs, ILP deps + `td.solvers.ILPSolver` import). This still uses the **proxy** metric.
- **Fully-fair (leader-requested) — feasibility scoped:**
  - **canqiang side: READY (SMALL, CPU, DONE).** `baseline/canqiang_csv_to_geff.py` converted the 15
    persisted density-CV prediction CSVs → geffs at `predictions/seshu/canqiang_full/split_{0,1}` (8+7). No
    GPU re-predict needed (the runner had saved node/edge CSVs, not geffs).
  - **official metric: VALIDATED (SMALL).** `evaluate.py` runs once `CELLMOT_DATA_DIR=<…/train>` is set
    (its default `data/dense_channel` is absent); scores an arbitrary `predictions/seshu/<method>/split_F`
    dir; the `dataset_splits.json` "missing" warning is non-blocking. Confirmed on a 1-sample dry-run —
    and the official `evaluate.py` **also drops the division term** ("No divisions present"), corroborating
    thread A that `div_tp=0` is genuine, not a proxy bug.
  - **pilkwang side: MEDIUM + GPU.** ⚠️ correction: the existing `predictions/seshu/pilk_loeodens/split_*`
    geffs are the **bare de-featured gate output** (ILP-off, pool 3.0, count_ratio 1.34) — **NOT** a full
    pipeline. No pilkwang-full geff on the density folds exists. `pilkwang_full/pipeline.py` has no CLI/
    splits support (directory-driven, hardcoded `split_0`, writes `submission.csv` only → needs a
    `score_full.py`-style CSV→geff step). Retargeting golden-12 → the 15 density datasets = stage zarrs into
    two `BIOHUB_TEST_DIR` dirs, run the notebook-script twice (GPU), convert, relabel folds. Not an
    algorithmic lift, but real GPU + wiring.
  - **Residual gap (even fully-fair):** pilkwang (ILP + `best.pt` fusion + heavy postproc) vs canqiang (own
    detector + linker) stay different *algorithms* — the gate equalizes split + GT + metric + scorer, i.e.
    it is a fair *whole-pipeline* comparison, not a same-algorithm one.

Interpretation either way: flip to pilkwang > canqiang → the gate measured a de-featured pipeline, CV
possibly salvageable; still pilkwang ≤ canqiang under the real pipeline + real metric → density CV is not
LB-faithful → escalate to human.
