# Biohub — LOCKED submission-ready artifact (honest ceiling, hack-free ~0.909 pilkwang recipe)

**Status: CLEAN + VALID + VERIFIED locally. READY to push. NOT pushed** (pushing is outward,
rate-limited, and requires the user's explicit go + a daily submission slot). 2026-07-20.

## Artifact paths

- **Kaggle notebook (produces `submission.csv` offline on the hidden test):**
  `research/yuske_clean_0909_kernel/biohub-yuske-clean-0909.ipynb`
  (+ `research/yuske_clean_0909_kernel/kernel-metadata.json` — private, GPU on, internet off,
  competition + `pilkwang/biohub-tracking-support-pack-50ep-v1` + figures dataset sources)
- **Locked local-proxy proof (12 held-out training datasets):**
  `research/yuske_clean_0909_kernel/local_proof/submission.csv`  (md5 `8f3dd1766d7eb85cb9e32a7884f50d4c`)
  + `official_score.json`, `run_stats.csv`, `pipeline_clean.py`, `validate_submission.py`
- **Recipe/env of record:** `docs/yuske_clean_0909_repro.md`

The recipe = pilkwang support-pack `split_0` edge_predictor weights
(md5 `fd822d8723cb3d1fa3139751308fa39e`) + tuned ILP/post-proc baked into notebook cell "recipe env":
det=0.9690, ILP appearance=0.0 / disappearance=1.5, density-adaptive gap-close 5.8µm (gap≤2),
safe-div caps (max 4.66 / sister 8.5 / child 7.65 / frame-frac 0.0076 / global-frac 0.00375),
min-track 6, motion-relink learned-bonus 1.0. This is the yusuketogashi "clean, no metric hacking"
recipe (public hidden-test LB = 0.909) — **not** the metric exploit (never submitted).

## Validation table (local proxy, 12 datasets)

| check | result |
|---|---|
| columns | **OK** — `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id` |
| row count | **595,024 rows** (303,781 node rows), 12 datasets |
| filename | **`submission.csv`** exactly |
| anti-hack: neg-time (t<0) | 0 → PASS |
| anti-hack: off-volume / neg coords | 0 (t∈[0,99], z∈[0,63], y∈[0,253], x∈[0,255]) → PASS |
| anti-hack: artificial hub / fake merge | max in-degree 1, 0 multi-parent → PASS |
| anti-hack: fake-fork | max out-degree 3 (1 node of 303,781 — benign in-volume ILP triple-link, the official metric itself keeps the 2 lowest edge-ids) → PASS |
| **ANTI-HACK GATE** | **PASS** |

## Official score (byte-verified patched metric, 7µm match, score = edge_J + 0.1·div_J)

| group | edge_J | div_J | score |
|---|---|---|---|
| **all 12 (micro)** | **0.9352** | 0.0 | **0.9352** |
| 44b6 (held-out, 6 ds) | 0.9174 | 0.0 | 0.9174 |
| 6bba (in-domain, 6 ds) | 0.9396 | 0.0 | 0.9396 |

Per-dataset (edge_J): 44b6 0.9216/0.9038/0.9718/0.9608/0.8933/1.0000; 6bba 0.9789/0.8385/0.9844/0.9593/0.8849/0.9957.
div_J = 0.0 everywhere (divTP=0, FP=17, FN=8) — division is dead on this sparse founder-lineage GT, so
100% of the score is honest edge_jaccard. Lands squarely in the honest 0.88–0.94 family, **not** a gamed >0.95.
Byte-identical to the independent reproduction in `docs/yuske_clean_0909_repro.md`.

> The 0.9352 local micro is on a 12-dataset held-out training slice and runs above the 0.909 hidden-test
> LB because these datasets are individually strong and division contributes nothing. 0.909 is the
> organizers' hidden-test number and is not locally recomputable.

## NB-preflight (per check)

| check | result |
|---|---|
| offline import-resolve (62 pack wheels, `--no-index --no-deps` in throwaway venv) | **GREEN** — 0 unresolved (tracksdata, zarr, blosc2, polars, IPython, …) |
| path-discovery by content (recursive glob, no hard-coded slug) | **GREEN** |
| GPU / accelerator | **GREEN** — accelerator NvidiaTeslaT4 (2×T4), not a multi-GPU notebook (uses 1 T4, no `invalid device ordinal` risk) |
| comment lines | 9 (informational; not a gate) |
| **VERDICT** | **GREEN — offline-import-safe** |

Push accelerator flag: **`--accelerator NvidiaTeslaT4`** (present/required).

## Notebook hardening applied (defensive, no scoring change)

The local pack clone ships a broken symlink (`repo/weights/ext_test1/split_0`) and 377 stale
`predictions/` geffs; both crash a naive run (broken-symlink copytree; "found N+377" glob mismatch).
The notebook cell that materializes the repo was hardened: `copytree(symlinks=True,
ignore_dangling_symlinks=True)` and a wipe of `REPO_DIR/predictions` after materialize so the
aggregation globs only freshly-inferred test graphs. No-op if the Kaggle-mounted pack omits them.

## Honest verdict

**YES — there is a CLEAN, VALID, VERIFIED submission ready to push.** All gates GREEN:
columns OK, anti-hack PASS (no exploit signatures), official score 0.9352 in the honest family,
nb-preflight GREEN, filename exactly `submission.csv`, accelerator flag NvidiaTeslaT4.

This is the honest-ceiling **~0.909** recipe — competitive, but **NOT top-10** per our bar
(top LB ≈ 0.968). It is the hack-free floor-of-honesty artifact, not a winner.

**Residual (not RED, but disclosed):**
- The hidden-test run was **not** executed on Kaggle (per instruction). End-to-end Kaggle runtime on
  ~199 datasets within 12h is not locally proven, but the identical public yuske pipeline achieved
  0.909 on the hidden test, so it demonstrably fits the budget.
- Local proxy = 12 training datasets; the true 0.909 is the organizers' hidden-test number.

**PUSHING requires the user's explicit go + a daily submission slot. NOT pushed here.**
