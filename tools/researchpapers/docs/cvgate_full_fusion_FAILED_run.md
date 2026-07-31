# CV-GATE full-fusion run — FAILED (invalid, re-run needed)

**Conclusion (one line):** The decisive full-fusion run (`train-32c5c76251`) is **INVALID** — pipeline.py could not resolve the model artifact (`BIOHUB_MODEL_ARTIFACTS` unset + no `/kaggle/input`), so it produced no fresh `submission.csv`; fold1 got 0 geffs and fold0's `0.8971` came from a **stale Jul-4 CSV** (12 mismatched datasets, div FP=31), not this run. **No valid `pilk_full` numbers → gate verdict cannot be computed.** Job reported `succeeded`/exit 0 only because the runner lacks `set -e` (errors were non-fatal).

## What happened

| Object | expected | actual | valid? |
|---|---|---|---|
| pilk_full_loeodens split_0 | n=8 (44b6) fresh fusion geffs | n=12, **65 missing**, div FP=31, score 0.8971 (from stale Jul-4 `submission_loeodens_f0.csv`) | ❌ contaminated |
| pilk_full_loeodens split_1 | n=7 (6bba) fresh fusion geffs | **0 geffs** — pipeline.py wrote no submission.csv (`submission not found`) | ❌ empty |
| canqiang_full split_0 | n=8 | score **0.7973** (n=8) | ✅ valid |
| canqiang_full split_1 | n=7 | score **0.7879** (n=7) | ✅ valid |

Canqiang official ≈ its proxy (0.7973/0.7879) — good cross-check, but useless without a valid pilk_full opponent.

## Root cause (read-only diagnosis)

- `learning/ensemble_work/pilkwang_full/pipeline.py::find_artifacts_root()` (L457) resolves the model artifact from env `BIOHUB_MODEL_ARTIFACTS`/`BIOHUB_ARTIFACTS`, else `/kaggle/input`; on miss it **raises** `FileNotFoundError` ("Attach the newly uploaded support dataset, or set BIOHUB_MODEL_ARTIFACTS", L493–497).
- `baseline/run_pilk_full_loeodens.sh` exports `CELLMOT_DATA_DIR` and `BIOHUB_ALLOW_ARTIFACT_FALLBACK=1` but **NOT `BIOHUB_MODEL_ARTIFACTS`**, and this host has no `/kaggle/input` → artifact resolution failed → pipeline.py aborted before inference (whole job ran far under the expected ~4 min/fold).
- Fold0 then converted a **pre-existing stale `submission_loeodens_f0.csv` (mtime Jul-4 21:24)** → the misleading 0.8971. Fold0's stale CSV must be cleared so it can't contaminate the re-run.

## Fix (researcher's lane — trainer did NOT modify the script)

1. `export BIOHUB_MODEL_ARTIFACTS=<local support-pack root>` in the runner. Candidate: `research/pilkwang_support_pack` (has `ARTIFACT_MANIFEST.json`); researcher to confirm the exact dir `find_artifacts_root`/`has_model_artifact` accept.
2. Remove/rename the stale `learning/ensemble_work/pilkwang_full/submission_loeodens_f0.csv` (and any stale `submission.csv`) before the re-run so fold0 can't reuse Jul-4 data.
3. Fail-fast: add `set -e` (or per-step guards) so an artifact/pipeline failure aborts instead of silently "succeeding".
4. Re-dry-run must actually exercise pipeline.py's `find_artifacts_root()` + model load (the prior GREEN dry-run evidently stubbed it).

## Evidence

- Job `train-32c5c76251`, log `baseline/train_log.txt`; geffs `research/official_repo/predictions/seshu/pilk_full_loeodens/split_{0,1}` (split_0=12 partial/stale, split_1=0).
- canqiang_full official: split_0 0.7973 (n=8), split_1 0.7879 (n=7).
