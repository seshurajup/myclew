# Auto-marched aug-ablation batch — CONFOUNDED / SUPERSEDED (not journey rows)

**Status: the entire fleet-auto-submitted `config/aug_ablation/{base,contrast,gamma,noise,rot90,...}.yml` batch is INVALID. Do NOT count any of them as kept/rejected data points.**

## Why invalid
1. **Confounded design** (human finding): `base.yml` = brightness+flip (NOT no-aug); every other config bakes brightness+flip in on top of its named aug → none isolate a single augmentation.
2. **Code-level confound** (researcher): `train_from_config` silently defaulted `augment:[]` to brightness — so even an "empty" aug list wasn't a true no-aug reference. Fixed (`augment is not None`) in the clean re-authoring.
3. **Out-of-order + off-mechanism**: auto-submitted by the fleet marcher (`FLEET_AUTO_SUBMIT=1`) to `:7799`, jumping the journey queue, instead of the human's intended one-at-a-time `:7788` POST flow.

## Runs disposed
- `train-9da500d0ec` contrast (train, succeeded) — training-proxy only, never officially scored. See `aug_journey_contrast_CONFOUNDED_superseded.md`.
- `train-fa432a45d5` gamma (train, succeeded) — training-proxy only (acc*recall), div_J=0, never officially scored.
- `train-cf3e228a46` score augabl_noise (predict+score, **failed**) — no official number journaled.
- `train-15670aaf02` rot90 (train, running at time of writing) — confounded, will be superseded.

## Replacement
Superseded by the clean, isolated **aug_journey**: `00_no_aug` (true reference) → `10_crop_scale`/`11_translate_static` (density) → `20/21` geometric → `30–35` photometric, screened on `splits_screen_matched` split0 (+`_k6` for close calls), judged by-embryo (adjJ_44b6/adjJ_6bba), POSTed one-at-a-time to `:7788`. Ledger: mark this batch `observation=confounded, superseded by clean aug_journey`.
