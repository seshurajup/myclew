# Post-gate frame-capped mini-screen — DESIGN ONLY (do not build/run until CVGATE-FAIR PASSES)

**task_id=CVGATE. Earmark for the POST-GATE idea-bracket screening. No build, no run — this is a plan the
instant the fully-fair gate (`EXP-CVGATE-FAIR`) confirms pilkwang-full > canqiang and the density CV is
declared LB-faithful.** Idea arms remain on HOLD per the human's directive until then.

## Why a frame cap
Idea screening via `baseline/successive_halving.py` ranks each rung on **mini-official** adjJ
(predict → pilk_post → official) over the density-fold test embryos. Predict cost scales ~linearly with the
number of **time frames** per video. The density-fold embryos are long (full T), so a full-video predict per
rung is the dominant screening cost. Capping each video to its **first N frames** cuts predict+score cost
~proportionally, letting the bracket screen many idea arms cheaply and promote only survivors to full T.

This is a **fidelity knob orthogonal** to the two the bracket already has (# embryos, # epochs):
`{embryos} × {epochs} × {frames}`. The frame cap is the cheapest lever to add.

## Capability check (grounded, read-only)
- `predict_unet_transformer.py::predict_video(max_frames=None)` (L303) already supports a hard cap:
  `T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)` (L326).
- **Gap:** `max_frames` is a function param, NOT CLI-wired (argparse has only `--slice`, which selects whole
  videos, not frames). Earmark: expose **`--max-frames`** the same 2-line way `--pool-kernel-um` was added
  (argparse arg → thread into the `predict(...)` call → `predict_video(max_frames=...)`). **Do not implement
  until the gate passes.**

## Proposed recipe (to build post-gate)
1. **Screen set = density-fold test embryos** (`splits_loeo_density.json`) — the validated CV (pending gate).
   Frame-cap is the cost lever; the embryo-disjoint split stays the faithfulness guarantee.
2. **Frame cap N**: start `N ≈ 24–40` frames/video (tune against step 4). Rung ladder becomes e.g.
   `round0: N=24, 6ep → round1: N=48, 18ep → round2: full T, 45ep` (few→many on BOTH frames and epochs).
3. **Wiring earmark** (post-gate build): add `--max-frames` to predict; add a `max_frames:` field to the
   bracket rung spec (`baseline/brackets/screen_loeodens_v1.yml`); thread it through
   `successive_halving.py::run_config → mini_official` so each rung predicts only the capped frames.
4. **MANDATORY faithfulness validation BEFORE trusting it** (same discipline that produced this whole gate):
   a frame-capped mini-official is only usable if it **rank-agrees** with the full-T density-fold official.
   Re-score known-delta arms on the capped set — e.g. **bare-pilkwang vs interim pilk_ilp_k5** (known full-T
   Δ) and any v1 arms — and confirm the capped screen preserves their ordering. If a frame cap N inverts a
   known ordering, raise N (or drop the cap). Never screen ideas on a cap that fails this check — exactly the
   lesson from [[both-local-cvs-invert-lb]] / the CV gate.
5. **Caveats to watch:** capping frames truncates tracks → fewer edges (weaker adjJ signal), fewer/zero
   divisions (div term already a wash at (1,4,4)), and biases toward early-timepoint density. These are why
   step 4's rank-agreement check is non-negotiable, and why the **final judge stays full-T density-fold
   official** (frame cap is a screening proxy only, like mini-official is to golden-12).

## Status
DESIGN ONLY. Gated on `EXP-CVGATE-FAIR` PASS. Ties into the held bracket draft
`baseline/brackets/screen_loeodens_v1.yml` (final judge = both-fold aggregate, approved-in-principle/held).
No idea arms designed or launched until the human/leader unblock post-gate.
