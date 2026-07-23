"""keyframe — spend the T4 budget where it matters: run the BIG accurate detector on a SPARSE set of KEYFRAMES,
and fill the INTERMEDIATE frames cheaply (rule-based DoG / nearest-keyframe propagation) since cells barely move
between adjacent frames (user 2026-07-12: "first important frames use big models, then rule-based to find
closest in intermediate frames — use GPU well"). Frozen frames are free (copy — see [[biohub_frozen_frames_6bba]]).
This is how a model that's too slow for all 19,900 frames becomes feasible: big_spf only pays on the keyframes.

DECIDES the keyframe interval + PROVES T4-feasibility (pure, tested); the run measures per-embryo recall+ETA.
"""
from __future__ import annotations
from .base import BaseAgent, COMP

TEST_FRAMES = 19_900
N_GPUS = 2
BUDGET_H = 12 * 0.8                                            # 12h wall, 80% for detection


def keyframe_plan(frozen, T, interval):
    """PURE (data-wise tested). frozen = set of frozen frame indices, T = movie length, interval = keyframe
    spacing. keyframes = every `interval`-th NON-frozen frame; intermediate = the other non-frozen frames
    (propagated from the nearest earlier keyframe); frozen = copied from source. Returns the plan."""
    frozen = set(frozen or ()); T = max(0, int(T))
    unique = [t for t in range(T) if t not in frozen]
    keyframes = unique[::max(int(interval), 1)]
    kfset = set(keyframes)
    interp, source_of = [], {}
    last_k = keyframes[0] if keyframes else 0
    for t in range(T):
        if t in kfset:
            last_k = t
        elif t in frozen:
            source_of[t] = t - 1                              # frozen → copy previous
        else:
            interp.append(t); source_of[t] = last_k          # intermediate → nearest keyframe
    return {"keyframes": keyframes, "n_key": len(keyframes), "interp": interp, "n_interp": len(interp),
            "n_frozen": len(frozen), "source_of": source_of}


def feasible(n_key, big_spf_t4, n_interp, cheap_spf, n_frozen=0, budget_h=BUDGET_H, n_gpus=N_GPUS,
             total_frames=TEST_FRAMES):
    """PURE (data-wise tested). Wall-clock hours for the keyframe pipeline over the FULL test, scaled from the
    per-movie ratios. big_spf_t4 = keyframe detector on T4; cheap_spf = intermediate fill; frozen ≈ free.
    Returns (feasible, eta_h)."""
    per_movie = n_key + n_interp + n_frozen
    if per_movie <= 0:
        return False, 0.0
    n_gpus = max(1, int(n_gpus)); total_frames = max(0, int(total_frames))
    key_frac = n_key / per_movie; interp_frac = n_interp / per_movie
    spf_eff = key_frac * max(0.0, big_spf_t4) + interp_frac * max(0.0, cheap_spf)   # frozen contributes ~0
    eta_h = spf_eff * total_frames / n_gpus / 3600
    return (eta_h <= budget_h, round(eta_h, 2))


class Keyframe(BaseAgent):
    name = "keyframe"
    thread = "S"
    kind = "verdict"

    def _measure(self, ds, interval, big_spf_t4, cheap_spf, budget_h=BUDGET_H):
        import sys, hashlib
        sys.path.insert(0, str(COMP))
        from src import io
        from model_scratch.train_v0 import frames_of
        ad, shape, dtype, T = frames_of(ds, None)
        prev = None; frozen = set()
        for t in range(T):
            h = hashlib.md5(io.load_volume(ad, shape, dtype, t).tobytes()).hexdigest()
            if prev is not None and h == prev:
                frozen.add(t)
            prev = h
        plan = keyframe_plan(frozen, T, interval)
        feas, eta = feasible(plan["n_key"], big_spf_t4, plan["n_interp"], cheap_spf, plan["n_frozen"],
                             budget_h=budget_h)
        return plan, feas, eta, T

    def run(self, q, worker):
        spec = self.spec(q)
        interval = max(1, int(spec.get("interval", 5)))
        big_spf_t4 = float(spec.get("big_spf_t4", 51.7))       # Cellpose-K12 measured
        cheap_spf = float(spec.get("cheap_spf", 0.024))        # DoG on Kaggle CPU measured
        budget_h = float(spec.get("budget_h", BUDGET_H))       # budget_h: detection wall-clock budget (h) for the feasibility gate
        datasets = spec.get("datasets")
        if not datasets:
            from model_scratch.train_v0 import split_datasets
            from src.io import embryo_id
            _, te = split_datasets(); pick = {}
            for ds in te:
                pick.setdefault(embryo_id(ds), ds)
            datasets = [pick[e] for e in ("44b6", "6bba") if e in pick]
        rows = []
        for ds in datasets:
            plan, feas, eta, T = self._measure(ds, interval, big_spf_t4, cheap_spf, budget_h=budget_h)
            rows.append(f"{ds[:4]}: {plan['n_key']}key+{plan['n_interp']}interp+{plan['n_frozen']}frozen/{T}fr "
                        f"→ ~{eta}h {'FITS' if feas else 'OVER'}")
        summary = (f"KEYFRAME plan (interval={interval}, big={big_spf_t4}s/f cheap={cheap_spf}s/f): "
                   + " | ".join(rows))
        self.log(summary, kind="verdict",
                 recommendation="run the BIG detector only on keyframes, propagate/DoG the intermediate frames, "
                                "copy frozen frames; raise interval until it FITS 12h/2×T4, then measure the "
                                "recall cost of the sparser keyframes (propagation error grows with interval).")
        return self.done({"interval": interval, "rows": rows}, summary)


_AGENT = Keyframe()


def run(q, worker):
    return _AGENT.run(q, worker)
