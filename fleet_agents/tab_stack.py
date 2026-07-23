"""tab-stack — blend the tab-train backends' out-of-fold predictions into one, choosing weights that
MAXIMIZE the CompConfig metric on OOF (hill-climb over the simplex — robust for any metric incl. AUC).
Falls back to the single best backend if blending doesn't beat it. Returns the blended test prediction.

Reuses comp_config.score (metric) and the _preds payload from tab-train. No training here — pure blend.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent
from . import comp_config as CC


def _score(cfg, y, p):
    return CC.score(cfg.metric, y, p)


def _better(cfg, a, b):
    return a > b if cfg.metric_direction == "max" else a < b


def optimize_blend(cfg, preds, y, n_iter=40, init_step=0.25):
    """preds = {backend:{'oof':.., 'test':..}}. Greedy hill-climb weight search on OOF. Returns
    (weights dict, blended_oof, blended_test, cv).
    n_iter: max hill-climb passes over the backend set.
    init_step: starting weight increment (halved when a pass makes no improvement)."""
    names = [b for b in preds if preds[b]["oof"] is not None]
    if not names:                                      # nothing to blend
        return {}, np.zeros(len(y)), None, float("nan")
    oofs = {b: np.nan_to_num(np.asarray(preds[b]["oof"], float), nan=0.0, posinf=0.0, neginf=0.0) for b in names}
    if len(names) == 1:                                # single backend → itself
        b = names[0]
        test = None if preds[b]["test"] is None else np.nan_to_num(np.asarray(preds[b]["test"], float),
                                                                   nan=0.0, posinf=0.0, neginf=0.0)
        return {b: 1.0}, oofs[b], test, _score(cfg, y, oofs[b])
    # single-best baseline
    singles = {b: _score(cfg, y, oofs[b]) for b in names}
    best_single = max(singles, key=lambda b: singles[b] if cfg.metric_direction == "max" else -singles[b])
    w = {b: (1.0 if b == best_single else 0.0) for b in names}
    cur = np.zeros_like(oofs[best_single])
    for b in names:
        cur = cur + w[b] * oofs[b]
    best_cv = _score(cfg, y, cur)
    # hill-climb: repeatedly add a small weight to whichever backend improves OOF most
    step = float(init_step)
    for _ in range(int(n_iter)):
        improved = False
        for b in names:
            trial = {k: v for k, v in w.items()}; trial[b] += step
            tot = sum(trial.values())
            blend = sum((trial[k] / tot) * oofs[k] for k in names)
            cv = _score(cfg, y, blend)
            if _better(cfg, cv, best_cv):
                best_cv, w, improved = cv, trial, True
        if not improved:
            step /= 2
            if step < 1e-3:
                break
    tot = sum(w.values()) or 1.0
    w = {b: w[b] / tot for b in names}
    hc_cv = _score(cfg, y, sum(w[b] * oofs[b] for b in names))

    # also try the grandmaster blenders (Caruana greedy + Nelder-Mead) and keep the best OOF — grounded in
    # s5e11/equity/s5e4/rsna top solutions where these beat plain hill-climb.
    best_w, best_cv, best_method = w, hc_cv, "hillclimb"
    try:
        from . import math_master as MM
        mfn = lambda a, b: _score(cfg, a, b)
        cw, ccv, _ = MM.caruana_ensemble_selection({b: oofs[b] for b in names}, y, mfn,
                                                   maximize=(cfg.metric_direction == "max"))
        if _better(cfg, ccv, best_cv):
            best_w = {b: cw.get(b, 0.0) for b in names}; best_cv, best_method = ccv, "caruana"
        nw, ncv = MM.nelder_mead_weights([oofs[b] for b in names], y, mfn,
                                         maximize=(cfg.metric_direction == "max"))
        if _better(cfg, ncv, best_cv):
            best_w = {b: nw[i] for i, b in enumerate(names)}; best_cv, best_method = ncv, "nelder_mead"
    except Exception:  # noqa: BLE001
        pass
    w = best_w
    blended_oof = sum(w[b] * oofs[b] for b in names)
    have_test = all(preds[b]["test"] is not None for b in names)
    blended_test = (sum(w[b] * np.nan_to_num(np.asarray(preds[b]["test"], float), nan=0.0, posinf=0.0, neginf=0.0)
                        for b in names) if have_test else None)
    return w, blended_oof, blended_test, _score(cfg, y, blended_oof)


class TabStack(BaseAgent):
    name = "tab-stack"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        missing = [k for k in ("config", "_preds", "y") if k not in spec]
        if missing:
            return self.escalate(worker, "leader", f"tab-stack needs spec keys {missing} — none provided")
        cfg = CC.CompConfig.from_dict(spec["config"])
        preds = spec["_preds"]; y = np.asarray(spec["y"])
        w, oof, test, cv = optimize_blend(cfg, preds, y, n_iter=int(spec.get("n_iter", 40)),
                                          init_step=float(spec.get("init_step", 0.25)))
        data = {"weights": w, "blend_cv": cv, "_blend_oof": oof, "_blend_test": test}
        msg = f"tab-stack {cfg.slug}: blend weights={ {k: round(v,3) for k,v in w.items()} } CV({cfg.metric})={cv:.5f}"
        self.log(msg, kind="finding", recommendation="write submission from _blend_test")
        return self.done(data, msg)


_AGENT = TabStack()


def run(q, worker):
    return _AGENT.run(q, worker)
