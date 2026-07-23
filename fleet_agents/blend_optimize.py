"""blend-optimize — the reusable ensemble blender used across the top tabular/vision solutions. Given a set
of models' OOF predictions + the truth + the CompConfig metric, it tries ALL the grandmaster blenders
(single-best, hill-climb, Caruana greedy selection, Nelder-Mead convex weights, Ridge linear stack) and
returns the BEST by OOF metric, plus the matching blend of the test predictions. Modality-agnostic — used by
tab-stack and by img/llm ensembles alike. Wraps math-master (no duplication).
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent
from . import comp_config as CC
from . import math_master as MM


def _rank01(a):
    """Rank-transform an array to [0,1] (isic rank-averaging primitive)."""
    a = np.asarray(a, float)
    return np.argsort(np.argsort(a)) / max(len(a) - 1, 1)


def optimize(metric, direction, oof_dict, y, test_dict=None, methods=None, n_restarts=1):
    """Returns {method, weights|coef, cv, test_pred}. oof_dict/test_dict = {name: array}.
    methods: optional subset of {'single','caruana','nelder_mead','ridge','rank_avg'} to try (default all).
    n_restarts: extra random-weight restarts added to the search pool for a more robust convex blend."""
    names = list(oof_dict)
    y = np.nan_to_num(np.asarray(y, float), nan=0.0, posinf=0.0, neginf=0.0) if np.asarray(y).dtype.kind == "f" else np.asarray(y)
    oofs = {n: np.nan_to_num(np.asarray(oof_dict[n], float), nan=0.0, posinf=0.0, neginf=0.0) for n in names}
    mfn = lambda a, b: CC.score(metric, a, b)
    maximize = direction == "max"
    better = (lambda a, b: a > b) if maximize else (lambda a, b: a < b)
    want = set(methods) if methods else {"single", "caruana", "nelder_mead", "ridge", "rank_avg"}

    results = {}
    # single best (always computed — the guaranteed fallback)
    singles = {n: mfn(y, oofs[n]) for n in names}
    sb = max(singles, key=lambda n: singles[n] if maximize else -singles[n])
    results["single"] = ({sb: 1.0}, singles[sb])
    # caruana
    if "caruana" in want:
        try:
            cw, ccv, _ = MM.caruana_ensemble_selection({n: oofs[n] for n in names}, y, mfn, maximize=maximize)
            results["caruana"] = (cw, ccv)
        except Exception:  # noqa: BLE001
            pass
    # nelder-mead (+ optional random restarts)
    if "nelder_mead" in want:
        try:
            nw, ncv = MM.nelder_mead_weights([oofs[n] for n in names], y, mfn, maximize=maximize)
            best_nw, best_ncv = nw, ncv
            if int(n_restarts) > 1 and len(names) > 1:
                rng = np.random.RandomState(0)
                for _ in range(int(n_restarts) - 1):
                    rw = rng.dirichlet(np.ones(len(names)))
                    blend = sum(rw[i] * oofs[n] for i, n in enumerate(names))
                    rcv = mfn(y, blend)
                    if better(rcv, best_ncv):
                        best_nw, best_ncv = rw.round(5).tolist(), round(float(rcv), 6)
            results["nelder_mead"] = ({n: best_nw[i] for i, n in enumerate(names)}, best_ncv)
        except Exception:  # noqa: BLE001
            pass
    # ridge stack
    if "ridge" in want:
        try:
            X = np.column_stack([oofs[n] for n in names])
            coef, inter, oof_pred = MM.ridge_stack(X, y)
            results["ridge"] = ({"__coef__": coef, "__intercept__": inter}, mfn(y, oof_pred))
        except Exception:  # noqa: BLE001
            pass
    # rank-average (isic) — robust for rank/AUC metrics
    if "rank_avg" in want and len(names) > 1:
        try:
            ra = np.mean([_rank01(oofs[n]) for n in names], axis=0)
            results["rank_avg"] = ({n: 1.0 / len(names) for n in names}, mfn(y, ra))
        except Exception:  # noqa: BLE001
            pass

    best_method = max(results, key=lambda m: results[m][1] if maximize else -results[m][1])
    weights, cv = results[best_method]
    test_pred = None
    if test_dict is not None and all(test_dict.get(n) is not None for n in names):
        tests = {n: np.nan_to_num(np.asarray(test_dict[n], float), nan=0.0, posinf=0.0, neginf=0.0) for n in names}
        if best_method == "ridge":
            X = np.column_stack([tests[n] for n in names])
            test_pred = (X @ np.asarray(weights["__coef__"]) + weights["__intercept__"])
        elif best_method == "rank_avg":
            test_pred = np.mean([_rank01(tests[n]) for n in names], axis=0)
        else:
            test_pred = sum(weights.get(n, 0.0) * tests[n] for n in names)
    return {"method": best_method, "weights": weights, "cv": round(float(cv), 6),
            "all_cv": {m: round(float(v[1]), 6) for m, v in results.items()},
            "test_pred": None if test_pred is None else np.asarray(test_pred)}


class BlendOptimize(BaseAgent):
    name = "blend-optimize"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        missing = [k for k in ("oof", "y") if k not in spec]
        if missing:
            return self.escalate(worker, "leader", f"blend-optimize needs spec keys {missing} — none provided")
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else None
        metric = spec.get("metric") or (cfg.metric if cfg else "roc_auc")
        direction = spec.get("direction") or (cfg.metric_direction if cfg else "max")
        res = optimize(metric, direction, spec["oof"], np.asarray(spec["y"]), spec.get("test"),
                       methods=spec.get("methods"), n_restarts=int(spec.get("n_restarts", 1)))
        out = {k: v for k, v in res.items() if k != "test_pred"}
        out["_test_pred"] = None if res["test_pred"] is None else res["test_pred"].tolist()
        msg = f"blend-optimize: best={res['method']} CV({metric})={res['cv']} (all={res['all_cv']})"
        self.log(msg, kind="finding", recommendation=f"use {res['method']} blend; write submission from _test_pred")
        return self.done(out, msg)


_AGENT = BlendOptimize()


def run(q, worker):
    return _AGENT.run(q, worker)
