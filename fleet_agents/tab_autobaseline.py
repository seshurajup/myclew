"""tab-autobaseline — the ONE-CALL turnkey tabular pipeline: profile → CV-train all backends → metric-
optimal blend → write submission.csv. Given only a CompConfig, it lands a competitive default with zero
hand-tuning (the playground-series requirement: speed + a strong default). Everything else (tab-fe, pseudo-
labels, deeper stacking) is refinement layered on top.

Composes the pack agents (does NOT duplicate them): tab_profile.profile, tab_train.train_backends,
tab_stack.optimize_blend, tab_common.write_submission. Reusable across any tabular/sequence comp.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from .base import BaseAgent, COMP
from . import comp_config as CC
from . import tab_common as TC
from . import tab_profile as TP
from . import tab_train as TT
from . import tab_stack as TS


def run_pipeline(cfg, out_path=None, seed=42, fe=True, n_folds=None, gpu=None,
                 early_stopping=False, backends=None, postprocess=True, params=None):
    """One-call turnkey tabular pipeline.
    n_folds/gpu/early_stopping/backends/params: forwarded to tab_train.train_backends.
    postprocess: if True, apply metric-specific post-processing (QWK round / clip guard)."""
    prof = TP.profile(cfg)
    preds, meta = TT.train_backends(cfg, seed=seed, fe=fe, n_folds=n_folds, gpu=gpu,
                                    early_stopping=early_stopping, backends=backends,
                                    params=params)  # GM tab-fe ON by default
    y = meta["y"]
    w, oof, test, blend_cv = TS.optimize_blend(cfg, preds, y)   # GM blender (hill-climb/Caruana/Nelder-Mead)
    single_cv = {b: preds[b]["cv"] for b in preds}
    # grounded metric-specific post-processing (child-mind QWK-round; s5e4 clip-guard)
    post = None
    if postprocess:
        try:
            from . import post_optimize as PO
            op = PO.auto_op(cfg.metric)
            if op == "qwk_round":
                oof_r, info = PO.apply("qwk_round", oof, y_true=y, metric=cfg.metric)
                if test is not None:
                    test, _ = PO.apply("qwk_round", test, y_true=y, metric=cfg.metric)
                blend_cv = CC.score(cfg.metric, y, oof_r); post = info
            elif op == "clip" and test is not None:
                test = PO.apply("clip", test, y_true=y)[0]; post = {"op": "clip"}
        except Exception:  # noqa: BLE001
            pass
    result = {"profile": prof, "per_backend_cv": single_cv, "blend_weights": w, "blend_cv": blend_cv,
              "n_train": meta["n_feats"], "backends": list(preds)}
    sub_path = None
    if test is not None and meta["test_ids"] is not None:
        out = out_path or (COMP / "config" / "_auto" / f"submission_{cfg.slug}.csv")
        sub_path = TC.write_submission(cfg, meta["test_ids"], test, out)
        result["submission"] = sub_path
    result["_oof"] = oof
    return result


class TabAutobaseline(BaseAgent):
    name = "tab-autobaseline"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        if "config" not in spec and "config_file" not in spec:
            return self.escalate(worker, "leader", "tab-autobaseline needs spec keys ['config' or 'config_file'] — none provided")
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else CC.CompConfig.load(spec["config_file"])
        res = run_pipeline(cfg, out_path=spec.get("out"), seed=int(spec.get("seed", 42)),
                           fe=bool(spec.get("fe", True)), n_folds=spec.get("n_folds"), gpu=spec.get("gpu"),
                           early_stopping=bool(spec.get("early_stopping", False)),
                           backends=spec.get("backends"), postprocess=bool(spec.get("postprocess", True)))
        # record a CV-ranked row so the turnkey result shows up in the journal, gated by an eval-set tag
        self.record(change=f"tab-autobaseline {cfg.slug}", cv=res["blend_cv"],
                    description=f"blend={res['blend_weights']} backends={res['backends']}",
                    train_set=f"cv:{cfg.cv_scheme}", kept=None)
        msg = (f"tab-autobaseline {cfg.slug}: backends={res['backends']} single_CV="
               f"{ {b: round(v,5) for b,v in res['per_backend_cv'].items()} } → blend CV({cfg.metric})="
               f"{res['blend_cv']:.5f}" + (f"; wrote {Path(res['submission']).name}" if res.get("submission") else ""))
        self.log(msg, kind="finding",
                 recommendation="human-gated submit; refine with tab-fe/pseudo before pushing if margin is thin")
        return self.done({k: v for k, v in res.items() if not k.startswith("_")}, msg)


_AGENT = TabAutobaseline()


def run(q, worker):
    return _AGENT.run(q, worker)
