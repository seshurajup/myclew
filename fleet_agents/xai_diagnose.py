"""xai_diagnose — the modality-agnostic XAI diagnostic battery that names WHY a solution will underperform,
grounded in the recurring failure modes the 61-comp mining exposed. It is the generalization of xai beyond
biohub: it REUSES math-master (no duplication) — adversarial_auc for drift/CV↔LB risk, expected_calibration
_error for calibration, bootstrap_ci for variance — and returns a ranked list of failure buckets with the
fix (naming the reusable agent that addresses each). Wired into xai.report() as family="diagnose".

Every check maps to a lever the winners used:
  • cv_lb_risk        (adversarial train/test AUC → if separable, local CV won't transfer; build a gap-fold / trust LB)
  • drift             (top adversarially-separating features → distribution shift; reuse adversarial-val)
  • metric_alignment  (is the training objective the competition metric? loss=metric was a repeated lever)
  • calibration       (ECE of predicted probs → for logloss/NLL metrics; fix with `calibrate`)
  • variance_risk     (bootstrap CI width of the CV score → public-LB overfit risk; hedge finals)
  • post_proc_gap     (is metric-specific post-processing applied? QWK-round / clip / quantile-thr)
Pure numpy; scores are honest (reused, tested math-master primitives).
"""
from __future__ import annotations
import numpy as np
from . import math_master as MM
from . import comp_config as CC

# training objective a model class typically optimizes → for metric-alignment flagging
_LOSS_FOR = {"mse": {"rmse", "rmsle", "mae", "r2"}, "logloss": {"logloss", "roc_auc", "average_precision"},
             "rank": {"roc_auc", "average_precision", "spearman_sharpe", "map_at_k"}}


def _san(x):
    """Coerce to a finite float array (nan/inf → 0) so a stray value never poisons a diagnostic."""
    return np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)


def cv_lb_risk(X_train, X_test, hi=0.75, mid=0.6):
    """Adversarial AUC train-vs-test. ~0.5 = CV transfers; high = shift, local CV unreliable → trust LB / gap-fold.
    hi/mid: optional AUC cut-points for the HIGH/MODERATE risk bands."""
    auc = MM.adversarial_auc(list(_san(X_train)), list(_san(X_test)))
    if auc is None:                                        # sklearn absent or degenerate → unknown, not a crash
        return {"adversarial_auc": None, "risk": "UNKNOWN — adversarial classifier unavailable"}
    risk = "HIGH — CV will NOT transfer to LB; build a gap/holdout fold or trust LB" if auc > hi else \
           ("MODERATE — watch CV↔LB" if auc > mid else "LOW — CV should transfer")
    return {"adversarial_auc": round(float(auc), 4), "risk": risk}


def drift(X_train, X_test, feature_names=None):
    """Per-feature train/test KS → top drifting features (reuses math-master ks_pvalue)."""
    Xtr = _san(X_train); Xte = _san(X_test)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1:
        Xte = Xte.reshape(-1, 1)
    n = Xtr.shape[1]; names = feature_names or [f"f{i}" for i in range(n)]
    ks = []
    for i in range(n):
        try:
            p = MM.ks_pvalue(Xtr[:, i], Xte[:, i])
        except Exception:  # noqa: BLE001
            p = 1.0
        ks.append((names[i], round(float(p), 4)))
    ks.sort(key=lambda t: t[1])
    return {"top_drift": ks[:5]}


def metric_alignment(train_loss, comp_metric):
    """Flag when the training objective is NOT aligned with the competition metric (loss=metric lever)."""
    aligned = comp_metric in _LOSS_FOR.get(train_loss, set())
    return {"train_loss": train_loss, "comp_metric": comp_metric, "aligned": aligned,
            "note": "OK" if aligned else f"MISALIGNED — winners optimize the metric; use a {comp_metric}-aligned loss"}


def calibration(y_true, y_prob, thresh=0.05):
    """thresh: ECE above which the probabilities are flagged MISCALIBRATED (default 0.05)."""
    ece = float(MM.expected_calibration_error(_san(y_true), _san(y_prob)))
    return {"ece": round(ece, 4), "note": "calibrated" if ece < thresh else f"MISCALIBRATED (ECE {ece:.3f}) — run `calibrate`"}


def variance_risk(oof_score_samples, wide=0.01):
    """Bootstrap CI width of the CV score across folds/seeds → public-LB overfit / small-sample risk.
    wide: optional CI-width above which the score is flagged high-variance (default 0.01)."""
    x = _san(oof_score_samples)
    if len(x) < 2:                                         # a single fold has no measurable spread
        v = float(x[0]) if len(x) else 0.0
        return {"ci": [v, v], "width": 0.0, "note": "single-sample — no variance estimate"}
    ci = MM.bootstrap_ci(x, stat="mean")
    lo, hi = ci["lo"], ci["hi"]
    width = float(hi - lo)
    return {"ci": [round(float(lo), 5), round(float(hi), 5)], "width": round(width, 5),
            "note": "WIDE — hedge with 2 diverse finals; do not chase public LB" if width > wide else "tight"}


def diagnose(spec):
    """Run whichever checks the provided arrays support. spec keys: X_train,X_test,y,oof_prob,train_loss,
    metric,cv_scores,feature_names. Returns a ranked findings dict."""
    out = {}; flags = []
    metric = spec.get("metric", "unknown")
    if "X_train" in spec and "X_test" in spec:
        out["cv_lb"] = cv_lb_risk(spec["X_train"], spec["X_test"])
        if "HIGH" in out["cv_lb"]["risk"]:
            flags.append("cv_lb_shift")
        out["drift"] = drift(spec["X_train"], spec["X_test"], spec.get("feature_names"))
    if "train_loss" in spec:
        out["metric_alignment"] = metric_alignment(spec["train_loss"], metric)
        if not out["metric_alignment"]["aligned"]:
            flags.append("metric_misaligned")
    if "y" in spec and "oof_prob" in spec:
        out["calibration"] = calibration(spec["y"], spec["oof_prob"])
        if "MISCAL" in out["calibration"]["note"]:
            flags.append("miscalibrated")
    if "cv_scores" in spec:
        out["variance"] = variance_risk(spec["cv_scores"])
        if "WIDE" in out["variance"]["note"]:
            flags.append("high_variance")
    if spec.get("post_processed") is False and metric in ("quadratic_weighted_kappa", "rmse", "rmsle"):
        flags.append("post_proc_gap")
        out["post_proc"] = {"note": f"no metric post-processing — winners apply {'QWK-round' if 'kappa' in metric else 'clip'} (`post-optimize`)"}
    out["flags"] = flags
    out["verdict"] = ("CLEAN — no grounded failure bucket flagged" if not flags
                      else f"{len(flags)} failure bucket(s): {flags} — fix with the named agents before trusting the score")
    return out
