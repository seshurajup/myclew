"""gm_toolkit_test — data-wise verifier for the cross-cutting GM post-modeling toolkit (offline, synthetic).

Each agent is exercised on real synthetic arrays with a behavioral assertion, not a smoke check:
  • pseudo-label selects the confident test rows (and respects max_frac),
  • blend-optimize returns a blend >= best single model,
  • post-optimize qwk_round improves QWK over naive rounding; clip guards outliers,
  • calibrate reduces (or holds) ECE,
  • target-transform round-trips + survival factorization recombines to a concordant risk score.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import comp_config as CC
from fleet_agents import pseudo_label as PL
from fleet_agents import blend_optimize as BO
from fleet_agents import post_optimize as PO
from fleet_agents import calibrate as CAL
from fleet_agents import target_transform as TT


def _run():
    print("=== GM TOOLKIT DATA-WISE VERIFIER (synthetic) ===")
    rng = np.random.RandomState(1); checks = {}

    # pseudo-label: confident binary test preds
    test_prob = np.concatenate([np.full(30, 0.98), np.full(30, 0.5), np.full(40, 0.02)])
    idx, lab, info = PL.select_pseudo(test_prob, kind="classification", conf_threshold=0.9, max_frac=0.9)
    checks["pseudo_selects_confident"] = info["n_selected"] == 70 and set(np.unique(lab)) <= {0, 1}  # 30@0.98 + 40@0.02
    idx2, _, info2 = PL.select_pseudo(test_prob, kind="classification", conf_threshold=0.9, max_frac=0.3)
    checks["pseudo_respects_maxfrac"] = info2["n_selected"] == 30

    # blend-optimize: 3 OOFs, blend >= best single (AUC)
    yb = rng.randint(0, 2, 600); base = yb + rng.normal(0, 1, 600)
    oof = {"a": base + rng.normal(0, 0.6, 600), "b": base + rng.normal(0, 0.7, 600), "c": rng.normal(0, 1, 600)}
    res = BO.optimize("roc_auc", "max", oof, yb, test_dict={k: oof[k] for k in oof})
    best_single = max(CC.score("roc_auc", yb, oof[k]) for k in oof)
    checks["blend_ge_best_single"] = res["cv"] >= best_single - 1e-9
    checks["blend_has_testpred"] = res["test_pred"] is not None
    print(f"  -> blend best={res['method']} cv={res['cv']} vs single {best_single:.4f}")

    # post-optimize qwk_round beats naive; clip guards
    yq = rng.randint(0, 4, 500); cont = yq + rng.normal(0, 0.5, 500)
    rounded, pinfo = PO.apply("qwk_round", cont, y_true=yq, metric="quadratic_weighted_kappa", n_classes=4)
    naive = CC.score("qwk", yq, np.rint(cont).clip(0, 3))
    checks["post_qwk_beats_naive"] = pinfo["score"] >= naive - 1e-9
    clipped, cinfo = PO.apply("clip", np.array([-999.0, 1.0, 999.0]), y_true=np.array([0.0, 2.0]))
    checks["post_clip_guards"] = clipped.max() < 3 and clipped.min() > -1

    # calibrate reduces/holds ECE
    scores = np.clip(1 / (1 + np.exp(-(base))) + rng.normal(0, 0.05, 600), 0, 1)
    before = CAL.ece(scores, yb); cal = CAL.calibrate(scores, yb, "isotonic"); after = CAL.ece(cal, yb)
    checks["calibrate_improves_ece"] = after <= before + 1e-6
    print(f"  -> calibrate ECE {before:.4f} -> {after:.4f}")

    # target-transform round-trip + survival factorization
    yv = rng.exponential(5, 400)
    yt = TT.forward(yv, "rank_gauss"); back = TT.inverse(yt, yv, "rank_gauss")
    checks["ttrans_roundtrip"] = np.mean(np.abs(np.sort(back) - np.sort(yv))) < 1e-6
    fac = TT.factorize_survival(time=np.array([5, 10, 15, 20.0]), event=np.array([1, 1, 1, 1]))
    # perfect: longer time → higher time-rank → higher recombined risk-score → concordant
    time_rank = np.argsort(np.argsort([5, 10, 15, 20.0])) / 3.0
    risk = fac["recombine"](prob_event=np.ones(4), time_rank=time_rank)
    surv = np.column_stack([[5, 10, 15, 20.0], [1, 1, 1, 1]])
    checks["ttrans_factorize_concordant"] = CC.score("concordance_index", surv, risk) == 1.0

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== gm-toolkit: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
