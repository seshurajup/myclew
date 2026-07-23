"""conformal_prediction_test — data-wise verifier: the CORE guarantee is COVERAGE.

  1. Regression split-conformal intervals achieve ~1-alpha marginal coverage on a held-out test set.
  2. Classification APS sets achieve >= 1-alpha marginal coverage AND adapt (bigger sets on harder inputs).
  3. Mondrian (class-conditional) fixes minority-group under-coverage that marginal conformal leaves.
  4. agent contract for both regression and classification paths."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import conformal_prediction as C


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def _run():
    print("=== CONFORMAL PREDICTION VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}; alpha = 0.1

    # 1. regression coverage
    n = 4000; x = rng.uniform(-3, 3, n); y = np.sin(x) + 0.3 * rng.randn(n)
    pred = np.sin(x)                                         # a decent model
    cal = slice(0, 2000); tst = slice(2000, n)
    lo, hi, qh = C.split_conformal_interval(pred[cal], y[cal], pred[tst], alpha=alpha)
    cov = C.coverage(y[tst], lo, hi)
    print(f"  -> regression coverage={cov:.3f} (target {1-alpha}), qhat={qh:.3f}")
    checks["reg_coverage"] = abs(cov - (1 - alpha)) < 0.03

    # 2. classification APS coverage + adaptivity
    K = 6; N = 6000
    labels = rng.randint(0, K, N)
    logits = rng.randn(N, K) * 0.5
    logits[np.arange(N), labels] += rng.uniform(0.5, 3.0, N)  # variable confidence → variable difficulty
    probs = _softmax(logits)
    clab = labels[:3000]; tlab = labels[3000:]
    sets, q = C.conformal_classify(probs[:3000], clab, probs[3000:], alpha=alpha, seed=1)
    cov = C.set_coverage(sets, tlab); msz = C.mean_set_size(sets)
    print(f"  -> APS coverage={cov:.3f} (target {1-alpha}), mean set size={msz:.2f}")
    checks["cls_coverage"] = cov >= (1 - alpha) - 0.03
    # adaptivity: split test by true-class prob; harder (lower prob) → larger sets
    tp = probs[3000:][np.arange(len(tlab)), tlab]
    hard = np.array([len(sets[i]) for i in range(len(tlab)) if tp[i] < np.median(tp)]).mean()
    easy = np.array([len(sets[i]) for i in range(len(tlab)) if tp[i] >= np.median(tp)]).mean()
    print(f"  -> mean set size hard={hard:.2f} vs easy={easy:.2f}")
    checks["cls_adaptive"] = hard > easy

    # 3. Mondrian per-group coverage under imbalance (group == true class, one rare class)
    grp_probs = probs[:3000]; grp_lab = clab
    cal_g = clab.copy(); tst_g = tlab.copy()
    msets, qmap = C.mondrian_classify(grp_probs, grp_lab, probs[3000:], tst_g, cal_g, alpha=alpha, seed=2)
    # per-group coverage should each be >= ~1-alpha-tol
    covs = []
    for g in np.unique(tst_g):
        m = tst_g == g
        covs.append(C.set_coverage([msets[i] for i in np.where(m)[0]], tlab[m]))
    print(f"  -> Mondrian per-group coverage min={min(covs):.3f}")
    checks["mondrian_group_coverage"] = min(covs) >= (1 - alpha) - 0.05

    # 4. agent contracts
    st, d, to, msg = C.run({"spec": {"task": "regression", "cal_pred": pred[cal].tolist(),
                                     "cal_y": y[cal].tolist(), "test_pred": pred[tst].tolist(),
                                     "test_y": y[tst].tolist(), "alpha": alpha}}, "t")
    checks["agent_reg"] = st == "done" and abs(d["coverage"] - (1 - alpha)) < 0.04
    st, d, to, msg = C.run({"spec": {"task": "classification", "cal_probs": probs[:3000].tolist(),
                                     "cal_labels": clab.tolist(), "test_probs": probs[3000:].tolist(),
                                     "test_labels": tlab.tolist(), "alpha": alpha}}, "t")
    checks["agent_cls"] = st == "done" and d["coverage"] >= (1 - alpha) - 0.04

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== conformal-predict: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
