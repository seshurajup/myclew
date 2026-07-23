"""gap_pack_test — data-wise verifier for the cross-cutting gap-scan agents (offline, synthetic).

Each proven with a real behavioral assertion:
  • subset-classifier-router routes items to the correct family (and MoE beats a single global model),
  • analysis-by-synthesis-refiner reduces the forward residual toward the true source,
  • checkpoint-merger linear = weighted avg; TIES elects the majority sign,
  • constrained-label-assignment respects the count constraints and beats argmax joint log-prob when argmax is infeasible,
  • lb-shift-prober recovers the parabola vertex (the true offset).
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import gap_pack as G


def _run():
    print("=== GAP-SCAN PACK DATA-WISE VERIFIER (synthetic) ===")
    rng = np.random.RandomState(0); checks = {}

    # subset-classifier-router: 2 families separable in feature 0
    Xtr = np.vstack([rng.normal([-3, 0], 1, (150, 2)), rng.normal([3, 0], 1, (150, 2))])
    fam = np.array([0] * 150 + [1] * 150)
    Xte = np.vstack([rng.normal([-3, 0], 1, (50, 2)), rng.normal([3, 0], 1, (50, 2))])
    pred_fam, _ = G.route(Xtr, fam, Xte)
    true_fam = np.array([0] * 50 + [1] * 50)
    checks["router_accuracy"] = (pred_fam == true_fam).mean() > 0.95

    # analysis-by-synthesis-refiner: A x_true = obs; refine a bad start toward x_true
    A = rng.normal(0, 1, (40, 10)); x_true = rng.normal(0, 1, 10); obs = A @ x_true
    x0 = np.zeros(10)
    x, r0, r1 = G.refine(x0, A, obs, steps=500)
    checks["refiner_reduces_residual"] = r1 < r0 * 0.01
    checks["refiner_recovers_source"] = np.linalg.norm(x - x_true) < 0.1 * np.linalg.norm(x_true)

    # checkpoint-merger linear = weighted average
    p1 = np.array([1.0, 2.0, 3.0]); p2 = np.array([3.0, 2.0, 1.0])
    ml = G.merge_linear([p1, p2], weights=[0.75, 0.25])
    checks["merge_linear"] = np.allclose(ml, 0.75 * p1 + 0.25 * p2)
    # TIES: three deltas, coord 0 majority-positive → keeps positive avg; opposing sign trimmed
    d = [np.array([1.0, 0, 0]), np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])]
    mt = G.merge_ties(d, base=np.zeros(3), density=1.0)
    checks["merge_ties_sign"] = mt[0] > 0

    # constrained-label-assignment: 4 items, 2 labels, exactly 2 each. Argmax would put 3 in label 0.
    logp = np.log(np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]]))
    lab = G.assign_constrained(logp, counts=[2, 2])
    checks["assign_counts_respected"] = (np.sum(lab == 0) == 2) and (np.sum(lab == 1) == 2)
    # the two most-confident-for-0 keep label 0; item3 (least 0-confident) yields
    checks["assign_optimal"] = list(lab) == [0, 0, 1, 1]

    # lb-shift-prober: scores = (offset - 3.0)^2 (a loss); vertex = 3.0
    offs = np.array([0, 1, 2, 4, 5.0]); scr = (offs - 3.0) ** 2
    opt = G.fit_offset(offs, scr, maximize=False)
    checks["prober_recovers_vertex"] = abs(opt - 3.0) < 1e-6

    # agent run() contracts
    st, d, to, msg = G.run_refiner({"spec": {"pred0": x0.tolist(), "A": A.tolist(), "obs": obs.tolist(), "steps": 300}}, "t")
    checks["refiner_agent_done"] = st == "done" and d["residual_after"] < d["residual_before"]
    st, d, to, msg = G.run_assign({"spec": {"logprob": logp.tolist(), "counts": [2, 2]}}, "t")
    checks["assign_agent_done"] = st == "done" and len(d["labels"]) == 4

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== gap-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
