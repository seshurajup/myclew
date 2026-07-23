"""moe_quantile_balance_test — data-wise verifier for Kimi-K3 Quantile Balancing MoE routing.

Core properties:
  1. Column quantile-normalization makes every expert column ~Uniform(0,1) (mean≈0.5, flat).
  2. On a MISCALIBRATED router (some experts biased high), baseline top-k skews load; quantile balancing
     flattens per-expert load (lower CV, lower max-load, higher utilisation) with NO aux loss.
  3. Balancing preserves signal: within a token's own experts the relative preference order is largely kept.
  4. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import moe_quantile_balance as Q


def _run():
    print("=== MOE QUANTILE-BALANCE VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}
    N, E, k = 512, 16, 2
    bias = rng.randn(E) * 1.5                              # miscalibrated columns
    logits = rng.randn(N, E) + bias

    # 1. quantile-normalization → each column ~Uniform(0,1)
    qn = Q.column_quantiles(logits)
    checks["quantile_range"] = bool(qn.min() >= 0.0 and qn.max() <= 1.0)
    checks["quantile_mean_half"] = bool(np.all(np.abs(qn.mean(axis=0) - 0.5) < 0.02))
    print(f"  -> column-quantile means: {np.round(qn.mean(axis=0)[:6],3)} ... (all ≈0.5)")

    # 2. load balancing without aux loss
    r = Q.aux_free_saving(logits, k)
    print(f"  -> load-CV top-k={r['baseline_cv']:.3f} → qbalance={r['qbalance_cv']:.3f} "
          f"({r['cv_reduction']*100:.0f}% flatter); max-frac {r['baseline_max_frac']:.3f}→{r['qbalance_max_frac']:.3f} "
          f"(ideal {1/E:.3f}); util {r['baseline_util']:.2f}→{r['qbalance_util']:.2f}")
    checks["cv_reduced"] = r["qbalance_cv"] < r["baseline_cv"]
    checks["cv_reduced_big"] = r["cv_reduction"] > 0.3
    checks["maxfrac_reduced"] = r["qbalance_max_frac"] <= r["baseline_max_frac"]
    checks["util_full"] = r["qbalance_util"] >= r["baseline_util"] and r["qbalance_util"] > 0.99
    # near-ideal balance: busiest expert not far above the 1/E ideal
    checks["near_ideal"] = r["qbalance_max_frac"] < 1.6 / E

    # 3. signal preserved (relative to RANDOM routing): quantile balancing deliberately strips the per-expert
    #    scale bias that dominates raw argmax, so the reference is random routing (k/E = 12.5% agreement), not
    #    raw argmax. Balanced routing must retain the raw top choice FAR more often than chance.
    raw_top = np.argmax(logits, axis=1); q_scores = qn
    agree = np.mean([raw_top[i] in np.argsort(-q_scores[i])[:k] for i in range(N)])
    random_baseline = k / E
    print(f"  -> raw-argmax expert still in quantile top-{k}: {agree*100:.0f}% (random={random_baseline*100:.0f}%)")
    checks["signal_beats_random"] = agree > 3.0 * random_baseline

    # 4. agent contract
    st, dta, to, msg = Q.run_qbalance({"spec": {"n_tokens": 512, "n_experts": 16, "k": 2}}, "t")
    checks["agent_done"] = st == "done" and dta["qbalance_cv"] < dta["baseline_cv"]

    for kk, v in checks.items():
        print(f"  {'OK' if v else 'X'} {kk}")
    ok = all(checks.values())
    print(f"=== moe-quantile-balance: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
