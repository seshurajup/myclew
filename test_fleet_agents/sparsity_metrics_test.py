"""sparsity_metrics_test — data-wise verifier for hidden-state sparsity difficulty signal (sparsityLLM).

Core properties:
  1. A SPARSE vector (few large dims) scores higher Gini / higher top-k energy / lower effective-rank than a
     DENSE uniform-ish vector (the defining behaviour the paper relies on).
  2. Metrics match hand-computed values on a tiny known vector (byte-faithful to compute_sparsity_metrics).
  3. difficulty_score ranks dense(hard) examples above sparse(easy) ones; curriculum_order returns easy→hard.
  4. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import sparsity_metrics as S


def _run():
    print("=== SPARSITY-METRICS VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}

    # 1. sparse vs dense separation
    D = 512
    sparse = np.zeros(D); sparse[rng.choice(D, 5, replace=False)] = rng.randn(5) * 5
    dense = rng.randn(D)
    ms, md = S.sparsity_metrics(sparse), S.sparsity_metrics(dense)
    print(f"  -> sparse: gini={ms['gini']:.3f} top5={ms['top5pct_ratio']:.3f} eff={ms['effective_rank']:.3f}")
    print(f"  -> dense : gini={md['gini']:.3f} top5={md['top5pct_ratio']:.3f} eff={md['effective_rank']:.3f}")
    checks["sparse_higher_gini"] = ms["gini"] > md["gini"]
    checks["sparse_higher_top5"] = ms["top5pct_ratio"] > md["top5pct_ratio"]
    checks["sparse_lower_effrank"] = ms["effective_rank"] < md["effective_rank"]

    # 2. hand-check on a known vector: one-hot → max Gini-ish, top1%=1.0, eff-rank=1/D
    onehot = np.zeros(100); onehot[3] = 4.0
    m = S.sparsity_metrics(onehot)
    checks["onehot_top1_full"] = abs(m["top1pct_ratio"] - 1.0) < 1e-9      # all energy in top 1%
    checks["onehot_effrank_min"] = abs(m["effective_rank"] - 1.0 / 100) < 1e-6
    # uniform vector → eff-rank ratio ≈ 1, gini ≈ 0
    uni = np.ones(100)
    mu = S.sparsity_metrics(uni)
    checks["uniform_effrank_one"] = abs(mu["effective_rank"] - 1.0) < 1e-6
    checks["uniform_gini_zero"] = abs(mu["gini"]) < 1e-6
    print(f"  -> one-hot top1={m['top1pct_ratio']:.3f} effrank={m['effective_rank']:.4f} | uniform gini={mu['gini']:.4f}")

    # 3. difficulty score + curriculum ordering
    N = 200
    easy = np.zeros((N // 2, D)); hard = rng.randn(N - N // 2, D)
    for i in range(easy.shape[0]):
        k = rng.randint(3, 10); easy[i, rng.choice(D, k, replace=False)] = rng.randn(k) * 5
    V = np.vstack([easy, hard]); label_hard = np.array([0] * (N // 2) + [1] * (N - N // 2))
    d = S.difficulty_score(V)
    eh, el = d[label_hard == 1], d[label_hard == 0]
    auc = (eh[:, None] > el[None, :]).mean()
    print(f"  -> difficulty separates hard>easy: AUC={auc:.3f}")
    checks["difficulty_separates"] = auc > 0.9
    order = S.curriculum_order(V)
    checks["curriculum_easy_first"] = d[order[0]] <= d[order[-1]]          # ascending difficulty
    checks["curriculum_is_permutation"] = sorted(order.tolist()) == list(range(N))

    # 4. agent contract
    st, dta, to, msg = S.run_sparsity({"spec": {"n": 200, "dim": 512}}, "t")
    checks["agent_done"] = st == "done" and dta["auc"] > 0.9

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== sparsity-metrics: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
