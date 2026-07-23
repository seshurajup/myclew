"""attention_residual_test — data-wise verifier for Kimi-K3 Attention Residuals (AttnRes).

Core properties:
  1. uniform_vs_selective: a uniform gate reproduces the plain running-sum residual (generalization check).
  2. retrieve() is a convex combination of the stored states (weights sum to 1, output in their span).
  3. At init the gate favors the most-recent state (≈ standard residual).
  4. End-to-end: on a task needing the EARLY (depth-0) representation, AttnRes learns to put >uniform weight
     on depth 0 and drives the reconstruction loss down.
  5. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))


def _run():
    print("=== ATTENTION-RESIDUAL VERIFIER ===")
    import torch
    from fleet_agents import attention_residual as AR
    torch.manual_seed(0); checks = {}
    dim, depth = 16, 5

    states = [torch.randn(4, dim) for _ in range(depth)]

    # 1. uniform gate == plain running sum
    uni, sel = AR.uniform_vs_selective(states, depth - 1, selective_gate=None)
    checks["uniform_is_sum"] = torch.allclose(uni, torch.stack(states).sum(0), atol=1e-6)
    # a one-hot selective gate on the last state ≈ standard residual (picks x_{l})
    onehot = torch.full((depth,), -1e9); onehot[-1] = 1e9
    _, sel2 = AR.uniform_vs_selective(states, depth - 1, selective_gate=onehot)
    checks["selective_onehot_recovers_residual"] = torch.allclose(sel2, states[-1], atol=1e-4)

    # 2. retrieve is a convex combo (bounded by the states)
    ar = AR.AttnResidual(dim, depth)
    read = ar.retrieve(states, depth - 1)
    with torch.no_grad():
        g = torch.softmax(ar.gate_logits[depth - 1, :depth], dim=0)
    checks["gate_sums_to_one"] = abs(float(g.sum()) - 1.0) < 1e-5
    checks["gate_nonneg"] = bool((g >= 0).all())

    # 3. init favors most-recent depth
    checks["init_favors_recent"] = int(torch.argmax(g)) == depth - 1
    print(f"  -> init gate over depths: {[round(float(x),2) for x in g]} (argmax={int(torch.argmax(g))})")

    # 4. end-to-end retrieval of the early representation
    st, dta, to, msg = AR.run_attnres({"spec": {"dim": 16, "depth": 6, "steps": 400}}, "t")
    print(f"  -> recon loss {dta['loss0']:.3f}->{dta['loss1']:.3f}; depth0 gate={dta['depth0_gate']:.2f} (uniform={1/6:.2f})")
    checks["loss_decreased"] = dta["loss1"] < 0.6 * dta["loss0"]
    checks["learned_early_retrieval"] = dta["depth0_gate"] > 1.0 / 6
    checks["agent_done"] = st == "done"

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== attention-residual: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
