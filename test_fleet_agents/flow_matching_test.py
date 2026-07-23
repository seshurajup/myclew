"""flow_matching_test — data-wise verifier for conditional/OT flow matching (ppflow).

Core properties:
  1. sample_conditional_xt: at t=0 → x0, at t=1 → x1; target field u = x1-x0 exactly (the OT-CFM identity).
  2. cfm_loss returns a finite positive scalar and its gradient flows to the network.
  3. End-to-end: training a small velocity net on a shifted Gaussian target and Euler-sampling reproduces
     the target's mean and std (the model actually learns to transport the prior to the data).
  4. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))


def _run():
    print("=== FLOW-MATCHING VERIFIER ===")
    import torch
    from fleet_agents import flow_matching as FM
    torch.manual_seed(0); checks = {}

    # 1. conditional path identities
    B, D = 64, 3
    x0 = torch.randn(B, D); x1 = torch.randn(B, D)
    xt0, u0 = FM.sample_conditional_xt(x0, x1, torch.zeros(B))
    xt1, u1 = FM.sample_conditional_xt(x0, x1, torch.ones(B))
    checks["path_t0_is_x0"] = torch.allclose(xt0, x0, atol=1e-6)
    checks["path_t1_is_x1"] = torch.allclose(xt1, x1, atol=1e-6)
    checks["target_field_is_x1_minus_x0"] = torch.allclose(u0, x1 - x0, atol=1e-6)
    # midpoint linear
    xth, _ = FM.sample_conditional_xt(x0, x1, torch.full((B,), 0.5))
    checks["path_midpoint_linear"] = torch.allclose(xth, 0.5 * (x0 + x1), atol=1e-6)

    # 2. loss is finite + grad flows
    v = FM.VectorField(D, hidden=32)
    loss = FM.cfm_loss(v, x1)
    loss.backward()
    gnorm = sum(p.grad.abs().sum() for p in v.parameters() if p.grad is not None)
    checks["loss_finite_pos"] = bool(torch.isfinite(loss) and loss > 0)
    checks["grad_flows"] = bool(gnorm > 0)
    print(f"  -> cfm_loss={float(loss):.3f}  grad|sum|={float(gnorm):.1f}")

    # 3. end-to-end transport: prior N(0,I) → target N(mu, sd)
    D2 = 2; mu = torch.tensor([3.0, -2.0]); sd = 0.5
    def data(bs): return mu + sd * torch.randn(bs, D2)
    vf = FM.VectorField(D2, hidden=64); opt = torch.optim.Adam(vf.parameters(), lr=2e-3)
    l_first = None
    for _ in range(800):
        opt.zero_grad(); L = FM.cfm_loss(vf, data(256)); L.backward(); opt.step()
        if l_first is None:
            l_first = float(L)
    gen = FM.sample(vf, 3000, D2, steps=60)
    mu_err = float((gen.mean(0) - mu).abs().max()); sd_err = float((gen.std(0) - sd).abs().max())
    print(f"  -> loss {l_first:.3f}->{float(L):.3f}; gen mean={gen.mean(0).tolist()} (tgt {mu.tolist()}), "
          f"mean-err={mu_err:.3f} std-err={sd_err:.3f}")
    checks["loss_decreased"] = float(L) < 0.5 * l_first
    checks["gen_mean_matches"] = mu_err < 0.3
    checks["gen_std_matches"] = sd_err < 0.25

    # 4. agent contract
    st, dta, to, msg = FM.run_flowmatch({"spec": {"dim": 2, "train_steps": 800}}, "t")
    checks["agent_done"] = st == "done" and dta["mean_err"] < 0.4

    for k, val in checks.items():
        print(f"  {'OK' if val else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== flow-matching: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
