"""latent_moe_test — data-wise verifier for Kimi-K3 latent-space MoE + MLA (Stable LatentMoE / Gated MLA).

Core properties:
  1. LatentMoE output shape = (B, d_model); it routes through a d_latent bottleneck.
  2. It LEARNS through the compression (loss falls on a synthetic target).
  3. latent_param_ratio < 1 when d_latent < d_model (experts are cheaper), and → ~1 as d_latent → d_model.
  4. MLA KV-cache is smaller than full K/V and shrinks as d_latent shrinks.
  5. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))


def _run():
    print("=== LATENT-MOE VERIFIER ===")
    import torch
    import torch.nn.functional as F
    from fleet_agents import latent_moe as LM
    torch.manual_seed(0); checks = {}
    d_model, d_latent, E, k = 128, 32, 8, 2

    # 1. shape
    moe = LM.LatentMoE(d_model, d_latent, E, k)
    y = moe(torch.randn(10, d_model))
    checks["output_shape"] = tuple(y.shape) == (10, d_model)

    # 2. learns through the bottleneck
    Wt = torch.randn(d_model, d_model) * 0.1
    opt = torch.optim.Adam(moe.parameters(), lr=3e-3); l0 = None
    for _ in range(300):
        x = torch.randn(64, d_model)
        loss = F.mse_loss(moe(x), torch.tanh(x @ Wt))
        opt.zero_grad(); loss.backward(); opt.step()
        if l0 is None:
            l0 = float(loss)
    print(f"  -> latent-MoE loss {l0:.3f} -> {float(loss):.3f}")
    checks["learns"] = float(loss) < 0.6 * l0

    # 3. param ratio
    pr_small = LM.latent_param_ratio(d_model, 32, E)
    pr_full = LM.latent_param_ratio(d_model, d_model, E)
    print(f"  -> expert param ratio: d_latent=32 → {pr_small:.2f}x ; d_latent=d_model → {pr_full:.2f}x")
    checks["latent_cheaper"] = pr_small < 0.5
    checks["ratio_approaches_one"] = pr_full > 1.0     # full-width latent + proj overhead ≥ plain full MoE
    checks["monotone_in_latent"] = LM.latent_param_ratio(d_model, 16, E) < LM.latent_param_ratio(d_model, 64, E)

    # 4. MLA kv-cache
    kv = LM.latent_kv_cache_bytes(4096, n_heads=8, head_dim=16, d_latent=32)
    kv_big = LM.latent_kv_cache_bytes(4096, n_heads=8, head_dim=16, d_latent=64)
    print(f"  -> MLA KV reduction: d_latent=32 → {kv['reduction']:.1f}x")
    checks["mla_smaller"] = kv["mla_bytes"] < kv["full_bytes"] and kv["reduction"] > 1
    checks["mla_scales"] = kv["reduction"] > kv_big["reduction"]

    # 5. agent
    st, dta, to, msg = LM.run_latentmoe({"spec": {"d_model": 128, "d_latent": 32, "steps": 300}}, "t")
    checks["agent_done"] = st == "done" and dta["loss1"] < dta["loss0"] and dta["param_ratio"] < 1.0

    for kk, v in checks.items():
        print(f"  {'OK' if v else 'X'} {kk}")
    ok = all(checks.values())
    print(f"=== latent-moe: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
