"""diffusion_sampler_test — data-wise verifier for the hackable_diffusion torch port (DDPM/DDIM).

Core properties:
  1. schedule: ᾱ monotone decreasing in [0,1], β in (0,1).
  2. q_sample at t=0 ≈ x0 (no noise); predict_x0 inverts q_sample exactly given the true ε.
  3. End-to-end: a denoiser trained on a shifted Gaussian, sampled via BOTH DDPM and DDIM, recovers the mean.
  4. DDIM with few steps is deterministic (same seed → same output).
  5. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))


def _run():
    print("=== DIFFUSION-SAMPLER VERIFIER ===")
    import torch
    from fleet_agents import diffusion_sampler as D
    torch.manual_seed(0); checks = {}

    # 1. schedule
    sched = D.make_schedule(100, kind="cosine")
    ab = sched["abar"]
    checks["abar_monotone"] = bool((ab[1:] <= ab[:-1] + 1e-6).all())
    checks["abar_in_range"] = bool((ab >= 0).all() and (ab <= 1.0 + 1e-6).all())
    checks["beta_valid"] = bool((sched["beta"] > 0).all() and (sched["beta"] < 1).all())

    # 2. q_sample / predict_x0 inversion
    x0 = torch.randn(16, 3)
    xt0, _ = D.q_sample(x0, torch.zeros(16, dtype=torch.long), sched, eps=torch.zeros_like(x0))
    # at t=0 the schedule has abar[0]=1-beta[0]≈0.9999, so x_0 = √abar[0]·x0 (near-identity, not exact)
    checks["qsample_t0_near_identity"] = torch.allclose(xt0, sched["abar"][0].sqrt() * x0, atol=1e-5) \
        and float(sched["abar"][0]) > 0.99
    t = torch.randint(1, 100, (16,))
    eps = torch.randn_like(x0)
    xt, _ = D.q_sample(x0, t, sched, eps=eps)
    x0_rec = D.predict_x0(xt, t, eps, sched)
    checks["predict_x0_inverts"] = torch.allclose(x0_rec, x0, atol=1e-3)
    print(f"  -> q_sample/predict_x0 inversion err {float((x0_rec-x0).abs().max()):.2e}")

    # 3. end-to-end train + sample (both samplers recover the mean)
    st, dta, to, msg = D.run_diffusion({"spec": {"dim": 2, "T": 100, "train_steps": 800}}, "t")
    print(f"  -> ε-loss {dta['loss0']:.3f}->{dta['loss1']:.3f}; DDIM err {dta['err_ddim']:.3f} DDPM err {dta['err_ddpm']:.3f}")
    checks["loss_decreased"] = dta["loss1"] < 0.7 * dta["loss0"]
    checks["ddim_recovers_mean"] = dta["err_ddim"] < 0.4
    checks["ddpm_recovers_mean"] = dta["err_ddpm"] < 0.4

    # 4. DDIM determinism
    sched2 = D.make_schedule(100, kind="cosine")
    m = D.Denoiser(2, hidden=32, T=100)
    torch.manual_seed(7); a = D.sample(m, 8, 2, sched2, sampler="ddim", steps=20)
    torch.manual_seed(7); b = D.sample(m, 8, 2, sched2, sampler="ddim", steps=20)
    checks["ddim_deterministic"] = torch.allclose(a, b)

    checks["agent_done"] = st == "done"

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== diffusion-sampler: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
