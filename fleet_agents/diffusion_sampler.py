"""diffusion_sampler — the portable CORE of Google's hackable_diffusion (a JAX/Flax educational diffusion
toolbox), reimplemented pure-torch. We do NOT adopt the framework (it drags in jax+flax+kauldron, which fights
our torch-first CUDA/5090 stack); we lift its load-bearing algorithm: Gaussian diffusion with an
initialize→update→finalize sampling loop, the noise/time schedule, and the score↔velocity↔x0↔eps
parameterization conversions that hackable_diffusion's gaussian_step_sampler is built around. This complements
`flow_matching` (which is the ODE / straight-path / OT view of generative transport) with the SDE / DDPM view:
a variance-preserving forward noising q(x_t|x_0)=√ᾱ_t x_0 + √(1-ᾱ_t) ε, a learned denoiser, and two reverse
samplers — stochastic DDPM (ancestral) and deterministic DDIM. Same trainer can be viewed either way; having
both lets the fleet pick stochastic (diverse) vs deterministic-few-step (fast) sampling.

Primitives (torch):
  • make_schedule(T, kind)        — β/α/ᾱ noise schedule (linear or cosine).
  • q_sample(x0, t, sched, eps)   — forward: noise x0 to timestep t.
  • predict_x0(x_t, t, eps, sched)/ to_eps / to_velocity — parameterization conversions.
  • ddpm_step / ddim_step         — one reverse update (stochastic / deterministic).
  • Denoiser(dim, hidden)         — tiny ε-prediction network; diffusion_loss trains it.
  • sample(model, n, dim, sched, sampler) — full reverse loop (initialize→update→finalize).
"""
from __future__ import annotations
import math
from .base import BaseAgent

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


if _HAS_TORCH:
    def make_schedule(T=200, kind="cosine", beta0=1e-4, beta1=0.02):
        """Return dict of β,α,ᾱ (alpha_bar) over T steps. 'cosine' (Nichol&Dhariwal) or 'linear' (DDPM)."""
        if kind == "cosine":
            s = 0.008
            f = torch.cos(((torch.arange(T + 1) / T + s) / (1 + s)) * math.pi / 2) ** 2
            abar_full = f / f[0]                                  # length T+1
            beta = (1 - abar_full[1:] / abar_full[:-1]).clamp(1e-8, 0.999)   # length T
        else:
            beta = torch.linspace(beta0, beta1, T)
        alpha = 1 - beta
        abar = torch.cumprod(alpha, dim=0)                        # length T, CONSISTENT: alpha[t]=abar[t]/abar[t-1]
        return {"beta": beta, "alpha": alpha, "abar": abar, "T": T}

    def q_sample(x0, t, sched, eps=None):
        """Forward noising: x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε. t: (B,) long. Returns (x_t, eps)."""
        eps = torch.randn_like(x0) if eps is None else eps
        ab = sched["abar"].to(x0.device)[t].unsqueeze(-1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps, eps

    def predict_x0(x_t, t, eps, sched):
        """Recover x0 estimate from x_t and predicted ε: x0 = (x_t - √(1-ᾱ) ε)/√ᾱ."""
        ab = sched["abar"].to(x_t.device)[t].unsqueeze(-1)
        return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-8)

    def to_velocity(x0, eps, t, sched):
        """v-parameterization (Salimans&Ho): v = √ᾱ ε − √(1-ᾱ) x0."""
        ab = sched["abar"][t].unsqueeze(-1).to(x0.device)
        return ab.sqrt() * eps - (1 - ab).sqrt() * x0

    @torch.no_grad()
    def ddpm_step(model, x_t, t, sched, x0_clip=5.0):
        """One STOCHASTIC (ancestral) reverse step via the STABLE posterior form: predict x0, clamp it, then
        use the closed-form q(x_{t-1}|x_t,x0) posterior mean/variance. This avoids the ε-form's division by
        √α (which blows up to NaN when a cosine schedule pushes α→0.001 at large t)."""
        dev = x_t.device
        B = x_t.shape[0]; tt = torch.full((B,), t, dtype=torch.long, device=dev)
        eps = model(x_t, tt)
        beta = sched["beta"].to(dev)[t]; alpha = sched["alpha"].to(dev)[t]
        ab = sched["abar"].to(dev)[t]
        ab_prev = sched["abar"].to(dev)[t - 1] if t > 0 else torch.tensor(1.0, device=dev)
        x0 = ((x_t - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-8)).clamp(-x0_clip, x0_clip)
        coef_x0 = (ab_prev.sqrt() * beta) / (1 - ab).clamp_min(1e-8)
        coef_xt = (alpha.sqrt() * (1 - ab_prev)) / (1 - ab).clamp_min(1e-8)
        mean = coef_x0 * x0 + coef_xt * x_t
        if t > 0:
            var = (beta * (1 - ab_prev) / (1 - ab).clamp_min(1e-8)).clamp_min(0)
            return mean + var.sqrt() * torch.randn_like(x_t)
        return mean

    @torch.no_grad()
    def ddim_step(model, x_t, t, t_prev, sched, x0_clip=5.0):
        """One DETERMINISTIC (DDIM, η=0) reverse step from t → t_prev. The x0 estimate is clamped (static
        thresholding) — at large t, √ᾱ is tiny under a cosine schedule and the raw x0 estimate explodes."""
        B = x_t.shape[0]; tt = torch.full((B,), t, dtype=torch.long, device=x_t.device)
        eps = model(x_t, tt)
        ab = sched["abar"].to(x_t.device)[t]
        ab_prev = sched["abar"].to(x_t.device)[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        x0 = ((x_t - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-8)).clamp(-x0_clip, x0_clip)
        return ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps

    class Denoiser(nn.Module):
        """Tiny ε-prediction net conditioned on a sinusoidal timestep embedding."""
        def __init__(self, dim, hidden=128, T=200):
            super().__init__()
            self.T = T
            self.net = nn.Sequential(nn.Linear(dim + 16, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim))

        def temb(self, t):
            t = t.float().unsqueeze(-1) / self.T
            freqs = torch.exp(torch.linspace(0, math.log(1000), 8, device=t.device))
            return torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)

        def forward(self, x, t):
            return self.net(torch.cat([x, self.temb(t)], dim=-1))

    def diffusion_loss(model, x0, sched):
        """Simple ε-prediction loss: sample t, noise x0, regress ε (the DDPM training objective)."""
        B = x0.shape[0]
        t = torch.randint(0, sched["T"], (B,), device=x0.device)
        x_t, eps = q_sample(x0, t, sched)
        return F.mse_loss(model(x_t, t), eps)

    @torch.no_grad()
    def sample(model, n, dim, sched, sampler="ddim", steps=50, device=None):
        """Reverse loop from x_T~N(0,I). sampler='ddpm' (stochastic, full T) or 'ddim' (deterministic, `steps`)."""
        x = torch.randn(n, dim, device=device)
        T = sched["T"]
        if sampler == "ddpm":
            for t in reversed(range(T)):
                x = ddpm_step(model, x, t, sched)
        else:
            ts = torch.linspace(T - 1, 0, steps).long().tolist()
            for i, t in enumerate(ts):
                t_prev = ts[i + 1] if i + 1 < len(ts) else -1
                x = ddim_step(model, x, t, t_prev, sched)
        return x
else:  # pragma: no cover
    make_schedule = q_sample = predict_x0 = to_velocity = ddpm_step = ddim_step = None
    Denoiser = diffusion_loss = sample = None


# ---------------------------------------------------------------- agent
class DiffusionSampler(BaseAgent):
    name = "diffusion-sampler"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        if not _HAS_TORCH:
            return self.escalate(q, "leader", "diffusion-sampler needs torch")
        s = self.spec(q)
        torch.manual_seed(int(s.get("seed", 0)))
        dim = int(s.get("dim", 2)); T = int(s.get("T", 100)); steps = int(s.get("train_steps", 800))
        mu = torch.tensor(s.get("target_mu", [2.5, -1.5])[:dim]); sd = float(s.get("target_sd", 0.4))
        sched = make_schedule(T, kind="cosine")
        model = Denoiser(dim, hidden=64, T=T)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3); l0 = None
        for _ in range(steps):
            x0 = mu + sd * torch.randn(256, dim)
            loss = diffusion_loss(model, x0, sched)
            opt.zero_grad(); loss.backward(); opt.step()
            if l0 is None:
                l0 = float(loss)
        g_ddim = sample(model, 2000, dim, sched, sampler="ddim", steps=40)
        g_ddpm = sample(model, 2000, dim, sched, sampler="ddpm")
        err_ddim = float((g_ddim.mean(0) - mu).abs().max()); err_ddpm = float((g_ddpm.mean(0) - mu).abs().max())
        msg = (f"diffusion-sampler: DDPM/DDIM trained (ε-loss {l0:.3f}→{float(loss):.3f}) on N({mu.tolist()},{sd}); "
               f"DDIM-40step mean-err={err_ddim:.3f}, DDPM-{T}step mean-err={err_ddpm:.3f}. Torch port of "
               f"hackable_diffusion's Gaussian sampler (SDE/DDPM view; complements flow-matching's ODE view) — "
               f"stochastic-diverse vs deterministic-fast sampling, no JAX")
        self.log(msg, kind="finding",
                 recommendation="use diffusion-sampler for generative augmentation needing DDPM diversity or "
                                "few-step DDIM; flow-matching for straight-path OT. Both pure-torch on the 5090")
        return self.done({"loss0": l0, "loss1": float(loss), "err_ddim": err_ddim, "err_ddpm": err_ddpm}, msg)


_AGENT = DiffusionSampler()


def run_diffusion(q, worker):
    return _AGENT.run(q, worker)
