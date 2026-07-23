"""flow_matching — conditional / optimal-transport Flow Matching (CFM) generative objective, lifted from
ppflow (EDAPINENUT/ppflow, ppflow/modules/flows/flow_sampler.py::R3FlowSampler). Flow matching is the modern,
simulation-free way to train a continuous generative model: instead of a many-step diffusion noise schedule,
you regress a network's velocity field onto a STRAIGHT probability path between a prior sample x0 and a data
sample x1. The conditional (OT) path is linear — x_t = t·x1 + (1-t)·x0 — so the target velocity is the CONSTANT
u = x1 - x0, and training is a plain MSE regression. Sampling integrates dx/dt = v(x,t) from t=0 (x0~prior) to
t=1 with a few Euler steps. It is faster and more stable to train than score/diffusion and drop-in for any
continuous target (embeddings, coordinates, tabular latents) — the R3 core here is domain-general.

Primitives (torch only, CPU-fine, offline-testable):
  • sample_conditional_xt(x0, x1, t, sigma) — the linear OT path point x_t and its constant target field u.
  • cfm_loss(v_net, x1, prior, sigma)       — one CFM training loss (MSE of predicted vs target velocity).
  • VectorField(dim, hidden)                — a tiny time-conditioned MLP velocity network v(x,t).
  • sample(v_net, n, dim, steps)            — Euler-integrate the ODE from prior to data (generation).
"""
from __future__ import annotations
from .base import BaseAgent

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


if _HAS_TORCH:
    def sample_conditional_xt(x0, x1, t, sigma: float = 0.0):
        """OT/CFM conditional path (ppflow R3FlowSampler): mu_t = t·x1 + (1-t)·x0, x_t = mu_t + sigma·ε, and
        the constant conditional vector field u = x1 - x0. t is a per-sample scalar in [0,1] shape (B,1).
        Returns (x_t, u)."""
        t = t.view(-1, *([1] * (x1.dim() - 1)))
        mu = t * x1 + (1.0 - t) * x0
        xt = mu + sigma * torch.randn_like(mu) if sigma > 0 else mu
        return xt, (x1 - x0)

    class VectorField(nn.Module):
        """Time-conditioned velocity network v(x,t): concat(x, t) → MLP → same-shape velocity."""
        def __init__(self, dim, hidden=128):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim + 1, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, dim))

        def forward(self, x, t):
            t = t.view(-1, 1).expand(x.shape[0], 1).to(x.dtype)
            return self.net(torch.cat([x, t], dim=-1))

    def cfm_loss(v_net, x1, prior=None, sigma: float = 0.0):
        """One conditional-flow-matching loss for a batch of data x1 (B,D). Draws x0 from `prior` (callable
        (B,D)->tensor; default standard normal) and t~U(0,1), then MSE(v(x_t,t), u). This is the whole training
        objective — no diffusion schedule, no simulation."""
        B = x1.shape[0]
        x0 = prior(x1.shape) if prior is not None else torch.randn_like(x1)
        t = torch.rand(B, device=x1.device)
        xt, u = sample_conditional_xt(x0, x1, t, sigma)
        v = v_net(xt, t)
        return ((v - u) ** 2).mean()

    @torch.no_grad()
    def sample(v_net, n, dim, steps: int = 50, prior=None, device=None):
        """Generate n samples by Euler-integrating dx/dt = v(x,t) from t=0 (x0~prior) to t=1 in `steps` steps."""
        x = prior((n, dim)) if prior is not None else torch.randn(n, dim, device=device)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((n,), k * dt, device=x.device)
            x = x + dt * v_net(x, t)
        return x
else:  # pragma: no cover
    sample_conditional_xt = VectorField = cfm_loss = sample = None


# ---------------------------------------------------------------- agent
class FlowMatching(BaseAgent):
    name = "flow-matching"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        if not _HAS_TORCH:
            return self.escalate(q, "leader", "flow-matching needs torch")
        s = self.spec(q)
        torch.manual_seed(int(s.get("seed", 0)))
        D = int(s.get("dim", 2)); steps = int(s.get("train_steps", 800))
        target_mu = torch.tensor(s.get("target_mu", [3.0, -2.0])[:D], dtype=torch.float32)
        target_sd = float(s.get("target_sd", 0.5))
        def data(bs): return target_mu + target_sd * torch.randn(bs, D)
        v = VectorField(D, hidden=64)
        opt = torch.optim.Adam(v.parameters(), lr=2e-3)
        l0 = None
        for _ in range(steps):
            opt.zero_grad(); loss = cfm_loss(v, data(256)); loss.backward(); opt.step()
            if l0 is None:
                l0 = float(loss)
        gen = sample(v, 2000, D, steps=60)
        mu_err = float((gen.mean(0) - target_mu).abs().max())
        sd_err = float((gen.std(0) - target_sd).abs().max())
        msg = (f"flow-matching: CFM trained {D}-D prior→N({target_mu.tolist()},{target_sd}) — loss {l0:.3f}→{float(loss):.3f}; "
               f"generated mean-err={mu_err:.3f}, std-err={sd_err:.3f}. Simulation-free straight-path generative "
               f"objective (u=x1-x0 MSE), Euler-sampled — drop-in for any continuous target (ppflow OT-CFM)")
        self.log(msg, kind="finding",
                 recommendation="use CFM for continuous generation/augmentation (embeddings, coords, latents): "
                                "regress v(x_t,t) onto x1-x0, integrate to sample — faster/steadier than diffusion")
        return self.done({"loss0": l0, "loss1": float(loss), "mean_err": mu_err, "std_err": sd_err}, msg)


_AGENT = FlowMatching()


def run_flowmatch(q, worker):
    return _AGENT.run(q, worker)
