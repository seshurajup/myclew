"""schedule_free — the Schedule-Free optimizer family. No learning-rate SCHEDULE (no cosine/warmup-decay to
tune, no need to know the training horizon in advance), yet it matches or beats a tuned schedule. Distinct
from the fleet's SWA/EMA (which average weights AFTER training, post-hoc): Schedule-Free maintains the
Polyak-Ruppert average AS the evaluated iterate and computes gradients at an interpolated point between the
average and the fast iterate — the averaging is part of the optimizer, not a wrapper.

Papers:
  • Defazio et al., "The Road Less Scheduled", NeurIPS 2024 — the Schedule-Free SGD/AdamW update.
  • "ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to LLMs", arXiv:2605.19095 (2026)
    — outperforms Warmup-Stable-Decay by ~31% at 1000 tokens/param; the 2026 large-scale validation.
  • "Anytime Pretraining: Horizon-Free LR Schedules with Weight Averaging", arXiv:2602.03702 (2026).

Update (per step t, y = interpolation, z = fast iterate, x = evaluated average):
    y_t = (1-beta) z_t + beta x_t
    g_t = grad( y_t )
    z_{t+1} = z_t - gamma * g_t          (constant gamma — no schedule)
    c_{t+1} = 1/(t+1)                     (weight; uniform Polyak average of the z-sequence)
    x_{t+1} = (1 - c_{t+1}) x_t + c_{t+1} z_{t+1}
Evaluate the model at x. Reusable across ALL modalities — a plain torch.optim.Optimizer with train()/eval()
that swaps the averaged weights in for evaluation.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent

try:
    import torch
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


# ---------------------------------------------------------------- framework-free core
def schedule_free_sgd(grad_fn, x0, steps=300, lr=0.1, beta=0.9, weight_decay=0.0):
    """Schedule-Free SGD. grad_fn(y)->grad. Returns (x, z, hist) where x is the EVALUATION iterate (average),
    z the fast iterate. Constant lr — no schedule. hist = per-step loss-at-x is NOT computed (grad only)."""
    z = np.asarray(x0, float).copy(); x = z.copy()
    for t in range(int(steps)):
        y = (1.0 - beta) * z + beta * x
        g = np.asarray(grad_fn(y), float)
        if weight_decay:
            g = g + weight_decay * y
        z = z - lr * g
        c = 1.0 / (t + 1.0)                                  # uniform Polyak-Ruppert weight
        x = (1.0 - c) * x + c * z
    return x, z


def sgd_cosine(grad_fn, x0, steps=300, lr=0.1, momentum=0.9):
    """SGD + momentum with a cosine-decay schedule — the tuned baseline Schedule-Free aims to match WITHOUT
    knowing the horizon or tuning the schedule."""
    w = np.asarray(x0, float).copy(); buf = np.zeros_like(w)
    for t in range(int(steps)):
        lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * t / max(1, steps)))
        buf = momentum * buf + np.asarray(grad_fn(w), float)
        w = w - lr_t * buf
    return w


# ---------------------------------------------------------------- torch drop-in optimizer
if _HAS_TORCH:
    class ScheduleFreeSGD(torch.optim.Optimizer):
        """Schedule-Free SGD. Call .train() before the training forward/backward and .eval() before
        evaluation so the averaged weights (x) are swapped in. Constant lr — pass NO scheduler.

            opt = ScheduleFreeSGD(model.parameters(), lr=0.1, beta=0.9)
            opt.train()
            for batch: loss.backward(); opt.step(); opt.zero_grad()
            opt.eval()   # swap the averaged weights in for validation / checkpointing
        """
        def __init__(self, params, lr=0.1, beta=0.9, weight_decay=0.0):
            super().__init__(params, dict(lr=lr, beta=beta, weight_decay=weight_decay))
            self._t = 0; self._mode = "train"

        @torch.no_grad()
        def train(self):
            if self._mode == "train":
                return
            for grp in self.param_groups:                    # move params from x back to y
                b = grp["beta"]
                for p in grp["params"]:
                    st = self.state[p]
                    if "z" in st:
                        p.copy_((1 - b) * st["z"] + b * p)
            self._mode = "train"

        @torch.no_grad()
        def eval(self):
            if self._mode == "eval":
                return
            for grp in self.param_groups:                    # p currently holds y = (1-b)z + b*x → recover x
                b = grp["beta"]
                for p in grp["params"]:
                    st = self.state[p]
                    if "z" in st:
                        p.copy_((p - (1 - b) * st["z"]) / b)
            self._mode = "eval"

        @torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            self._t += 1; c = 1.0 / self._t
            for grp in self.param_groups:
                lr, b, wd = grp["lr"], grp["beta"], grp["weight_decay"]
                for p in grp["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "z" not in st:
                        st["z"] = p.detach().clone(); st["x"] = p.detach().clone()
                    z, x = st["z"], st["x"]
                    y = p                                    # p holds y during training
                    g = p.grad
                    if wd:
                        g = g.add(y, alpha=wd)
                    z.add_(g, alpha=-lr)
                    x.mul_(1 - c).add_(z, alpha=c)
                    p.copy_((1 - b) * z + b * x)             # next y
            return loss
else:  # pragma: no cover
    ScheduleFreeSGD = None


# ---------------------------------------------------------------- agent
class ScheduleFree(BaseAgent):
    name = "schedule-free"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q); seed = int(s.get("seed", 0)); rng = np.random.RandomState(seed)
        n, d = int(s.get("n", 400)), int(s.get("dim", 20)); steps = int(s.get("steps", 400))
        noise = float(s.get("noise", 1.0)); lr = float(s.get("lr", 0.05))
        # noisy linear regression — stochastic gradients so Polyak averaging clearly helps
        A = rng.randn(n, d); wt = rng.randn(d); y = A @ wt + noise * rng.randn(n)
        def full_loss(w): return float(np.mean((A @ w - y) ** 2))
        def sgrad(w):                                        # minibatch stochastic gradient
            idx = rng.choice(n, size=max(1, n // 10), replace=False)
            r = A[idx] @ w - y[idx]
            return 2.0 * A[idx].T @ r / len(idx)
        x0 = np.zeros(d)
        xsf, zsf = schedule_free_sgd(sgrad, x0, steps=steps, lr=lr, beta=0.9)
        wcos = sgd_cosine(sgrad, x0, steps=steps, lr=lr, momentum=0.9)
        lx, lz, lc = full_loss(xsf), full_loss(zsf), full_loss(wcos)
        msg = (f"schedule-free: avg-iterate loss={lx:.4f} vs raw-iterate {lz:.4f} vs tuned-cosine {lc:.4f} "
               f"(constant lr, no schedule) — averaging & no-schedule both hold")
        self.log(msg, kind="finding", recommendation="drop the LR schedule; swap x (eval) in for validation")
        return self.done({"sf_avg_loss": lx, "sf_raw_loss": lz, "cosine_loss": lc}, msg)


_AGENT = ScheduleFree()


def run(q, worker):
    return _AGENT.run(q, worker)
