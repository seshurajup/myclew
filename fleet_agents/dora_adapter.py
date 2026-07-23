"""dora_adapter — DoRA (Weight-Decomposed Low-Rank Adaptation), the PEFT method that beats plain LoRA at
equal parameter budget. The fleet had `lora-train`/`lora-validate` (plain LoRA) but NO DoRA math — only a
`--use-dora` CLI flag passthrough with no implementation. DoRA decomposes each frozen weight W0 into a
per-output MAGNITUDE vector m and a DIRECTION matrix, and adapts them separately: the direction gets a LoRA
update, the magnitude is a directly-trainable vector. This decouples "how big" from "which way", which is
exactly what plain low-rank LoRA cannot represent (a per-row rescale is full-rank).

Papers:
  • Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024, arXiv:2402.09353 — the method.
  • "DoRAN: Stabilizing DoRA via Noise Injection and Auxiliary Networks", arXiv:2510.04331 (2026 frontier).
  • "OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting", arXiv:2510.13003.

Decomposition (W0 shape [out,in]):  m = ||W0||_row (length out);  V = W0 + scaling*(B@A);
    W' = m[:,None] * V / ||V||_row   → at init (B=0) W' == W0 exactly (identity), then m and BA train.
Reusable: a drop-in nn.Linear replacement for ANY torch model (tabular MLP head, vision conv-as-linear,
transformer projection). Trainable params = out + rank*(in+out) << out*in.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


if _HAS_TORCH:
    class DoRALinear(nn.Module):
        """DoRA-adapted linear layer wrapping a FROZEN base weight W0 (and optional bias)."""
        def __init__(self, weight, bias=None, rank=4, alpha=None):
            super().__init__()
            W0 = weight.detach().clone().float()
            self.register_buffer("W0", W0)
            self.register_buffer("bias", None if bias is None else bias.detach().clone().float())
            out, inp = W0.shape; self.rank = int(rank)
            self.scaling = (alpha / rank) if alpha else 1.0
            # trainable magnitude initialised to the base per-row norm → identity at init
            self.m = nn.Parameter(W0.norm(dim=1).clone())
            # LoRA factors: B zero-init so B@A = 0 at init (direction == W0)
            self.A = nn.Parameter(torch.randn(self.rank, inp) * 0.01)
            self.B = nn.Parameter(torch.zeros(out, self.rank))

        def effective_weight(self):
            V = self.W0 + self.scaling * (self.B @ self.A)
            Vn = V.norm(dim=1, keepdim=True) + 1e-12
            return self.m.unsqueeze(1) * V / Vn

        def forward(self, x):
            return F.linear(x, self.effective_weight(), self.bias)

        def n_trainable(self):
            return int(self.m.numel() + self.A.numel() + self.B.numel())


    class LoRALinear(nn.Module):
        """Plain LoRA baseline (no magnitude decomposition) for comparison: W' = W0 + scaling*(B@A)."""
        def __init__(self, weight, bias=None, rank=4, alpha=None):
            super().__init__()
            W0 = weight.detach().clone().float(); self.register_buffer("W0", W0)
            self.register_buffer("bias", None if bias is None else bias.detach().clone().float())
            out, inp = W0.shape; self.rank = int(rank)
            self.scaling = (alpha / rank) if alpha else 1.0
            self.A = nn.Parameter(torch.randn(self.rank, inp) * 0.01)
            self.B = nn.Parameter(torch.zeros(out, self.rank))

        def effective_weight(self):
            return self.W0 + self.scaling * (self.B @ self.A)

        def forward(self, x):
            return F.linear(x, self.effective_weight(), self.bias)

        def n_trainable(self):
            return int(self.A.numel() + self.B.numel())
else:  # pragma: no cover
    DoRALinear = LoRALinear = None


def _fit(layer, X, Ttarget, steps=400, lr=0.05):
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(); pred = layer(X); loss = F.mse_loss(pred, Ttarget); loss.backward(); opt.step()
    with torch.no_grad():
        return float(F.mse_loss(layer(X), Ttarget))


# ---------------------------------------------------------------- agent
class DoraAdapt(BaseAgent):
    name = "dora-adapt"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        if not _HAS_TORCH:
            return self.escalate(worker, "leader", "dora-adapt: torch unavailable")
        s = self.spec(q); torch.manual_seed(int(s.get("seed", 0)))
        out, inp, rank = int(s.get("out", 16)), int(s.get("in", 16)), int(s.get("rank", 2))
        W0 = torch.randn(out, inp)
        # target = per-row RESCALE of W0 (a full-rank change plain low-rank LoRA can't fit, but DoRA's
        # magnitude vector captures directly) → the canonical DoRA-vs-LoRA separation
        scale = (0.4 + 2.0 * torch.rand(out)).unsqueeze(1)
        Wt = scale * W0
        X = torch.randn(256, inp); Ttarget = X @ Wt.T
        dora = DoRALinear(W0, rank=rank); lora = LoRALinear(W0, rank=rank)
        with torch.no_grad():
            init_err = float(F.mse_loss(dora(X), X @ W0.T))   # identity-at-init check value
            m0 = dora.m.clone()
        ld = _fit(dora, X, Ttarget, steps=int(s.get("steps", 500)))
        ll = _fit(lora, X, Ttarget, steps=int(s.get("steps", 500)))
        m_moved = float((dora.m - m0).abs().mean())
        msg = (f"dora-adapt: rank={rank} per-row-rescale target — DoRA MSE={ld:.4e} vs LoRA MSE={ll:.4e} "
               f"(init-identity err={init_err:.2e}, magnitude moved {m_moved:.3f}, "
               f"trainable={dora.n_trainable()} vs full {out*inp})")
        self.log(msg, kind="finding", recommendation="use DoRA over LoRA when per-channel scaling matters; same rank")
        return self.done({"dora_mse": ld, "lora_mse": ll, "init_identity_err": init_err,
                          "magnitude_moved": m_moved, "n_trainable": dora.n_trainable()}, msg)


_AGENT = DoraAdapt()


def run(q, worker):
    return _AGENT.run(q, worker)
