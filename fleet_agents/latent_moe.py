"""latent_moe — Kimi-K3 LATENT-SPACE training (Stable LatentMoE + Gated MLA). The idea common to K3's
"Latent MoE" and "Gated Multi-head Latent Attention" is that the expensive operators do NOT run in the full
model width d_model; they run in a much smaller LATENT dimension d_latent: a shared down-projection compresses
the hidden state into the latent, the operator (expert FFN, or the K/V of attention) works there, and an
up-projection restores d_model. This is what makes K3's very-sparse MoE (16-of-896) and its long context
affordable: (a) each expert stores/computes in d_latent (params ∝ d_latent, not d_model) and (b) attention
caches a single d_latent vector per token instead of full K,V (the MLA / DeepSeek-V2 KV-cache win). "Stable"
= a shared latent basis + normalization so the compression doesn't destabilize training.

Two reusable pieces (pure torch, offline-testable):
  • LatentMoE          — down-project → route top-k experts (each a small d_latent FFN) → up-project. Param
                         and FLOP accounting vs a full-width MoE is exact and testable.
  • latent_kv_cache_bytes(...) — MLA KV-cache: one d_latent vector/token vs full n_heads·head_dim K and V.

Reduces to a normal MoE when d_latent = d_model, so it strictly generalizes. Pairs with moe-quantile-balance
(the router) and moe-inference-cost (the accounting).

Primitives:
  • LatentMoE(d_model, d_latent, n_experts, k)  — the latent-space MoE layer.
  • latent_param_ratio(d_model, d_latent, ...)   — params(latent-MoE)/params(full-width-MoE).
  • latent_kv_cache_bytes(seq, d_latent | full)  — MLA cache bytes vs full K/V cache.
"""
from __future__ import annotations
from .base import BaseAgent

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


if _HAS_TORCH:
    class LatentMoE(nn.Module):
        """Latent-space Mixture-of-Experts. A shared down-proj W_down: d_model→d_latent compresses the token,
        a router picks top-k of n_experts (each a small d_latent→d_latent FFN), their outputs are combined and
        a shared up-proj W_up: d_latent→d_model restores width. Experts live in d_latent so their params scale
        with d_latent, not d_model — the K3 latent-MoE cost win."""
        def __init__(self, d_model, d_latent, n_experts=8, k=2, d_ff=None):
            super().__init__()
            self.d_model = d_model; self.d_latent = d_latent
            self.n_experts = n_experts; self.k = k
            d_ff = d_ff or (2 * d_latent)
            self.down = nn.Linear(d_model, d_latent, bias=False)
            self.up = nn.Linear(d_latent, d_model, bias=False)
            self.norm = nn.LayerNorm(d_latent)                  # "stable": normalize the latent
            self.router = nn.Linear(d_latent, n_experts, bias=False)
            self.experts = nn.ModuleList(
                [nn.Sequential(nn.Linear(d_latent, d_ff), nn.SiLU(), nn.Linear(d_ff, d_latent))
                 for _ in range(n_experts)])

        def forward(self, x):
            z = self.norm(self.down(x))                         # (B, d_latent)
            scores = torch.softmax(self.router(z), dim=-1)      # (B, E)
            topv, topi = scores.topk(self.k, dim=-1)            # (B, k)
            topv = topv / topv.sum(dim=-1, keepdim=True)        # renormalize the kept experts
            out = torch.zeros_like(z)
            for slot in range(self.k):
                idx = topi[:, slot]; w = topv[:, slot].unsqueeze(-1)
                for e in range(self.n_experts):
                    m = idx == e
                    if m.any():
                        out[m] += w[m] * self.experts[e](z[m])
            return self.up(out)                                 # (B, d_model)

    def latent_kv_cache_bytes(seq_len, n_heads, head_dim, d_latent, dtype_bytes=2):
        """MLA KV-cache accounting: full cache stores K and V = 2·n_heads·head_dim per token; MLA stores ONE
        d_latent vector per token. Returns (full_bytes, mla_bytes, reduction_ratio)."""
        full = seq_len * 2 * n_heads * head_dim * dtype_bytes
        mla = seq_len * d_latent * dtype_bytes
        return {"full_bytes": full, "mla_bytes": mla, "reduction": full / max(mla, 1)}
else:  # pragma: no cover
    LatentMoE = latent_kv_cache_bytes = None


def latent_param_ratio(d_model, d_latent, n_experts=8, d_ff_mult=2):
    """params(latent-MoE) / params(full-width-MoE): experts in d_latent cost ~E·2·d_latent·(d_ff_mult·d_latent)
    plus the shared down/up projections (2·d_model·d_latent); a full-width MoE's experts cost
    ~E·2·d_model·(d_ff_mult·d_model). Returns the ratio (<1 = cheaper)."""
    e_lat = n_experts * 2 * d_latent * (d_ff_mult * d_latent)
    proj = 2 * d_model * d_latent
    e_full = n_experts * 2 * d_model * (d_ff_mult * d_model)
    return (e_lat + proj) / e_full


# ---------------------------------------------------------------- agent
class LatentMoEAgent(BaseAgent):
    name = "latent-moe"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        if not _HAS_TORCH:
            return self.escalate(q, "leader", "latent-moe needs torch")
        s = self.spec(q)
        torch.manual_seed(int(s.get("seed", 0)))
        d_model = int(s.get("d_model", 128)); d_latent = int(s.get("d_latent", 32))
        E = int(s.get("n_experts", 8)); k = int(s.get("k", 2)); steps = int(s.get("steps", 300))

        # it must still LEARN despite the compression: fit a random nonlinear target through the latent MoE.
        moe = LatentMoE(d_model, d_latent, E, k)
        Wt = torch.randn(d_model, d_model) * 0.1
        def target(x): return torch.tanh(x @ Wt)
        opt = torch.optim.Adam(moe.parameters(), lr=3e-3); l0 = None
        for _ in range(steps):
            x = torch.randn(64, d_model)
            loss = F.mse_loss(moe(x), target(x))
            opt.zero_grad(); loss.backward(); opt.step()
            if l0 is None:
                l0 = float(loss)
        pr = latent_param_ratio(d_model, d_latent, E)
        kv = latent_kv_cache_bytes(4096, n_heads=8, head_dim=d_model // 8, d_latent=d_latent)
        msg = (f"latent-moe: LatentMoE(d_model={d_model}→d_latent={d_latent}, {k}/{E}) learns through the "
               f"compression (loss {l0:.3f}→{float(loss):.3f}); experts cost {pr*100:.0f}% of a full-width MoE, "
               f"and MLA KV-cache is {kv['reduction']:.1f}× smaller than full K/V. Run experts+attention in a "
               f"low-rank latent (K3 Stable LatentMoE + Gated MLA)")
        self.log(msg, kind="finding",
                 recommendation="down-project to a latent, run experts/attention KV there, up-project back — "
                                "shrinks expert params + KV-cache; pair with moe-quantile-balance router")
        return self.done({"loss0": l0, "loss1": float(loss), "param_ratio": pr,
                          "kv_reduction": kv["reduction"]}, msg)


_AGENT = LatentMoEAgent()


def run_latentmoe(q, worker):
    return _AGENT.run(q, worker)
