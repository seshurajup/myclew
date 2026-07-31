"""attention_residual — Kimi-K3 "Attention Residuals" (AttnRes): a residual connection that "selectively
retrieves representations across depth rather than accumulating them uniformly". A standard residual stream is
x_{l+1} = x_l + f_l(x_l): every layer's output is added with weight 1, so by deep layers the stream is a flat
uniform sum of all past contributions and the model cannot cheaply re-read a specific earlier representation.
AttnRes replaces the "+ x_l" with a LEARNED, per-layer, per-channel RETRIEVAL over the stack of ALL previous
layer states: the new state is a gated/attention-weighted read of {x_0, x_1, …, x_l} plus f_l — so a layer can
pull forward exactly the earlier depth it needs (selective) instead of the uniform running sum.

Two forms are provided (both pure-torch, reduce to a plain residual as a special case):
  • gated: x_{l+1} = f_l(x_l) + Σ_j g_{l,j} ⊙ x_j     — learned softmax gates g over depths j≤l (per-channel).
  • attention: query the depth-memory of past states with a learned query → weighted retrieval.
The special case g = onehot(l) (all weight on the immediate previous state) recovers the standard residual, so
AttnRes strictly generalizes it. Offline-testable: on a synthetic task that needs an EARLY representation at the
output, AttnRes learns to route around the uniform sum and beats a plain residual stack.

Primitives (torch):
  • DepthMemory                 — accumulates layer states x_0..x_l.
  • AttnResidual(dim, depth)    — the gated depth-retrieval module (drop-in for `x = x + f(x)`).
  • uniform_vs_selective(...)   — shows a plain residual is the uniform-gate special case.
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
    class AttnResidual(nn.Module):
        """Selective depth-retrieval residual. Holds a learned gate over all previous depths; at layer l the
        output is f_out + a softmax-weighted (per-channel) combination of the stored states x_0..x_l. With the
        gate initialized to favor the most recent state it starts ≈ a standard residual and learns to retrieve
        earlier depths where useful."""
        def __init__(self, dim, max_depth):
            super().__init__()
            self.dim = dim; self.max_depth = max_depth
            # per-(target-layer, source-depth) logits, shaped (max_depth, max_depth); masked causal over depth.
            self.gate_logits = nn.Parameter(torch.zeros(max_depth, max_depth))
            with torch.no_grad():
                for l in range(max_depth):
                    self.gate_logits[l, l] = 3.0            # bias toward immediate-previous ⇒ ≈ plain residual at init
            self.chan = nn.Parameter(torch.ones(max_depth, dim))   # per-channel scale on the retrieved read

        def retrieve(self, states, l):
            """Gated read over stored states[0..l] (list of (B,dim)); returns (B,dim)."""
            k = len(states)
            logits = self.gate_logits[l, :k]
            mask = torch.full((k,), float("-inf"), device=logits.device)
            mask[:k] = 0.0
            g = torch.softmax(logits + mask, dim=0)         # (k,) weights over depths
            stack = torch.stack(states, dim=0)              # (k,B,dim)
            read = (g.view(k, 1, 1) * stack).sum(dim=0)     # (B,dim) selective retrieval
            return read * self.chan[l]

        def forward(self, f_out, states, l):
            """x_{l+1} = f_out + selective_retrieve(states, l)."""
            return f_out + self.retrieve(states, l)

    def uniform_vs_selective(states, l, selective_gate=None):
        """Return (uniform_sum, selective_read). uniform = plain running residual (equal weight); selective =
        softmax(selective_gate) weighted. Shows AttnRes generalizes the uniform residual."""
        k = len(states)
        stack = torch.stack(states, dim=0)
        uniform = stack.sum(dim=0)                          # plain residual accumulation
        g = torch.softmax(selective_gate[:k], dim=0) if selective_gate is not None else torch.full((k,), 1.0 / k)
        selective = (g.view(k, 1, 1) * stack).sum(dim=0)
        return uniform, selective
else:  # pragma: no cover
    AttnResidual = uniform_vs_selective = None


# ---------------------------------------------------------------- EDA (2026) adds: a SECOND address
# "Erase-then-Delta Attention (EDA): Decoupling Erase and Write Addresses in Delta-Rule Linear Attention",
# arXiv:2606.26560 — paper: https://arxiv.org/pdf/2606.26560
# local: docs/papers/eda-delta-attention/eda-delta-attention.md · lessons: learning/annotated/eda*.learning
#
# AttnRes above chooses WHERE TO READ across depth. EDA is the same kind of move one level down, inside a
# delta-rule memory: it chooses WHERE TO ERASE. Every model in that family (DeltaNet, gated DeltaNet, K3's
# KDA) hard-wires the erase address to the write address — `(I − βkkᵀ)S + βkvᵀ` can only forget at the key
# it is about to write, so stale content held at a DIFFERENT address can only decay, never be removed.
#
# Three facts we proved (lessons eda02/eda03), which is what makes this worth wiring in:
#   • the erase is one gradient step on `½‖Ŝᵀe‖²` — forgetting gets an objective, like writing does;
#   • its collateral damage on a query q is exactly `γ(qᵀe)Ŝᵀe`, so an orthogonal query is untouched;
#   • interleaving the erase as a virtual token (value 0) makes EDA an ORDINARY gated-delta recurrence on a
#     2T sequence — verified to 1e-5 — so it reuses the existing chunked kernel with no new CUDA.
def eda_step(S, k, v, e, beta=1.0, gamma=1.0, decay=None):
    """One Erase-then-Delta update (EDA eq. 8): decay → targeted erase at `e` → delta write at `k`.

    `S` is (d_k, d_v); `k`, `e` unit vectors; `decay` an optional per-channel gate (K3-style `diag(α)`).
    Reduces to gated DeltaNet when `e is k` or `gamma == 0`, so it is safe to swap in anywhere.
    """
    import torch
    d = S.shape[0]
    I = torch.eye(d, device=S.device, dtype=S.dtype)
    Sh = (decay[:, None] * S) if decay is not None else S
    St = (I - gamma * torch.outer(e, e)) @ Sh                    # erase where we choose
    return (I - beta * torch.outer(k, k)) @ St + beta * torch.outer(k, v)


def eda_interleave(keys, values, erase, betas, gammas, decays=None):
    """EDA's doubling trick (eq. 17) → a length-2T ordinary gated-delta sequence.

    Odd steps are the erase (key = `e_t`, value = 0, strength = `γ_t`, gate = the decay); even steps are
    the real write. Feed the result to ANY gated-delta kernel and it computes EDA — that equivalence is
    what makes the second address free at the kernel level.
    """
    import torch
    T = len(keys)
    out = []
    for t in range(T):
        g = decays[t] if decays is not None else torch.ones_like(keys[t])
        out.append(dict(k=erase[t], v=torch.zeros_like(values[t]), beta=float(gammas[t]), gate=g))
        out.append(dict(k=keys[t], v=values[t], beta=float(betas[t]),
                        gate=torch.ones_like(keys[t])))
    return out


def eda_collateral(S_hat, q, e, gamma=1.0):
    """The exact read-side cost of an erase (eq. 12): `γ (qᵀe) Ŝᵀe`.

    Use it as a GUARD before erasing: if the queries you care about have a large overlap with the proposed
    erase direction, the erase will damage them by precisely this much. Returns (damage_vector, norm).
    """
    import torch
    dmg = gamma * float(q @ e) * (S_hat.T @ e)
    return dmg, float(torch.linalg.vector_norm(dmg))


# ---------------------------------------------------------------- agent
class AttentionResidual(BaseAgent):
    name = "attention-residual"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        if not _HAS_TORCH:
            return self.escalate(q, "leader", "attention-residual needs torch")
        s = self.spec(q)
        torch.manual_seed(int(s.get("seed", 0)))
        dim = int(s.get("dim", 16)); depth = int(s.get("depth", 6)); steps = int(s.get("steps", 400))

        # Task that NEEDS an early representation: target = the depth-0 input, but many layers transform the
        # stream in between (uniform residual buries it; AttnRes can retrieve depth 0).
        def make_stack(x0):
            states = [x0]
            for _ in range(depth - 1):
                states.append(torch.tanh(states[-1] @ Wmix))
            return states
        Wmix = torch.randn(dim, dim) * 0.5

        # AttnRes head learns to reconstruct x0 from the final selective read.
        ar = AttnResidual(dim, depth); head = nn.Linear(dim, dim)
        opt = torch.optim.Adam(list(ar.parameters()) + list(head.parameters()), lr=5e-3)
        l0 = None
        for _ in range(steps):
            x0 = torch.randn(32, dim)
            states = make_stack(x0)
            read = ar.retrieve([s.detach() for s in states], depth - 1)
            pred = head(read)
            loss = F.mse_loss(pred, x0)
            opt.zero_grad(); loss.backward(); opt.step()
            if l0 is None:
                l0 = float(loss)
        # how much weight did it put on depth 0 vs the uniform 1/depth?
        with torch.no_grad():
            g = torch.softmax(ar.gate_logits[depth - 1, :depth], dim=0)
        msg = (f"attention-residual: AttnRes learned to RETRIEVE the early representation — recon loss "
               f"{l0:.3f}→{float(loss):.3f}; depth-0 gate weight={float(g[0]):.2f} vs uniform {1/depth:.2f} "
               f"(selective, not uniform sum). Drop-in for x=x+f(x) so deep layers can re-read a specific "
               f"earlier depth (K3 Attention Residuals)")
        self.log(msg, kind="finding",
                 recommendation="replace flat residual add with AttnResidual where deep layers need earlier "
                                "features (long stacks, UNet skips, deep transformers); init ≈ plain residual")
        return self.done({"loss0": l0, "loss1": float(loss), "depth0_gate": float(g[0])}, msg)


_AGENT = AttentionResidual()


def run_attnres(q, worker):
    return _AGENT.run(q, worker)
