"""muon_optimizer — the Muon optimizer (MomentUm Orthogonalized by Newton-schulz), the 2025-26 SOTA
matrix-parameter optimizer the fleet was missing entirely (we had SAM/AWP/EMA/SWA but no orthogonalizing
optimizer). Muon orthogonalizes the momentum buffer of every 2D weight before the step — replacing its
singular values with ones via a fixed 5-step Newton-Schulz quintic iteration — so every direction of an
ill-conditioned weight matrix gets an equal-sized update. This whitens the update WITHOUT a per-parameter
second-moment estimate (unlike Adam) and is stable on GPU tensor cores.

Papers (2025-26 frontier — the winners haven't published these yet):
  • Keller Jordan et al., "Muon: An optimizer for hidden layers in neural networks" (2024, muon writeup) —
    the original zeropower-via-Newton-Schulz quintic (a,b,c = 3.4445, -4.7750, 2.0315).
  • "The Newton-Muon Optimizer", Du & Su, arXiv:2604.01472 (2026) — surrogate model behind Muon's design.
  • "Hierarchical Muon: Tiled Newton-Schulz Updates", arXiv:2606.27216 (2026) — scalable tiled NS.
  • Kim & Oh, "On the Convergence of Muon with Newton-Schulz", ICLR 2026 (arXiv:2509.15816) — convergence.
  • "Spectral Flattening Is All Muon Needs", arXiv:2605.13079 (2026) — WHY orthogonalization controls the LR.

Reusable: a drop-in torch.optim.Optimizer for ANY model with 2D weight matrices (conv/linear/attention) —
tabular MLPs, vision UNets, transformers. 1D params (biases/norms) fall back to plain momentum SGD.
"""
from __future__ import annotations
import numpy as np
try:
    from .base import BaseAgent
except Exception:  # noqa: BLE001 — allow top-level import (e.g. trackC_run does `from muon_optimizer import Muon`)
    class BaseAgent:  # minimal stand-in when imported outside the fleet_agents package
        pass

try:
    import torch
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


# ---------------------------------------------------------------- Newton-Schulz quintic orthogonalization
def newton_schulz5(G, steps: int = 5, eps: float = 1e-7):
    """Orthogonalize a 2D matrix G by the quintic Newton-Schulz iteration (numpy). Pushes every singular
    value toward 1 (semi-orthogonal output) using ONLY matrix products X <- aX + (bA + cA^2)X, A=XX^T.
    Works for tall OR wide G (transposes internally so the small dimension is contracted)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = np.asarray(G, dtype=np.float64)
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    X = X / (np.linalg.norm(X) + eps)          # normalise so the iteration's basin holds
    for _ in range(int(steps)):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


def _orthogonality_error(O):
    """||O O^T - I|| (or O^T O for tall) — 0 means perfectly orthonormal rows/cols."""
    O = np.asarray(O, float)
    m, n = O.shape
    if m <= n:
        G = O @ O.T; I = np.eye(m)
    else:
        G = O.T @ O; I = np.eye(n)
    return float(np.linalg.norm(G - I) / np.sqrt(I.shape[0]))


# ---------------------------------------------------------------- numpy Muon step (framework-free core)
def muon_update(grad, momentum_buf, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
    """One Muon update for a 2D weight. Returns (delta, new_buf) where new_weight = weight + delta.
    buf <- momentum*buf + grad ; g <- grad + momentum*buf (nesterov) ; O <- NS5(g) ; scaled by sqrt(fan)."""
    g = np.asarray(grad, float); buf = momentum * np.asarray(momentum_buf, float) + g
    upd = g + momentum * buf if nesterov else buf
    O = newton_schulz5(upd, steps=ns_steps)
    scale = max(1.0, upd.shape[0] / upd.shape[1]) ** 0.5      # RMS-match the update magnitude to fan-out
    return -lr * scale * O, buf


def per_head_muon_update(grad, momentum_buf, n_heads, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
    """Kimi-K3 "Per-Head Muon": orthogonalize an attention projection weight ONE HEAD AT A TIME instead of
    as a single fused matrix. An attention weight has shape (n_heads*head_dim, in_dim) (q/k/v/o proj); the
    heads are independent linear maps that standard Muon fuses and whitens jointly — so a well-conditioned
    head is dragged by an ill-conditioned one and the sqrt(fan) RMS-match uses the WRONG (fused) fan.
    Per-Head Muon runs the Newton-Schulz quintic on each head's (head_dim, in_dim) block separately, giving
    every head its own conditioning-blind, correctly-fan-scaled update (K3's report credits this for the
    stability of its very-sparse LatentMoE attention at scale). Reduces to plain Muon when n_heads==1.

    grad/momentum_buf: 2D arrays shape (n_heads*head_dim, in_dim). Returns (delta, new_buf), same shape."""
    g = np.asarray(grad, float); buf = momentum * np.asarray(momentum_buf, float) + g
    upd = g + momentum * buf if nesterov else buf
    H = int(n_heads); rows = upd.shape[0]
    if H <= 1 or rows % H != 0:                                   # not head-structured → fall back to fused Muon
        O = newton_schulz5(upd, steps=ns_steps)
        scale = max(1.0, upd.shape[0] / upd.shape[1]) ** 0.5
        return -lr * scale * O, buf
    hd = rows // H
    out = np.empty_like(upd)
    for h in range(H):
        blk = upd[h * hd:(h + 1) * hd]                            # this head's (head_dim, in_dim) slice
        Oh = newton_schulz5(blk, steps=ns_steps)
        scale = max(1.0, blk.shape[0] / blk.shape[1]) ** 0.5     # per-HEAD fan RMS-match (the fix)
        out[h * hd:(h + 1) * hd] = scale * Oh
    return -lr * out, buf


def optimize_matrix(grad_fn, W0, steps=200, lr=0.02, momentum=0.95, ns_steps=5, nesterov=True):
    """Minimise a matrix objective whose gradient is grad_fn(W) using Muon. Returns (W, loss_history?)."""
    W = np.asarray(W0, float).copy(); buf = np.zeros_like(W)
    for _ in range(int(steps)):
        g = grad_fn(W)
        delta, buf = muon_update(g, buf, lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        W = W + delta
    return W


def sgd_momentum_matrix(grad_fn, W0, steps=200, lr=0.02, momentum=0.95):
    """SGD+momentum baseline at the SAME lr."""
    W = np.asarray(W0, float).copy(); buf = np.zeros_like(W)
    for _ in range(int(steps)):
        buf = momentum * buf + grad_fn(W)
        W = W - lr * buf
    return W


def gd_matrix(grad_fn, W0, steps=200, lr=0.02):
    """Plain gradient descent at the SAME lr — stalls on the flat (small-singular-value) directions of an
    ill-conditioned problem; the reference Muon beats because its orthogonalized update is conditioning-blind."""
    W = np.asarray(W0, float).copy()
    for _ in range(int(steps)):
        W = W - lr * grad_fn(W)
    return W


# ---------------------------------------------------------------- torch drop-in optimizer
if _HAS_TORCH:
    class Muon(torch.optim.Optimizer):
        """Drop-in Muon for 2D weights; 1D params fall back to momentum SGD. Use for hidden matrices;
        keep embeddings/heads on AdamW in a separate param group as the paper recommends.

            opt = Muon(model.parameters(), lr=0.02, momentum=0.95)
            loss.backward(); opt.step(); opt.zero_grad()

        Per-Head Muon (Kimi-K3): put attention projections in a param group with n_heads=H so each head's
        (head_dim, in) block is orthogonalized independently:
            opt = Muon([{"params": attn_w, "n_heads": 16}, {"params": mlp_w}], lr=0.02)
        """
        def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0,
                     n_heads=1):
            super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                          ns_steps=ns_steps, weight_decay=weight_decay, n_heads=n_heads))

        @torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for grp in self.param_groups:
                for p in grp["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if grp["weight_decay"]:
                        g = g.add(p, alpha=grp["weight_decay"])
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(p)
                    buf = st["buf"]; buf.mul_(grp["momentum"]).add_(g)
                    upd = g.add(buf, alpha=grp["momentum"]) if grp["nesterov"] else buf
                    if p.ndim >= 2:
                        G2 = upd.reshape(upd.shape[0], -1)
                        H = int(grp.get("n_heads", 1))
                        if H > 1 and G2.shape[0] % H == 0:        # Per-Head Muon: orthogonalize each head block
                            delta, _ = per_head_muon_update(
                                G2.detach().cpu().numpy(), np.zeros_like(G2.detach().cpu().numpy()),
                                n_heads=H, lr=1.0, momentum=0.0, nesterov=False, ns_steps=grp["ns_steps"])
                            p.add_(torch.from_numpy(delta).to(p).reshape_as(p), alpha=grp["lr"])
                        else:
                            O = torch.from_numpy(newton_schulz5(G2.detach().cpu().numpy(),
                                                                steps=grp["ns_steps"])).to(p)
                            scale = max(1.0, G2.shape[0] / G2.shape[1]) ** 0.5
                            p.add_(O.reshape_as(p), alpha=-grp["lr"] * scale)
                    else:
                        p.add_(buf, alpha=-grp["lr"])
            return loss
else:  # pragma: no cover
    Muon = None


# ---------------------------------------------------------------- agent
class MuonOptimizer(BaseAgent):
    name = "muon-optimizer"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        rng = np.random.RandomState(int(s.get("seed", 0)))
        d = int(s.get("dim", 12)); cond = float(s.get("cond", 1000.0)); steps = int(s.get("steps", 300))
        lr = float(s.get("lr", 0.05))
        # ill-conditioned linear regression min ||X W - Y||_F^2, X anisotropic (cond number `cond`)
        U, _ = np.linalg.qr(rng.randn(d, d)); Vt, _ = np.linalg.qr(rng.randn(d, d))
        sv = np.linspace(1.0, 1.0 / cond, d); X = U @ np.diag(sv) @ Vt
        Wt = rng.randn(d, d); Y = X @ Wt
        def loss(W): return float(np.linalg.norm(X @ W - Y) ** 2)
        def grad(W): return 2.0 * X.T @ (X @ W - Y)
        W0 = np.zeros((d, d)); l0 = loss(W0)
        lm = loss(optimize_matrix(grad, W0, steps=steps, lr=lr))
        lp = loss(gd_matrix(grad, W0, steps=steps, lr=lr))            # plain GD stalls on flat directions
        msg = (f"muon-optimizer: cond={cond:.0f} {d}x{d} regression — Muon loss={lm:.3e} vs "
               f"plain-GD {lp:.3e} (init {l0:.3e}); Muon ratio={lm/max(lp,1e-30):.2e}× (conditioning-blind orthogonalized update)")
        self.log(msg, kind="finding", recommendation="use Muon for 2D hidden weights; keep heads/embeddings on AdamW")
        return self.done({"muon_loss": lm, "gd_loss": lp, "init_loss": l0}, msg)


_AGENT = MuonOptimizer()


def run(q, worker):
    return _AGENT.run(q, worker)
