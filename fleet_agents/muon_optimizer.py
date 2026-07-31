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


def polargrad_update(grad, momentum_buf, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
    """One PolarGrad step for a 2D weight (NVIDIA Emerging-Optimizers, arXiv:2505.21799). Muon replaces the
    update's singular values with ONES (LMO w.r.t. the spectral norm); PolarGrad instead orthogonalizes the
    momentum and rescales by its NUCLEAR norm — steepest descent w.r.t. the spectral norm. Concretely the
    Muon direction O=NS5(g) is kept but scaled by <O,g>=sum(O*g) (=||g||_* when O is the exact polar factor),
    so the step magnitude tracks the momentum's own energy instead of being flattened to a constant. This
    keeps Muon's conditioning-blindness while recovering a data-adaptive step size on top. Returns
    (delta, new_buf); new_weight = weight + delta."""
    g = np.asarray(grad, float); buf = momentum * np.asarray(momentum_buf, float) + g
    upd = g + momentum * buf if nesterov else buf
    O = newton_schulz5(upd, steps=ns_steps)
    scale = float((O * upd).sum())                              # nuclear-norm-scaled (PolarGrad), not sqrt(fan)
    return -lr * scale * O, buf


def polar_factor(X, steps=25, eps=1e-12):
    """Orthogonal polar factor (matrix sign) U V^T of X = U diag(s) V^T, via the cubic Newton-Schulz iteration
    Y <- 1.5 Y - 0.5 Y (Y^T Y). SVD-free (tensor-core friendly). We first divide by the Frobenius norm (an
    upper bound on the spectral norm) so every singular value lands in (0, 1], the convergence basin of the
    cubic map, which then drives them ACCURATELY to 1 (unlike Muon's fixed 5-step quintic which only
    approximates). Handles tall/wide by contracting the small dimension."""
    X = np.asarray(X, float)
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    Y = X / (np.linalg.norm(X) + eps)
    for _ in range(int(steps)):
        Y = 1.5 * Y - 0.5 * (Y @ (Y.T @ Y))
    return Y.T if transpose else Y


def spectral_hardcap(X, beta=1.0, ns_steps=25):
    """Spectral hardcap: return X with every singular value clipped ABOVE to <= beta, leaving smaller ones
    untouched (leloykun.github.io/ponder/spectral-clipping, ported from Emerging-Optimizers). Bounds the
    spectral norm of an update/weight for stability WITHOUT an SVD — uses only Newton-Schulz matrix-sign
    (polar-factor) iterations, so it runs on tensor cores. For X = U diag(s) V^T it computes
    U diag(min(s, beta)) V^T. Returns an array the same shape as X."""
    X = np.asarray(X, float)
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    OX = polar_factor(X, steps=ns_steps)                       # polar factor U V^T (matrix sign)
    aX = beta * OX - X
    result = beta * OX + X
    result = result - aX @ (polar_factor(aX, steps=ns_steps).T @ OX)
    result = result * 0.5
    return result.T if transpose else result


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
    def _newton_schulz5_torch(G, steps=5, eps=1e-7):
        """Torch quintic Newton-Schulz orthogonalization (device-native; keeps the update on GPU). Same
        (a,b,c) as the numpy core; pushes every singular value of G toward 1 (semi-orthogonal / polar factor)."""
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.to(torch.float32)
        transpose = X.shape[0] > X.shape[1]
        if transpose:
            X = X.T
        X = X / (X.norm() + eps)
        for _ in range(int(steps)):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        return (X.T if transpose else X).to(G.dtype)

    class AdaptiveMuon(torch.optim.Optimizer):
        """Adaptive Muon (NVIDIA Emerging-Optimizers / AdaMuon arXiv:2507.11005): Muon's Newton-Schulz
        orthogonalized momentum, then an AdamW-style ELEMENTWISE second-moment normalization on the
        orthogonalized update before the sqrt(fan) Muon scale. Plain Muon gives every direction a
        unit-magnitude step; AdaMuon adds a per-parameter adaptive learning rate on top (divide by
        sqrt(EMA of squared orthogonalized grad) + eps), which stabilizes noisy/heavy-tailed gradients and
        makes the step far less LR-sensitive. 2D params are orthogonalized; 1D params fall back to Adam.

            opt = AdaptiveMuon(model.parameters(), lr=0.02, momentum=0.95, beta2=0.95)
            loss.backward(); opt.step(); opt.zero_grad()
        """
        def __init__(self, params, lr=0.02, momentum=0.95, beta2=0.95, nesterov=True, ns_steps=5,
                     eps=1e-8, weight_decay=0.0):
            super().__init__(params, dict(lr=lr, momentum=momentum, beta2=beta2, nesterov=nesterov,
                                          ns_steps=ns_steps, eps=eps, weight_decay=weight_decay))

        @torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for grp in self.param_groups:
                b2, eps = grp["beta2"], grp["eps"]
                for p in grp["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if grp["weight_decay"]:
                        g = g.add(p, alpha=grp["weight_decay"])
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(p); st["v"] = torch.zeros_like(p)
                    buf = st["buf"]; buf.mul_(grp["momentum"]).add_(g)
                    upd = g.add(buf, alpha=grp["momentum"]) if grp["nesterov"] else buf
                    v = st["v"]
                    if p.ndim >= 2:
                        G2 = upd.reshape(upd.shape[0], -1)
                        O = _newton_schulz5_torch(G2, steps=grp["ns_steps"]).reshape_as(p)
                        v.mul_(b2).addcmul_(O, O, value=1 - b2)            # EMA of squared orthogonalized grad
                        step_dir = O / (v.sqrt() + eps)
                        scale = max(1.0, G2.shape[0] / G2.shape[1]) ** 0.5
                        p.add_(step_dir, alpha=-grp["lr"] * scale)
                    else:                                                   # 1D → Adam-style fallback
                        v.mul_(b2).addcmul_(upd, upd, value=1 - b2)
                        p.add_(upd / (v.sqrt() + eps), alpha=-grp["lr"])
            return loss
else:  # pragma: no cover
    Muon = None
    AdaptiveMuon = None


# ---------------------------------------------------------------- Nested Learning (NeurIPS 2025) adds
# Behrouz, Razaviyayn, Zhong & Mirrokni, "Nested Learning: The Illusion of Deep Learning Architecture",
# NeurIPS 2025 — paper: https://alibehrouz.com/files/NL.pdf
# local: docs/papers/nested-learning/nested-learning.md · lessons: learning/annotated/nl*.learning
# They read an optimizer as an ASSOCIATIVE MEMORY over gradients, which turns two of its knobs
# into design choices rather than folklore (lessons `nl04`/`nl07`, every step proved in PyTorch):
#   • the momentum's DECAY is the retention gate of an L2-regression objective → make it depend on the
#     gradient (eq. 49, "Delta Momentum") instead of being a constant low-pass filter;
#   • one momentum is one TIME-SCALE. With beta=0.9 the last ~6 gradients hold ~47% of the buffer and
#     ~43 hold ~99% (measured), so nothing older than ~43 steps survives — which is why an optimizer has
#     no record of the gradient subspace it should avoid (§4.3). M3 (Algorithm 1) adds a slow, chunked
#     second memory and orthogonalises BOTH before aggregating.
def delta_momentum_update(m, g, alpha=0.9, eta=0.1, precond=None):
    """Delta Momentum (NL eq. 49): momentum whose decay is `alpha - eta*<g,g>` instead of a constant.

    Standard momentum is gradient descent on a DOT-PRODUCT objective, so its update ignores its own
    state; the L2-regression objective gives `m <- m (alpha - eta g^T g) - eta P g`, i.e. the memory
    forgets exactly when the gradient is large (a real forget gate). MEASURED on the paper's own
    time-varying-curvature landscape (eq. 53): comparable to standard momentum at its best-tuned step
    size and markedly more ROBUST when the schedule is mistuned (worst case ~3x better) — so use it when
    the schedule cannot be tuned per run, not as a free speed-up.

    `g` is normalised as the paper's derivation assumes (`||x||=lambda`), keeping the decay in (0, alpha].
    """
    gn = g / (1.0 + np.linalg.norm(g.reshape(-1)))
    decay = max(0.0, float(alpha) - float(eta) * float((gn.reshape(-1) ** 2).sum()))
    step = gn if precond is None else precond @ gn
    return m * decay - float(eta) * step, decay


def m3_state(shape, xp=None):
    """Fresh M3 state: fast memory, slow memory, second moment, and the chunk accumulator."""
    xp = xp or np
    z = xp.zeros(shape)
    return {"M1": z.copy(), "M2": z.copy(), "V": z.copy(), "acc": z.copy(), "t": 0}


def m3_update(state, g, lr=1e-3, beta1=1.0, beta2=1.0, beta3=1.0, alpha=0.3, freq=8, ns_steps=5):
    """Multi-scale Momentum Muon (NL Algorithm 1) → (update, state).

    Fast memory every step, slow memory every `freq` steps (a two-level Continuum Memory System inside
    the optimizer), each orthogonalised by Newton-Schulz, then aggregated `O1 + alpha*O2` and scaled by
    Adam's second moment.

    TWO measured caveats, both from running it (lesson `nl07`):
      1. Algorithm 1 divides by `sqrt(V)+eps` where V is a running SUM starting at zero, so the first
         steps divide by ~0 and diverge. We normalise the denominator by its own mean — the smallest
         guard that makes the pseudocode runnable. This deviation is deliberate and reported.
      2. Cost: ~2x a Muon step (a second memory + a second Newton-Schulz). The paper says the same
         (Fig. 12: slower than Muon, on par with AdaMuon) — do not adopt it for throughput.
    """
    xp = np
    st = state
    st["t"] += 1
    st["M1"] = st["M1"] + beta1 * g
    st["V"] = st["V"] + beta2 * (g * g)
    st["acc"] = st["acc"] + g
    if st["t"] % int(freq) == 0:                      # the SLOW memory: one write per chunk (NL eq. 75)
        st["M2"] = st["M2"] + beta3 * st["acc"]
        st["acc"] = xp.zeros_like(st["acc"])
    two_d = g.ndim == 2
    o1 = newton_schulz5(st["M1"], steps=ns_steps) if two_d else st["M1"] / (
        xp.linalg.norm(st["M1"].reshape(-1)) + 1e-9)
    o2 = newton_schulz5(st["M2"], steps=ns_steps) if two_d else st["M2"] / (
        xp.linalg.norm(st["M2"].reshape(-1)) + 1e-9)
    u = o1 + float(alpha) * o2
    den = xp.sqrt(st["V"])
    den = den / max(float(den.mean()), 1e-12) + 1e-2              # the guard (caveat 1 above)
    return -float(lr) * u / den, st


def momentum_horizon(beta=0.9, mass=(0.5, 0.99)):
    """How much of the past a momentum buffer actually holds (NL §4.3), as a dict {mass: n_gradients}.

    The contribution of the i-th previous gradient is `beta^i (1-beta)`. This is the number that makes
    "the optimizer has no memory of the old gradient subspace" concrete: at beta=0.9 the 99% horizon is
    43 steps, so a task learned 200 steps ago is invisible to the update direction.
    """
    xp = np
    contrib = xp.array([beta ** i * (1 - beta) for i in range(5000)])
    cum = xp.cumsum(contrib)
    out = {}
    for m in mass:
        idx = int((cum < m).sum())
        out[m] = idx + 1
    return out


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
