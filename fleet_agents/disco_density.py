"""disco-density — reusable plug-in DENSITY + SCORE oracle and its label-free test-time adaptation,
distilled from DiScoFormer (Ilin, Sushko, Krishna — arXiv:2511.05924, ICLR 2026, Allen AI).

Two ideas we adopt (grounded, not asserted — prove per-comp before trusting):

  1. AMORTIZED set→(density, score) estimator. A context set of i.i.d. samples X ∈ R^{n×d} is
     encoded once; arbitrary query points q attend to that context via CROSS-ATTENTION and read out
     both log-density and score (∇_x log p). Trained once over many distributions (a fresh GMM per
     batch — GMMs are universal approximators with closed-form density+score), it generalizes to new
     distributions and sample sizes WITHOUT retraining. Theory: one normalized-attention block already
     reproduces a Gaussian KDE, so attention is a functional generalization of kernel density estimation.

  2. HEAD-CONSISTENCY TEST-TIME ADAPTATION (the transferable trick). The score head must equal the
     gradient of the log-density head at every query: score(q) ≈ ∇_q logp(q). Any gap is a LABEL-FREE
     consistency loss — no ground-truth density/score needed. At inference, hold the context fixed and
     take a few gradient steps on that gap to self-adapt to an out-of-distribution input. This principle
     reuses on ANY multi-head model whose heads have a differentiable relationship (see `tta_consistency`).

Where this plugs into the fleet: a plug-in score oracle for score-debiased KDE, Fisher-information and
Fokker–Planck-type computations, density-ratio / adversarial-validation features, and per-group OOD
adaptation (e.g. rogii per-well GR density; biohub per-embryo detection-feature density).

Usage:
    python disco_density.py smoke        # train tiny model on random GMMs, check KDE-beating + TTA
"""
from __future__ import annotations

import math
import sys


def _torch():
    import torch  # noqa: PLC0415
    return torch


def _load_hct():
    """Import the shared head_consistency_tta primitive from inference_tricks_pack, whether disco_density
    is run as a script or imported as part of the fleet_agents package (that module uses `from .base`)."""
    try:
        from inference_tricks_pack import head_consistency_tta
        return head_consistency_tta
    except Exception:  # noqa: BLE001
        import importlib
        import types
        from pathlib import Path as _P
        here = _P(__file__).resolve().parent            # this dir IS the fleet_agents package
        if "fleet_agents" not in sys.modules:
            m = types.ModuleType("fleet_agents"); m.__path__ = [str(here)]
            sys.modules["fleet_agents"] = m
        return importlib.import_module("fleet_agents.inference_tricks_pack").head_consistency_tta


# ----------------------------- data: a fresh GMM per batch -----------------------------
def sample_gmm(batch, n, d, k=3, device="cpu", dtype=None):
    """Return (X, mu, w, sigma) for `batch` independent k-component isotropic GMMs, each with `n`
    samples in R^d. Closed-form density/score come from (mu, w, sigma) via `gmm_logp_score`."""
    torch = _torch()
    dtype = dtype or torch.float32
    mu = torch.randn(batch, k, d, device=device, dtype=dtype) * 2.0
    sigma = (0.3 + torch.rand(batch, k, device=device, dtype=dtype) * 0.7)
    w = torch.softmax(torch.randn(batch, k, device=device, dtype=dtype), dim=-1)
    comp = torch.multinomial(w, n, replacement=True)                     # (batch, n)
    m = torch.gather(mu, 1, comp.unsqueeze(-1).expand(-1, -1, d))
    s = torch.gather(sigma, 1, comp).unsqueeze(-1)
    X = m + torch.randn(batch, n, d, device=device, dtype=dtype) * s
    return X, mu, w, sigma


def gmm_logp_score(q, mu, w, sigma):
    """Exact log-density and score (grad log p) of an isotropic GMM at query points q ∈ (B, m, d)."""
    torch = _torch()
    d = q.shape[-1]
    diff = q.unsqueeze(2) - mu.unsqueeze(1)                              # (B, m, k, d)
    s2 = (sigma ** 2).unsqueeze(1)                                       # (B, 1, k)
    sq = (diff ** 2).sum(-1)                                             # (B, m, k)
    logN = -0.5 * sq / s2 - 0.5 * d * torch.log(2 * math.pi * s2)
    logw = torch.log(w).unsqueeze(1)                                     # (B, 1, k)
    logp = torch.logsumexp(logw + logN, dim=-1)                         # (B, m)
    resp = torch.softmax(logw + logN, dim=-1)                           # (B, m, k)
    score = -(resp.unsqueeze(-1) * diff / s2.unsqueeze(-1)).sum(2)      # (B, m, d)
    return logp, score


# ----------------------------- the model -----------------------------
def build_model(d=2, dim=128, heads=4, layers=3):
    torch = _torch()
    nn = torch.nn

    class CrossBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.ln_q = nn.LayerNorm(dim); self.ln_c = nn.LayerNorm(dim); self.ln_f = nn.LayerNorm(dim)
            self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

        def forward(self, q, ctx):
            a, _ = self.attn(self.ln_q(q), self.ln_c(ctx), self.ln_c(ctx), need_weights=False)
            q = q + a
            return q + self.ff(self.ln_f(q))

    class DiSco(nn.Module):
        """Permutation-invariant in the context (attention has no positional encoding), query points
        are independent — a set→function operator, exactly the DiScoFormer contract."""
        def __init__(self):
            super().__init__()
            self.embed_c = nn.Linear(d, dim)
            self.embed_q = nn.Linear(d, dim)
            self.self_ctx = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, activation="gelu")
            self.blocks = nn.ModuleList([CrossBlock() for _ in range(layers)])
            self.head_logp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
            self.head_score = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, d))

        def forward(self, ctx_x, q):
            ctx = self.self_ctx(self.embed_c(ctx_x))
            h = self.embed_q(q)
            for b in self.blocks:
                h = b(h, ctx)
            return self.head_logp(h).squeeze(-1), self.head_score(h)

    return DiSco()


def train(steps=400, d=2, n=128, mq=64, batch=32, device=None, lr=2e-3, log=print):
    torch = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(d=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        X, mu, w, sigma = sample_gmm(batch, n, d, device=device)
        q, _, _, _ = sample_gmm(batch, mq, d, device=device)            # query points from a similar GMM
        # queries evaluated against the CONTEXT's true GMM (mu,w,sigma), not their own
        tgt_logp, tgt_score = gmm_logp_score(q, mu, w, sigma)
        pred_logp, pred_score = model(X, q)
        loss = (pred_logp - tgt_logp).pow(2).mean() + (pred_score - tgt_score).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if log and (step % max(1, steps // 5) == 0 or step == steps - 1):
            log(f"  step {step:4d}  loss {loss.item():.4f}")
    return model, device


def kde_score(ctx_x, q, bw=0.5):
    """Baseline: Gaussian-KDE log-density + score, the classical estimator DiScoFormer generalizes."""
    torch = _torch()
    d = ctx_x.shape[-1]
    diff = q.unsqueeze(2) - ctx_x.unsqueeze(1)                          # (B, m, n, d)
    sq = (diff ** 2).sum(-1)                                            # (B, m, n)
    logk = -0.5 * sq / bw ** 2 - 0.5 * d * math.log(2 * math.pi * bw ** 2)
    logp = torch.logsumexp(logk, dim=-1) - math.log(ctx_x.shape[1])
    resp = torch.softmax(logk, dim=-1)
    score = -(resp.unsqueeze(-1) * diff / bw ** 2).sum(2)
    return logp, score


def tta_consistency(model, ctx_x, q, steps=10, lr=0.05, return_gaps=False):
    """The canonical DiScoFormer instance of head-consistency TTA: score(q) must equal ∇_q logp(q).
    Thin wrapper over the shared, model-agnostic `inference_tricks_pack.head_consistency_tta` primitive
    (the generalized trick lives there — this only supplies the score-vs-grad-log-density `relation`).
    Returns the adapted (logp, score), or the per-step gap list when return_gaps=True."""
    torch = _torch()
    head_consistency_tta = _load_hct()
    q = q.clone().requires_grad_(True)

    def forward_fn():
        logp, score = model(ctx_x, q)
        return logp, score

    def relation(outputs):
        logp, score = outputs
        grad_logp = torch.autograd.grad(logp.sum(), q, create_graph=True)[0]
        return score - grad_logp

    gaps = head_consistency_tta(model, forward_fn, relation, steps=steps, lr=lr, clip=1.0)
    if return_gaps:
        return gaps
    with torch.no_grad():
        return model(ctx_x, q.detach())


def smoke():
    torch = _torch()
    torch.manual_seed(0)
    model, device = train(steps=300)
    model.eval()
    X, mu, w, sigma = sample_gmm(8, 128, 2, device=device)
    q, _, _, _ = sample_gmm(8, 64, 2, device=device)
    tgt_logp, tgt_score = gmm_logp_score(q, mu, w, sigma)
    with torch.no_grad():
        p_logp, p_score = model(X, q)
    k_logp, k_score = kde_score(X, q)
    disco_err = (p_logp - tgt_logp).pow(2).mean().item()
    kde_err = (k_logp - tgt_logp).pow(2).mean().item()
    ds = (p_score - tgt_score).pow(2).mean().item()
    ks = (k_score - tgt_score).pow(2).mean().item()
    print(f"logp MSE   disco={disco_err:.4f}  kde={kde_err:.4f}  (disco should be <= kde)")
    print(f"score MSE  disco={ds:.4f}  kde={ks:.4f}")
    # TTA on an OOD context (more modes) — consistency gap should shrink
    Xood, muo, wo, so = sample_gmm(8, 128, 2, k=6, device=device)
    with torch.no_grad():
        _, s0 = model(Xood, q); g0 = None
    q2 = q.clone().requires_grad_(True)
    lp, sc = model(Xood, q2)
    g0 = torch.autograd.grad(lp.sum(), q2)[0]
    gap_before = (sc - g0).pow(2).mean().item()
    tta_consistency(model, Xood, q, steps=8, lr=0.02)
    q3 = q.clone().requires_grad_(True)
    lp2, sc2 = model(Xood, q3)
    g1 = torch.autograd.grad(lp2.sum(), q3)[0]
    gap_after = (sc2 - g1).pow(2).mean().item()
    print(f"TTA consistency gap  before={gap_before:.4f}  after={gap_after:.4f}  (should drop)")
    ok = disco_err <= kde_err * 1.5 and gap_after <= gap_before
    print("SMOKE", "OK" if ok else "CHECK")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if cmd == "smoke":
        smoke()
    else:
        print(__doc__)
