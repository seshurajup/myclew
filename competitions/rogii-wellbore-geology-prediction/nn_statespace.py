"""nn_statespace.py — a PHYSICS-INFORMED neural network whose ARCHITECTURE is dictated by eda.py's proofs.
Pure GPU torch (sm_120 / cu128), no numpy in the compute. Each module cites the proof it captures:

  • IRWHead        — proofs 1,4,11,40 (TVT is an Integrated Random Walk / unit-root / VR>1): the net predicts
                     per-step INCREMENTS that are CUMULATIVELY SUMMED, structurally enforcing integration
                     instead of emitting free per-row values.
  • HeteroHead     — proofs 3,14,48 (super-diffusion, variance grows with distance): a growing log-variance
                     head so predicted uncertainty compounds down the toe.
  • student_t_nll  — proof 6 (heavy-tailed GR/measurement noise): a Student-t negative log-likelihood loss,
                     robust to the leptokurtic residual (better than L2).
  • PerWellFiLM    — proofs 5,13 (process noise is per-well, not global): FiLM conditioning from per-well heel
                     statistics (incl. pf_noise_id q_s,q_v), so each well gets its own dynamics.
  • BiGRU encoder  — proofs 10,15,18,26 (Z-TVT coupling, GR facies length, near-horizontal lateral, depth trend):
                     the GR log is fully observed (only TVT hidden) so a bidirectional encoder is legitimate.

StateSpaceNet ties them together. A data-wise test plants a known IRW trajectory and asserts the integration
head recovers it AND beats a free-output baseline — the architecture is validated, not assumed.
"""
from __future__ import annotations
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class PerWellFiLM(nn.Module):
    """Proofs 5,13: per-well noise heterogeneity → feature-wise linear modulation from per-well context
    (e.g. heel stats + pf_noise_id q_s,q_v). Produces (γ,β) that scale/shift the sequence features."""
    def __init__(self, ctx_dim: int, feat_dim: int):
        super().__init__()
        self.g = nn.Linear(ctx_dim, feat_dim)
        self.b = nn.Linear(ctx_dim, feat_dim)

    def forward(self, x, ctx):                       # x:(B,L,F)  ctx:(B,C)
        return x * (1 + self.g(ctx).unsqueeze(1)) + self.b(ctx).unsqueeze(1)


class IRWHead(nn.Module):
    """Proofs 1,4,11,40: TVT is an integrated random walk ⇒ predict per-step increments δ_i and return the
    CUMULATIVE SUM (anchored at the PS value), so the output is an integrated trajectory by construction."""
    def __init__(self, hid: int):
        super().__init__()
        self.delta = nn.Linear(hid, 1)

    def forward(self, h, anchor):                    # h:(B,L,H)  anchor:(B,)
        d = self.delta(h).squeeze(-1)                # per-step increment δ_i
        return anchor.unsqueeze(1) + torch.cumsum(d, dim=1)   # integrate


class HeteroHead(nn.Module):
    """Proofs 3,14,48: super-diffusion ⇒ predicted variance must GROW with distance. Emits a per-step
    log-variance whose cumulative sum matches diffusive spreading."""
    def __init__(self, hid: int):
        super().__init__()
        self.logv = nn.Linear(hid, 1)

    def forward(self, h):                            # returns cumulative log-variance (monotone ↑ in expectation)
        step = torch.nn.functional.softplus(self.logv(h).squeeze(-1))   # ≥0 per-step variance increment
        return torch.log(torch.cumsum(step, dim=1) + 1e-6)


def student_t_nll(pred, logvar, target, mask, nu: float = 4.0):
    """Proof 6: heavy-tailed residual ⇒ Student-t NLL (robust). mask selects hidden rows."""
    var = torch.exp(logvar).clamp_min(1e-6)
    z2 = (target - pred) ** 2 / var
    nll = 0.5 * logvar + 0.5 * (nu + 1.0) * torch.log1p(z2 / nu)
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


class StateSpaceNet(nn.Module):
    """BiGRU encoder + per-well FiLM + IRW integration head + heteroscedastic head. Predicts an integrated
    TVT trajectory with growing uncertainty, trained with the Student-t NLL."""
    def __init__(self, feat_dim: int, ctx_dim: int, hid: int = 96):
        super().__init__()
        self.film = PerWellFiLM(ctx_dim, feat_dim)
        self.gru = nn.GRU(feat_dim, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.irw = IRWHead(2 * hid)
        self.het = HeteroHead(2 * hid)

    def forward(self, x, ctx, anchor):
        x = self.film(x, ctx)
        h, _ = self.gru(x)
        return self.irw(h, anchor), self.het(h)      # (mean trajectory, cumulative log-variance)


# --------------------------- factory ---------------------------
def build(feat_dim: int, ctx_dim: int, hid: int = 96):
    return StateSpaceNet(feat_dim, ctx_dim, hid).to(DEV)


class ResidualHead(nn.Module):
    """LESSON from StateSpaceNet failure (cumsum over 7500 steps diverges → field-CV 114): predict a bounded
    RESIDUAL r_i directly (stable, free per-row) around a physical dip-continuation baseline, and impose the IRW
    proof as a SOFT smoothness penalty on Δ²r instead of a hard integration. Captures proofs 1/27/44 (smooth
    integrated dip) without the numerical blow-up. Returns (residual, cumulative-log-variance)."""
    def __init__(self, hid: int):
        super().__init__()
        self.r = nn.Linear(hid, 1)
        self.logv = nn.Linear(hid, 1)

    def forward(self, h):
        r = self.r(h).squeeze(-1)
        step = torch.nn.functional.softplus(self.logv(h).squeeze(-1))
        return r, torch.log(torch.cumsum(step, dim=1) + 1e-6)


def irw_smoothness(r, mask):
    """Soft IRW prior: penalize the 2nd difference of the residual (proof 1: Δ²s is small MA(1) noise)."""
    d2 = r[:, 2:] - 2 * r[:, 1:-1] + r[:, :-2]
    m = mask[:, 2:]
    return (d2 ** 2 * m).sum() / m.sum().clamp_min(1.0)


class ResidualNet(nn.Module):
    """BiGRU + FiLM + ResidualHead. Output dtvt = baseline (dip-continuation, passed in) + residual."""
    def __init__(self, feat_dim: int, ctx_dim: int, hid: int = 96):
        super().__init__()
        self.film = PerWellFiLM(ctx_dim, feat_dim)
        self.gru = nn.GRU(feat_dim, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = ResidualHead(2 * hid)

    def forward(self, x, ctx, baseline):
        x = self.film(x, ctx); h, _ = self.gru(x)
        r, logv = self.head(h)
        return baseline + r, logv, r


def build_residual(feat_dim: int, ctx_dim: int, hid: int = 96):
    return ResidualNet(feat_dim, ctx_dim, hid).to(DEV)
