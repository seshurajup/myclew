"""mup_scaling — Maximal-Update Parametrization (μP) width-scaling rules, the scaling half of microsoft/
ArchScale (its optimizer half, Muon/Newton-Schulz, is already in muon_optimizer.py). μP lets you tune
hyperparameters (LR, init) on a SMALL model and transfer them to a WIDE model unchanged, because it keeps
every layer's activation and per-step update O(1) as width→∞. The load-bearing rule the standard fan-in
parametrization gets wrong is the READOUT/output layer: a plain constant LR makes the per-step change in the
logits grow with width (∝ width), so the wide model needs a different LR than the narrow one. μP fixes this by
scaling the readout LR by 1/width (and its init by 1/fan_in), making Δlogit width-independent → the same LR is
optimal at every width. This is the "coordinate check": plot per-step activation change vs width; μP is flat.

Why it matters for the fleet: any time we scale a model up (bigger UNet, wider tabular MLP, larger transformer)
μP means we DON'T re-sweep LR — tune it once small, apply the rules, scale up. Pure-math, offline-testable.

Primitives (numpy, deps = numpy):
  • mup_init_std(fan_in, readout=False)      — μP init std: 1/sqrt(fan_in) hidden, 1/fan_in readout.
  • mup_lr_scale(width, base_width, layer)   — LR multiplier: 1 hidden (Adam), base_width/width readout.
  • delta_logit(width, lr, mup)              — per-step logit change from a readout SGD update (coord check).
  • coordinate_check(widths, mup)            — Δlogit across widths; μP → flat, standard → grows ∝ width.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


def mup_init_std(fan_in, readout=False):
    """μP init standard deviation. Hidden layers use fan-in init 1/sqrt(fan_in) (keeps preactivations O(1));
    the readout layer uses 1/fan_in so its OUTPUT is O(1/sqrt(width)) at init, part of the μP recipe."""
    fan_in = max(1, int(fan_in))
    return (1.0 / fan_in) if readout else (1.0 / np.sqrt(fan_in))


def mup_lr_scale(width, base_width=256, layer="hidden", optimizer="adam"):
    """Per-layer LR multiplier for μP under Adam (the ArchScale default). Hidden layers: 1 (constant LR
    transfers). Readout/output: base_width/width — the key rule so Δlogit stays O(1) as width grows.
    Input/embedding: 1. (SGD would use different exponents; Adam is the common case.)"""
    width = max(1, int(width)); base = max(1, int(base_width))
    if layer in ("readout", "output", "unembed"):
        return base / width
    return 1.0


def delta_logit(width, lr, mup=True, base_width=64, seed=0):
    """One SGD step on a readout w (shape (width,)) with y = wᵀh; measure |Δy| = lr·||h||² (with dL/dy=1).
    ||h||² ~ width, so a constant lr makes |Δy| grow ∝ width (standard param); μP scales lr by base/width so
    |Δy| is width-independent. Returns |Δy| for one random hidden vector h with O(1) coordinates."""
    rng = np.random.RandomState(seed + width)
    h = rng.randn(int(width))                                    # O(1)-coordinate hidden activation
    eff_lr = lr * (mup_lr_scale(width, base_width, "readout") if mup else 1.0)
    # y = w·h ; grad wrt w = (dL/dy)·h ; with dL/dy=1, Δw = -eff_lr·h ; Δy = Δw·h = -eff_lr·||h||²
    return float(eff_lr * (h @ h))


def coordinate_check(widths, mup=True, lr=0.1, base_width=64):
    """Per-step |Δlogit| across a list of widths. μP → roughly flat (width-independent); standard param →
    grows ∝ width. Returns dict width→|Δy|."""
    return {int(w): delta_logit(int(w), lr, mup=mup, base_width=base_width) for w in widths}


# ---------------------------------------------------------------- agent
class MuPScaling(BaseAgent):
    name = "mup-scaling"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        widths = list(s.get("widths", [64, 128, 256, 512, 1024]))
        base = int(s.get("base_width", 64)); lr = float(s.get("lr", 0.1))
        mu = coordinate_check(widths, mup=True, lr=lr, base_width=base)
        sp = coordinate_check(widths, mup=False, lr=lr, base_width=base)
        mu_spread = max(mu.values()) / max(min(mu.values()), 1e-9)
        sp_spread = max(sp.values()) / max(min(sp.values()), 1e-9)
        msg = (f"mup-scaling: coordinate check over widths {widths} — μP |Δlogit| spread={mu_spread:.2f}× "
               f"(flat → LR transfers) vs standard-param spread={sp_spread:.1f}× (grows ∝ width → needs re-tuning). "
               f"Tune LR/init small, apply μP rules (readout LR ×base/width, readout init 1/fan_in), scale up "
               f"without re-sweeping (ArchScale μP)")
        self.log(msg, kind="finding",
                 recommendation="when widening any model, use mup_init_std + mup_lr_scale so the small-model LR "
                                "transfers; readout LR must scale 1/width or the wide model diverges/underfits")
        return self.done({"mup_spread": mu_spread, "sp_spread": sp_spread}, msg)


_AGENT = MuPScaling()


def run_mup(q, worker):
    return _AGENT.run(q, worker)
