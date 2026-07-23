"""mtp_speculative_pack — the SPECULATIVE-DECODING lever from Gemma 4 (arXiv 2607.02770, §2.6 + Figure 1):
a small autoregressive MTP *drafter* head proposes γ future tokens, the big model VERIFIES them in one
parallel pass, and the longest correct prefix is accepted. This turns serial 1-token-per-pass decoding
into several-tokens-per-verify. All of it is pure control arithmetic — testable offline with no model:

  • mtp-speculative-decode  — expected accepted tokens per verify pass, decode speedup vs a target/draft
                              cost ratio, and the draft length γ that maximizes speedup.

Math (Leviathan et al., 2023, the paper Gemma 4 cites for speculative decoding): with i.i.d. per-token
acceptance probability α and draft length γ, the expected number of tokens produced per iteration
(including the one bonus token the target samples on a rejection/at the end) is

    E[tokens] = (1 - α^(γ+1)) / (1 - α)          (α<1;  = γ+1 when α→1)

One iteration costs one target forward pass plus γ cheap drafter passes. If c = drafter_cost/target_cost,
the wall-clock speedup over standard decoding (1 token per target pass) is

    speedup = E[tokens] / (1 + γ·c)

This is the exact lever Figure 1 (the MTP drafter fed the main model's activations + KV cache) buys.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- core math
def expected_accepted(alpha, gamma):
    """Expected tokens produced per speculative iteration (incl. the bonus target token).
    alpha: per-token acceptance prob in [0,1]. gamma: draft length (int >=0)."""
    a = float(np.clip(alpha, 0.0, 1.0)); g = int(max(0, gamma))
    if a >= 1.0:
        return float(g + 1)
    return float((1.0 - a ** (g + 1)) / (1.0 - a))


def decode_speedup(alpha, gamma, cost_ratio=0.0):
    """Wall-clock speedup vs standard 1-token decoding. cost_ratio c = drafter_cost/target_cost (>=0).
    speedup = E[tokens] / (1 + gamma*c). With c=0 (free drafter) it is just E[tokens]."""
    g = int(max(0, gamma)); c = max(0.0, float(cost_ratio))
    denom = 1.0 + g * c
    return expected_accepted(alpha, g) / denom if denom > 0 else 0.0


def optimal_draft_length(alpha, cost_ratio=0.0, gamma_max=16):
    """The draft length γ in [1, gamma_max] that maximizes decode_speedup, and that speedup.
    Longer γ helps while acceptance holds but is eventually eaten by γ·c drafter cost."""
    gmax = int(max(1, gamma_max))
    best_g, best_s = 1, -1.0
    for g in range(1, gmax + 1):
        s = decode_speedup(alpha, g, cost_ratio)
        if s > best_s:
            best_s, best_g = s, g
    return best_g, float(best_s)


# ---------------------------------------------------------------- agent
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class MTPSpeculativeDecode(_B):
    name = "mtp-speculative-decode"
    def run(self, q, worker):
        s = self.spec(q)
        if "alpha" not in s and "acceptance" not in s:
            return self.escalate(worker, "leader",
                "mtp-speculative-decode needs spec key 'alpha' (per-token acceptance prob) — none provided")
        alpha = float(s.get("alpha", s.get("acceptance")))
        c = float(s.get("cost_ratio", 0.0))
        gmax = int(s.get("gamma_max", 16))
        if "gamma" in s:
            g = int(s["gamma"])
            e = expected_accepted(alpha, g); sp = decode_speedup(alpha, g, c)
            gbest, sbest = optimal_draft_length(alpha, c, gmax)
            msg = (f"mtp-speculative-decode: α={alpha:.2f} γ={g} → {e:.2f} tok/verify, "
                   f"{sp:.2f}× decode speedup (c={c:g}); best γ={gbest} → {sbest:.2f}×")
            data = {"expected_tokens": e, "speedup": sp, "best_gamma": gbest, "best_speedup": sbest}
        else:
            gbest, sbest = optimal_draft_length(alpha, c, gmax)
            e = expected_accepted(alpha, gbest)
            msg = (f"mtp-speculative-decode: α={alpha:.2f} → best γ={gbest} gives {e:.2f} tok/verify, "
                   f"{sbest:.2f}× decode speedup (c={c:g})")
            data = {"best_gamma": gbest, "best_speedup": sbest, "expected_tokens": e, "speedup": sbest}
        self.log(msg, kind="finding",
                 recommendation="raise α (better drafter) or cut c (smaller draft head, e.g. Gemma-4's top-k-cluster projection) to widen the speedup")
        return self.done(data, msg)


_MTP = MTPSpeculativeDecode()


def run_mtp(q, worker): return _MTP.run(q, worker)
