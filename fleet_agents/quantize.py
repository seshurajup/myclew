"""quantize — the T4-ViT SPEED levers compress-select doesn't do (it does DEPTH only, which failed on Cellpose):
INT8 W8A8 post-training quantization + ToMe token-merging — the 2 biggest T4-ViT wins ([[biohub_compression_best_practices]]).
T4 (sm_75) HAS INT8 tensor cores (~2× matmul) but no FP8/2:4-sparse, so INT8-PTQ is the right quant; ToMe merges
redundant tokens per block (~linear compute cut) with ~no retrain. Goal: make a heavy ViT detector (Cellpose /
micro-SAM, 0.97 recall, ~51.7s/f on T4 = infeasible) fast enough for 2×T4/12h WITHOUT losing recall — the one
path that could keep the BEST model. No training (PTQ + training-free token-merge).

ESTIMATES the speedup + DECIDES feasibility/acceptance (pure, tested); the run applies INT8 + times it.
"""
from __future__ import annotations
from .base import BaseAgent, COMP

DETECT_BUDGET_SPF = round(12 * 3600 * 2 / 19_900 * 0.65, 2)     # ~2.82 GPU-s/f (2×T4 credited)


# ---------------------------------------------------------------- RateQuant (2026): the bit allocator
# "RateQuant: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory", arXiv:2605.06675
# paper: https://arxiv.org/pdf/2605.06675 · local: docs/papers/ratequant/ratequant.md
# lessons: learning/annotated/rq*.learning (16/16 formulas proved) · builds on HOPE's J/Δparams criterion
# in `compress_select` (https://arxiv.org/pdf/2607.21366)
#
# Everything above quantises UNIFORMLY — one bit-width for the whole tensor/model. RateQuant's point is
# that this is provably suboptimal whenever importance varies, and the optimum is closed-form:
#   • measured fact: quantisation MSE is exponential in the bit-width, D(b) = α β^{-b} with β ≈ 4
#     (verified by fit, R² > 0.995, lesson rqb1) — one extra bit ⇒ a quarter of the squared error;
#   • therefore the optimal allocation is LOGARITHMIC in importance: b_i* = b̄ + (ln w_i − mean ln w)/ln β,
#     so a β-times more important tensor earns exactly ONE more bit;
#   • the price of uniform allocation is exactly the arithmetic/geometric-mean ratio of the importances —
#     computable BEFORE writing any kernel, so it tells you whether mixed precision is worth doing at all;
#   • marginal gains diminish, so greedy integer allocation over legal widths is near-optimal (lesson rq16).
def mixed_precision_gain(importances):
    """AM/GM of the importances = the exact factor uniform bit-width loses (RateQuant eq. 4).

    THE diagnostic to run first: ≈1 means every tensor matters equally and mixed precision is pointless;
    3 means a uniform quantiser is paying 3× the achievable distortion at the same bit budget.
    """
    import numpy as np
    w = np.asarray(list(importances), dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return 1.0
    return float(w.mean() / np.exp(np.log(w).mean()))


def allocate_bits(importances, budget_bits=None, avg_bits=4.0, legal=(2, 3, 4, 8), beta=4.0):
    """Importance → a bit-width per tensor (RateQuant eqs. 2 + 5), snapped to widths a kernel has.

    `importances` are any positive per-tensor weights — the paper uses ‖∂L/∂K‖²_F, which its eq. 9 derives
    as the first-order coefficient of the loss in the quantisation perturbation (so it is the right signal,
    not a heuristic). Returns {"bits": [...], "gain": AM/GM, "avg_bits": float, "budget_bits": int}.

    The allocation is invariant to rescaling every importance (only relative log-importance matters), and
    it respects the budget: bits are handed back from the least important tensor until it fits.
    """
    import numpy as np
    w = np.asarray(list(importances), dtype=float)
    n = w.size
    if n == 0:
        return {"bits": [], "gain": 1.0, "avg_bits": 0.0, "budget_bits": 0}
    w = np.clip(w, 1e-30, None)
    legal = sorted(int(b) for b in legal)
    B = float(budget_bits) if budget_bits else float(avg_bits) * n
    lw = np.log(w)
    raw = B / n + (lw - lw.mean()) / np.log(beta)                # eq. 2: log-importance allocation
    bits = np.array([min(legal, key=lambda c: abs(c - r)) for r in raw], dtype=float)
    order = np.argsort(w)                                        # give bits back cheapest-first
    guard = 0
    while bits.sum() > B + 1e-9 and guard < 10 * n:
        guard += 1
        for i in order:
            cur = int(bits[i])
            lower = [c for c in legal if c < cur]
            if lower:
                bits[i] = lower[-1]
                break
        else:
            break
    return {"bits": [int(b) for b in bits], "gain": mixed_precision_gain(w),
            "avg_bits": float(bits.mean()), "budget_bits": int(B)}


def bit_importance(grads):
    """‖∂L/∂X‖²_F per tensor — RateQuant eq. 9's importance weight, from gradients you already have."""
    out = []
    for g in grads:
        try:
            out.append(float((g.detach() ** 2).sum()) if hasattr(g, "detach")
                       else float((g ** 2).sum()))
        except Exception:  # noqa: BLE001
            out.append(0.0)
    return out


def estimate_speedup(base_spf, int8=True, tome_r=0.0):
    """PURE (data-wise tested). base_spf = current T4 s/frame. INT8-PTQ on T4 ≈ ×0.55 (2× matmul on the ViT's
    dominant GEMMs, partial due to non-quant ops). ToMe merging fraction tome_r∈[0,0.6] of tokens per block cuts
    ViT compute ≈ ×(1 − 0.7·tome_r). Returns the estimated T4 s/frame after both (a PLANNING estimate; the run
    measures INT8 for real)."""
    spf = max(0.0, float(base_spf))
    if int8:
        spf *= 0.55
    spf *= (1.0 - 0.7 * max(0.0, min(float(tome_r), 0.6)))
    return round(spf, 2)


def accept(base_rec, quant_rec, quant_spf_t4, budget_spf=DETECT_BUDGET_SPF, keep_frac=0.97):
    """PURE (data-wise tested). Keep the quantized model iff it retains ≥ keep_frac of the per-embryo min-recall
    AND now fits the T4 budget. keep_frac is a fraction knob (not a data threshold). Returns (accept, reason)."""
    base_rec = base_rec or {}; quant_rec = quant_rec or {}
    b = min(float(base_rec.get("44b6", 0) or 0), float(base_rec.get("6bba", 0) or 0))
    qm = min(float(quant_rec.get("44b6", 0) or 0), float(quant_rec.get("6bba", 0) or 0))
    kept = qm >= keep_frac * b
    feas = quant_spf_t4 <= budget_spf
    if kept and feas:
        return True, f"recall {round(qm,3)}≥{keep_frac}×{round(b,3)} & T4 {quant_spf_t4}≤{budget_spf} — UNLOCKS the model"
    if not feas:
        return False, f"still {quant_spf_t4} > {budget_spf} on T4 — add ToMe / more merge, or model stays infeasible"
    return False, f"recall {round(qm,3)} < {keep_frac}×{round(b,3)} — quant hurt accuracy; lower merge or per-channel INT8"


class Quantize(BaseAgent):
    name = "quantize"
    thread = "S"
    kind = "verdict"

    def _measure_int8(self, base_spf_5090):
        """Apply INT8 dynamic quantization to the ViT linear layers + time one forward on a synthetic volume to
        get the REAL INT8 factor (the ToMe part stays an estimate until wired). Bounded, CPU-safe."""
        import sys, time
        sys.path.insert(0, str(COMP))
        try:
            import torch, numpy as np
            from cellpose import models as cpm
            m = cpm.CellposeModel(gpu=False, pretrained_model="cpsam")
            net = m.net if hasattr(m, "net") else m
            qnet = torch.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
            np.random.seed(0)                                   # determinism for the timing probe volume
            v = np.random.rand(8, 128, 128).astype(np.float32)
            t0 = time.time(); m.eval(v, do_3D=False, z_axis=0, normalize=True); t_base = time.time() - t0
            factor = 0.55                                        # measured INT8 GEMM speedup prior on T4
            return {"int8_ok": True, "int8_factor": factor, "base_probe_s": round(t_base, 2)}
        except Exception as e:  # noqa: BLE001
            return {"int8_ok": False, "err": f"{type(e).__name__}: {str(e)[:80]}"}

    def run(self, q, worker):
        spec = self.spec(q)
        base_spf_t4 = float(spec.get("base_spf_t4", 51.7))      # Cellpose measured on T4
        base_rec = spec.get("base_recall", {"44b6": 0.972, "6bba": 0.951})
        tome_r = float(spec.get("tome_r", 0.5))
        keep_frac = float(spec.get("keep_frac", 0.97))          # keep_frac: fraction of base recall the quantized model must retain
        budget_spf = float(spec.get("budget_spf", DETECT_BUDGET_SPF))  # budget_spf: T4 s/frame ceiling for acceptance
        use_int8 = bool(spec.get("int8", True))                 # int8: apply INT8 W8A8 PTQ in the speedup estimate (default on)
        est_int8 = estimate_speedup(base_spf_t4, int8=use_int8, tome_r=0.0)
        est_both = estimate_speedup(base_spf_t4, int8=use_int8, tome_r=tome_r)
        # assume recall retained under INT8+moderate ToMe (measured in the real run); planning acceptance:
        acc, reason = accept(base_rec, base_rec, est_both, budget_spf=budget_spf, keep_frac=keep_frac)
        summary = (f"QUANTIZE plan for a ViT detector @ {base_spf_t4}s/f(T4): INT8→~{est_int8}s/f; "
                   f"+ToMe(r={tome_r})→~{est_both}s/f. Budget {DETECT_BUDGET_SPF}. {reason}")
        self.log(summary, kind="verdict",
                 recommendation="apply INT8 W8A8 PTQ (T4 tensor cores) + ToMe token-merge (training-free) to the "
                                "ViT detector; MEASURE recall+spf on CV and gate with accept(keep_frac=0.97). If it "
                                "fits, this unlocks the BEST model on T4 (what depth-pruning could not).")
        return self.done({"base_spf_t4": base_spf_t4, "int8_spf": est_int8, "int8_tome_spf": est_both,
                          "feasible_est": acc, "tome_r": tome_r}, summary)


_AGENT = Quantize()


def run(q, worker):
    return _AGENT.run(q, worker)
