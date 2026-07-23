"""moe_inference_pack — the MIXTURE-OF-EXPERTS inference-cost lever behind Gemma 4's 26B-A4B (arXiv
2607.02770, Table 1 + §2): an MoE holds many experts (large TOTAL parameter count → sets the memory
footprint) but a router fires only top-k per token (small ACTIVE parameter count → sets the compute).
The whole trade — "26B total for ~4B active" — is pure accounting, testable offline with no model:

  • moe-inference-cost  — active-vs-total params, per-token FLOPs, compute-saving ratio, and the memory
                          the resident experts still cost, from (E experts, top-k active, expert size,
                          shared/dense params).

    active_params = shared + k · expert_params
    total_params  = shared + E · expert_params
    compute_ratio = active_params / total_params      (compute you pay vs a same-total dense model)
    per_token_flops ≈ 2 · active_params               (one forward MAC pair per active weight)
    memory_params = total_params                       (all experts must be resident)
"""
from __future__ import annotations
from .base import BaseAgent


# ---------------------------------------------------------------- core math
def moe_cost(n_experts, active_experts, expert_params, shared_params=0.0):
    """Parameter/compute accounting for one MoE config.
    n_experts E: total experts. active_experts k: experts fired per token (top-k).
    expert_params: params per expert. shared_params: always-on (attention, embeds, shared MLP)."""
    E = max(1, int(n_experts)); k = int(max(0, min(active_experts, E)))
    ep = max(0.0, float(expert_params)); sh = max(0.0, float(shared_params))
    active = sh + k * ep
    total = sh + E * ep
    ratio = active / total if total > 0 else 1.0
    return {"active_params": active, "total_params": total,
            "compute_ratio": ratio, "per_token_flops": 2.0 * active,
            "memory_params": total, "speedup_vs_dense_total": (1.0 / ratio if ratio > 0 else float("inf"))}


def dense_equivalent_flops(total_params):
    """Per-token FLOPs a DENSE model of the same total size would pay (the compute MoE avoids)."""
    return 2.0 * max(0.0, float(total_params))


# ---------------------------------------------------------------- agent
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class MoEInferenceCost(_B):
    name = "moe-inference-cost"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("n_experts", "active_experts", "expert_params") if k not in s]
        if missing:
            return self.escalate(worker, "leader",
                f"moe-inference-cost needs spec keys {missing} — none provided")
        r = moe_cost(int(s["n_experts"]), int(s["active_experts"]), float(s["expert_params"]),
                     float(s.get("shared_params", 0.0)))
        act_b = r["active_params"] / 1e9; tot_b = r["total_params"] / 1e9
        msg = (f"moe-inference-cost: {act_b:.2f}B active / {tot_b:.2f}B total "
               f"(compute {r['compute_ratio']*100:.1f}% of dense-total → {r['speedup_vs_dense_total']:.1f}× cheaper "
               f"compute), but {tot_b:.2f}B must stay resident")
        self.log(msg, kind="finding",
                 recommendation="MoE buys big-model capacity at small-model compute; memory (total params) is the constraint, not FLOPs — pair with QAT (lowbit-qat) to fit total params")
        return self.done({"active_b": act_b, "total_b": tot_b, "compute_ratio": r["compute_ratio"],
                          "per_token_flops": r["per_token_flops"], "memory_params": r["memory_params"],
                          "speedup_vs_dense_total": r["speedup_vs_dense_total"]}, msg)


_MOE = MoEInferenceCost()


def run_moe(q, worker): return _MOE.run(q, worker)
