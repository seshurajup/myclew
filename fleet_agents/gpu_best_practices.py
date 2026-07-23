"""gpu-best-practices — from our precision + 5090/Blackwell research: catalogue every GPU best practice
(compile, kernels, memory layout, precision) with its accuracy/speed effect and which hardware it needs,
and emit the cheap free-wins + the candidates arch-search must PROVE. A BaseAgent subclass (the enforced
pattern: extends BaseAgent, has a data-wise test).

Grounded in the session research: RTX 5090 = Blackwell (sm_120, native FP8/FP4), Kaggle T4 = Turing
(FP16/INT8 only). Speed is first-class (T4 runtime-limited code comp). Nothing is asserted to help — the
value-uncertain ones are search candidates for arch-search; only the safe free-wins are auto-adopt.
"""
from __future__ import annotations
import json
from .base import BaseAgent

# name -> (scope, accuracy, speed, needs, note, status)
PRACTICES = {
    "torch.compile":        ("train+infer", "0", "++", "any", "fuse+graph-capture; big speedup, ~no acc change", "adopt-cheap"),
    "TF32 matmul":          ("train", "-", "+", "ampere+", "torch.set_float32_matmul_precision('high'); tiny acc cost", "adopt-cheap"),
    "channels_last (3d)":   ("train+infer", "0", "+", "any", "memory format → better tensor-core utilisation", "adopt-cheap"),
    "AMP bf16 autocast":    ("train", "0", "+", "ampere+", "mixed precision; 2x mem/speed, stable", "adopt-cheap"),
    "fused AdamW":          ("train", "0", "+", "any", "fused optimiser kernel; less launch overhead", "adopt-cheap"),
    "CUDA graphs":          ("train+infer", "0", "++", "any", "capture static graph; kills per-step launch overhead", "search"),
    "FP8 (E4M3)":           ("train", "0", "++", "hopper/blackwell", "native FP8 on 5090; NOT on T4", "search"),
    "NVFP4":                ("train", "-", "++", "blackwell", "native 4-bit on 5090; stochastic rounding needed; NOT on T4", "search"),
    "INT8 PTQ":             ("infer", "-", "+", "any", "post-train int8 for inference; T4-friendly", "search"),
    "Flash-Attention":      ("train+infer", "0", "++", "ampere+", "IO-aware exact attention (if any attn head)", "adopt-if-attn"),
    "pin_memory + workers": ("train", "0", "+", "any", "overlap host→device copy with compute", "adopt-cheap"),
    "grad checkpointing":   ("train", "0", "-", "any", "trade compute for memory (only if OOM)", "conditional"),
    "set_to_none grads":    ("train", "0", "+", "any", "zero_grad(set_to_none=True); cheaper", "adopt-cheap"),
}


class GpuBestPractices(BaseAgent):
    name = "gpu-best-practices"
    thread = "A"

    def run(self, q, worker):
        spec = self.spec(q) or {}
        extra = spec.get("catalog")
        cat = {**PRACTICES, **(extra if isinstance(extra, dict) else {})}
        hardware = spec.get("hardware", "blackwell")   # 5090 by default; "turing" for the Kaggle T4
        try:
            sw = float(spec.get("speed_weight", 1.5))  # speed_weight: relative weight of speed vs accuracy
        except (TypeError, ValueError):
            sw = 1.5
        v = {"++": 2, "+": 1, "0": 0, "-": -1}

        rows = []
        for n, entry in cat.items():
            try:
                scope, acc, spd, needs, note, status = entry
            except (TypeError, ValueError):
                continue                               # skip malformed custom catalog rows, never crash
            ok_hw = needs == "any" or (hardware == "blackwell") or (hardware == "turing" and needs == "ampere+")
            score = v.get(acc, 0) + sw * v.get(spd, 0)
            rows.append((n, scope, acc, spd, needs, note, status, ok_hw, score))
        rows.sort(key=lambda r: -r[-1])

        adopt = [r[0] for r in rows if r[6].startswith("adopt") and r[7]]
        search = [r[0] for r in rows if r[6] == "search" and r[7]]
        blocked = [r[0] for r in rows if not r[7]]
        self.save_state({"hardware": hardware, "adopt": adopt, "search": search, "blocked_on_hw": blocked})
        self.log(summary=f"{len(rows)} GPU best practices ranked for {hardware}; {len(adopt)} free-wins, {len(search)} to prove",
                 detail="; ".join(f"{r[0]}(acc{r[2]},spd{r[3]})" for r in rows[:8]),
                 recommendation="adopt the free-wins (torch.compile/AMP/channels_last/fused-AdamW); arch-search proves FP8/FP4/CUDA-graphs")
        top = "\n".join(f"{'✅' if r[6].startswith('adopt') and r[7] else ('🔬' if r[7] else '⛔')} **{r[0]}** "
                        f"({r[1]}) acc`{r[2]}` speed`{r[3]}` needs`{r[4]}` — {r[5]}" for r in rows[:8])
        msg = (f"[{worker}] **GPU-BEST-PRACTICES** [{hardware}] · {len(rows)} practices · speed-weighted\n"
               f"{top}\n"
               f"**Free-wins to adopt:** {', '.join(adopt[:6])}\n"
               f"**To PROVE via arch-search:** {', '.join(search[:6])}"
               + (f"\n**Blocked on hardware (T4 lacks):** {', '.join(blocked)}" if blocked else ""))
        self.post(worker, "all", msg, routine=False, kind="finding")
        return self.done({"hardware": hardware, "adopt": adopt, "search": search, "blocked": blocked}, msg)


# module-level handler so the fleet registry (build_agents) can wrap/register it uniformly
_AGENT = GpuBestPractices()


def report(q, worker):
    return _AGENT.run(q, worker)
