"""nvfp4_loader — practical loading + finetuning planner for NVIDIA NVFP4 models on the RTX 5090 (Blackwell,
sm_120), built for Unsloth's Gemma-4 NVFP4 quants. NVFP4 is Blackwell's NATIVE 4-bit tensor-core format (see
lowbit_qat.nvfp4_quantize): 16-elem blocks, FP8-E4M3 block scale, FP32 global scale → ~4.5 bits/weight, 1.5×
faster inference than fp16 and higher accuracy than MXFP4. The 5090's 32GB fits every Gemma-4 variant with room
for KV-cache; Gemma-4-12B fits in ~11GB.

This agent does the VRAM ARITHMETIC (weights at 4.5 b/w + KV-cache + activation headroom) so you know what fits
and how much context, and emits the concrete loader/finetune recipe. Pure-python accounting (offline-testable);
the actual model load is left to vLLM/Unsloth at run time.

Model registry (Gemma-4 NVFP4, params from the model cards):
  E2B/E4B (edge), 12B dense, 26B-A4B MoE (25.2B total / 3.8B active, 8/128 experts +1 shared), 31B dense.

Primitives:
  • nvfp4_weight_gb(total_params)             — weight VRAM at 4.5 bits/param.
  • kv_cache_gb(...)                          — KV-cache VRAM for a context length.
  • fits_5090(model, ctx, vram_gb=32)         — {weights, kv, total, fits, headroom}.
  • plan(model)                               — full loading + finetune recipe for the 5090.
"""
from __future__ import annotations
from .base import BaseAgent
from . import lowbit_qat as _q

# Gemma-4 NVFP4 registry: (total_params, active_params, n_layers, kv_heads, head_dim)
GEMMA4 = {
    "e2b":      (2.0e9,  2.0e9,  26, 4, 256),
    "e4b":      (4.0e9,  4.0e9,  34, 8, 256),
    "12b":      (12.0e9, 12.0e9, 48, 8, 256),
    "26b-a4b":  (25.2e9, 3.8e9,  48, 8, 256),   # MoE: 128 experts, 8 active + 1 shared
    "31b":      (31.0e9, 31.0e9, 56, 8, 256),
}
_NVFP4_BITS = 4.5      # lowbit_qat.nvfp4_effective_bits(16)


def nvfp4_weight_gb(total_params):
    """Weight VRAM (GiB) at NVFP4 ~4.5 bits/param (all params resident, incl. inactive MoE experts)."""
    return total_params * (_NVFP4_BITS / 8.0) / (1024 ** 3)


def kv_cache_gb(ctx_len, n_layers, kv_heads, head_dim, dtype_bytes=2, batch=1):
    """KV-cache VRAM (GiB): 2 (K,V) · layers · kv_heads · head_dim · ctx · batch · dtype_bytes."""
    return 2 * n_layers * kv_heads * head_dim * ctx_len * batch * dtype_bytes / (1024 ** 3)


def fits_5090(model, ctx_len=8192, vram_gb=32.0, batch=1, act_overhead_gb=2.0):
    """Does a Gemma-4 NVFP4 model fit the 5090 at a given context? Returns the VRAM breakdown + headroom."""
    tot, act, nl, kvh, hd = GEMMA4[model]
    w = nvfp4_weight_gb(tot)
    kv = kv_cache_gb(ctx_len, nl, kvh, hd, batch=batch)
    total = w + kv + act_overhead_gb
    return {"weights_gb": w, "kv_gb": kv, "act_gb": act_overhead_gb, "total_gb": total,
            "fits": total <= vram_gb, "headroom_gb": vram_gb - total, "active_params": act, "total_params": tot}


def max_context_5090(model, vram_gb=32.0, act_overhead_gb=2.0):
    """Largest context that fits on the 5090 after weights + activation headroom (KV fills the rest)."""
    tot, act, nl, kvh, hd = GEMMA4[model]
    free = vram_gb - nvfp4_weight_gb(tot) - act_overhead_gb
    per_tok = 2 * nl * kvh * hd * 2 / (1024 ** 3)               # GiB per token of context (bf16 KV)
    return max(0, int(free / per_tok)) if per_tok > 0 else 0


def plan(model="12b"):
    """Loading + finetune recipe for a Gemma-4 NVFP4 model on the 5090."""
    f = fits_5090(model)
    infer = ("uv pip install 'vllm>=0.25.0' 'flashinfer-python>=0.6.13'\n"
             f"vllm serve unsloth/gemma-4-{model}-it-NVFP4   # let vLLM auto-pick the NVFP4 kernel; "
             "do NOT force Marlin (~2× slower on Blackwell)")
    finetune = (f"from unsloth import FastModel\n"
                f"model, tok = FastModel.from_pretrained('unsloth/gemma-4-{model}-it-NVFP4', max_seq_length=2048)\n"
                "# QLoRA-style: adapters in bf16 over frozen NVFP4 base (5090 sm_120 native NVFP4 matmul).\n"
                "# For custom NVFP4 QAT of your OWN model, use lowbit_qat.QuantLinear(scheme='nvfp4').")
    return {"fits": f["fits"], "weights_gb": round(f["weights_gb"], 1), "headroom_gb": round(f["headroom_gb"], 1),
            "infer": infer, "finetune": finetune}


# ---------------------------------------------------------------- agent
class NVFP4Loader(BaseAgent):
    name = "nvfp4-loader"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        model = str(s.get("model", "12b")).lower()
        if model not in GEMMA4:
            return self.escalate(q, "leader", f"nvfp4-loader: unknown model {model}; have {list(GEMMA4)}")
        ctx = int(s.get("ctx_len", 8192))
        f = fits_5090(model, ctx_len=ctx)
        maxc = max_context_5090(model)
        eb = _q.nvfp4_effective_bits(16)
        msg = (f"nvfp4-loader [gemma-4-{model} NVFP4 @{eb}b/w on RTX 5090 32GB]: weights {f['weights_gb']:.1f}GB "
               f"+ KV@{ctx}={f['kv_gb']:.1f}GB + {f['act_gb']:.0f}GB act = {f['total_gb']:.1f}GB → "
               f"{'FITS' if f['fits'] else 'OOM'} ({f['headroom_gb']:.1f}GB headroom); max context ≈ {maxc:,} tok. "
               f"Load: vLLM auto-NVFP4 kernel (NOT Marlin) or Unsloth FastModel; finetune = bf16 LoRA over frozen "
               f"NVFP4 base. Active {f['active_params']/1e9:.1f}B/{f['total_params']/1e9:.1f}B total")
        self.log(msg, kind="finding",
                 recommendation="run Gemma-4 NVFP4 on the 5090 via vLLM (native Blackwell FP4 kernel, 1.5× faster); "
                                "for own-model NVFP4 QAT use lowbit_qat.QuantLinear(scheme='nvfp4')")
        return self.done({"model": model, "fits": f["fits"], "weights_gb": f["weights_gb"],
                          "max_context": maxc, "total_gb": f["total_gb"]}, msg)


_AGENT = NVFP4Loader()


def run_nvfp4loader(q, worker):
    return _AGENT.run(q, worker)
