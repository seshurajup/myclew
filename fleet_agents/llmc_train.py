"""llmc-train — REUSABLE GPT-2 training cost / MFU / kernel-precision planner distilled from
karpathy/llm.c (minimal C/CUDA GPT-2 pre-training). PURE, data-wise tested.

llm.c value we keep (the *arithmetic*, not the C): the exact FLOPs-per-token, MFU, tokens/sec and
wall-clock estimate for a GPT-2-family decoder at a given precision, plus the vocab-padding trick
(pad V to a multiple of 128 so the classifier matmul is tensor-core friendly). Lets any fleet comp
that pre-trains a small GPT (text / agent-config, on the 5090 or 2×T4) decide batch/precision and
predict cost BEFORE launching a trainer — pairs with hardware-tune (measured throughput) and
lowbit-qat (training-side quant). [[github_repo_integrations]] [[fp8_sm120_ecosystem_verdict]]
"""
from __future__ import annotations

try:
    from .base import BaseAgent
except Exception:  # noqa: BLE001
    BaseAgent = object

# llm.c GPT-2 family (train_gpt2.py): (n_layer, n_head, n_embd). vocab_size=50257, block_size=1024.
GPT2 = {
    "gpt2":        (12, 12, 768),    # 124M
    "gpt2-medium": (24, 16, 1024),   # 350M
    "gpt2-large":  (36, 20, 1280),   # 774M
    "gpt2-xl":     (48, 25, 1600),   # 1558M
}
VOCAB = 50257
# device peak bf16/fp16 TFLOPs (dense, tensor-core) used for MFU — extend as hardware is added.
PEAK_TFLOPS = {"A100": 312.0, "T4": 65.0, "5090": 209.5, "V100": 125.0, "H100": 989.0}
# llm.c precision speed multipliers vs fp32 baseline (tf32~8x, bf16~16x, fp8~2x over bf16 on Hopper+).
PREC_MULT = {"fp32": 1.0, "tf32": 8.0, "bf16": 16.0, "fp8": 32.0}


def pad_vocab(v: int = VOCAB, to: int = 128) -> int:
    """llm.c pads vocab_size up to a multiple of 128 (padded_vocab_size) so the final classifier and
    embedding matmuls hit tensor cores cleanly. 50257 -> 50304."""
    return ((v + to - 1) // to) * to


def param_count(n_layer: int, n_embd: int, vocab: int = VOCAB, block: int = 1024) -> int:
    """GPT-2 params: wte + wpe + L*(2 LayerNorm(2*n_embd) + attn(3*n_embd^2 qkv + n_embd^2 proj)
    + mlp(2*4*n_embd^2)) + final LN. Weight-tied head (wte reused), matching llm.c."""
    per_layer = 4 * n_embd + 4 * n_embd * n_embd + 8 * n_embd * n_embd
    return vocab * n_embd + block * n_embd + n_layer * per_layer + 2 * n_embd


def flops_per_token(n_layer: int, n_embd: int, n_head: int, seq: int, vocab: int = VOCAB) -> float:
    """Forward+backward FLOPs per token ~= 6*N + 12*L*n_embd*seq (the 6N approx plus the attention
    term that grows with context), the estimator llm.c prints for MFU."""
    N = param_count(n_layer, n_embd, vocab, seq)
    return 6 * N + 12 * n_layer * n_embd * seq


def estimate(model: str = "gpt2", device: str = "5090", precision: str = "bf16",
             batch_tokens: int = 2 ** 19, total_tokens: int = 10 ** 10, mfu: float = 0.4) -> dict:
    """Predict tokens/sec + wall-clock for pre-training a GPT-2-family model. batch_tokens is the
    global tokens/step (llm.c total_batch_size); mfu is realized model-FLOPs utilisation (0.3-0.5
    typical). Returns the numbers you need to decide feasibility on a 2×T4/12h code-comp budget."""
    if model not in GPT2:
        raise ValueError(f"unknown model {model}; know {list(GPT2)}")
    if device not in PEAK_TFLOPS:
        raise ValueError(f"unknown device {device}; know {list(PEAK_TFLOPS)}")
    n_layer, n_head, n_embd = GPT2[model]
    fpt = flops_per_token(n_layer, n_embd, n_head, seq=1024)
    peak = PEAK_TFLOPS[device] * 1e12 * (PREC_MULT.get(precision, 1.0) / PREC_MULT["bf16"])
    achieved = peak * mfu
    tok_per_s = achieved / fpt
    steps = max(1, total_tokens // batch_tokens)
    wall_s = total_tokens / tok_per_s
    return {
        "model": model, "params": param_count(n_layer, n_embd),
        "padded_vocab": pad_vocab(), "flops_per_token": fpt,
        "device": device, "precision": precision, "mfu": mfu,
        "tokens_per_sec": round(tok_per_s), "steps": steps,
        "wall_hours": round(wall_s / 3600, 2),
    }


class LlmcTrain(BaseAgent):
    name = "llmc-train"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        rep = estimate(
            model=spec.get("model", "gpt2"), device=spec.get("device", "5090"),
            precision=spec.get("precision", "bf16"),
            batch_tokens=int(spec.get("batch_tokens", 2 ** 19)),
            total_tokens=int(spec.get("total_tokens", 10 ** 10)),
            mfu=float(spec.get("mfu", 0.4)),
        )
        msg = (f"[{worker}] **llmc-train** · {rep['model']} {rep['params']/1e6:.0f}M · "
               f"{rep['precision']}@{rep['device']} · {rep['tokens_per_sec']:,} tok/s · "
               f"{rep['wall_hours']}h for {rep['steps']:,} steps")
        if hasattr(self, "done"):
            self.save_state(rep)
            self.post(worker, "leader", msg, routine=False, kind="verdict")
            return self.done(rep, msg, to="leader")
        return rep


_AGENT = LlmcTrain()


def run(q, worker):
    return _AGENT.run(q, worker)
