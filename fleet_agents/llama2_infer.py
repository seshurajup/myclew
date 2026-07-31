"""llama2-infer — REUSABLE minimal-transformer inference + int8-export planner distilled from
karpathy/llama2.c (single-file C inference of a Llama-2 style decoder). PURE, data-wise tested.

llama2.c value we keep (not a dependency, the *math*): exact param count, KV-cache memory, and
int8 group-wise quantized export sizing for a decoder-only transformer with GQA (n_kv_heads <=
n_heads), RoPE + RMSNorm + SwiGLU FFN. Lets any fleet comp that ships a tiny in-house LM (text /
agent-config modalities, T4-offline) size the model, its KV cache, and a Q8_0-style export BEFORE
building it — so we pick dims that fit VRAM and the group_size that keeps int8 error bounded.
Complements fp8/lowbit-qat (training-side) with the inference/export side. [[github_repo_integrations]]
"""
from __future__ import annotations

try:
    from .base import BaseAgent
except Exception:  # noqa: BLE001 — standalone (tests / offline)
    BaseAgent = object

# llama2.c Config (run.c): dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len.
CONFIG_FIELDS = ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len")


def param_count(cfg: dict, tied: bool = True) -> int:
    """Exact fp32 parameter count of a llama2.c decoder. head_size = dim // n_heads; GQA shrinks
    wk/wv to n_kv_heads. Matches the export.py layout: tok_emb + per-layer(2 rmsnorm + wq/wk/wv/wo
    + 3 FFN mats w1/w2/w3) + final rmsnorm + classifier. `tied` (shared_weights, the stories*.bin
    default) reuses tok_emb as the classifier, so no extra V*dim."""
    dim, hid, L = cfg["dim"], cfg["hidden_dim"], cfg["n_layers"]
    nh, nkv, V = cfg["n_heads"], cfg["n_kv_heads"], cfg["vocab_size"]
    head = dim // nh
    kv_dim = nkv * head
    per_layer = (
        2 * dim                       # attention + ffn RMSNorm gains
        + dim * (nh * head)           # wq
        + dim * kv_dim                # wk
        + dim * kv_dim                # wv
        + (nh * head) * dim           # wo
        + dim * hid + hid * dim + dim * hid  # SwiGLU: w1, w2, w3
    )
    cls = 0 if tied else V * dim
    return V * dim + L * per_layer + dim + cls  # tok_emb + layers + final norm + (classifier)


def kv_cache_bytes(cfg: dict, dtype_bytes: int = 4, batch: int = 1) -> int:
    """key_cache + value_cache = 2 * L * seq_len * kv_dim elements (llama2.c RunState). This is the
    memory that grows with context and usually decides whether a long-context tiny LM fits on a T4."""
    head = cfg["dim"] // cfg["n_heads"]
    kv_dim = cfg["n_kv_heads"] * head
    return 2 * cfg["n_layers"] * cfg["seq_len"] * kv_dim * dtype_bytes * batch


def int8_export_bytes(cfg: dict, group_size: int = 64) -> dict:
    """Q8_0-style (llama2.c export.py `version 2`) group-wise int8: every quantized matrix — token
    embedding, attention/FFN weights and the (tied) classifier — stores 1 int8 per weight + 1 fp32
    scale per `group_size` weights. Only the RMSNorm gains stay fp32. Returns the byte breakdown +
    the fp32->int8 compression ratio actually achieved."""
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    total = param_count(cfg)
    fp32_kept = (2 * cfg["n_layers"] * cfg["dim"] + cfg["dim"])  # rmsnorm gains stay fp32
    quantized = total - fp32_kept                                # everything else is Q8_0
    if quantized % group_size:
        # llama2.c requires every quantized matrix's size divisible by group_size
        quantized -= quantized % group_size
    scales = quantized // group_size
    q_bytes = quantized * 1 + scales * 4
    fp32_bytes = fp32_kept * 4
    return {
        "quantized_params": quantized,
        "group_size": group_size,
        "int8_bytes": q_bytes + fp32_bytes,
        "fp32_bytes": total * 4,
        "ratio": round(total * 4 / (q_bytes + fp32_bytes), 3),
    }


def fits(cfg: dict, vram_gb: float, dtype_bytes: int = 4, group_size: int | None = None) -> dict:
    """Weights + KV cache vs a VRAM budget (e.g. T4=15GB usable). If group_size given, weights are
    sized from the int8 export instead of fp32 — the lever for shipping a bigger tiny-LM offline."""
    w = int8_export_bytes(cfg, group_size)["int8_bytes"] if group_size else param_count(cfg) * dtype_bytes
    kv = kv_cache_bytes(cfg, dtype_bytes)
    need = w + kv
    budget = vram_gb * (1024 ** 3)
    return {"weight_bytes": w, "kv_bytes": kv, "need_gb": round(need / 1024 ** 3, 3),
            "budget_gb": vram_gb, "fits": need <= budget}


class Llama2Infer(BaseAgent):
    name = "llama2-infer"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q) if hasattr(self, "spec") else (q.get("spec", {}) if isinstance(q, dict) else {})
        cfg = spec.get("config") or {"dim": 288, "hidden_dim": 768, "n_layers": 6, "n_heads": 6,
                                     "n_kv_heads": 6, "vocab_size": 32000, "seq_len": 256}
        vram = float(spec.get("vram_gb", 15.0))
        gs = spec.get("group_size")
        rep = {"params": param_count(cfg), "kv_cache_bytes": kv_cache_bytes(cfg),
               "int8_export": int8_export_bytes(cfg, gs or 64), "fits_fp32": fits(cfg, vram),
               "fits_int8": fits(cfg, vram, group_size=gs or 64)}
        msg = (f"[{worker}] **llama2-infer** · {rep['params']/1e6:.1f}M params · "
               f"int8 ×{rep['int8_export']['ratio']} · fits@{vram}GB fp32={rep['fits_fp32']['fits']} "
               f"int8={rep['fits_int8']['fits']}")
        if hasattr(self, "done"):
            self.save_state(rep)
            self.post(worker, "leader", msg, routine=False, kind="verdict")
            return self.done(rep, msg, to="leader")
        return rep


_AGENT = Llama2Infer()


def run(q, worker):
    return _AGENT.run(q, worker)
