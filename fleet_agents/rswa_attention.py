"""rswa_attention — Reference Sliding Window Attention (R-SWA) from Baidu's "Unlimited OCR Works"
(arXiv:2606.23050), the mechanism that lets a 3B (500M-active) model transcribe dozens of pages in ONE 32K
forward pass with a CONSTANT KV cache. Standard causal attention caches K,V for every past token → the cache
grows linearly with decode length and blows up on long documents. Plain sliding-window attention bounds the
cache but throws away the document/prompt context once it scrolls out of the window. R-SWA keeps BOTH bounded
AND grounded: every query attends to (a) a fixed set of persistent REFERENCE tokens — the vision/prompt prefix
that encodes the page(s), which never leave the cache — plus (b) a LOCAL sliding window of the last W decoded
tokens. So the resident KV cache is exactly (n_reference + W) entries — constant in the output length — while
generation stays conditioned on the source document through the reference block.

Reconstructed from the paper's description (the released repo ships only a vLLM inference wrapper; R-SWA is in
the weights). The reusable, offline-testable pieces are the attention MASK and the CACHE-SIZE accounting:

Primitives (numpy/torch):
  • rswa_mask(seq_len, n_ref, window)      — boolean attend-mask: ref block (always) + causal local window.
  • kv_cache_size(n_ref, window, dtype…)   — constant resident cache vs linear full-causal cache.
  • rswa_attention(q, k, v, n_ref, window) — masked scaled-dot-product attention using the R-SWA mask.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


def rswa_mask(seq_len, n_ref, window):
    """Boolean (seq_len, seq_len) attend-mask (True = query i may attend to key j) for R-SWA:
      • key j < n_ref            → reference token, ALWAYS attendable by every query (persistent context).
      • n_ref ≤ j ≤ i and i-j<W  → local causal sliding window of width `window` over the generated tokens.
    Everything else is masked. The number of attendable keys per query is ≤ n_ref + window (constant)."""
    n = int(seq_len); r = int(n_ref); W = int(window)
    i = np.arange(n)[:, None]; j = np.arange(n)[None, :]
    ref = j < r                                                   # reference block: always visible
    local = (j >= r) & (j <= i) & (i - j < W)                    # causal window on the tail
    return ref | local


def kv_cache_size(n_ref, window, decode_len, dtype_bytes=2, kv_dim=1):
    """Resident KV-cache entries and bytes. R-SWA holds a CONSTANT (n_ref + window) entries no matter how long
    decode_len is; a full-causal cache holds (n_ref + decode_len). Returns both + the reduction ratio."""
    rswa = (n_ref + window)
    full = (n_ref + decode_len)
    b = 2 * kv_dim * dtype_bytes                                  # K and V
    return {"rswa_entries": rswa, "full_entries": full,
            "rswa_bytes": rswa * b, "full_bytes": full * b,
            "reduction": full / max(rswa, 1)}


def rswa_attention(q, k, v, n_ref, window):
    """Scaled-dot-product attention under the R-SWA mask. q,k,v: (seq, d). Returns (seq, d) output. Masked
    positions get -inf pre-softmax so they contribute zero — numerically identical to only caching the
    reference block + local window."""
    q = np.asarray(q, float); k = np.asarray(k, float); v = np.asarray(v, float)
    seq, d = q.shape
    scores = q @ k.T / np.sqrt(d)
    mask = rswa_mask(seq, n_ref, window)
    scores = np.where(mask, scores, -np.inf)
    scores -= scores.max(axis=1, keepdims=True)
    w = np.exp(scores); w /= w.sum(axis=1, keepdims=True)
    return w @ v


# ---------------------------------------------------------------- agent
class RSWAAttention(BaseAgent):
    name = "rswa-attention"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        rng = np.random.RandomState(int(s.get("seed", 0)))
        n_ref = int(s.get("n_ref", 64)); window = int(s.get("window", 128))
        decode = int(s.get("decode_len", 8000)); d = int(s.get("dim", 32))
        seq = n_ref + min(decode, 512)                           # bounded demo seq for the mask/attn check
        Q = rng.randn(seq, d); K = rng.randn(seq, d); V = rng.randn(seq, d)
        out = rswa_attention(Q, K, V, n_ref, window)
        mask = rswa_mask(seq, n_ref, window)
        max_keys = int(mask.sum(axis=1).max())
        cache = kv_cache_size(n_ref, window, decode)
        msg = (f"rswa-attention: R-SWA over seq={seq} (ref={n_ref}, window={window}) — each query attends ≤ "
               f"{max_keys} keys (ref+window, constant); at decode_len={decode} the KV-cache is "
               f"{cache['reduction']:.0f}× smaller than full-causal ({cache['rswa_entries']} vs "
               f"{cache['full_entries']} entries). Constant-memory long-document decoding (Unlimited-OCR R-SWA)")
        self.log(msg, kind="finding",
                 recommendation="for long-sequence decoding (OCR/doc parsing, long generation) keep a persistent "
                                "reference/prefix block + a local window → constant KV-cache instead of linear growth")
        return self.done({"max_keys_per_query": max_keys, "cache_reduction": cache["reduction"],
                          "rswa_entries": cache["rswa_entries"]}, msg)


_AGENT = RSWAAttention()


def run_rswa(q, worker):
    return _AGENT.run(q, worker)
