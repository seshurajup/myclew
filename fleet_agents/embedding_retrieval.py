"""embedding_retrieval — a dense-embedding retrieval / RAG / dedup agent, with two self-hostable, permissively
licensed backends the user flagged:
  • NVIDIA Nemotron-3-Embed-1B-BF16 (~1.14B) — strong retrieval at small size, 34 languages incl. CROSS-lingual
    (query in one language, match docs in another). Linux Foundation license → free self-host, commercial, tunable.
    For: RAG over large corpora (low cost/query), multilingual + cross-lingual search, semantic dedup, clustering.
  • minishlab/potion-code-16M-v2 — a TINY (16M) STATIC (model2vec-distilled) CODE embedding model: no transformer
    forward, just a token→vector lookup + mean-pool, so it embeds at CPU speed for code retrieval / technical Q&A.

The retrieval MECHANICS are model-agnostic and pure-numpy (offline-testable): L2-normalize embeddings, cosine
top-k search, MMR diversification, and near-duplicate detection by cosine threshold. The model FORWARD is
import-guarded (sentence-transformers / model2vec) and falls back to a deterministic hashing embedder so tests
and dry-runs need no download. Pairs with `turboquant` (quantize the embedding bank 8× for cheap NN) and feeds
any RAG/skill loop via `llm_backend`.

Primitives (numpy; model load guarded):
  • normalize(V)                          — L2-normalize rows (unit vectors → cosine = dot).
  • search(bank, q, k)                    — cosine top-k retrieval.
  • mmr(bank, q, k, lambda_)              — Maximal Marginal Relevance: relevance − redundancy (diverse RAG ctx).
  • dedup(V, threshold)                   — near-duplicate clusters by cosine ≥ threshold.
  • embed(texts, model=...)               — Nemotron/potion via sentence-transformers/model2vec; hashing fallback.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent

NEMOTRON = "nvidia/Nemotron-3-Embed-1B-BF16"
POTION_CODE = "minishlab/potion-code-16M-v2"


def normalize(V, eps=1e-12):
    V = np.asarray(V, float)
    return V / np.clip(np.linalg.norm(V, axis=-1, keepdims=True), eps, None)


def search(bank, q, k=10):
    """Cosine top-k. bank (N,D), q (D,) or (Q,D). Returns (scores, idx). Assumes rows will be normalized here."""
    B = normalize(bank); q = normalize(np.atleast_2d(q))
    sims = q @ B.T                                          # (Q,N) cosine
    idx = np.argsort(-sims, axis=1)[:, :k]
    scores = np.take_along_axis(sims, idx, axis=1)
    return (scores[0], idx[0]) if scores.shape[0] == 1 else (scores, idx)


def mmr(bank, q, k=5, lambda_=0.5):
    """Maximal Marginal Relevance selection: iteratively pick the doc maximizing
    λ·sim(q,d) − (1−λ)·max_{s∈selected} sim(d,s) — relevance traded against redundancy, for diverse RAG context."""
    B = normalize(bank); qv = normalize(np.atleast_2d(q))[0]
    rel = B @ qv                                            # (N,)
    selected, cand = [], list(range(B.shape[0]))
    while cand and len(selected) < k:
        if not selected:
            i = int(np.argmax(rel[cand])); selected.append(cand.pop(i)); continue
        red = np.max(B[cand] @ B[selected].T, axis=1)      # max sim to already-selected
        score = lambda_ * rel[cand] - (1 - lambda_) * red
        j = int(np.argmax(score)); selected.append(cand.pop(j))
    return selected


def dedup(V, threshold=0.95):
    """Greedy near-duplicate clustering by cosine ≥ threshold. Returns list of clusters (index lists);
    singletons included. For deduping knowledge bases / tickets / corpora."""
    B = normalize(V); n = B.shape[0]; unassigned = set(range(n)); clusters = []
    while unassigned:
        i = min(unassigned); sims = B[list(unassigned)] @ B[i]
        members = [idx for idx, s in zip(sorted(unassigned), sims) if s >= threshold]
        clusters.append(members); unassigned -= set(members)
    return clusters


def _hash_embed(texts, dim=256, seed=0):
    """Deterministic hashing embedder (bag-of-token-hashes) — the offline fallback so retrieval works with no
    model download. NOT semantic, but stable and unit-testable; real semantics come from embed(model=...)."""
    out = np.zeros((len(texts), dim))
    for r, t in enumerate(texts):
        for tok in str(t).lower().split():
            h = (hash((seed, tok)) % dim + dim) % dim
            out[r, h] += 1.0
    return normalize(out)


def embed(texts, model=NEMOTRON, batch_size=32):
    """Embed texts with the named model. Tries model2vec (static, for potion-code) then sentence-transformers
    (Nemotron); falls back to a deterministic hashing embedder if neither/model is unavailable. Returns (N,D)."""
    texts = list(texts)
    if "potion" in model or "model2vec" in model:
        try:
            from model2vec import StaticModel
            return normalize(StaticModel.from_pretrained(model).encode(texts))
        except Exception:  # noqa: BLE001
            return _hash_embed(texts)
    try:
        from sentence_transformers import SentenceTransformer
        return normalize(SentenceTransformer(model).encode(texts, batch_size=batch_size,
                                                            normalize_embeddings=True))
    except Exception:  # noqa: BLE001
        return _hash_embed(texts)


# ---------------------------------------------------------------- agent
class EmbeddingRetrieval(BaseAgent):
    name = "embedding-retrieval"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        model = s.get("model", NEMOTRON)
        # deterministic offline proof of the retrieval mechanics on a tiny synthetic corpus
        docs = s.get("docs") or ["cell tracking in microscopy", "protein folding prediction",
                                 "microscopy cell segmentation", "stock market forecast",
                                 "tracking cells across frames", "portfolio optimization"]
        query = s.get("query", "track cells in microscopy video")
        V = embed(docs, model=model); qv = embed([query], model=model)[0]
        scores, idx = search(V, qv, k=3)
        div = mmr(V, qv, k=3, lambda_=0.6)
        clusters = dedup(V, threshold=0.99)
        used_real = not np.array_equal(V, _hash_embed(docs))  # heuristic: did a real model load?
        msg = (f"embedding-retrieval [{model.split('/')[-1]}{'' if used_real else ' (hashing-fallback: no model)'}]: "
               f"top-3 for {query!r} → docs {idx.tolist()} (cosine {np.round(scores,3).tolist()}); MMR-diverse "
               f"picks {div}; {len(clusters)} dedup clusters. Nemotron-3-Embed (34-lang, cross-lingual) / "
               f"potion-code (static, CPU) for RAG/dedup/code-search; quantize the bank with turboquant for 8×")
        self.log(msg, kind="finding",
                 recommendation="self-host Nemotron-3-Embed-1B for multilingual RAG (free, permissive); "
                                "potion-code-16M for CPU-speed code retrieval; pair with turboquant + llm_backend")
        return self.done({"top_idx": idx.tolist(), "n_clusters": len(clusters), "used_real_model": bool(used_real)}, msg)


_AGENT = EmbeddingRetrieval()


def run_embedding(q, worker):
    return _AGENT.run(q, worker)
