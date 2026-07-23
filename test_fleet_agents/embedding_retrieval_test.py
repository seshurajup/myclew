"""embedding_retrieval_test — data-wise verifier for the Nemotron/potion embedding retrieval agent.

Core properties (offline via the deterministic hashing embedder; no model download):
  1. normalize → unit rows; search returns exact cosine top-k (a query equal to a doc retrieves it first).
  2. MMR trades relevance for diversity (avoids returning near-duplicates back-to-back).
  3. dedup clusters near-identical vectors together and keeps distinct ones apart.
  4. embed() falls back to the hashing embedder deterministically when no model is present.
  5. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import embedding_retrieval as E


def _run():
    print("=== EMBEDDING-RETRIEVAL VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}

    # 1. normalize + search exactness
    V = rng.randn(20, 16)
    checks["normalize_unit"] = np.allclose(np.linalg.norm(E.normalize(V), axis=1), 1.0)
    scores, idx = E.search(V, V[7], k=3)
    checks["search_self_first"] = idx[0] == 7 and scores[0] > 0.999
    print(f"  -> query=doc7 → top idx {idx.tolist()} cos {np.round(scores,3).tolist()}")

    # 2. MMR diversity: with two near-duplicates, MMR should not pick both before a distinct doc
    base = rng.randn(16)
    bank = np.stack([base, base + 1e-3 * rng.randn(16),          # 0,1 near-duplicates
                     rng.randn(16), rng.randn(16)])
    q = base
    picks = E.mmr(bank, q, k=2, lambda_=0.5)
    checks["mmr_diverse"] = not (0 in picks and 1 in picks)      # shouldn't take both dupes
    picks_rel = E.mmr(bank, q, k=2, lambda_=1.0)                 # λ=1 → pure relevance, dupes allowed
    checks["mmr_lambda_relevance"] = picks_rel[0] in (0, 1)

    # 3. dedup
    D = np.stack([base, base + 1e-4 * rng.randn(16), rng.randn(16)])
    clusters = E.dedup(D, threshold=0.99)
    sizes = sorted(len(c) for c in clusters)
    checks["dedup_groups_dupes"] = sizes == [1, 2]
    print(f"  -> dedup cluster sizes: {sizes}")

    # 4. embed fallback deterministic
    e1 = E.embed(["cell tracking", "protein fold"], model="nonexistent/model")
    e2 = E.embed(["cell tracking", "protein fold"], model="nonexistent/model")
    checks["embed_fallback_deterministic"] = np.allclose(e1, e2) and e1.shape[0] == 2
    checks["embed_semantic_sanity"] = E.search(e1, E.embed(["cell tracking"], model="x")[0], k=1)[1][0] == 0

    # 5. agent
    st, dta, to, msg = E.run_embedding({"spec": {}}, "t")
    checks["agent_done"] = st == "done" and len(dta["top_idx"]) == 3

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== embedding-retrieval: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
