"""perf-choice — benchmark the compute BACKENDS for a hot operation and recommend the fastest, so no agent
ever ships the slow choice again (the per-node Python loop that starved the GPU).

The lesson this codifies: for KNN/feature extraction the ranking is usually
  per-node Python loop  <<  vectorised numpy/scipy (batched query)  ≤  GPU-batched torch (large N).
A NAIVE 'move to GPU' per-item is often SLOWER than CPU (kernel-launch overhead) — batching is what wins.
This agent measures all three on a representative workload and returns the data-driven winner + the rule.

A BaseAgent subclass with its own data-wise test. Reusable/spec-driven: {n_frames, n_cells, k}.
"""
from __future__ import annotations
from .base import BaseAgent


class PerfChoice(BaseAgent):
    name = "perf-choice"
    thread = "A"

    def run(self, q, worker):
        import time
        try:
            import numpy as np
            from scipy.spatial import cKDTree
        except Exception as e:  # noqa: BLE001 — missing numpy/scipy → clean escalate, not a crash
            return self.escalate(worker, "researcher",
                                 f"[{worker}] perf-choice: numpy/scipy unavailable ({type(e).__name__}) — cannot benchmark.")
        spec = self.spec(q)

        def _pos(v, default):  # sanitised positive int
            try:
                return max(1, int(v))
            except (TypeError, ValueError):
                return default
        nf = _pos(spec.get("n_frames", 200), 200); nc = _pos(spec.get("n_cells", 400), 400)
        k = min(_pos(spec.get("k", 2), 2), nc)         # k cannot exceed the cell count
        try:
            seed = int(spec.get("seed", 0))            # seed: RNG seed for the synthetic workload (reproducible)
        except (TypeError, ValueError):
            seed = 0
        force_cpu = str(spec.get("device", "") or "").lower() in ("cpu", "-1")  # device: 'cpu' skips the GPU bench
        rng = np.random.RandomState(seed)
        frames = [rng.rand(nc, 3).astype("float32") * np.array([10, 40, 40], "float32") for _ in range(nf)]
        res = {}

        # 1) per-NODE python loop (the anti-pattern)
        t = time.time()
        for f in frames:
            tr = cKDTree(f)
            for i in range(len(f)):
                tr.query(f[i], k=k)
        res["per_node_loop"] = round(time.time() - t, 3)

        # 2) VECTORISED numpy/scipy (batched query — one call per frame)
        t = time.time()
        for f in frames:
            cKDTree(f).query(f, k=k)
        res["vectorised_cpu"] = round(time.time() - t, 3)

        # 3) GPU-batched torch (all frames one op) — only if torch+cuda (skipped when device='cpu')
        try:
            import torch
            if not force_cpu and torch.cuda.is_available():
                t = time.time()
                big = torch.tensor(np.stack(frames), device="cuda")
                torch.cdist(big, big).topk(k, largest=False)
                torch.cuda.synchronize()
                res["gpu_batched"] = round(time.time() - t, 3)
            else:
                res["gpu_batched"] = None
        except Exception:  # noqa: BLE001
            res["gpu_batched"] = None

        ranked = sorted(((k_, v) for k_, v in res.items() if v is not None), key=lambda kv: kv[1])
        best = ranked[0][0]
        speedup = round(res["per_node_loop"] / max(ranked[0][1], 1e-6), 1)
        self.save_state({"timings": res, "best": best, "speedup_vs_loop": speedup})
        self.log(summary=f"perf-choice: BEST backend = {best} ({speedup}x vs per-node loop) for {nf}f×{nc}c KNN",
                 detail="; ".join(f"{k_}={v}" for k_, v in res.items()), kind="verdict",
                 recommendation="NEVER per-node loop; vectorise (batched query); GPU-batched only for large N (per-item GPU is slower)")
        rows = "\n".join(f"| {'🏆' if k_ == best else ' '} | {k_} | {v if v is not None else 'n/a'}s |" for k_, v in res.items())
        msg = (f"[{worker}] **PERF-CHOICE** · KNN backend benchmark ({nf} frames × {nc} cells)\n"
               f"| | backend | time |\n|:-|:--|--:|\n{rows}\n"
               f"**Winner: `{best}` ({speedup}× vs the per-node loop).** Rule: vectorise (batched query); "
               f"GPU-batched only helps at large N — per-item GPU is *slower* than CPU (launch overhead).")
        self.post(worker, "all", msg, routine=False, kind="verdict")
        return self.done({"best": best, "timings": res, "speedup_vs_loop": speedup}, msg)


_AGENT = PerfChoice()


def run(q, worker):
    return _AGENT.run(q, worker)
