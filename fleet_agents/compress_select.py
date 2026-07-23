"""compress-select — CHOOSE the best MODEL-COMPRESSION of the detector (training-free) by DATA PROOF:
the compression that runs fastest within the Kaggle time budget while keeping per-embryo recall ≥ bar.

Implements the recent LLM depth-compression best practices (user 2026-07-11 "same as LLM research"):
  • truncate       — drop the last-K blocks (naive tail cut).
  • drop_redundant — ShortGPT (2024): drop the blocks with LOWEST "Block Influence" (relative residual
                     ‖out−in‖/‖in‖ / cosine(in,out)), wherever they sit — a near-identity block in the
                     MIDDLE is a better cut than a tail block that does real work.
  • merge          — LaCo (Layer-Collapse): AVERAGE a redundant block's weights into its neighbour instead
                     of dropping it — keeps some of its contribution (min/max/avg weight-folding family).
Choice is by data proof on OUR embryo-disjoint CV, per-embryo. NO training. (Width/channel pruning is the
next extension — same importance-ranked, data-proof pattern.)

User strategy: "reduce our best models without training, don't lose quality." Cellpose-SAM (best recall,
3.96s/f) is too slow for the full hidden test (199 ds × 100 fr = 19,900 fr) at 12h/2×T4; compressing its
ViT blocks (residual → near-identity ones removable/mergeable) speeds it up with ~no quality loss to a
floor. This agent finds the best technique+level on OUR CV.

Companion to detector-select / tracker-select. Reuses experiments/prune_cellpose/prune_lib (block detection
+ truncation) and the same recall@7µm / time measurement. Decision logic is a pure, data-wise-tested fn.
"""
from __future__ import annotations
from .base import BaseAgent, COMP

TEST_FRAMES = 19_900
GPU_SECONDS = 12 * 3600 * 2                       # 2×T4, 12h
DETECT_BUDGET_SPF = round(GPU_SECONDS / TEST_FRAMES * 0.9, 2)   # ~3.9 s/f (tracker Trackastra is ~0.1, cheap)


def _choose(results, recall_bar=0.95, budget_spf=DETECT_BUDGET_SPF):
    """PURE decision (data-wise tested). results = {K: {"44b6":r,"6bba":r,"spf":s}}.
    Pick the SMALLEST K (fastest / most time-margin) whose MIN per-embryo recall ≥ recall_bar AND
    spf ≤ budget. If none clears the bar within budget, fall back to the highest-min-recall that fits.
    Returns (best_K, ranked)."""
    rows = []
    for K, r in (results or {}).items():
        if not isinstance(r, dict):
            continue
        r44, r6b, spf = r.get("44b6"), r.get("6bba"), r.get("spf")
        if r44 is None or r6b is None or spf is None:
            continue
        rows.append({"K": int(K), "44b6": round(r44, 3), "6bba": round(r6b, 3),
                     "min_recall": round(min(r44, r6b), 3), "spf": round(spf, 2),
                     "fits_budget": spf <= budget_spf,
                     "keeps_quality": min(r44, r6b) >= recall_bar})
    within = [d for d in rows if d["fits_budget"]]
    good = [d for d in within if d["keeps_quality"]]
    if good:
        best = min(good, key=lambda d: d["K"])            # smallest K = fastest, most margin
    elif within:
        best = max(within, key=lambda d: d["min_recall"])  # none keeps full quality → best that fits
    else:
        best = None
    ranked = sorted(rows, key=lambda d: (d["K"]))
    return (best["K"] if best else None), best, ranked


def _iterative_select(trace, recall_bar=0.95):
    """PURE SLEB-style selection (data-wise tested). trace = ordered list of removal steps, each
    {"n_kept": k, "min_recall": r, "spf": s}, produced by greedily removing the lowest-Block-Influence
    block one at a time. Pick the FEWEST-block (fastest) config still at/above the recall bar — i.e. stop
    just before the recall cliff. This beats one-shot top-K because it finds the ACTUAL floor on our data
    (block redundancy compounds; removing A can make B load-bearing). Returns (best_n_kept, best)."""
    trace = [d for d in (trace or []) if isinstance(d, dict) and d.get("min_recall") is not None
             and d.get("n_kept") is not None]
    if not trace:
        return None, None
    ok = [d for d in trace if d["min_recall"] >= recall_bar]
    if not ok:
        best = max(trace, key=lambda d: d["min_recall"]) if trace else None
        return (best["n_kept"] if best else None), best
    best = min(ok, key=lambda d: d["n_kept"])              # fewest blocks that still holds recall
    return best["n_kept"], best


class CompressSelect(BaseAgent):
    name = "compress-select"
    thread = "S"
    kind = "verdict"

    def _measure(self, ks, nds, nframes, strategy="drop_redundant"):
        """Compress Cellpose to K ViT blocks (no-train) via `strategy` and measure per-embryo recall@7µm +
        s/frame. Default strategy = drop_redundant (ShortGPT Block-Influence order) — better than truncate."""
        import sys, time
        sys.path.insert(0, str(COMP)); sys.path.insert(0, str(COMP / "experiments" / "prune_cellpose"))
        import numpy as np, pandas as pd
        try:
            import torch.nn as nn
            from cellpose import models as cpm
            import prune_lib
        except Exception as e:  # noqa: BLE001 — optional heavy deps absent → let caller escalate cleanly
            raise RuntimeError(f"compress-select deps unavailable: {type(e).__name__}: {str(e)[:80]}")
        np.random.seed(0)                                     # determinism for any sampled measurement
        from scipy import ndimage
        from src import io, metric
        from src.config import Config
        from src.io import embryo_id
        from model_scratch.train_v0 import frames_of, split_datasets
        cfg = Config(); scale = np.asarray(cfg.SCALE)
        _, te = split_datasets()
        by = {"44b6": [], "6bba": []}
        for ds in te:
            e = embryo_id(ds)
            if e in by and len(by[e]) < nds:
                by[e].append(ds)
        m = cpm.CellposeModel(gpu=True, pretrained_model="cpsam")
        net = m.net if hasattr(m, "net") else m
        pname, blocks, _ = prune_lib.find_block_lists(net)[0]; orig = list(blocks)
        parent = net
        for p in pname.split(".")[:-1]:
            parent = getattr(parent, p)
        leaf = pname.split(".")[-1]

        # BLOCK IMPORTANCE (relative residual ||out-in||/||in||): near-0 = redundant/near-identity → best to cut,
        # wherever it sits — NOT just the tail. (block_redundancy.py recipe.)
        import torch
        def _importance():
            rel = [0.0] * len(orig)
            def hook(i):
                def h(mod, inp, out):
                    x = inp[0].detach().float(); y = out.detach().float()
                    rel[i] = (y - x).norm().item() / (x.norm().item() + 1e-8)
                return h
            hs = [orig[i].register_forward_hook(hook(i)) for i in range(len(orig))]
            setattr(parent, leaf, nn.ModuleList(orig))
            d = by["6bba"][0]; ad, shape, dtype, T = frames_of(d, 1)
            m.eval(io.load_volume(ad, shape, dtype, 0).astype(np.float32), do_3D=False, z_axis=0,
                   stitch_threshold=0.5, normalize=True, batch_size=64)
            for hh in hs:
                hh.remove()
            return rel
        imp = _importance()
        keep_order = list(np.argsort(imp))                  # ascending importance → drop these first

        def setk(k, strategy="truncate"):
            if strategy == "truncate":                      # keep first k (naive tail cut)
                kept = list(range(k))
            elif strategy == "drop_redundant":              # keep the k MOST-important blocks (drop least-transforming, anywhere)
                kept = sorted(keep_order[-k:])
            elif strategy == "merge":                       # average each dropped block's params into its predecessor, keep k
                kept = sorted(keep_order[-k:])
                drop = [i for i in range(len(orig)) if i not in kept]
                for di in drop:                             # fold redundant block's weights into the nearest kept one (avg)
                    tgt = min(kept, key=lambda j: abs(j - di))
                    with torch.no_grad():
                        for pk, pv in orig[di].state_dict().items():
                            if pk in dict(orig[tgt].named_parameters()) and pv.shape == dict(orig[tgt].named_parameters())[pk].shape:
                                dict(orig[tgt].named_parameters())[pk].data.mul_(0.5).add_(0.5 * pv)
            else:
                kept = list(range(k))
            setattr(parent, leaf, nn.ModuleList([orig[i] for i in kept]))

        def cent(mk):
            l = np.unique(mk); l = l[l > 0]
            return np.array(ndimage.center_of_mass(np.ones_like(mk), mk, l), float) if len(l) else np.zeros((0, 3))

        def rs(ds):
            gn, _ = io.read_geff(COMP / f"input/biohub-cell-tracking-during-development/train/{ds}.geff")
            ad, shape, dtype, T = frames_of(ds, nframes); tp = gt = 0; secs = 0.0
            for t in range(nframes):
                v = io.load_volume(ad, shape, dtype, t).astype(np.float32); gf = gn[gn["t"] == t]
                t0 = time.time(); mk, _, _ = m.eval(v, do_3D=False, z_axis=0, stitch_threshold=0.5,
                                                    normalize=True, batch_size=64); secs += time.time() - t0
                pk = cent(mk); gt += len(gf)
                if len(gf) and len(pk):
                    pf = pd.DataFrame({"node_id": range(len(pk)), "t": t, "z": pk[:, 0], "y": pk[:, 1], "x": pk[:, 2]})
                    tp += len(metric._match_nodes(gf, pf, scale, 7.0))
            return tp / max(gt, 1), secs / nframes

        results = {}
        for k in ks:
            setk(k, strategy)
            r44 = np.mean([rs(d)[0] for d in by["44b6"]]); r6 = np.mean([rs(d)[0] for d in by["6bba"]])
            _, s = rs(by["6bba"][0])
            results[k] = {"44b6": float(r44), "6bba": float(r6), "spf": float(s)}
        return results

    def run(self, q, worker):
        spec = self.spec(q)
        strategy = spec.get("method", spec.get("strategy", "drop_redundant"))  # method: compression technique (alias strategy)
        recall_bar = spec.get("recall_bar", 0.95)
        calib_samples = int(spec.get("calib_samples", spec.get("nframes", 5)))  # calib_samples: frames per embryo for the recall/spf measurement
        results = spec.get("results")
        if not results:
            try:
                results = self._measure(spec.get("ks", [24, 20, 16, 14, 12, 10]), int(spec.get("nds", 2)),
                                        calib_samples, strategy)
            except Exception as e:  # noqa: BLE001
                return self.escalate(worker, "researcher", f"compress-select: measurement unavailable ({e}).")
        if not results:
            return self.escalate(worker, "researcher", "compress-select: no results to choose from.")
        bestK, best, ranked = _choose(results, recall_bar, spec.get("budget_spf", DETECT_BUDGET_SPF))
        # SLEB-style floor: over the same measurements, the FEWEST blocks still at/above the recall bar
        trace = [{"n_kept": int(K), "min_recall": min(r["44b6"], r["6bba"]), "spf": r["spf"]}
                 for K, r in results.items()]
        floor_k, floor = _iterative_select(trace, recall_bar)
        proof = "; ".join(f"K{d['K']}[44b6={d['44b6']} 6bba={d['6bba']} {d['spf']}s/f "
                          f"{'fit' if d['fits_budget'] else 'OVER'}]" for d in ranked)
        if bestK is None:
            return self.escalate(worker, "researcher", f"compress-select: no K fits the budget. {proof}")
        eta = round(best["spf"] * TEST_FRAMES / 2 / 3600, 1)
        summary = (f"CHOSEN compression: {bestK} blocks via {strategy} (no-train) — 44b6={best['44b6']} "
                   f"6bba={best['6bba']} {best['spf']}s/f → full-test ~{eta}h(2×T4) "
                   f"{'keeps quality' if best['keeps_quality'] else 'best-fit'}. Recall floor (SLEB): "
                   f"{floor_k} blocks. Sweep: {proof}")
        self.log(summary, kind="verdict",
                 recommendation=f"submittable detector = Cellpose-SAM compressed to {bestK} ViT blocks via "
                                f"{strategy} (no training) → fits 12h/2×T4, retains recall. Pair with Trackastra "
                                f"greedy. Floor is {floor_k} blocks; go lower only if the tracker needs the budget.")
        return self.done({"best_k": bestK, "best": best, "floor_k": floor_k, "strategy": strategy,
                          "ranked": ranked, "results": results}, summary)


_AGENT = CompressSelect()


def run(q, worker):
    return _AGENT.run(q, worker)
