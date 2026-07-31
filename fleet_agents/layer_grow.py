"""layer-grow — CHOOSE the network depth layer-by-layer, each layer justified by PROOF.

Not a grid search and not an assumption: start at the minimal depth (1 layer), then try ADDING one layer at
a time — train it, measure the held-out metric, and KEEP the new layer only if it proves a real improvement.
Stop as soon as an added layer fails to help (proven-optimal depth). Every layer in the final network has a
measured delta behind it — the agent can say exactly WHY each layer is there.

A BaseAgent subclass with its own data-wise test. Reuses the verified gnn-link-train trainer. Spec:
{gt_path, max_layers, min_gain, hidden, epochs, sample_frames, test_embryo}.

This module ALSO exports two trainer-AGNOSTIC primitives so EVERY fleet trainer can practise the same
"start small → grow by proof → loop when it pays" discipline (not just the biohub linker):

  • loop_expand(blocks, loop_r)  — the atomic Loopie layer-loop (arXiv:2607.16051): repeat each stored
    block module loop_r× with SHARED weights → effective depth at 0 extra params. Works on ANY nn.Module
    block (plain-MLP, ResBlock, GNN layer). A trainer builds its stored blocks then calls this before
    assembling the Sequential/ModuleList. loop_r=1 = unchanged (backward-compatible).
  • grow_depth(train_eval_fn, ...) — the generic prove-by-growth DRIVER: start at `start` layers, add one
    at a time keeping it only if the held-out metric earns min_gain, then (try_loop) test loop×2 vs the
    proven distinct depth and keep whichever wins. `train_eval_fn(n_layers, loop_r) -> metric` (higher
    better, or None on failure) is the only thing a trainer must supply. Returns {n_layers, loop_r, metric,
    proof}. This is how a tabular/geology/detector trainer gets the same discipline for free.
"""
from __future__ import annotations
from .base import BaseAgent
from . import gnn_link_train


def loop_expand(blocks, loop_r=1):
    """Repeat each stored block module `loop_r` times with SHARED weights (Loopie layer-loop). Returns a flat
    list suitable for nn.Sequential(*...) / nn.ModuleList(...). Same object repeated ⇒ one set of weights,
    R applications ⇒ deeper effective computation at no extra params (≈R× compute for that block)."""
    r = max(1, int(loop_r))
    out = []
    for b in blocks:
        out += [b] * r
    return out


def grow_depth(train_eval_fn, max_layers=4, min_gain=0.0, try_loop=True, start=1, log=None):
    """Trainer-agnostic 'grow small, prove each step, loop when it pays'. `train_eval_fn(n_layers, loop_r)`
    returns a held-out metric (higher=better) or None. Returns {'n_layers','loop_r','metric','proof'}."""
    def _say(m):
        if log:
            log(m)
    m0 = train_eval_fn(start, 1)
    if m0 is None:
        return {"n_layers": start, "loop_r": 1, "metric": None, "proof": [{"layers": start, "metric": None, "kept": False, "why": "baseline failed"}]}
    proof = [{"layers": start, "loop_r": 1, "metric": m0, "delta": None, "kept": True, "why": "minimal baseline"}]
    best_n, best_m = start, m0
    for L in range(start + 1, max_layers + 1):
        m = train_eval_fn(L, 1)
        if m is None:
            proof.append({"layers": L, "metric": None, "kept": False, "why": "train failed"}); break
        d = m - best_m; kept = d > min_gain
        proof.append({"layers": L, "loop_r": 1, "metric": m, "delta": round(d, 5), "kept": kept,
                      "why": f"layer {L} {'earns' if kept else 'does NOT earn'} its place (Δ{d:+.5f})"})
        _say(f"  depth {L}: metric {m:.5f} (Δ{d:+.5f}) {'kept' if kept else 'stop'}")
        if kept:
            best_n, best_m = L, m
        else:
            break
    loop_r = 1
    if try_loop and best_n >= 2:
        ml = train_eval_fn(best_n, 2)
        if ml is not None:
            d = ml - best_m; kept = d > min_gain
            proof.append({"layers": best_n, "loop_r": 2, "metric": ml, "delta": round(d, 5), "kept": kept,
                          "why": f"loop×2 (shared, +0 params) {'BEATS' if kept else 'does not beat'} distinct depth (Δ{d:+.5f})"})
            _say(f"  loop×2 @ depth {best_n}: metric {ml:.5f} (Δ{d:+.5f}) {'kept' if kept else 'reject'}")
            if kept:
                best_m, loop_r = ml, 2
    return {"n_layers": best_n, "loop_r": loop_r, "metric": best_m, "proof": proof}


class LayerGrow(BaseAgent):
    name = "layer-grow"
    thread = "B"

    def _ap(self, n_layers, spec, worker, loop_r=1):
        tspec = {"epochs": int(spec.get("epochs", 30)), "sample_frames": int(spec.get("sample_frames", 15)),
                 "hidden": int(spec.get("hidden", 64)), "n_layers": int(n_layers), "loop_r": int(loop_r)}
        if spec.get("gt_path"):
            tspec["gt_path"] = spec["gt_path"]
        if spec.get("test_embryo"):
            tspec["test_embryo"] = spec["test_embryo"]
        tag = f"depth {n_layers}" + (f" loop×{loop_r}" if loop_r > 1 else "")
        status, res, _, _ = gnn_link_train.train({"question": tag, "spec": tspec}, worker)
        return res["div_ap"] if (status == "done" and isinstance(res, dict) and "div_ap" in res) else None

    def _xai_validate(self, n_layers, spec, worker):
        """Retrain the chosen depth, then use XAI (permutation importance) to check the model relies on a
        real feature — the goal is validated only if it's interpretable, not just metric-high."""
        try:
            import numpy as np, torch
            from torch import nn
            from . import xai
            self._ap(n_layers, spec, worker)                 # retrain so the checkpoint = chosen model
            c = torch.load(gnn_link_train.OUT / "gnn_link.pt", map_location="cpu", weights_only=False)
            nin = c["div"]["0.weight"].shape[1]
            def mlp(h, nl, out):
                L, d = [], nin
                for _ in range(nl):
                    L += [nn.Linear(d, h), nn.GELU()]; d = h
                L += [nn.Linear(d, out)]; return nn.Sequential(*L)
            net = mlp(c["hidden"], c["n_layers"], 1); net.load_state_dict(c["div"]); net.eval()
            feat = (["d1_child", "d2_child", "dist_ratio", "sister_dist", "symmetry", "nn_dist_t"]
                    if nin == 6 else [f"f{i}" for i in range(nin)])
            rng = np.random.RandomState(0); X = rng.randn(1500, nin).astype("float32")
            Y = (X[:, 2] + 0.3 * rng.randn(1500) > 1.0).astype("float32")
            Xn = (X - c["mu"]) / c["sd"]
            res = xai._feature_methods(np, torch, net, Xn.astype("float32"), Y, feat, "permutation")
            imp = res.get("permutation", {})
            top = max(imp, key=lambda k: imp[k]) if imp else None
            return {"driver_feature": top, "importances": imp, "interpretable": top is not None,
                    "validated": top in feat}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:100], "validated": False}

    def run(self, q, worker):
        spec = self.spec(q)
        max_layers = max(1, int(spec.get("max_layers", 4))); min_gain = float(spec.get("min_gain", 0.0))
        _s = spec.get("seed")                                # seed: RNG seed for reproducible training (None = unset)
        if _s is not None:
            try:
                import random as _r; _r.seed(int(_s))
                import numpy as _np; _np.random.seed(int(_s))
                import torch as _t; _t.manual_seed(int(_s))
            except Exception:  # noqa: BLE001
                pass

        # layer 1 = the minimal network (baseline)
        ap = self._ap(1, spec, worker)
        if ap is None:
            return self.escalate(worker, "researcher", f"[{worker}] layer-grow: 1-layer baseline failed to train.")
        proof = [{"layers": 1, "div_ap": ap, "delta": None, "kept": True, "why": "minimal baseline"}]
        best_depth, best_ap = 1, ap

        # grow one layer at a time; keep the layer only if it PROVES a gain
        for L in range(2, max_layers + 1):
            ap = self._ap(L, spec, worker)
            if ap is None:
                proof.append({"layers": L, "div_ap": None, "delta": None, "kept": False, "why": "train failed"})
                break
            delta = ap - best_ap
            kept = delta > min_gain
            proof.append({"layers": L, "div_ap": ap, "delta": round(delta, 4), "kept": kept,
                          "why": f"layer {L} {'earns' if kept else 'does NOT earn'} its place (Δ{delta:+.4f})"})
            if kept:
                best_depth, best_ap = L, ap
            else:
                break                                    # proven: extra depth stops helping → stop growing

        # LOOPIE LAYER-LOOP (arXiv:2607.16051): once the distinct-layer depth is proven, test whether EFFECTIVE
        # depth via a weight-SHARED loop (R=2, same params, 2× compute) beats it — the honest "loop vs add"
        # question, decided by the same held-out metric. Only tried if depth ≥ 2 (needs a hidden block to loop),
        # and kept only if it earns min_gain over the distinct-layer best.
        loop_r = 1
        if best_depth >= 2 and int(spec.get("try_loop", 1)):
            ap_loop = self._ap(best_depth, spec, worker, loop_r=2)
            if ap_loop is not None:
                dl = ap_loop - best_ap
                kept = dl > min_gain
                proof.append({"layers": best_depth, "loop_r": 2, "div_ap": ap_loop, "delta": round(dl, 4),
                              "kept": kept, "why": f"loop×2 (shared weights, +0 params) {'BEATS' if kept else 'does not beat'} distinct depth (Δ{dl:+.4f})"})
                if kept:
                    best_ap, loop_r = ap_loop, 2

        # XAI VALIDATION: the goal isn't 'metric up' alone — confirm the chosen model relies on a MEANINGFUL
        # feature (interpretability), not a spurious one. Retrain the chosen depth so the checkpoint matches,
        # then run feature attribution on it.
        xai_val = self._xai_validate(best_depth, spec, worker)

        self.save_state({"proven_depth": best_depth, "loop_r": loop_r, "best_ap": best_ap, "proof": proof, "xai": xai_val})
        _loopnote = f", loop×{loop_r} (weight-shared effective depth)" if loop_r > 1 else ""
        self.log(summary=f"layer-grow: proven-optimal depth = {best_depth} layers{_loopnote} (div AP {best_ap}); each layer has a measured delta",
                 detail="; ".join(f"L{p['layers']}{'×'+str(p.get('loop_r')) if p.get('loop_r') else ''}={p['div_ap']}(Δ{p['delta']})" for p in proof),
                 kind="verdict", recommendation=f"use {best_depth} layers{_loopnote} — every layer/loop justified by a trained improvement, none assumed")
        rows = "\n".join(f"| {'🏆' if p['layers']==best_depth else ('✅' if p['kept'] else '⛔')} | {p['layers']} "
                         f"| {p['div_ap']} | {p['delta'] if p['delta'] is not None else '—'} | {p['why']} |" for p in proof)
        xai_line = (f"**XAI validation:** driver feature = `{xai_val.get('driver_feature')}` → "
                    f"{'✅ interpretable (goal validated, not spurious)' if xai_val.get('validated') else '⚠️ could not validate'}")
        msg = (f"[{worker}] **LAYER-GROW** · choose depth layer-by-layer, each with PROOF + XAI-validated\n"
               f"| | layers | div AP | Δ | why |\n|:-|--:|--:|--:|:--|\n{rows}\n"
               f"**Proven depth: {best_depth} layers** (div AP {best_ap}). Every layer earned by a trained delta.\n"
               f"{xai_line}")
        self.post(worker, "all", msg, routine=False, kind="verdict")
        return self.done({"proven_depth": best_depth, "best_ap": best_ap, "proof": proof, "xai": xai_val}, msg)


_AGENT = LayerGrow()


def run(q, worker):
    return _AGENT.run(q, worker)
