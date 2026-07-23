"""arch-search — PROVE arch-builder's search space by starting SIMPLE and growing one axis at a time.

The user's rule: don't jump to '8 heads' or grid-search — start from the BASIC architecture (the smallest
value on every axis) and change ONE axis at a time, keeping a change only if it TRAINS to a better held-out
metric. Greedy coordinate-ascent from the minimal baseline: every added bit of complexity must earn its
place with data. Reuses the verified gnn-link-train trainer.

A BaseAgent subclass with its own data-wise test. Reusable/spec-driven: {gt_path, search_space, epochs,
sample_frames, test_embryo}. Axis value order matters — put the SIMPLEST value first (it's the baseline).
"""
from __future__ import annotations
from .base import BaseAgent
from . import gnn_link_train


def _seed(seed):
    """Best-effort determinism across numpy/torch/random; no-op if seed is None or deps absent."""
    if seed is None:
        return
    try:
        import random as _r
        _r.seed(int(seed))
    except Exception:  # noqa: BLE001
        pass
    for mod, fn in (("numpy", "seed"), ("torch", "manual_seed")):
        try:
            m = __import__(mod)
            getattr(m.random, fn)(int(seed)) if mod == "numpy" else getattr(m, fn)(int(seed))
        except Exception:  # noqa: BLE001
            pass


class ArchSearch(BaseAgent):
    name = "arch-search"
    thread = "B"

    def _train(self, cfg, spec, worker):
        tspec = {"epochs": int(spec.get("epochs", 40)), "sample_frames": int(spec.get("sample_frames", 20)),
                 "hidden": int(cfg.get("hidden_dim", 64)), "n_layers": int(cfg.get("n_layers", 2))}
        # forward the training-data + augmentation knobs so arch-search can design on the EXTERNAL set
        for k in ("gt_path", "test_embryo", "include_embryos", "feat_jitter_std", "div_pos_weight"):
            if spec.get(k) is not None:
                tspec[k] = spec[k]
        status, res, _, _ = gnn_link_train.train({"question": f"arch {cfg}", "spec": tspec}, worker)
        if status == "done" and isinstance(res, dict) and "div_ap" in res:
            return res["div_ap"]
        return None

    def _competition_ap(self, spec):
        """DECISION METRIC = transfer to the COMPETITION embryos (the real target), not held-out external.
        Reuses ext-transfer's per-embryo eval on the model gnn-link-train just saved. Returns mean AP."""
        from pathlib import Path
        from . import ext_transfer
        C = Path(__file__).resolve().parent.parent
        ckpt = C / "results" / "gnn_link" / "gnn_link.pt"
        comp = C / spec.get("comp_parquet", "results/flow_gt/competition_flow.parquet")
        if not (ckpt.exists() and comp.exists()):
            return None
        try:
            t = ext_transfer.ExtTransfer()._eval_competition(str(ckpt), str(comp), spec)
        except Exception:  # noqa: BLE001
            return None
        aps = [v["ap"] for v in t.values() if isinstance(v, dict) and isinstance(v.get("ap"), (int, float))]
        return round(sum(aps) / len(aps), 4) if aps else None

    def _decide(self, cfg, step, spec, worker):
        """Train the candidate, then score it by the DECISION metric (competition transfer if we trained on
        external, else held-out external), and RECORD it as a row in the new-dataset journal section."""
        ext_ap = self._train(cfg, spec, worker)
        if ext_ap is None:
            return None
        decide_on = spec.get("decide_on") or ("competition" if spec.get("include_embryos") else "external")
        comp_ap = self._competition_ap(spec) if decide_on == "competition" else None
        metric = comp_ap if (decide_on == "competition" and comp_ap is not None) else ext_ap
        # NEW JOURNAL SECTION: trn_set groups the table, so a new dataset name = a new table on /journal
        try:
            self.record(change=f"arch_{step.replace('→','_')}_{cfg.get('hidden_dim')}h{cfg.get('n_layers')}L",
                        script="fleet_dispatch arch-search", cv=metric, train_set=spec.get("journal_set", "ext_comp"),
                        description=f"arch {step}: {cfg} — decide-on {decide_on} (comp transfer AP {comp_ap}, ext held-out {ext_ap})")
        except Exception:  # noqa: BLE001
            pass
        return metric

    def run(self, q, worker):
        spec = self.spec(q)
        space = spec.get("search_space") or {"hidden_dim": [32, 64, 128], "n_layers": [2, 3, 4]}
        space = {a: list(v) for a, v in space.items() if v}   # drop empty axes (degenerate space)
        axes = list(space.keys())
        if not axes:
            return self.escalate(worker, "researcher", f"[{worker}] arch-search: empty search_space.")
        decide_on = spec.get("decide_on") or ("competition" if spec.get("include_embryos") else "external")
        budget = spec.get("budget")                           # budget: max candidate trainings (None = full coordinate sweep)
        budget = int(budget) if budget is not None else None
        early_stop = spec.get("early_stop")                  # early_stop: stop an axis after this many non-improving trials (None = try all)
        early_stop = int(early_stop) if early_stop is not None else None
        _seed(spec.get("seed"))                              # seed: RNG seed for reproducible training (None = unset)

        # 1) BASIC baseline = the simplest value on every axis (first in each list)
        best_cfg = {a: space[a][0] for a in axes}
        base_ap = self._decide(best_cfg, "basic baseline", spec, worker)
        if base_ap is None:
            return self.escalate(worker, "researcher",
                                 f"[{worker}] arch-search: the BASIC baseline failed to train (check GT/trainer).")
        trail = [{"config": dict(best_cfg), "div_ap": base_ap, "step": "basic baseline", "kept": True}]
        best_ap = base_ap

        # 2) grow ONE axis at a time; keep a bump ONLY if it scores better on the DECISION metric
        n_trained = 1                                        # the basic baseline already counts
        for a in axes:
            misses = 0
            for val in space[a][1:]:                 # try the next-more-complex values, in order
                if val == best_cfg[a]:
                    continue
                if budget is not None and n_trained >= budget:
                    break
                cand = dict(best_cfg); cand[a] = val
                ap = self._decide(cand, f"{a}→{val}", spec, worker)
                if ap is None:
                    continue
                n_trained += 1
                improved = ap > best_ap + 1e-6
                trail.append({"config": dict(cand), "div_ap": ap, "step": f"{a}→{val}", "kept": improved})
                if improved:
                    best_cfg, best_ap = cand, ap      # adopt only proven complexity
                    misses = 0
                else:
                    misses += 1
                    if early_stop is not None and misses >= early_stop:
                        break                         # this axis stopped paying off
            if budget is not None and n_trained >= budget:
                break

        self.save_state({"best_config": best_cfg, "best_ap": best_ap, "base_ap": base_ap,
                         "decide_on": decide_on, "steps": trail})
        self.log(summary=f"arch-search (simple→complex): basic AP {base_ap} → proven best {best_cfg} AP {best_ap} in {len(trail)} steps",
                 detail="; ".join(f"{s['step']}={s['div_ap']}({'kept' if s['kept'] else 'reverted'})" for s in trail),
                 kind="verdict",
                 recommendation="adopt the proven config — started basic, added complexity ONLY where training earned it")
        rows = "\n".join(f"| {'🏆' if s['config']==best_cfg and s['div_ap']==best_ap else ('✅' if s['kept'] else '↩︎')} "
                         f"| {s['step']} | `{s['config']}` | {s['div_ap']} |" for s in trail)
        # GROUNDED modern-technique proposal from the catalog (non-fatal) — surfaces MoE/int8/QAT/graft/gate
        # options with their MEASURED constraints for this comp's target, so the search verdict isn't arch-blind.
        prop_note = ""
        try:
            from . import arch_builder
            prop = arch_builder.propose(spec.get("target_profile"))
            names = ", ".join(p["name"] for p in prop["recommended"][:5])
            exc = "; ".join(f"{e['name']} ({e['reason']})" for e in prop["excluded"])
            prop_note = (f"\n🧩 **modern-technique catalog** ({prop['target']['_label']}): {names}"
                         + (f" · ⛔ {exc}" if exc else "") + f" · GATE {prop['gate']['name']}")
        except Exception:  # noqa: BLE001
            pass
        msg = (f"[{worker}] **ARCH-SEARCH** · start BASIC, grow ONE axis at a time (kept only if trained better)\n"
               f"| | step | config | held-out div AP |\n|:-|:--|:--|--:|\n{rows}\n"
               f"**Proven best:** `{best_cfg}` → div AP **{best_ap}** (basic baseline was {base_ap}). "
               f"No complexity assumed — each step earned by data." + prop_note)
        self.post(worker, "all", msg, routine=False, kind="verdict")
        return self.done({"best": {"config": best_cfg, "div_ap": best_ap, "base_ap": base_ap},
                          "candidates": len(trail), "steps": trail}, msg)


_AGENT = ArchSearch()


def run(q, worker):
    return _AGENT.run(q, worker)
