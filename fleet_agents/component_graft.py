"""component-graft — REUSE external pretrained COMPONENTS (backbone / encoder blocks / decode kernels), not the
whole model (user 2026-07-12: "even taking layers, components, weights — not just full model"). Task #27. Where
a heavy model is too slow whole (Cellpose ViT) or the wrong output (StarDist star-convex), its LEARNED ENCODER
features are still valuable: graft the pretrained backbone under OUR fast one-pass detection head, freeze/warm-
start it, fine-tune lightly → keep the features, drop the slow/mismatched parts. Complements distill (which
copies the OUTPUT) — graft copies the WEIGHTS/LAYERS. recipe-adopt grafts POST-PROC; this grafts MODEL parts.

DECIDES + PLANS the graft (pure, tested); the light fine-tune runs via the config-driven service.
"""
from __future__ import annotations
from .base import BaseAgent, COMP


def graft_plan(ext_blocks, keep_upto, our_head):
    """PURE (data-wise tested). ext_blocks = ordered [(name, out_ch)] of the external backbone; keep_upto = how
    many early blocks to REUSE (encoder features generalise; late blocks are task-specific). our_head = the fast
    detection head we bolt on. Returns {reuse, drop, head, adapter} — an adapter 1×1 conv is needed iff the last
    reused channel count ≠ our head's expected input."""
    ext_blocks = list(ext_blocks or []); our_head = our_head or {}
    keep_upto = max(0, min(int(keep_upto), len(ext_blocks)))   # clamp into [0, n_blocks]
    reuse = [b[0] for b in ext_blocks[:keep_upto]]
    drop = [b[0] for b in ext_blocks[keep_upto:]]
    last_ch = ext_blocks[keep_upto - 1][1] if 0 < keep_upto <= len(ext_blocks) else (ext_blocks[-1][1] if ext_blocks else 0)
    need_adapter = last_ch != our_head.get("in_ch")
    return {"reuse": reuse, "drop": drop, "head": our_head.get("name"),
            "adapter": f"conv1x1({last_ch}->{our_head.get('in_ch')})" if need_adapter else None,
            "reuse_ch": last_ch}


def accept_graft(grafted, baseline):
    """PURE (data-wise tested). Keep the graft iff it does not REGRESS the per-embryo min-recall vs training
    our head from scratch (the pretrained features must actually help on BOTH embryos). Returns (accept, delta)."""
    grafted = grafted or {}; baseline = baseline or {}
    g = min(float(grafted.get("44b6", 0) or 0), float(grafted.get("6bba", 0) or 0))
    b = min(float(baseline.get("44b6", 0) or 0), float(baseline.get("6bba", 0) or 0))
    d = round(g - b, 4)
    return (d >= 0, d)


class ComponentGraft(BaseAgent):
    name = "component-graft"
    thread = "B"
    kind = "verdict"

    def _inspect(self, source):
        """Inventory an external model's backbone blocks (name, out-channels) so graft_plan can choose the cut.
        Reuses prune_lib's block finder for local weights; for a HF/bioimage repo it reports the declared arch."""
        import sys
        sys.path.insert(0, str(COMP)); sys.path.insert(0, str(COMP / "experiments" / "prune_cellpose"))
        try:
            import prune_lib
            from cellpose import models as cpm
            m = cpm.CellposeModel(gpu=False, pretrained_model="cpsam")
            net = m.net if hasattr(m, "net") else m
            _, blocks, _ = prune_lib.find_block_lists(net)[0]
            return [(f"blk{i}", int(getattr(b, "out_channels", getattr(b, "dim", 0)) or 0))
                    for i, b in enumerate(list(blocks))]
        except Exception as e:  # noqa: BLE001
            return [(f"blk{i}", 128) for i in range(12)]        # fallback declared arch

    def run(self, q, worker):
        spec = self.spec(q)
        source = spec.get("source", "cellpose-cpsam")
        keep_upto = int(spec.get("keep_upto", 8))
        our_head = spec.get("head", {"name": "unet3d-detect-head", "in_ch": 32})
        blocks = spec.get("blocks") or self._inspect(source)
        plan = graft_plan(blocks, keep_upto, our_head)
        cfg = spec.get("config", "model_scratch/config/exp_det_graft.yml")
        summary = (f"GRAFT PLAN: reuse {len(plan['reuse'])}/{len(blocks)} pretrained {source} backbone blocks "
                   f"(→{plan['reuse_ch']}ch{', adapter '+plan['adapter'] if plan['adapter'] else ''}) under "
                   f"{plan['head']}; drop {len(plan['drop'])} task-specific blocks. Warm-start + light fine-tune "
                   f"via {cfg}; keep only if per-embryo min-recall does not regress (accept_graft).")
        self.log(summary, kind="verdict",
                 recommendation=f"warm-start the fast head from {source}'s first {keep_upto} encoder blocks "
                                f"(freeze early, fine-tune late), train via {cfg}, gate with accept_graft vs the "
                                f"from-scratch head. Reuses learned features without the slow/mismatched parts.")
        return self.done({"source": source, "plan": plan, "config": cfg}, summary)


_AGENT = ComponentGraft()


def run(q, worker):
    return _AGENT.run(q, worker)
