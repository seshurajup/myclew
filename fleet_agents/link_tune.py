"""link-tune — the NEW lever's agent. Node-recall is now SATURATED (~0.998 on the support-pack detector,
verified on golden-12), so the gap to LB-top is EDGE-PRECISION / LINKING, not detection. This agent
sweeps the LINKING knobs (motion-relink on/off, gap-close µm, min-track-len, safe-divisions, ILP weights)
and keeps the config that maximises canonical golden-12 — the lever recipe-adopt/det-sweep don't own.

Grounded in measurement (2026-07-10): the ablation showed motion-relink COSTS −0.017 on golden-12
([[biohub_grandmaster_agents]]); this agent operationalises that — it tests motion-relink OFF and every
other linking knob, and only KEEPS a change that measurably raises canonical CV (evidence-gated, like
recipe-adopt). Distinct from fullconfig-search (which sweeps the whole 53-param config incl. detection);
link-tune is FOCUSED on the linking/edge knobs now that detection is settled.

Reusable / spec-driven: {base: {knob: val}, grid: {knob: [vals]}, eps: 0.0005,
   score_fn: injectable scorer(config)->float (real default = canonical golden-12)}.
A BaseAgent subclass with its own data-wise test.
"""
from __future__ import annotations
from .base import BaseAgent

# the linking knobs (BIOHUB_* env read by pilk_post) — the edge-precision lever
_DEFAULT_GRID = {
    "BIOHUB_MOTION_RELINK": ["0", "1"],           # the ablation says OFF wins on golden-12 — prove it here
    "BIOHUB_GAP_CLOSE_UM": ["3.5", "4.0", "5.0", "6.0"],
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": ["4", "6", "10"],
    "BIOHUB_OUTPUT_SAFE_DIVISIONS": ["0", "1"],
}


class LinkTune(BaseAgent):
    name = "link-tune"
    thread = "B"
    kind = "verdict"

    def _canonical_scorer(self):
        def score(config):
            import subprocess, json, os
            from pathlib import Path
            COMP = Path(__file__).resolve().parent.parent
            py = COMP / "research" / "cellmot_venv" / "bin" / "python"
            script = COMP / "scripts" / "score_golden12_official.py"
            env = dict(os.environ)
            for k, v in config.items():
                env[k] = str(v)
            r = subprocess.run([str(py), str(script), "--emit-json"], capture_output=True, text=True,
                               cwd=str(COMP), env=env, timeout=3600)
            d = json.loads([l for l in r.stdout.splitlines() if l.strip().startswith("{")][-1])
            if "tracking_cellmot.metrics.canonical" not in json.dumps(d):
                raise RuntimeError("not canonical")
            return float(d.get("score")), float(d.get("edge_jaccard", d.get("adj_edge_jaccard", 0.0)))
        return score

    def run(self, q, worker):
        spec = self.spec(q)
        base = dict(spec.get("base") or {})
        grid = spec.get("grid") or _DEFAULT_GRID
        eps = float(spec.get("eps", 0.0005))
        sf = spec.get("score_fn") or self._canonical_scorer()
        seed = spec.get("seed", None)                 # seed RNG so stochastic injectable scorers are reproducible
        if seed is not None:
            try:
                import random as _rnd, numpy as _np
                _rnd.seed(int(seed)); _np.random.seed(int(seed))
            except Exception:  # noqa: BLE001
                pass

        def _clean(x, default=0.0):
            try:
                import numpy as _np
                return float(_np.nan_to_num(float(x), nan=default, posinf=default, neginf=default))
            except Exception:  # noqa: BLE001
                return default

        def _score(cfg):
            v = sf(cfg)
            return (_clean(v[0]), _clean(v[1])) if isinstance(v, (list, tuple)) else (_clean(v), None)

        try:
            base_cv, base_ep = _score(dict(base))
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] link-tune: base scoring failed: {e}")

        # coordinate-ascent over the linking knobs: for each knob, try each value, keep the best-if-improved
        best = dict(base); best_cv = base_cv; trail = []
        for knob, vals in grid.items():
            local = None
            for v in vals:
                if str(best.get(knob)) == str(v):
                    continue
                cand = dict(best); cand[knob] = v
                try:
                    cv, ep = _score(cand)
                except Exception:  # noqa: BLE001
                    continue
                if local is None or cv > local[1]:
                    local = (v, cv, ep)
            if local and local[1] > best_cv + eps:
                best[knob] = local[0]; delta = round(local[1] - best_cv, 4); best_cv = local[1]
                trail.append({"knob": knob, "set": local[0], "cv": round(local[1], 4), "delta": delta, "keep": True})
            elif local:
                trail.append({"knob": knob, "best_try": local[0], "cv": round(local[1], 4),
                              "delta": round(local[1] - best_cv, 4), "keep": False})

        self.save_state({"base_cv": round(base_cv, 4), "best_cv": round(best_cv, 4),
                         "best_config": best, "gain": round(best_cv - base_cv, 4), "trail": trail})
        kept = {t["knob"]: t["set"] for t in trail if t.get("keep")}
        self.log(summary=f"link-tune: base {base_cv:.4f} → {best_cv:.4f} (+{best_cv-base_cv:.4f}) via {kept or 'no change'}",
                 detail="; ".join(f"{t['knob']}={t.get('set',t.get('best_try'))}:{t['delta']}({'keep' if t.get('keep') else 'drop'})" for t in trail),
                 kind="verdict", recommendation="adopt the kept linking knobs (edge-precision lever); re-verify full golden-12")
        rows = "\n".join(f"| `{t['knob']}` | {t.get('set', t.get('best_try'))} | {t.get('cv')} | {t.get('delta')} | {'✅' if t.get('keep') else '—'} |" for t in trail)
        msg = (f"[{worker}] **LINK-TUNE** (edge-precision lever) · base **{base_cv:.4f}** → **{best_cv:.4f}** "
               f"(+{best_cv-base_cv:.4f})\n| knob | val | cv | Δ | keep |\n|---|---|--:|--:|---|\n{rows}\n"
               f"→ kept: {kept or 'nothing beat base'}")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"base_cv": round(base_cv, 4), "best_cv": round(best_cv, 4),
                          "best_config": best, "gain": round(best_cv - base_cv, 4), "kept": kept}, msg, to="leader")


_AGENT = LinkTune()


def run(q, worker):
    return _AGENT.run(q, worker)
