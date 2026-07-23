"""recipe-adopt — GRAFT a reproduced public notebook's winning recipe onto OUR pipeline, one component
at a time, keeping ONLY the components that measurably help on canonical golden-12.

Motivation (measured, 2026-07-10): abhijith's public 0.900-LB notebook reproduces to 0.9257 canonical
golden-12 — beating our best real pipeline (0.8837) by +0.042 — and the win is NOT a new model, it's a
DETECTION+POSTPROC recipe (det=0.97, pool=3.0, min_track_len=6, 6-way TTA, motion-relink, gap-close,
safe-divisions, linefit-smooth, gap-refine). This agent turns "port their recipe" into a disciplined,
evidence-gated sweep: for each recipe knob that differs from our base, build base+knob, score canonical,
KEEP it only if Δcanonical > eps. The merged config = base + every load-bearing knob. No knob is adopted
on faith — each earns its place by a measured golden-12 delta (see [[feedback_ground_changes_in_domain]]).

Reusable / spec-driven:
  {base: {knob: val}, recipe: {knob: val}, components: [knob,...] (default = recipe keys),
   eps: 0.0005, score_fn: optional injectable scorer(config)->float (real default = canonical golden-12)}

A BaseAgent subclass with its own data-wise test (mock scorer with a known winning subset).
"""
from __future__ import annotations
from .base import BaseAgent


class RecipeAdopt(BaseAgent):
    name = "recipe-adopt"
    thread = "B"
    kind = "verdict"

    def _canonical_scorer(self, timeout=3600):
        """Real scorer: apply a config's env knobs → predict/postproc → canonical golden-12 score.
        Wired lazily so the data-wise test can inject a cheap mock instead (no GPU in unit tests).
        timeout (s): cap each canonical scoring subprocess."""
        to = max(1, int(timeout))
        def score(config: dict) -> float:
            import subprocess, json, os
            from pathlib import Path
            COMP = Path(__file__).resolve().parent.parent
            py = COMP / "research" / "cellmot_venv" / "bin" / "python"
            script = COMP / "scripts" / "score_golden12_official.py"
            if not (py.exists() and script.exists()):
                raise RuntimeError("canonical scorer not available")
            env = dict(os.environ)
            for k, v in config.items():                       # config knobs → BIOHUB_* env for pilk_post
                env[k] = str(v)
            r = subprocess.run([str(py), str(script), "--emit-json"], capture_output=True, text=True,
                               cwd=str(COMP), env=env, timeout=to)
            line = [l for l in r.stdout.splitlines() if l.strip().startswith("{")]
            if not line:
                raise RuntimeError(f"scorer produced no JSON: {r.stderr[-200:]}")
            d = json.loads(line[-1])
            if "tracking_cellmot.metrics.canonical" not in json.dumps(d):
                raise RuntimeError("scorer output is not the canonical metric")
            return float(d.get("score"))
        return score

    def run(self, q, worker):
        spec = self.spec(q)
        base = dict(spec.get("base") or {})
        recipe = dict(spec.get("recipe") or {})
        components = spec.get("components") or list(recipe.keys())
        try:
            eps = float(spec.get("eps", 0.0005))
        except Exception:  # noqa: BLE001
            eps = 0.0005
        # OPTIONAL timeout (s) for the canonical scorer subprocess; max_components caps the sweep breadth.
        score_fn = spec.get("score_fn") or self._canonical_scorer(spec.get("timeout", 3600))

        # only consider knobs whose recipe value actually DIFFERS from ours
        diff = [k for k in components if k in recipe and base.get(k) != recipe[k]]
        if spec.get("max_components"):
            try:
                diff = diff[:int(spec["max_components"])]
            except Exception:  # noqa: BLE001
                pass
        if not diff:
            return self.escalate(worker, "researcher",
                                 f"[{worker}] recipe-adopt: recipe matches base on all components — nothing to graft.")
        try:
            base_cv = float(score_fn(dict(base)))
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] recipe-adopt: base scoring failed: {e}")

        results = []                                          # (knob, cv_with_knob, delta, load_bearing)
        merged = dict(base)
        for k in diff:
            cand = dict(base); cand[k] = recipe[k]
            try:
                cv = float(score_fn(cand))
            except Exception as e:  # noqa: BLE001
                results.append({"knob": k, "cv": None, "delta": None, "keep": False, "error": str(e)[:80]})
                continue
            delta = cv - base_cv
            keep = delta > eps
            if keep:
                merged[k] = recipe[k]
            results.append({"knob": k, "cv": round(cv, 4), "delta": round(delta, 4), "keep": keep})

        # score the fully-merged config (all load-bearing knobs together) — synergy check
        merged_cv = None
        kept = [r["knob"] for r in results if r.get("keep")]
        if kept:
            try:
                merged_cv = round(float(score_fn(merged)), 4)
            except Exception:  # noqa: BLE001
                merged_cv = None

        results.sort(key=lambda r: (r.get("delta") is not None, r.get("delta") or -9), reverse=True)
        self.save_state({"base_cv": round(base_cv, 4), "kept": kept, "merged_cv": merged_cv,
                         "merged_config": merged, "results": results})
        self.log(summary=f"recipe-adopt: base {base_cv:.4f} → merged {merged_cv} keeping {kept or 'none'}",
                 detail="; ".join(f"{r['knob']}:{r['delta']}({'keep' if r.get('keep') else 'drop'})" for r in results),
                 kind="verdict", recommendation="adopt the kept knobs into our config; re-verify on full golden-12")
        tbl = "\n".join(f"| `{r['knob']}` | {r.get('cv')} | {r.get('delta')} | {'✅ keep' if r.get('keep') else '— drop'} |"
                        for r in results)
        msg = (f"[{worker}] **RECIPE-ADOPT** · base canonical **{base_cv:.4f}** → merged **{merged_cv}**\n"
               f"| knob | cv | Δ | verdict |\n|---|--:|--:|---|\n{tbl}\n"
               f"→ adopt: {', '.join(f'`{k}`' for k in kept) or 'nothing beat base'}")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"base_cv": round(base_cv, 4), "kept": kept, "merged_cv": merged_cv,
                          "merged_config": merged, "results": results}, msg, to="leader")


_AGENT = RecipeAdopt()


def run(q, worker):
    return _AGENT.run(q, worker)
