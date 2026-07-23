"""det-sweep — find the detection operating point (det_threshold × pool_kernel_um) that MAXIMISES node
recall at a CALIBRATED node count on canonical golden-12. This is the lever that actually moved the metric:
adjJ ≈ node_rec² · edge_prec ([[biohub_node_recall_lever]]), and the count-penalty term punishes
over-/under-detection, so raw recall isn't enough — recall must be won at pred≈true count.

abhijith's win (det=0.97, pool=3.0) is a point in exactly this grid. This agent sweeps the grid, and for
each point records (node_recall, count_ratio=pred/true, canonical_cv). The pick = the point with the
HIGHEST node_recall whose count_ratio stays inside [lo, hi] (default 0.8–1.25) — i.e. recall gained by
recovering true nuclei, NOT by flooding false detections ([[biohub_detectors_complementary]]).

Reusable / spec-driven:
  {det_grid: [...], pool_grid: [...], count_lo: 0.8, count_hi: 1.25,
   eval_fn: optional injectable (det,pool) -> {node_recall, count_ratio, cv}  (real default = detector+scorer)}

A BaseAgent subclass with its own data-wise test (planted recall/count surface with a known optimum).
"""
from __future__ import annotations
from .base import BaseAgent


class DetSweep(BaseAgent):
    name = "det-sweep"
    thread = "B"
    kind = "verdict"

    def _real_eval(self):
        """Real evaluator: run the detector at (det,pool), score canonical golden-12, read node_recall +
        count_ratio from the scorer JSON. Lazy so the data-wise test injects a cheap surface instead."""
        def ev(det, pool):
            import subprocess, json, os
            from pathlib import Path
            COMP = Path(__file__).resolve().parent.parent
            py = COMP / "research" / "cellmot_venv" / "bin" / "python"
            script = COMP / "scripts" / "score_golden12_official.py"
            env = dict(os.environ, BIOHUB_DET_THRESHOLD=str(det), BIOHUB_POOL_KERNEL_UM=str(pool))
            r = subprocess.run([str(py), str(script), "--emit-json"], capture_output=True, text=True,
                               cwd=str(COMP), env=env, timeout=3600)
            d = json.loads([l for l in r.stdout.splitlines() if l.strip().startswith("{")][-1])
            return {"node_recall": float(d.get("node_recall", 0.0)),
                    "count_ratio": float(d.get("count_ratio", d.get("predN_ratio", 1.0))),
                    "cv": float(d.get("score", 0.0))}
        return ev

    def run(self, q, worker):
        spec = self.spec(q)
        det_grid = spec.get("det_grid") or [0.90, 0.95, 0.97, 0.99]
        pool_grid = spec.get("pool_grid") or [3.0, 5.0]
        lo = float(spec.get("count_lo", 0.8)); hi = float(spec.get("count_hi", 1.25))
        ev = spec.get("eval_fn") or self._real_eval()
        seed = spec.get("seed", None)                 # seed RNG so stochastic injectable evaluators are reproducible
        if seed is not None:
            try:
                import random as _rnd, numpy as _np
                _rnd.seed(int(seed)); _np.random.seed(int(seed))
            except Exception:  # noqa: BLE001
                pass
        # explicit list of [det, pool] points; overrides the det_grid×pool_grid cartesian product when given
        pairs = spec.get("grid")
        if pairs:
            try:
                points = [(float(p[0]), float(p[1])) for p in pairs]
            except Exception:  # noqa: BLE001
                points = [(d, p) for d in det_grid for p in pool_grid]
        else:
            points = [(d, p) for d in det_grid for p in pool_grid]
        if not points:                                # empty grid → nothing to sweep; escalate cleanly
            return self.escalate(worker, "researcher",
                                 f"[{worker}] det-sweep: empty sweep grid — no (det,pool) points to evaluate.")

        def _num(x, default=0.0):
            try:
                import numpy as _np
                return float(_np.nan_to_num(float(x), nan=default, posinf=default, neginf=default))
            except Exception:  # noqa: BLE001
                return default

        grid = []
        for det, pool in points:
                try:
                    m = ev(det, pool)
                except Exception as e:  # noqa: BLE001
                    grid.append({"det": det, "pool": pool, "error": str(e)[:80]}); continue
                if not isinstance(m, dict):
                    grid.append({"det": det, "pool": pool, "error": "eval_fn did not return a dict"}); continue
                cr = _num(m.get("count_ratio", 1.0), 1.0)
                grid.append({"det": det, "pool": pool, "node_recall": round(_num(m.get("node_recall", 0.0)), 4),
                             "count_ratio": round(cr, 3), "cv": round(_num(m.get("cv", 0.0)), 4),
                             "calibrated": lo <= cr <= hi})
        valid = [g for g in grid if g.get("calibrated")]
        if not valid:
            return self.escalate(worker, "researcher",
                                 f"[{worker}] det-sweep: NO grid point kept count_ratio in [{lo},{hi}] — detector mis-calibrated across the whole grid.")
        # pick: highest node_recall among calibrated points; tie-break by cv
        pick = max(valid, key=lambda g: (g["node_recall"], g["cv"]))
        grid.sort(key=lambda g: (g.get("calibrated", False), g.get("node_recall", -1)), reverse=True)
        self.save_state({"pick": pick, "n_calibrated": len(valid), "grid": grid})
        self.log(summary=f"det-sweep: best recall-at-count = det {pick['det']} / pool {pick['pool']} "
                         f"(recall {pick['node_recall']}, count {pick['count_ratio']}, cv {pick['cv']})",
                 detail=f"{len(valid)}/{len(grid)} points calibrated (count in [{lo},{hi}])",
                 kind="verdict", recommendation="set the detector to this det/pool; recall won at calibrated count")
        rows = "\n".join(f"| {g['det']} | {g['pool']} | {g.get('node_recall','—')} | {g.get('count_ratio','—')} "
                         f"| {g.get('cv','—')} | {'✅' if g.get('calibrated') else '⚠️ off-count'} |" for g in grid)
        msg = (f"[{worker}] **DET-SWEEP** · pick **det {pick['det']} / pool {pick['pool']}** "
               f"→ recall **{pick['node_recall']}** @ count {pick['count_ratio']}, cv **{pick['cv']}**\n"
               f"| det | pool | node_rec | count | cv | calibrated |\n|--:|--:|--:|--:|--:|---|\n{rows}")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"pick": pick, "grid": grid}, msg, to="leader")


_AGENT = DetSweep()


def run(q, worker):
    return _AGENT.run(q, worker)
