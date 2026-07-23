"""tracker-select — CHOOSE the end-to-end TRACKER by DATA PROOF on the competition CV, per-embryo, on the
FULL official metric (adj_edge_jaccard + 0.1·division_jaccard). Companion to detector-select.

User rule (2026-07-11): we already found publicly-shared TRAINED models for the WHOLE task — don't rebuild
linking, USE them and choose with a python AGENT, verified on our REAL data. So: fix the chosen detector
(Cellpose-SAM), feed its detections into each trained tracker (Trackastra / ultrack / our mh-ilp), score
the full metric per-embryo on the embryo-disjoint CV, and _choose() the tracker best on BOTH embryos.

None of the eligible trackers is trained on OUR data (Trackastra=CTC, ultrack=unsupervised-at-link, mh-ilp
=geometric) → no leakage. The measurement is done by experiments/pipeline/tracker_compare_cv.py.
"""
from __future__ import annotations
from .base import BaseAgent, COMP


def _choose(results):
    """PURE decision logic (data-wise tested). results = {tracker: {"44b6":score,"6bba":score, ...}}.
    Pick the tracker maximising the MIN per-embryo full-metric score (good on BOTH embryos — the
    per-embryo rule), excluding any flagged leaky. Returns (winner, ranked)."""
    ranked = []
    for name, r in results.items():
        if r.get("leaky"):
            continue
        s44, s6b = r.get("44b6"), r.get("6bba")
        if s44 is None or s6b is None:
            continue
        if s44 != s44 or s6b != s6b:                # NaN score → skip (mis-measured tracker)
            continue
        ranked.append({"name": name, "44b6": round(s44, 4), "6bba": round(s6b, 4),
                       "edge_44b6": r.get("edge_44b6"), "edge_6bba": r.get("edge_6bba"),
                       "min_score": round(min(s44, s6b), 4)})
    ranked.sort(key=lambda d: -d["min_score"])
    return (ranked[0]["name"] if ranked else None), ranked


def _results_from_compare(compare_json):
    """Fold experiments/pipeline/tracker_compare_cv.py output → {tracker: per-embryo full-metric score}."""
    import numpy as np
    out = {}
    for name, r in (compare_json or {}).items():
        d = {"leaky": False}
        for emb in ("44b6", "6bba"):
            vals = r.get(emb) or []
            svals = [x["score"] for x in vals if isinstance(x, dict) and "score" in x]
            evals = [x["edge"] for x in vals if isinstance(x, dict) and "edge" in x]
            d[emb] = float(np.mean(svals)) if svals else None
            d[f"edge_{emb}"] = float(np.mean(evals)) if evals else None
        out[name] = d
    return out


class TrackerSelect(BaseAgent):
    name = "tracker-select"
    thread = "S"
    kind = "verdict"

    def _measure(self, nds, frames):
        """Run the trained-tracker comparison on the CV and fold to per-embryo scores."""
        import subprocess, json, sys
        py = str(COMP / "research" / "cellmot_venv" / "bin" / "python")
        script = str(COMP / "experiments" / "pipeline" / "tracker_compare_cv.py")
        env = {"FLEET_COMPETITION_ROOT": str(COMP),
               "PYTHONPATH": f"{COMP}:{COMP}/tools/researchpapers"}
        import os
        e = dict(os.environ); e.update(env)
        try:
            subprocess.run([py, script, "--nds", str(nds), "--frames", str(frames)],
                           cwd=str(COMP), env=e, timeout=7200, check=False)
            with open("/tmp/tracker_compare_cv.json") as fh:
                return _results_from_compare(json.load(fh))
        except Exception:  # noqa: BLE001 — subprocess/JSON failure → empty results (run() escalates on no winner)
            return {}

    def run(self, q, worker):
        spec = self.spec(q)
        results = spec.get("results")
        if results is None:
            import json, os
            if spec.get("use_cached") and os.path.exists("/tmp/tracker_compare_cv.json"):
                results = _results_from_compare(json.load(open("/tmp/tracker_compare_cv.json")))
            else:
                results = self._measure(int(spec.get("nds", 2)), int(spec.get("frames", 50)))
        winner, ranked = _choose(results)
        proof = "; ".join(f"{d['name']} [44b6={d['44b6']} 6bba={d['6bba']} min={d['min_score']}]" for d in ranked)
        if winner is None:
            return self.escalate(worker, "researcher", f"tracker-select: no eligible tracker scored. {proof}")
        w = next(d for d in ranked if d["name"] == winner)
        summary = (f"CHOSEN tracker (data-proof, CV full metric per-embryo, Cellpose detections): {winner} — "
                   f"44b6={w['44b6']} 6bba={w['6bba']} (min={w['min_score']}). Ranked: {proof}")
        self.log(summary, kind="verdict",
                 recommendation=f"adopt Cellpose-SAM → {winner} as the pipeline. Tune {winner}'s CV-tunable "
                                f"costs per-embryo (division_weight for div_jaccard). None trained on our data.")
        return self.done({"winner": winner, "ranked": ranked, "results": results}, summary)


_AGENT = TrackerSelect()


def run(q, worker):
    return _AGENT.run(q, worker)
