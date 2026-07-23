"""cv-lb-calibrate — the grandmaster's first discipline: KNOW your CV↔LB relationship so you never submit
blind. Reads every journal row that has BOTH a measured canonical golden-12 CV and a real Kaggle LB,
fits the CV→LB map (robust linear: slope+offset, falls back to a constant offset when <3 anchors), and
PREDICTS the LB for any candidate CV — with an honest confidence flag based on anchor count + residual.

Grounded in measurements (2026-07-10): abhijith canonical 0.9257 → LB 0.900 (offset −0.026); yaroslav
0.8803 → LB 0.897. CV and LB do NOT agree 1:1 ([[feedback_local_lb_kaggle_lb_correlation]],
[[biohub_golden_cv_lb_calibration]]) — so a raw CV win is NOT an LB win until mapped. This agent is what
lets submit-decisions be evidence-based: "candidate CV 0.90 → predicted LB 0.876 ± resid, anchors=2 (LOW
confidence)".

Reusable / spec-driven:
  {anchors: optional [{cv, lb}] (default = journal rows with both), predict: [cv,...] (CVs to map)}

A BaseAgent subclass with its own data-wise test (planted cv→lb line, assert recovery + prediction).
"""
from __future__ import annotations
from .base import BaseAgent


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v  # v==v rejects nan


def _fit(anchors, robust=False):
    """Return (slope, intercept, kind). ≥3 anchors → least-squares line; 2 → line through them;
    1 → constant offset (slope 1); 0 → identity with a warning kind.
    robust: with ≥3 anchors, fit a Theil–Sen (median-of-pairwise-slopes) line, resistant to outlier anchors."""
    pts = [(float(a["cv"]), float(a["lb"])) for a in anchors
           if _is_num(a.get("cv")) and _is_num(a.get("lb"))]
    n = len(pts)
    if n == 0:
        return (1.0, 0.0, "none")
    if n == 1:
        cv, lb = pts[0]
        return (1.0, lb - cv, "offset")
    if robust and n >= 3:
        from statistics import median
        slopes = [(pts[j][1] - pts[i][1]) / (pts[j][0] - pts[i][0])
                  for i in range(n) for j in range(i + 1, n) if abs(pts[j][0] - pts[i][0]) > 1e-12]
        if slopes:
            slope = median(slopes)
            intercept = median([y - slope * x for x, y in pts])
            return (slope, intercept, "robust")
        # all same CV → constant offset
        return (1.0, median([y - x for x, y in pts]), "offset")
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:                                 # degenerate (all same CV) → constant offset
        return (1.0, (sy - sx) / n, "offset")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return (slope, intercept, "linear" if n >= 3 else "two-point")


class CvLbCalibrate(BaseAgent):
    name = "cv-lb-calibrate"
    thread = "B"
    kind = "verdict"

    def _journal_anchors(self):
        from . import ledger
        out = []
        for r in ledger.entries():
            cv, lb = r.get("cv"), r.get("lb")
            if isinstance(cv, (int, float)) and not isinstance(cv, bool) \
               and isinstance(lb, (int, float)) and not isinstance(lb, bool) and lb < 1.0:
                out.append({"cv": cv, "lb": lb, "exp": r.get("exp")})
        return out

    @staticmethod
    def _dedupe_by_cv(anchors):
        """Collapse anchors that share the same CV into ONE point (median LB). Many rows carrying the same
        placeholder/shared CV (e.g. 26×0.8708) are NOT independent CV↔LB evidence — counting them all lets a
        vertical noise-cluster dominate the fit. Returns (clean_anchors, dup_fraction)."""
        from statistics import median
        groups = {}
        for a in anchors:
            cv = a.get("cv")
            if isinstance(cv, (int, float)):
                groups.setdefault(round(float(cv), 4), []).append(float(a["lb"]))
        clean = [{"cv": cv, "lb": median(lbs), "n": len(lbs)} for cv, lbs in groups.items()]
        clean.sort(key=lambda a: a["cv"])
        raw_n = sum(len(v) for v in groups.values())
        dup_frac = round(1 - len(groups) / raw_n, 2) if raw_n else 0.0
        return clean, dup_frac

    def run(self, q, worker):
        spec = self.spec(q)
        raw = spec.get("anchors") or self._journal_anchors()
        # grandmaster hygiene: distinct-CV anchors only (dup CVs are noise, not evidence)
        anchors, dup_frac = self._dedupe_by_cv(raw) if spec.get("dedupe", True) else (raw, 0.0)
        slope, intercept, kind = _fit(anchors, robust=bool(spec.get("robust", False)))
        # DECOUPLING check: over the top third of CVs, how much does LB actually move per CV point?
        top = sorted([a for a in anchors if isinstance(a.get("cv"), (int, float))], key=lambda a: a["cv"])[-3:]
        decoupled = False
        if len(top) >= 2:
            dcv = top[-1]["cv"] - top[0]["cv"]; dlb = top[-1]["lb"] - top[0]["lb"]
            if dcv > 0.02 and (dlb / dcv) < 0.25:          # LB moves <0.25 per 1.0 CV at the top → saturating
                decoupled = True
        # residuals on the anchors (how trustworthy the fit is)
        resid = [abs((slope * float(a["cv"]) + intercept) - float(a["lb"])) for a in anchors
                 if isinstance(a.get("cv"), (int, float)) and isinstance(a.get("lb"), (int, float))]
        mean_resid = round(sum(resid) / len(resid), 4) if resid else None
        n = len(resid)
        conf = "high" if n >= 4 and (mean_resid or 0) < 0.01 else ("medium" if n >= 2 else "low")

        def predict(cv):
            return round(slope * float(cv) + intercept, 4)

        preds = [{"cv": cv, "predicted_lb": predict(cv)} for cv in (spec.get("predict") or [])]
        self.save_state({"slope": round(slope, 4), "intercept": round(intercept, 4), "kind": kind,
                         "n_anchors": n, "raw_anchors": len(raw), "dup_frac": dup_frac, "decoupled": decoupled,
                         "mean_resid": mean_resid, "confidence": conf, "preds": preds})
        self.log(summary=f"cv→lb: lb ≈ {slope:.3f}·cv + {intercept:+.4f} ({kind}, {n} distinct-CV anchors, conf {conf})"
                         + (" · ⚠️ LB DECOUPLED from CV at the top (saturating)" if decoupled else ""),
                 detail=f"{len(raw)} raw anchors → {n} distinct CVs (dup {dup_frac:.0%}); "
                        + "; ".join(f"cv {a['cv']}→lb {round(a['lb'],3)}(n{a.get('n',1)})" for a in anchors[-6:]),
                 kind="verdict", recommendation="map candidate CV through this before submitting; if decoupled, canonical-CV gains may NOT move LB")
        ptxt = ("\n".join(f"| {p['cv']} | **{p['predicted_lb']}** |" for p in preds)) if preds else ""
        msg = (f"[{worker}] **CV→LB CALIBRATE** · `lb ≈ {slope:.3f}·cv {intercept:+.4f}` "
               f"({kind}, {n} distinct-CV anchors from {len(raw)} rows, dup {dup_frac:.0%}, **{conf}** conf)\n"
               + (f"| candidate CV | predicted LB |\n|--:|--:|\n{ptxt}\n" if preds else "")
               + ("🛑 **LB DECOUPLED from CV at the top** — LB barely moves as canonical CV rises; "
                  "chasing more canonical CV likely won't raise LB. Diversify / verify on real submits.\n" if decoupled else "")
               + ("⚠️ few distinct-CV anchors — predictions rough; submit known configs to get more (CV,LB) pairs."
                  if conf == "low" else ""))
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"slope": round(slope, 4), "intercept": round(intercept, 4), "kind": kind,
                          "n_anchors": n, "dup_frac": dup_frac, "decoupled": decoupled,
                          "mean_resid": mean_resid, "confidence": conf, "preds": preds}, msg, to="leader")


_AGENT = CvLbCalibrate()


def run(q, worker):
    return _AGENT.run(q, worker)
