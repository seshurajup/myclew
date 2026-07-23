"""final_pure_pack — the last PURE tools from the one-by-one backlog (the rest genuinely need GPU/models).
All numpy, offline-verified:

  • geometric-spatial-augmentor — coordinate/tracking augmentation: rotation about a center, horizontal/vertical
                                  flip (with label symmetry), agent dropout, jitter — the top anti-overfit lever
                                  for trajectory/pose comps (NFL/MABe).
  • infer-cascade               — multi-stage confidence cascade: items the small model is unsure about escalate
                                  to the next (bigger) model — fits the ensemble in the time budget (LLM comps).
  • llm-synthetic-drill-generator — fabricate synthetic supervised pairs from templates + vocab (dictionary/
                                  grammar drills) to teach a model a skill (nemotron/deep-past).
  • heteroscedastic-uncertainty — turn an ENSEMBLE of predictions into (mu, sigma) for GaussianNLL / calibrated
                                  uncertainty (ariel/CSIRO/NFL) — sigma = model disagreement. No torch needed.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- geometric-spatial-augmentor
def augment_coords(points, op, center=None, angle=0.0, drop_idx=None, jitter=0.0, seed=0):
    """Apply a geometry-preserving augmentation to (n,2) coordinates. op in {rotate,flipx,flipy,dropout,jitter}."""
    p = np.asarray(points, float).copy()
    if p.size == 0:
        return p
    c = np.asarray(center if center is not None else p.mean(0), float)
    if op == "rotate":
        q = p - c; co, si = np.cos(angle), np.sin(angle)
        R = np.array([[co, -si], [si, co]]); return (q @ R.T) + c
    if op == "flipx":
        q = p.copy(); q[:, 0] = 2 * c[0] - q[:, 0]; return q
    if op == "flipy":
        q = p.copy(); q[:, 1] = 2 * c[1] - q[:, 1]; return q
    if op == "dropout":
        q = p.copy()
        for i in (drop_idx or []):
            q[i] = c
        return q
    if op == "jitter":
        rng = np.random.RandomState(seed); return p + rng.normal(0, jitter, p.shape)
    return p


# ---------------------------------------------------------------- infer-cascade
def cascade_stage(confidences, thresholds):
    """Assign each item the FIRST stage whose confidence clears its threshold; else the last (biggest) stage.
    confidences = (n_items, n_stages) confidence of each stage. Returns the chosen stage index per item."""
    C = np.asarray(confidences, float)
    if C.size == 0 or C.ndim != 2:
        return np.zeros(0, int)
    n, k = C.shape
    thresholds = list(thresholds)
    if len(thresholds) < k:                        # pad missing thresholds with +inf (never triggers early)
        thresholds = thresholds + [np.inf] * (k - len(thresholds))
    # first stage over threshold
    stage = np.full(n, k - 1, int)
    for i in range(n):
        for j in range(k):
            if C[i, j] >= thresholds[j]:
                stage[i] = j; break
    return stage


# ---------------------------------------------------------------- llm-synthetic-drill-generator
def generate_drills(templates, slots, n=100, seed=0):
    """Fill templates with sampled slot values → synthetic (prompt, answer) drills. templates = list of
    (prompt_tmpl, answer_tmpl) with {name} fields; slots = {name: [values]}."""
    templates = list(templates or [])
    if not templates or int(n) <= 0:
        return []
    slots = {k: list(v) for k, v in (slots or {}).items() if len(v)}
    rng = np.random.RandomState(int(seed)); out = []
    for _ in range(int(n)):
        vals = {k: v[rng.randint(len(v))] for k, v in slots.items()}
        pt, at = templates[rng.randint(len(templates))]
        out.append({"prompt": pt.format(**vals), "answer": at.format(**vals)})
    return out


# ---------------------------------------------------------------- heteroscedastic-uncertainty
def ensemble_uncertainty(ensemble_preds):
    """ensemble_preds = (n_models, n_samples). Returns (mu, sigma) where sigma = cross-model disagreement —
    the calibrated-uncertainty output a GaussianNLL metric (ariel) rewards."""
    E = np.asarray(ensemble_preds, float)
    if E.size == 0 or E.ndim < 2:
        return np.zeros(0), np.zeros(0)
    return E.mean(0), E.std(0)


def gaussian_nll_loss(y, mu, sigma):
    """The GaussianNLL objective value (for validating a heteroscedastic head)."""
    y = np.asarray(y, float); mu = np.asarray(mu, float); sigma = np.clip(np.asarray(sigma, float), 1e-6, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)))


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class SpatialAugmentor(_B):
    name = "geometric-spatial-augmentor"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("points",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"geometric-spatial-augmentor needs spec keys {missing} — none provided")
        out = augment_coords(s["points"], s.get("op", "rotate"), s.get("center"),
                             float(s.get("angle", 0.0)), s.get("drop_idx"), float(s.get("jitter", 0.0)),
                             seed=int(s.get("seed", 0)))
        msg = f"geometric-spatial-augmentor: applied '{s.get('op','rotate')}' to {len(out)} points"
        self.log(msg, kind="finding", recommendation="rotation/flip/dropout is the top anti-overfit lever for tracking")
        return self.done({"_points": np.asarray(out).tolist()}, msg)


class InferCascade(_B):
    name = "infer-cascade"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("confidences", "thresholds") if k not in s]
        if missing: return self.escalate(worker, "leader", f"infer-cascade needs spec keys {missing} — none provided")
        stage = cascade_stage(s["confidences"], s["thresholds"])
        counts = {int(j): int((stage == j).sum()) for j in np.unique(stage)}
        msg = f"infer-cascade: stage assignment {counts} (small→big; unsure items escalate)"
        self.log(msg, kind="finding", recommendation="run only escalated items on the big model to fit the budget")
        return self.done({"stage": stage.tolist(), "counts": counts}, msg)


class DrillGenerator(_B):
    name = "llm-synthetic-drill-generator"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("templates", "slots") if k not in s]
        if missing: return self.escalate(worker, "leader", f"llm-synthetic-drill-generator needs spec keys {missing} — none provided")
        drills = generate_drills([tuple(t) for t in s["templates"]], s["slots"],
                                                  int(s.get("n", 100)), seed=int(s.get("seed", 0)))
        msg = f"llm-synthetic-drill-generator: generated {len(drills)} synthetic supervised drills"
        self.log(msg, kind="finding", recommendation="mix drills into SFT to teach a skill (nemotron/deep-past)")
        return self.done({"n": len(drills), "sample": drills[:3]}, msg)


class HeteroscedasticHead(_B):
    name = "heteroscedastic-uncertainty-head"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("ensemble_preds",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"heteroscedastic-uncertainty-head needs spec keys {missing} — none provided")
        mu, sigma = ensemble_uncertainty(s["ensemble_preds"])
        nll = gaussian_nll_loss(s["y"], mu, sigma) if "y" in s else None
        msg = f"heteroscedastic-uncertainty-head: mu+sigma from ensemble disagreement" + (f"; GaussianNLL={nll:.4f}" if nll is not None else "")
        self.log(msg, kind="finding", recommendation="output (mu,sigma) for distributional-scoring metrics (ariel)")
        return self.done({"_mu": mu.tolist(), "_sigma": sigma.tolist(), "gaussian_nll": nll}, msg)


_GA = SpatialAugmentor(); _IC = InferCascade(); _DG = DrillGenerator(); _HH = HeteroscedasticHead()


def run_augment(q, worker): return _GA.run(q, worker)
def run_cascade(q, worker): return _IC.run(q, worker)
def run_drills(q, worker): return _DG.run(q, worker)
def run_hetero(q, worker): return _HH.run(q, worker)
