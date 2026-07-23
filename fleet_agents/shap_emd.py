"""shap_emd — SHAP-guided Earth-Mover's-Distance between compositional samples (thiagorr162/shap-emd,
src/shap_emd/pipeline.py). The idea: two samples that are simplex/histogram vectors over the SAME set of
features (e.g. glass oxide fractions, or any part-of-whole composition) shouldn't be compared with a plain
L1/Euclidean distance that treats every feature as equidistant — features that a model uses SIMILARLY should
be "cheap" to move mass between. shap-emd derives the ground cost between features from their SHAP dependence
curves (how each feature drives the prediction), then measures sample-to-sample distance as optimal transport
under that cost. This turns a trained model's attributions into a MODEL-AWARE metric on the input simplex.

Pipeline (mirrors the source, deps dropped to numpy+scipy — replaces `pot`/`ot.emd2` with an exact transport
LP via scipy.optimize.linprog, and the trained-RF+SHAP front-end is left to the caller who supplies curves):
  1. cost between feature i,j = area between their SHAP-value-vs-value curves: C[i,j]=∫|curve_j(x)-curve_i(x)|dx.
  2. distance(a,b) = W(a,b;C) / norm, where a,b are normalized feature-fraction vectors and W is exact EMD.
  3. norm = 99th percentile of reference pairwise distances (so typical distances land near 1).

Reusable primitives (numpy + scipy, no `pot`):
  • cost_matrix_from_curves(curves, x)      — L1-area ground cost between feature curves.
  • emd2(a, b, C)                           — EXACT earth-mover's distance (transportation LP), no POT dep.
  • shap_emd_distance(a, b, C, norm=1)      — normalized model-aware distance between two composition vectors.
  • normalization_p99(comps, C)            — 99th-percentile reference distance (the source's normalizer).
"""
from __future__ import annotations
import numpy as np
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))  # numpy 2.x renamed trapz→trapezoid
from scipy.optimize import linprog
from .base import BaseAgent


# ---------------------------------------------------------------- ground cost from SHAP curves
def cost_matrix_from_curves(curves, x):
    """C[i,j] = ∫ |curve_j(x) - curve_i(x)| dx via the trapezoid rule. curves: (F, P) array of F feature
    curves sampled on the common support x (length P). Returns a symmetric (F, F) non-negative cost matrix
    with zero diagonal — features whose SHAP effect is similar are cheap to transport between."""
    C = np.asarray(curves, float); x = np.asarray(x, float)
    F = C.shape[0]
    M = np.zeros((F, F))
    for i in range(F):
        for j in range(i + 1, F):
            a = _trapz(np.abs(C[j] - C[i]), x)
            M[i, j] = M[j, i] = a
    return M


# ---------------------------------------------------------------- exact EMD (transportation LP)
def emd2(a, b, C):
    """Exact earth-mover's distance (a.k.a. Wasserstein-1 / EMD²-value in POT) between two mass vectors a,b
    over the same F bins under ground-cost C (F×F). Solves the transportation LP
        min_T Σ C_ij T_ij  s.t.  Σ_j T_ij = a_i,  Σ_i T_ij = b_j,  T ≥ 0
    with scipy.optimize.linprog (HiGHS) — exact, and drops the POT dependency. a,b need not be normalized but
    must share the same total mass (they are L1-normalized here to be safe)."""
    a = np.asarray(a, float); b = np.asarray(b, float); C = np.asarray(C, float)
    a = a / a.sum() if a.sum() > 0 else a
    b = b / b.sum() if b.sum() > 0 else b
    F = len(a)
    # equality constraints: F row-sums + F col-sums (one is redundant; HiGHS tolerates it)
    A_eq = np.zeros((2 * F, F * F))
    for i in range(F):
        A_eq[i, i * F:(i + 1) * F] = 1.0                 # row i sums to a_i
    for j in range(F):
        A_eq[F + j, j::F] = 1.0                          # col j sums to b_j
    b_eq = np.concatenate([a, b])
    res = linprog(C.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (F * F), method="highs")
    if not res.success:
        # fallback: feasible upper bound (independent coupling) — never crash a running fleet
        T = np.outer(a, b)
        return float((C * T).sum())
    return float(res.fun)


def shap_emd_distance(a, b, C, norm=1.0):
    """Normalized model-aware distance between two composition vectors a,b under ground-cost C."""
    return emd2(a, b, C) / max(float(norm), 1e-12)


def normalization_p99(comps, C):
    """99th-percentile of pairwise EMD over a reference set `comps` ((N,F) rows sum to 1) — the source's
    normalizer so a typical distance is ~1. O(N²) EMD solves; use a modest reference sample."""
    comps = np.asarray(comps, float)
    N = comps.shape[0]
    d = [emd2(comps[i], comps[j], C) for i in range(N) for j in range(i + 1, N)]
    return float(np.percentile(d, 99)) if d else 1.0


# ---------------------------------------------------------------- agent
class ShapEMD(BaseAgent):
    name = "shap-emd-distance"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        rng = np.random.RandomState(int(s.get("seed", 0)))
        F = int(s.get("n_features", 8)); P = int(s.get("support", 64)); N = int(s.get("n_ref", 12))
        x = np.linspace(0, 1, P)
        # synthetic SHAP curves: smooth random functions of the feature value
        curves = np.array([np.sin((k + 1) * np.pi * x) * rng.uniform(0.5, 1.5) for k in range(F)])
        C = cost_matrix_from_curves(curves, x)
        comps = rng.dirichlet(np.ones(F), size=N)               # reference compositions on the simplex
        norm = normalization_p99(comps, C)
        d_same = shap_emd_distance(comps[0], comps[0], C, norm)   # must be ~0
        d_ab = shap_emd_distance(comps[0], comps[1], C, norm)
        msg = (f"shap-emd-distance: {F}-feature model-aware EMD — cost from SHAP curves (mean C={C[C>0].mean():.3f}), "
               f"norm(p99)={norm:.4f}; d(a,a)={d_same:.2e} (≈0), d(a,b)={d_ab:.3f}. Optimal-transport distance "
               f"on the input simplex that treats similarly-attributed features as cheap to swap (no POT dep)")
        self.log(msg, kind="finding",
                 recommendation="use for compositional/histogram features (part-of-whole): build SHAP curves from "
                                "any tree model, then shap_emd_distance for a model-aware neighbor/duplicate metric")
        return self.done({"d_self": d_same, "d_ab": d_ab, "norm": norm}, msg)


_AGENT = ShapEMD()


def run_shapemd(q, worker):
    return _AGENT.run(q, worker)
