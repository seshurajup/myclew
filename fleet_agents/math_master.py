"""math-master — the advanced-mathematics toolkit a Kaggle grandmaster reaches for when comparing two
distributions, matching a sampler to a target, checking train/test drift, and building trustworthy CV.

It exists so every other agent can ask ONE place "are these two distributions the same, and if not where and by
how much?" with rigorous math instead of eyeballing medians ([[feedback_agents_only_no_adhoc_python]]).

Everything is a module-level pure function (importable + unit-testable) with a BaseAgent wrapper on top.

DISTRIBUTION DISTANCES (2-sample, per column):
  • Kolmogorov–Smirnov   D = sup|F_a−F_b|                    sup-norm CDF gap, ∈[0,1]
  • 1-Wasserstein        W1 = ∫|F⁻¹_a−F⁻¹_b|                 optimal-transport cost (earth-mover)
  • Energy distance      2E|X−Y|−E|X−X'|−E|Y−Y'| (Székely)   0 ⇔ equal law
  • MMD²  (RBF kernel, median-heuristic bandwidth)           kernel two-sample test
  • Cramér–von Mises     ∫(F_a−F_b)²dF                        L2 CDF gap
  • Anderson–Darling (k-sample)                              tail-sensitive CDF gap
  • Jensen–Shannon divergence (shared histogram)             bounded, symmetric KL
  • Population Stability Index (PSI)                          the grandmaster drift staple
  • Moment errors  |Δmean| |Δstd| |Δskew| |Δkurt| (relative)
  • QQ R²  (quantile–quantile linear fit)                    distributional-shape agreement
DRIFT / ADVERSARIAL:
  • adversarial_auc(A,B) — GBM/logistic classifier CV-AUC distinguishing two multivariate samples
    (0.5 = indistinguishable = well matched; the CV↔LB-gap diagnostic).
JOINT STRUCTURE (what per-column metrics miss):
  • Spearman rank-correlation matrix distance (copula check) — cross-column dependence (e.g. tracks↔len).
SAMPLING / CV UTILITIES:
  • quantile_transform(src, target) — monotone inverse-CDF map → forces src's marginal onto target EXACTLY.
  • gaussian_copula_sample(...) — resample a joint with a target rank-dependence.
"""
from __future__ import annotations
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
PERCROP = ("tracks_per_crop", "z_std", "divisions_per_crop", "total_frames")   # per-crop-aligned → joint/copula/adv


# ───────────────────────── distribution distances (numpy core, scipy when present) ─────────────────────────
def _np():
    import numpy as np
    return np


def _cap(a, n=2000):
    """Random-subsample (fixed seed) so O(n²) kernel/energy metrics stay tractable on huge arrays (e.g. 190k
    nn-distances → a 271 GiB pairwise matrix). O(n log n) metrics (KS, Wasserstein) get the full array."""
    import numpy as np
    a = np.asarray(a, float)
    if len(a) <= n: return a
    return a[np.random.RandomState(0).choice(len(a), n, replace=False)]


def _san(a):
    """Coerce to a finite float array (nan/inf → 0). No-op on already-clean data; stops a stray nan poisoning a metric."""
    import numpy as np
    return np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=0.0, neginf=0.0)


def ks(a, b):
    np = _np(); a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    if not len(a) or not len(b): return None
    v = np.concatenate([a, b])
    return round(float(np.max(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b)))), 4)


def wasserstein(a, b):
    try:
        from scipy.stats import wasserstein_distance
        return round(float(wasserstein_distance(a, b)), 4)
    except Exception:  # noqa: BLE001
        np = _np(); q = np.linspace(0, 1, 201)
        return round(float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q)))), 4)


def wasserstein_norm(a, b):                                    # scale-free (normalized by |mean_b|)
    np = _np(); w = wasserstein(a, b); sc = abs(float(np.mean(b))) or 1.0
    return round(w / sc, 4) if w is not None else None


def energy(a, b):
    try:
        from scipy.stats import energy_distance
        return round(float(energy_distance(a, b)), 4)
    except Exception:  # noqa: BLE001
        np = _np(); a = _cap(a); b = _cap(b)
        def m(x, y): return float(np.mean(np.abs(x[:, None] - y[None, :])))
        return round(float(2 * m(a, b) - m(a, a) - m(b, b)), 4)


def mmd_rbf(a, b):                                             # MMD² with RBF, median-heuristic bandwidth
    np = _np(); a = _cap(a)[:, None]; b = _cap(b)[:, None]
    if len(a) < 2 or len(b) < 2: return None
    z = np.vstack([a, b]); d2 = np.abs(z - z.T) ** 2
    sig = np.median(d2[d2 > 0]) or 1.0
    def k(x, y): return np.exp(-np.abs(x - y.T) ** 2 / sig)
    return round(float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()), 4)


def cramer_von_mises(a, b):
    try:
        from scipy.stats import cramervonmises_2samp
        return round(float(cramervonmises_2samp(a, b).statistic), 4)
    except Exception:  # noqa: BLE001
        return None


def anderson_darling(a, b):
    try:
        from scipy.stats import anderson_ksamp
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return round(float(anderson_ksamp([np_asarray(a), np_asarray(b)]).statistic), 4)
    except Exception:  # noqa: BLE001
        return None


def np_asarray(a):
    return _np().asarray(a, float)


def js_divergence(a, b, bins=20):
    np = _np(); a = np.asarray(a, float); b = np.asarray(b, float)
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    if hi <= lo: return 0.0
    edg = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, edg)[0] + 1e-9; pb = np.histogram(b, edg)[0] + 1e-9
    pa /= pa.sum(); pb /= pb.sum(); m = 0.5 * (pa + pb)
    def kl(p, q): return float(np.sum(p * np.log(p / q)))
    return round(0.5 * kl(pa, m) + 0.5 * kl(pb, m), 4)


def psi(a, b, bins=10):                                        # Population Stability Index (drift): b=expected/base
    np = _np(); a = np.asarray(a, float); b = np.asarray(b, float)
    edg = np.quantile(b, np.linspace(0, 1, bins + 1)); edg[0], edg[-1] = -np.inf, np.inf
    ea = np.histogram(a, edg)[0] / len(a) + 1e-6; eb = np.histogram(b, edg)[0] / len(b) + 1e-6
    return round(float(np.sum((ea - eb) * np.log(ea / eb))), 4)


def moments(a, b):
    from scipy.stats import skew, kurtosis
    np = _np(); a = np.asarray(a, float); b = np.asarray(b, float)
    def rel(x, y): return round(abs(x - y) / (abs(y) or 1.0), 4)
    return {"mean_relerr": rel(a.mean(), b.mean()), "std_relerr": rel(a.std(), b.std()),
            "skew_abserr": round(abs(float(skew(a)) - float(skew(b))), 4),
            "kurt_abserr": round(abs(float(kurtosis(a)) - float(kurtosis(b))), 4)}


def qq_r2(a, b):
    np = _np(); q = np.linspace(0.02, 0.98, 50)
    qa, qb = np.quantile(a, q), np.quantile(b, q)
    if np.std(qa) == 0 or np.std(qb) == 0: return None
    return round(float(np.corrcoef(qa, qb)[0, 1] ** 2), 4)


# ── histogram-family divergences (shared bins over the pooled support) ──
def _hist2(a, b, bins=25):
    np = _np(); a = np.asarray(a, float); b = np.asarray(b, float)
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    if hi <= lo: return None, None
    edg = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, edg)[0].astype(float) + 1e-9; pb = np.histogram(b, edg)[0].astype(float) + 1e-9
    return pa / pa.sum(), pb / pb.sum()


def kl_divergence(a, b, bins=25):                              # KL(a‖b), asymmetric
    np = _np(); pa, pb = _hist2(a, b, bins)
    return None if pa is None else round(float(np.sum(pa * np.log(pa / pb))), 4)


def hellinger(a, b, bins=25):
    np = _np(); pa, pb = _hist2(a, b, bins)
    return None if pa is None else round(float(np.sqrt(np.sum((np.sqrt(pa) - np.sqrt(pb)) ** 2)) / np.sqrt(2)), 4)


def bhattacharyya(a, b, bins=25):
    np = _np(); pa, pb = _hist2(a, b, bins)
    return None if pa is None else round(float(-np.log(np.sum(np.sqrt(pa * pb)))), 4)


def total_variation(a, b, bins=25):
    np = _np(); pa, pb = _hist2(a, b, bins)
    return None if pa is None else round(float(0.5 * np.sum(np.abs(pa - pb))), 4)


def chi_square(a, b, bins=25):
    np = _np(); pa, pb = _hist2(a, b, bins)
    return None if pa is None else round(float(0.5 * np.sum((pa - pb) ** 2 / (pa + pb))), 4)


def overlap_coef(a, b, bins=25):                              # histogram intersection (1=identical)
    np = _np(); pa, pb = _hist2(a, b, bins)
    return None if pa is None else round(float(np.sum(np.minimum(pa, pb))), 4)


# ── more CDF / optimal-transport distances ──
def kuiper(a, b):                                             # KS cousin: V=D+ + D− (tail + cyclic sensitive)
    np = _np(); a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    if not len(a) or not len(b): return None
    v = np.concatenate([a, b]); ca = np.searchsorted(a, v, "right") / len(a); cb = np.searchsorted(b, v, "right") / len(b)
    return round(float(np.max(ca - cb) + np.max(cb - ca)), 4)


def wasserstein2(a, b):                                       # quadratic OT (L2 between quantile functions)
    np = _np(); q = np.linspace(0, 1, 201)
    return round(float(np.sqrt(np.mean((np.quantile(a, q) - np.quantile(b, q)) ** 2))), 4)


def crps(a, b):                                              # continuous ranked probability score (forecast vs sample b)
    np = _np(); a = np.asarray(a, float); b = np.sort(np.asarray(b, float))
    grid = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 200)
    Fa = np.searchsorted(np.sort(a), grid, "right") / len(a); Fb = np.searchsorted(b, grid, "right") / len(b)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return round(float(_trap((Fa - Fb) ** 2, grid)), 4)


def sinkhorn(a, b, eps=None, iters=50):                       # entropic-regularized OT (1-D, closed grids)
    np = _np(); a = np.asarray(a, float); b = np.asarray(b, float)
    xs = np.sort(a)[:80]; ys = np.sort(b)[:80]
    C = np.abs(xs[:, None] - ys[None, :]); eps = eps or (np.median(C) or 1.0)
    K = np.exp(-C / eps); u = np.ones(len(xs)) / len(xs); v = np.ones(len(ys)) / len(ys)
    r = np.ones(len(xs)) / len(xs); c = np.ones(len(ys)) / len(ys)
    for _ in range(iters):
        u = r / (K @ v + 1e-12); v = c / (K.T @ u + 1e-12)
    P = np.diag(u) @ K @ np.diag(v)
    return round(float(np.sum(P * C)), 4)


# ── hypothesis tests (p-value; high p ⇒ cannot reject "same") ──
def ks_pvalue(a, b):
    try:
        from scipy.stats import ks_2samp
        return round(float(ks_2samp(a, b).pvalue), 4)
    except Exception:  # noqa: BLE001
        return None


def mannwhitney_p(a, b):                                      # location shift
    try:
        from scipy.stats import mannwhitneyu
        return round(float(mannwhitneyu(a, b, alternative="two-sided").pvalue), 4)
    except Exception:  # noqa: BLE001
        return None


def levene_p(a, b):                                          # variance/scale equality
    try:
        from scipy.stats import levene
        return round(float(levene(a, b).pvalue), 4)
    except Exception:  # noqa: BLE001
        return None


def epps_singleton_p(a, b):                                  # characteristic-function test (robust, discrete-ok)
    try:
        from scipy.stats import epps_singleton_2samp
        return round(float(epps_singleton_2samp(a, b).pvalue), 4)
    except Exception:  # noqa: BLE001
        return None


# ── effect sizes ──
def cohens_d(a, b):
    np = _np(); a = np.asarray(a, float); b = np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)) or 1.0
    return round(float((a.mean() - b.mean()) / sp), 4)


def cliffs_delta(a, b):                                      # non-parametric dominance ∈[−1,1]
    np = _np(); a = np.asarray(a, float); b = np.sort(np.asarray(b, float))
    if not len(a) or not len(b): return None
    gt = np.searchsorted(b, a, "left").sum(); lt = (len(b) - np.searchsorted(b, a, "right")).sum()
    return round(float((gt - lt) / (len(a) * len(b))), 4)


# ── multivariate distribution distances ──
def frechet_distance(A, B):                                  # FID-style: ‖μA−μB‖² + Tr(ΣA+ΣB−2√(ΣAΣB))
    np = _np(); from scipy.linalg import sqrtm
    A = np.asarray(A, float); B = np.asarray(B, float)
    mA, mB = A.mean(0), B.mean(0); CA, CB = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    cov = sqrtm(CA @ CB); cov = cov.real if np.iscomplexobj(cov) else cov
    return round(float(np.sum((mA - mB) ** 2) + np.trace(CA + CB - 2 * cov)), 4)


def mmd_multivariate(A, B):                                  # RBF MMD² on vectors (median-heuristic σ)
    np = _np(); A = np.asarray(A, float); B = np.asarray(B, float)
    if len(A) > 1500: A = A[np.random.RandomState(0).choice(len(A), 1500, False)]
    if len(B) > 1500: B = B[np.random.RandomState(1).choice(len(B), 1500, False)]
    if len(A) < 2 or len(B) < 2: return None
    Z = np.vstack([A, B]); D = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    sig = np.median(D[D > 0]) or 1.0
    def k(X, Y): return np.exp(-np.sum((X[:, None] - Y[None]) ** 2, -1) / sig)
    return round(float(k(A, A).mean() + k(B, B).mean() - 2 * k(A, B).mean()), 4)


def sliced_wasserstein(A, B, n_proj=60, seed=0):            # multivariate OT: mean 1-Wasserstein over random 1D projections
    np = _np(); A = np.asarray(A, float); B = np.asarray(B, float)
    if A.ndim == 1: A = A[:, None]; B = B[:, None]
    rng = np.random.RandomState(seed); d = A.shape[1]; tot = 0.0
    for _ in range(n_proj):
        v = rng.randn(d); v /= (np.linalg.norm(v) + 1e-12)
        tot += wasserstein(A @ v, B @ v)
    return round(tot / n_proj, 4)


def _perm_pvalue(a, b, stat_fn, n_perm=150, seed=0):        # permutation test: P(shuffled stat ≥ observed)
    np = _np(); a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) > 400: a = a[np.random.RandomState(1).choice(len(a), 400, False)]   # independent subsample per side
    if len(b) > 400: b = b[np.random.RandomState(2).choice(len(b), 400, False)]   # (not index-paired → unbiased null)
    obs = stat_fn(a, b)
    if obs is None: return None
    pooled = np.concatenate([a, b]); na = len(a); rng = np.random.RandomState(seed); cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pooled); s = stat_fn(pooled[:na], pooled[na:])
        if s is not None and s >= obs: cnt += 1
    return round((cnt + 1) / (n_perm + 1), 4)


def mmd_permutation_pvalue(a, b, n_perm=150):               # rigorous kernel two-sample TEST (not just the statistic)
    return _perm_pvalue(a, b, mmd_rbf, n_perm)


def energy_permutation_pvalue(a, b, n_perm=150):            # rigorous energy-distance two-sample TEST
    return _perm_pvalue(a, b, energy, n_perm)


def welch_t_p(a, b):                                        # unequal-variance t-test on means
    try:
        from scipy.stats import ttest_ind
        return round(float(ttest_ind(a, b, equal_var=False).pvalue), 4)
    except Exception:  # noqa: BLE001
        return None


def kendall_tau_matrix_dist(A, B):                          # Kendall-τ rank-dependence matrices distance (copula, robust)
    np = _np(); from scipy.stats import kendalltau
    def cm(T):
        T = np.asarray(T, float)
        if T.ndim == 1 or T.shape[0] < 3: return None
        d = T.shape[1]; M = np.eye(d)
        for i in range(d):
            for j in range(i + 1, d):
                t = kendalltau(T[:, i], T[:, j]).correlation; M[i, j] = M[j, i] = 0.0 if np.isnan(t) else t
        return M
    ca, cb = cm(A), cm(B)
    if ca is None or cb is None: return None
    return round(float(np.sqrt(np.nanmean((ca - cb) ** 2))), 4)


def kde_resample(data, n, rng=None):                       # smooth Gaussian-KDE resample (vs the discrete bootstrap)
    np = _np(); from scipy.stats import gaussian_kde
    rng = rng or np.random.RandomState(0)
    return gaussian_kde(np.asarray(data, float)).resample(n, seed=rng)[0]


def gmm_fit_sample(data, n, k=3, seed=0):                  # fit a Gaussian mixture, then sample from it
    np = _np()
    try:
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(int(k), random_state=seed).fit(np.asarray(data, float)[:, None])
        return g.sample(n)[0].ravel()
    except Exception:  # noqa: BLE001
        return None


def gaussian_copula_sample(target_table, n, rng):
    """Parametric Gaussian copula: fit the rank-correlation as a Gaussian dependence, sample, and push back
    through the empirical marginals → reproduces BOTH marginals and (linear rank) dependence smoothly."""
    np = _np(); from scipy.stats import norm, rankdata
    T = np.asarray(target_table, float); m, d = T.shape
    Z = np.column_stack([norm.ppf((rankdata(T[:, j]) - 0.5) / m) for j in range(d)])
    C = np.corrcoef(Z, rowvar=False); L = np.linalg.cholesky(C + 1e-6 * np.eye(d))
    U = norm.cdf(rng.standard_normal((n, d)) @ L.T); out = np.empty((n, d))
    for j in range(d):
        col = np.sort(T[:, j]); out[:, j] = col[np.clip((U[:, j] * m).astype(int), 0, m - 1)]
    return out


def energy_multivariate(A, B):                              # multivariate energy distance (two-sample, ⇔ equal law)
    np = _np(); A = np.asarray(A, float); B = np.asarray(B, float)
    if len(A) > 1200: A = A[np.random.RandomState(0).choice(len(A), 1200, False)]
    if len(B) > 1200: B = B[np.random.RandomState(1).choice(len(B), 1200, False)]
    def m(X, Y): return float(np.sqrt(((X[:, None] - Y[None]) ** 2).sum(-1)).mean())
    return round(float(2 * m(A, B) - m(A, A) - m(B, B)), 4)


def distance_correlation(x, y):                             # Székely dCor: detects ANY (incl. nonlinear) dependence, 0⇔independent
    np = _np(); x = _cap(x, 800); y = _cap(y, 800)
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    a = np.abs(x[:, None] - x[None, :]); b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean(); B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov = np.sqrt(max((A * B).mean(), 0)); vx = np.sqrt(max((A * A).mean(), 0)); vy = np.sqrt(max((B * B).mean(), 0))
    return round(float(dcov / np.sqrt(vx * vy)) if vx * vy > 0 else 0.0, 4)


def dcor_matrix_dist(A, B):                                 # nonlinear cross-column dependence (dCor matrices distance)
    np = _np(); A = np.asarray(A, float); B = np.asarray(B, float)
    if A.ndim == 1 or A.shape[0] < 4: return None
    d = A.shape[1]
    def cm(T):
        M = np.ones((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                M[i, j] = M[j, i] = distance_correlation(T[:, i], T[:, j])
        return M
    return round(float(np.sqrt(np.nanmean((cm(A) - cm(B)) ** 2))), 4)


def hsic(x, y):                                            # Hilbert-Schmidt Independence Criterion (RBF), 0⇔independent
    np = _np(); x = _cap(x, 400)[:, None]; y = _cap(y, 400)[:, None]
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    def K(z):
        d = (z - z.T) ** 2; s = np.median(d[d > 0]) or 1.0; return np.exp(-d / s)
    H = np.eye(n) - 1.0 / n; Kx = H @ K(x) @ H; Ky = H @ K(y) @ H
    return round(float((Kx * Ky).sum() / (n - 1) ** 2), 4)


def jensen_shannon_distance(a, b, bins=25):               # metric form (√JS-divergence) ∈[0,1]
    d = js_divergence(a, b, bins); return None if d is None else round(float(d) ** 0.5, 4)


def jeffreys(a, b, bins=25):                              # symmetric KL: KL(a‖b)+KL(b‖a)
    np = _np(); pa, pb = _hist2(a, b, bins)
    return None if pa is None else round(float(np.sum((pa - pb) * np.log(pa / pb))), 4)


def renyi_divergence(a, b, alpha=2.0, bins=25):           # Rényi-α (generalizes KL; α=2 ≈ collision divergence)
    np = _np(); pa, pb = _hist2(a, b, bins)
    if pa is None: return None
    if abs(alpha - 1) < 1e-6: return kl_divergence(a, b, bins)
    return round(float(1.0 / (alpha - 1) * np.log(np.sum(pa ** alpha * pb ** (1 - alpha)))), 4)


def mutual_information(x, y, bins=16):                    # binned MI — nonlinear dependence (0 ⇔ independent)
    np = _np()
    n = min(len(x), len(y)); x, y = _san(x)[:n], _san(y)[:n]
    if n == 0: return None
    c = np.histogram2d(x, y, bins)[0]; pxy = c / (c.sum() + 1e-12) + 1e-12
    px = pxy.sum(1, keepdims=True); py = pxy.sum(0, keepdims=True)
    return round(float(np.sum(pxy * np.log(pxy / (px * py)))), 4)


def shapiro_p(a):                                        # Shapiro–Wilk normality (small-n gold standard)
    try:
        from scipy.stats import shapiro
        import numpy as np
        return round(float(shapiro(np.asarray(a, float)[:5000]).pvalue), 4)
    except Exception:  # noqa: BLE001
        return None


def fligner_p(a, b):                                     # Fligner–Killeen: robust (non-normal) scale-equality test
    try:
        from scipy.stats import fligner
        return round(float(fligner(a, b).pvalue), 4)
    except Exception:  # noqa: BLE001
        return None


def rank_gauss(x):                                       # rank-based Gaussianization → exact N(0,1) marginal (grandmaster transform)
    np = _np(); from scipy.stats import norm
    x = np.asarray(x, float); return norm.ppf((np.argsort(np.argsort(x)) + 0.5) / len(x))


def kl_gaussian_mv(A, B):                                # closed-form KL between fitted multivariate Gaussians
    np = _np(); A = np.asarray(A, float); B = np.asarray(B, float); d = A.shape[1]
    mA, mB = A.mean(0), B.mean(0)
    CA = np.cov(A, rowvar=False) + 1e-6 * np.eye(d); CB = np.cov(B, rowvar=False) + 1e-6 * np.eye(d)
    CBi = np.linalg.pinv(CB); dm = mB - mA
    sign, logdet = np.linalg.slogdet(CB @ np.linalg.pinv(CA))
    return round(float(0.5 * (np.trace(CBi @ CA) + dm @ CBi @ dm - d + logdet)), 4)


def kruskal_wallis_p(*groups):                            # omnibus >2-group equality (multiple embryos/folds)
    try:
        from scipy.stats import kruskal
        return round(float(kruskal(*groups).pvalue), 4)
    except Exception:  # noqa: BLE001
        return None


def normality_test(a):                                    # is this column Gaussian? Jarque–Bera + D'Agostino K²
    out = {}
    try:
        from scipy.stats import jarque_bera, normaltest
        out["jarque_bera_p"] = round(float(jarque_bera(a).pvalue), 4)
        out["dagostino_p"] = round(float(normaltest(a).pvalue), 4)
        out["is_normal"] = out["jarque_bera_p"] > 0.05 and out["dagostino_p"] > 0.05
    except Exception:  # noqa: BLE001
        out["is_normal"] = None
    return out


def adversarial_weights(A, B, seed=None, n_splits=3):
    """Propensity weights p(x∈B)/p(x∈A) from an adversarial-validation classifier → reweight sample A so its
    (weighted) distribution looks like B. The CV-building move: downweight train rows unlike test.
    seed: optional RNG for the classifier; n_splits: max CV folds (auto-capped)."""
    np = _np()
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_predict
    except Exception:  # noqa: BLE001
        return None
    A = _san(A); B = _san(B)
    if A.ndim == 1: A = A[:, None]
    if B.ndim == 1: B = B[:, None]
    if len(A) < 6 or len(B) < 6: return None
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=60, random_state=seed), X, y,
                          cv=int(min(n_splits, len(A), len(B))), method="predict_proba")[:, 1]
    pa = np.clip(p[:len(A)], 1e-3, 1 - 1e-3); w = pa / (1 - pa)
    return (w / (w.mean() + 1e-12)).round(4)


def mahalanobis_dist(A, B):                                  # between-mean distance in pooled-covariance metric
    np = _np(); A = np.asarray(A, float); B = np.asarray(B, float)
    d = A.mean(0) - B.mean(0); S = np.cov(np.vstack([A, B]), rowvar=False)
    try: Si = np.linalg.pinv(S); return round(float(np.sqrt(d @ Si @ d)), 4)
    except Exception: return None  # noqa: BLE001


# ── covariate-shift reweighting (make source match target WITHOUT resampling) ──
def kmm_weights(src, target, bins=20):
    """Kernel-mean-matching-style importance weights w(x)=p_target(x)/p_src(x) via histogram density ratio →
    reweight source samples so their weighted distribution matches the target (grandmaster covariate-shift fix)."""
    np = _np(); src = np.asarray(src, float); target = np.asarray(target, float)
    lo, hi = float(min(src.min(), target.min())), float(max(src.max(), target.max()))
    edg = np.linspace(lo, hi, bins + 1)
    ps = np.histogram(src, edg, density=True)[0] + 1e-9; pt = np.histogram(target, edg, density=True)[0] + 1e-9
    idx = np.clip(np.digitize(src, edg) - 1, 0, bins - 1); w = (pt / ps)[idx]
    return (w / w.mean()).round(4)


# ── sampling transforms ──
def moment_match_affine(src, target):                        # affine x→(x−μs)/σs·σt+μt (match mean+var exactly)
    np = _np(); src = np.asarray(src, float); target = np.asarray(target, float)
    ss = src.std() or 1.0
    return (src - src.mean()) / ss * target.std() + target.mean()


def power_transform(src):                                     # Yeo–Johnson → Gaussianize (handles zeros/negatives)
    try:
        from sklearn.preprocessing import PowerTransformer
        import numpy as np
        return PowerTransformer(method="yeo-johnson").fit_transform(np.asarray(src, float)[:, None]).ravel()
    except Exception:  # noqa: BLE001
        return None


def best_fit_distribution(data, families=None):
    """FIND THE CLOSEST parametric distribution: fit each candidate family by MLE, rank by KS statistic +
    AIC, return the best fit and the full ranked table. The grandmaster 'what law is this column?' move —
    once you know it's e.g. lognormal you can sample/augment analytically instead of only resampling."""
    np = _np()
    from scipy import stats as st
    data = np.asarray(data, float); data = data[np.isfinite(data)]
    if len(data) < 8: return None
    fam = families or ["norm", "lognorm", "gamma", "expon", "beta", "weibull_min", "pareto", "poisson_like"]
    ranked = []
    for name in fam:
        try:
            if name == "poisson_like":                         # discrete counts → Poisson (KS via CDF at integers)
                lam = float(max(data.mean(), 1e-6)); D = ks(data, st.poisson.rvs(lam, size=max(200, len(data)), random_state=0))
                k = 1; ll = float(np.sum(st.poisson.logpmf(np.round(data).clip(0), lam)))
            else:
                dist = getattr(st, name)
                shift = data if name not in ("lognorm", "gamma", "expon", "beta", "weibull_min", "pareto") else data - data.min() + 1e-6
                par = dist.fit(shift)
                xs = np.sort(shift); emp = np.arange(1, len(xs) + 1) / len(xs)   # KS vs fitted CDF (avoids scipy kstest arg bug)
                D = float(np.max(np.abs(emp - dist.cdf(xs, *par))))
                k = len(par); ll = float(np.sum(dist.logpdf(shift, *par)))
            aic = 2 * k - 2 * ll
            ranked.append({"family": name, "ks": round(D, 4), "aic": round(aic, 1)})
        except Exception:  # noqa: BLE001
            continue
    ranked.sort(key=lambda r: r["ks"])
    return {"best": ranked[0]["family"] if ranked else None, "ranked": ranked} if ranked else None


def morph_to_target(src, target=None, target_family=None, target_params=None, method="quantile"):
    """CHANGE one distribution INTO a target. method='quantile' → exact empirical inverse-CDF match (monotone
    OT); 'moment' → affine mean/std match; 'parametric' → map src ranks through a fitted target CDF⁻¹. Returns
    the morphed array whose distribution ≈ the target. The 'make this column look like that one' transform."""
    np = _np()
    if method == "moment" and target is not None:
        return moment_match_affine(src, target)
    if method == "parametric" and target_family:
        from scipy import stats as st
        dist = getattr(st, target_family); par = target_params or (dist.fit(target) if target is not None else ())
        r = (np.argsort(np.argsort(np.asarray(src, float))) + 0.5) / len(src)
        return dist.ppf(r, *par)
    return quantile_transform(src, target)                     # default: exact empirical match


def empirical_copula_sample(target_table, n, rng):
    """Resample a JOINT table preserving its empirical rank-dependence (empirical copula) — draws row indices
    per column by shared uniform ranks so cross-column correlations are reproduced, not just the marginals."""
    np = _np(); T = np.asarray(target_table, float); m, d = T.shape
    U = rng.random(n)                                         # shared latent uniform → common rank structure
    out = np.empty((n, d))
    for j in range(d):
        col = np.sort(T[:, j]); out[:, j] = col[np.clip((U * m).astype(int), 0, m - 1)]
    return out


def compare_columns(boxed: dict, comp: dict, ks_thresh=0.34) -> dict:
    """Full per-column metric suite for two dicts of {column: array}. Returns {column: {metrics...}} + verdict."""
    out = {}
    for col, ba in boxed.items():
        ca = comp.get(col)
        if ca is None or ba is None or len(ba) < 2 or len(ca) < 2:
            continue
        row = {"n_boxed": len(ba), "n_comp": len(ca),
               # CDF / optimal-transport
               "ks": ks(ba, ca), "kuiper": kuiper(ba, ca), "wasserstein_norm": wasserstein_norm(ba, ca),
               "wasserstein2": wasserstein2(ba, ca), "crps": crps(ba, ca), "sinkhorn": sinkhorn(ba, ca),
               "energy": energy(ba, ca), "mmd_rbf": mmd_rbf(ba, ca), "cramer_von_mises": cramer_von_mises(ba, ca),
               "anderson_darling": anderson_darling(ba, ca),
               # histogram-family divergences
               "js_div": js_divergence(ba, ca), "kl_div": kl_divergence(ba, ca), "hellinger": hellinger(ba, ca),
               "bhattacharyya": bhattacharyya(ba, ca), "total_variation": total_variation(ba, ca),
               "chi_square": chi_square(ba, ca), "overlap": overlap_coef(ba, ca), "psi": psi(ba, ca),
               # hypothesis-test p-values (high ⇒ can't reject "same")
               "ks_p": ks_pvalue(ba, ca), "mannwhitney_p": mannwhitney_p(ba, ca), "levene_p": levene_p(ba, ca),
               "epps_singleton_p": epps_singleton_p(ba, ca),
               # effect sizes + shape
               "cohens_d": cohens_d(ba, ca), "cliffs_delta": cliffs_delta(ba, ca), "qq_r2": qq_r2(ba, ca),
               **{f"moment_{k}": v for k, v in moments(ba, ca).items()}}
        row["close"] = row["ks"] is not None and row["ks"] <= ks_thresh
        out[col] = row
    return out


# ───────────────────────── drift / adversarial validation ─────────────────────────
def adversarial_auc(A, B, seed=None, n_splits=5):
    """Train a classifier to tell multivariate sample A from B; CV-AUC≈0.5 ⇒ indistinguishable (well matched).
    A,B: lists of equal-length feature vectors (per-crop). Grandmaster CV↔LB-gap diagnostic.
    seed: optional RNG for the discriminator (reproducibility); n_splits: max CV folds (auto-capped)."""
    np = _np()
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
    except Exception:  # noqa: BLE001
        return None
    A = _san(A); B = _san(B)
    if A.ndim == 1: A = A[:, None]
    if B.ndim == 1: B = B[:, None]
    if len(A) < 4 or len(B) < 4: return None
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    k = int(min(n_splits, len(A), len(B)))
    try:
        return round(float(np.mean(cross_val_score(HistGradientBoostingClassifier(max_iter=80, random_state=seed),
                                                    X, y, cv=k, scoring="roc_auc"))), 4)
    except Exception:  # noqa: BLE001
        return None


def spearman_matrix_dist(A, B, cols):
    """Frobenius distance between the Spearman rank-correlation matrices of two per-crop feature tables →
    measures how well the CROSS-column dependence (copula) is reproduced (per-column metrics can't see this)."""
    np = _np()
    from scipy.stats import rankdata
    def cm(T):
        T = np.asarray(T, float)
        if T.ndim == 1 or T.shape[0] < 3: return None
        R = np.vstack([rankdata(T[:, j]) for j in range(T.shape[1])]).T
        return np.corrcoef(R, rowvar=False)
    ca, cb = cm(A), cm(B)
    if ca is None or cb is None: return None
    return round(float(np.sqrt(np.nanmean((ca - cb) ** 2))), 4)


# ───────────────────────── sampling / CV utilities ─────────────────────────
def quantile_transform(src, target):
    """Monotone inverse-CDF (histogram-matching) map that forces `src`'s marginal onto `target`'s EXACTLY —
    1D optimal transport. Returns the transformed src array. The analytic 'make it match' move."""
    np = _np(); src = np.asarray(src, float); target = np.asarray(target, float)
    r = (np.argsort(np.argsort(src)) + 0.5) / len(src)         # src ranks → uniform
    return np.quantile(target, r)


# ═══════════════════ AUDIT ADDITIONS (5-round completeness sweep) ═══════════════════
# ── CDF-topology distances (critic 1) ──
def wasserstein_inf(a, b):                                 # worst-case quantile displacement supᵤ|F⁻¹_a−F⁻¹_b|
    np = _np(); q = np.linspace(0, 1, 201)
    return round(float(np.max(np.abs(np.quantile(a, q) - np.quantile(b, q)))), 4)


def levy_metric(a, b, grid=200):                           # Lévy–Prokhorov: robust to horizontal shift, metrizes weak conv.
    np = _np(); a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    lo, hi = float(min(a[0], b[0])), float(max(a[-1], b[-1]))
    xs = np.linspace(lo, hi, grid); Fa = np.searchsorted(a, xs, "right") / len(a)
    def within(eps):
        Fam = np.searchsorted(a, xs - eps, "right") / len(a); Fap = np.searchsorted(a, xs + eps, "right") / len(a)
        Fb = np.searchsorted(b, xs, "right") / len(b)
        return np.all(Fam - eps <= Fb + 1e-9) and np.all(Fb <= Fap + eps + 1e-9)
    lo_e, hi_e = 0.0, hi - lo + 1.0
    for _ in range(30):                                    # binary search smallest ε
        mid = 0.5 * (lo_e + hi_e); lo_e, hi_e = (lo_e, mid) if within(mid) else (mid, hi_e)
    return round(float(hi_e), 4)


def watson_u2(a, b):                                       # mean-centred CvM: isolates SHAPE from a pure location offset
    np = _np(); a = np.asarray(a, float); b = np.asarray(b, float)
    v = np.sort(np.concatenate([a, b]))
    d = np.searchsorted(np.sort(a), v, "right") / len(a) - np.searchsorted(np.sort(b), v, "right") / len(b)
    return round(float(np.mean((d - d.mean()) ** 2)), 4)


# ── hypothesis tests (critic 2) ──
def brunner_munzel_p(a, b):                                # nonparametric location under UNEQUAL variance (Behrens–Fisher)
    try:
        from scipy.stats import brunnermunzel
        return round(float(brunnermunzel(a, b).pvalue), 4)
    except Exception:  # noqa: BLE001
        return None


def anderson_gof(a, dist="norm"):                          # 1-sample Anderson–Darling GoF vs a named family (tail-weighted)
    try:
        from scipy.stats import anderson
        return round(float(anderson(np_asarray(a), dist).statistic), 4)
    except Exception:  # noqa: BLE001
        return None


def mood_median_p(a, b):                                   # robust equality-of-medians (heavy-tail safe)
    try:
        from scipy.stats import median_test
        return round(float(median_test(a, b)[1]), 4)
    except Exception:  # noqa: BLE001
        return None


# ── multivariate / dependence (critic 3) ──
def tail_dependence(x, y, u=0.95):                          # joint-extreme co-movement λ_U, λ_L (empirical copula)
    np = _np(); x = np.asarray(x, float); y = np.asarray(y, float)
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    U = (np.argsort(np.argsort(x)) + 0.5) / n; V = (np.argsort(np.argsort(y)) + 0.5) / n
    up = np.sum((U > u) & (V > u)) / max(1, np.sum(U > u)); lo = np.sum((U < 1 - u) & (V < 1 - u)) / max(1, np.sum(U < 1 - u))
    return {"upper": round(float(up), 4), "lower": round(float(lo), 4)}


def chatterjee_xi(x, y):                                    # ASYMMETRIC ξ→1 iff Y=f(X); catches NON-monotonic dependence
    np = _np(); x = np.asarray(x, float); y = np.asarray(y, float)
    n = min(len(x), len(y)); order = np.argsort(x[:n], kind="stable"); r = np.argsort(np.argsort(y[:n][order])) + 1
    return round(float(1 - 3 * np.sum(np.abs(np.diff(r))) / (n * n - 1)), 4)


def knn_kl_divergence(A, B, k=5):                          # distribution-free multivariate KL (Wang k-NN; non-Gaussian ok)
    np = _np()
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:  # noqa: BLE001
        return None
    A = np.atleast_2d(np.asarray(A, float)); B = np.atleast_2d(np.asarray(B, float))
    if A.shape[0] < k + 1 or B.shape[0] < k: return None
    d = A.shape[1]; n, m = len(A), len(B)
    rho = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A)[0][:, k]
    nu = NearestNeighbors(n_neighbors=k).fit(B).kneighbors(A)[0][:, k - 1]
    return round(float(d / n * np.sum(np.log((nu + 1e-12) / (rho + 1e-12))) + np.log(m / (n - 1))), 4)


def friedman_rafsky(A, B):                                 # graph-based, tuning-free MST two-sample test (permutation p)
    np = _np()
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial.distance import cdist
    except Exception:  # noqa: BLE001
        return None
    A = np.atleast_2d(np.asarray(A, float)); B = np.atleast_2d(np.asarray(B, float))
    if len(A) > 300: A = A[np.random.RandomState(0).choice(len(A), 300, False)]
    if len(B) > 300: B = B[np.random.RandomState(1).choice(len(B), 300, False)]
    Z = np.vstack([A, B]); lab = np.r_[np.zeros(len(A)), np.ones(len(B))]
    def cross(labels):
        mst = minimum_spanning_tree(cdist(Z, Z)).tocoo()
        return int(np.sum(labels[mst.row] != labels[mst.col]))
    obs = cross(lab); rng = np.random.RandomState(0); cnt = 0
    for _ in range(120):
        p = lab.copy(); rng.shuffle(p)
        if cross(p) <= obs: cnt += 1
    return round((cnt + 1) / 121, 4)                       # low p ⇒ few cross-edges ⇒ distributions differ


def partial_distance_correlation(x, y, z):                 # dependence of X,Y CONTROLLING for Z (Székely–Rizzo)
    np = _np()
    def uc(v):                                             # U-centred distance matrix
        v = np.asarray(v, float); v = v[:, None] if v.ndim == 1 else v
        D = np.sqrt(((v[:, None] - v[None]) ** 2).sum(-1)); n = len(D)
        M = D - D.sum(0, keepdims=True) / (n - 2) - D.sum(1, keepdims=True) / (n - 2) + D.sum() / ((n - 1) * (n - 2))
        np.fill_diagonal(M, 0); return M
    def ip(P, Q): return float((P * Q).sum())
    n = min(len(x), len(y), len(z)); Ax, Ay, Az = uc(x[:n]), uc(y[:n]), uc(z[:n])
    def pc(P, Q): return ip(P, Q) / np.sqrt(ip(P, P) * ip(Q, Q) + 1e-12)
    rxy, rxz, ryz = pc(Ax, Ay), pc(Ax, Az), pc(Ay, Az)
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return round(float((rxy - rxz * ryz) / den) if den > 1e-9 else 0.0, 4)


def multivariate_spearman_rho(A):                          # single joint rank-dependence scalar (Schmid–Schmidt)
    np = _np(); A = np.asarray(A, float); n, d = A.shape
    U = np.column_stack([(np.argsort(np.argsort(A[:, j])) + 1) / (n + 1) for j in range(d)])
    h = (d + 1) / (2 ** d - (d + 1))
    return round(float(h * (2 ** d * np.mean(np.prod(1 - U, axis=1)) - 1)), 4)


# ── sampling / morphing (critic 4) ──
def t_copula_sample(target_table, n, rng, nu=5):           # joint sample WITH tail dependence (Gaussian copula has none)
    np = _np(); from scipy.stats import norm, t as tdist, rankdata
    T = np.asarray(target_table, float); m, d = T.shape
    Z = np.column_stack([norm.ppf((rankdata(T[:, j]) - 0.5) / m) for j in range(d)])
    L = np.linalg.cholesky(np.corrcoef(Z, rowvar=False) + 1e-6 * np.eye(d))
    g = rng.standard_normal((n, d)) @ L.T; chi = rng.chisquare(nu, n)[:, None]
    U = tdist.cdf(g * np.sqrt(nu / chi), nu); out = np.empty((n, d))
    for j in range(d):
        col = np.sort(T[:, j]); out[:, j] = col[np.clip((U[:, j] * m).astype(int), 0, m - 1)]
    return out


def sir_resample(data, weights, n, rng):                   # weighted bootstrap: turn density-ratio weights → equal-weight sample
    np = _np(); data = np.asarray(data, float); w = np.asarray(weights, float); w = w / w.sum()
    return data[rng.choice(len(data), n, replace=True, p=w)]


def stratified_quantile_match(src, target, src_strata, target_strata):
    """Per-stratum empirical inverse-CDF match → matches P(X | stratum), preserving conditional structure that
    a global quantile_transform flattens."""
    np = _np(); src = np.asarray(src, float); out = src.copy()
    for s in np.unique(src_strata):
        ms = np.asarray(src_strata) == s; mt = np.asarray(target_strata) == s
        if ms.sum() and mt.sum():
            out[ms] = quantile_transform(src[ms], np.asarray(target, float)[mt])
    return out


def smote_sample(A, n, k=5, rng=None):                     # synthetic points via k-NN convex interpolation (manifold aug)
    np = _np(); rng = rng or np.random.RandomState(0)
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:  # noqa: BLE001
        return None
    A = np.atleast_2d(np.asarray(A, float)); nn = NearestNeighbors(n_neighbors=min(k + 1, len(A))).fit(A)
    idx = rng.randint(0, len(A), n); out = []
    for i in idx:
        neigh = nn.kneighbors(A[i:i + 1])[1][0][1:]
        j = neigh[rng.randint(0, len(neigh))] if len(neigh) else i
        lam = rng.random(); out.append(A[i] + lam * (A[j] - A[i]))
    return np.array(out)


def ot_barycentric_map(src, target, eps=None, iters=60):   # JOINT multivariate morph via the Sinkhorn coupling
    np = _np(); src = np.atleast_2d(np.asarray(src, float)); target = np.atleast_2d(np.asarray(target, float))
    if len(src) > 400: src = src[np.random.RandomState(0).choice(len(src), 400, False)]
    if len(target) > 400: target = target[np.random.RandomState(1).choice(len(target), 400, False)]
    C = ((src[:, None] - target[None]) ** 2).sum(-1); eps = eps or (np.median(C) or 1.0)
    K = np.exp(-C / eps); u = np.ones(len(src)) / len(src); v = np.ones(len(target)) / len(target)
    r = np.ones(len(src)) / len(src); c = np.ones(len(target)) / len(target)
    for _ in range(iters):
        u = r / (K @ v + 1e-12); v = c / (K.T @ u + 1e-12)
    P = np.diag(u) @ K @ np.diag(v)
    return (P @ target) / (P.sum(1, keepdims=True) + 1e-12)  # each src point → barycentre of matched targets


def coral_align(src, target):                              # whiten with src cov, recolour with target cov (full 2nd-moment)
    np = _np(); src = np.atleast_2d(np.asarray(src, float)); target = np.atleast_2d(np.asarray(target, float))
    from scipy.linalg import sqrtm
    d = src.shape[1]; Cs = np.cov(src, rowvar=False) + np.eye(d); Ct = np.cov(target, rowvar=False) + np.eye(d)
    Ws = np.linalg.pinv(sqrtm(Cs).real); Wt = sqrtm(Ct).real
    return (src - src.mean(0)) @ Ws @ Wt + target.mean(0)


# ── drift / practical (critic 5) ──
def label_shift_weights(y_train, train_proba, test_proba):
    """BBSE label-shift: recover test class priors from the train confusion matrix + soft predictions, giving
    per-class importance weights P_test(y)/P_train(y). The one reweighting regime KMM/adversarial (covariate) miss."""
    np = _np(); y = np.asarray(y_train, int); Pt = np.asarray(train_proba, float); Pte = np.asarray(test_proba, float)
    K = Pt.shape[1]; Ctr = np.array([[Pt[y == j, i].sum() for j in range(K)] for i in range(K)]) / len(y)
    mu = Pte.mean(0); w = np.linalg.pinv(Ctr) @ mu
    prior = np.array([(y == j).mean() for j in range(K)]) + 1e-9
    return np.clip(w / prior, 0, None).round(4)


def support_coverage_rate(train, test):                    # fraction of test outside train's observed support (novelty/OOB)
    np = _np(); tr = np.asarray(train, float); te = np.asarray(test, float)
    return round(float(np.mean((te < tr.min()) | (te > tr.max()))), 4)


def woe_iv(feature, target, bins=10):                      # Weight-of-Evidence encoding + Information Value (binary target)
    np = _np(); f = np.asarray(feature, float); y = np.asarray(target, int)
    edg = np.quantile(f, np.linspace(0, 1, bins + 1)); edg[0], edg[-1] = -np.inf, np.inf
    b = np.digitize(f, edg) - 1; tot_g = max((y == 1).sum(), 1); tot_b = max((y == 0).sum(), 1)
    woe = {}; iv = 0.0
    for k in range(bins):
        g = (y[b == k] == 1).sum() / tot_g + 1e-6; bd = (y[b == k] == 0).sum() / tot_b + 1e-6
        w = float(np.log(g / bd)); woe[k] = round(w, 4); iv += (g - bd) * w
    return {"iv": round(float(iv), 4), "woe": woe}


# ── grandmaster ensemble / post-processing optimizers (mined from real 2025-26 top solutions) ──
def optimized_rounder(y_true, y_pred, metric_fn, n_classes=None, maximize=True):
    """Find the CONTINUOUS→ordinal cut-points that optimize `metric_fn` on OOF predictions (Nelder-Mead).
    The child-mind-PIU 1st-place lever: regress a continuous target, then optimize the QWK rounding
    thresholds instead of predicting the class directly. Returns (thresholds, best_score, rounded)."""
    np = _np(); from scipy.optimize import minimize
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred, float)
    k = int(n_classes or (int(np.max(y_true)) + 1))
    q = np.quantile(y_pred, np.linspace(0, 1, k + 1)[1:-1])            # init at empirical quantiles

    def _round(th):
        th = np.sort(th)
        return np.digitize(y_pred, th)

    def _neg(th):
        s = metric_fn(y_true, _round(th))
        return -s if maximize else s
    res = minimize(_neg, q, method="Nelder-Mead", options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-6})
    th = np.sort(res.x); rounded = _round(th)
    best = metric_fn(y_true, rounded)
    return th.round(5).tolist(), round(float(best), 6), rounded


def caruana_ensemble_selection(oof_dict, y_true, metric_fn, maximize=True, n_iter=100, init_top=3):
    """Greedy ensemble selection WITH replacement (Caruana 2004) — the canonical GM blender used across
    s5e11/equity/s5e4. oof_dict = {name: oof_array}. Returns (weights dict, cv, order). Robust to many
    correlated models; a model can be picked multiple times → integer weights normalized at the end."""
    np = _np()
    names = list(oof_dict); oofs = {n: np.asarray(oof_dict[n], float) for n in names}
    singles = {n: metric_fn(y_true, oofs[n]) for n in names}
    order0 = sorted(names, key=lambda n: singles[n], reverse=maximize)
    counts = {n: 0 for n in names}
    for n in order0[:init_top]:
        counts[n] += 1                                                 # warm start with the best few
    ens = sum(counts[n] * oofs[n] for n in names)
    tot = sum(counts.values()) or 1
    best = metric_fn(y_true, ens / tot)
    picks = list(order0[:init_top])
    for _ in range(n_iter):
        cand_best, cand_name = None, None
        for n in names:
            trial = (ens + oofs[n]) / (tot + 1)
            s = metric_fn(y_true, trial)
            if cand_best is None or (s > cand_best if maximize else s < cand_best):
                cand_best, cand_name = s, n
        improved = cand_best > best if maximize else cand_best < best
        if not improved:
            break
        counts[cand_name] += 1; tot += 1; ens = ens + oofs[cand_name]; best = cand_best; picks.append(cand_name)
    weights = {n: counts[n] / tot for n in names if counts[n] > 0}
    return weights, round(float(best), 6), picks


def nelder_mead_weights(oof_list, y_true, metric_fn, maximize=True):
    """Optimize convex blend weights over a list of OOF arrays via Nelder-Mead (rsna/s5e4 stacking)."""
    np = _np(); from scipy.optimize import minimize
    oofs = [np.asarray(o, float) for o in oof_list]; n = len(oofs)
    w0 = np.ones(n) / n

    def _blend(w):
        w = np.clip(w, 0, None); s = w.sum() or 1.0
        return sum((w[i] / s) * oofs[i] for i in range(n))

    def _neg(w):
        v = metric_fn(y_true, _blend(w)); return -v if maximize else v
    res = minimize(_neg, w0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-7})
    w = np.clip(res.x, 0, None); w = w / (w.sum() or 1.0)
    return w.round(5).tolist(), round(float(metric_fn(y_true, _blend(res.x))), 6)


def ridge_stack(oof_matrix, y_true, alpha=1.0):
    """Linear (Ridge) meta-stacker over OOF columns — linear L2 stackers often beat non-linear ones when
    base models overfit (s5e11). oof_matrix = (n_samples, n_models). Returns (coef, intercept, oof_pred)."""
    np = _np(); from sklearn.linear_model import Ridge
    X = np.asarray(oof_matrix, float); y = np.asarray(y_true, float)
    m = Ridge(alpha=alpha, positive=False).fit(X, y)
    return m.coef_.round(5).tolist(), float(round(m.intercept_, 5)), m.predict(X)


def platt_scale(scores, y_true):
    """Platt/logistic calibration of scores→probabilities (equity/rsna use sigmoid calibration on OOF)."""
    np = _np(); from sklearn.linear_model import LogisticRegression
    X = np.asarray(scores, float).reshape(-1, 1); y = np.asarray(y_true, int)
    m = LogisticRegression().fit(X, y)
    return m.predict_proba(X)[:, 1]


def isotonic_calibrate(scores, y_true):
    """Isotonic (monotone, non-parametric) calibration — the other GM calibrator."""
    np = _np(); from sklearn.isotonic import IsotonicRegression
    x = np.asarray(scores, float); y = np.asarray(y_true, float)
    return IsotonicRegression(out_of_bounds="clip").fit(x, y).predict(x)


def soft_spearman(y_true, y_pred):
    """Spearman rank correlation (mitsui metric core). As a LOSS use 1 - this (0.2*MSE + 0.8*(1-ρ) was the
    winning objective). Reused for the spearman_sharpe scorer."""
    np = _np(); from scipy.stats import spearmanr
    r = spearmanr(np.asarray(y_true, float), np.asarray(y_pred, float)).correlation
    return float(0.0 if r != r else r)


def spearman_sharpe(daily_true, daily_pred):
    """Mitsui official metric: mean(daily rank-corr) / std(daily rank-corr) — a stability-penalized ranking
    score. daily_true/daily_pred = lists (per day) of arrays. Higher = better AND more consistent."""
    np = _np()
    corrs = [soft_spearman(t, p) for t, p in zip(daily_true, daily_pred) if len(t) > 1]
    if not corrs:
        return 0.0
    sd = float(np.std(corrs))
    return float(np.mean(corrs) / sd) if sd > 1e-9 else float(np.mean(corrs))


def clip_guard(preds, lo=None, hi=None, train_y=None, pad=0.02):
    """Clip predictions to a safe range to defuse test outliers — the s5e4 lesson (unclipped→RMSE 177).
    If lo/hi are None they are taken from the train target range with a small pad."""
    np = _np(); p = np.asarray(preds, float)
    if (lo is None or hi is None) and train_y is not None:
        ty = np.asarray(train_y, float); span = ty.max() - ty.min()
        lo = ty.min() - pad * span if lo is None else lo
        hi = ty.max() + pad * span if hi is None else hi
    return np.clip(p, lo, hi)


class MathMaster(BaseAgent):
    name = "math-master"
    thread = "B"
    kind = "verdict"

    def run(self, q, worker):
        import json as _j
        np = _np()
        spec = self.spec(q)
        # Mode ADV: two inline feature matrices A (e.g. competition crops) & B (external crops) → does B match A?
        # The full multivariate two-sample battery. adversarial_auc→0.5 = INDISTINGUISHABLE (matched); →1.0 = a
        # classifier separates them (domain gap). Used to prove an augmentation makes external ≈ training BEFORE
        # spending a train run (user 2026-07-12).
        if spec.get("A") is not None and spec.get("B") is not None:
            A = _san(spec["A"]); B = _san(spec["B"])
            if A.ndim == 1: A = A[:, None]
            if B.ndim == 1: B = B[:, None]
            seed = spec.get("seed")
            adv = adversarial_auc(A.tolist(), B.tolist(), seed=seed)
            def _r(fn):                                        # None-safe distance (degenerate input → None, never crash)
                try:
                    v = fn(A.tolist(), B.tolist()); return round(float(v), 4) if v is not None else None
                except Exception:  # noqa: BLE001
                    return None
            out = {"label_A": spec.get("label_A", "A"), "label_B": spec.get("label_B", "B"),
                   "n_A": int(len(A)), "n_B": int(len(B)), "n_feat": int(A.shape[1] if A.ndim > 1 else 1),
                   "adversarial_auc": round(float(adv), 4) if adv is not None else None,
                   "frechet_distance": _r(frechet_distance),
                   "mmd_multivariate": _r(mmd_multivariate),
                   "sliced_wasserstein": _r(sliced_wasserstein),
                   "energy_multivariate": _r(energy_multivariate),
                   "mahalanobis": _r(mahalanobis_dist)}
            # matched if a classifier can't separate them (adv≈0.5). 0.5–0.6 excellent, ≤0.7 acceptable.
            _auc = out["adversarial_auc"] if out["adversarial_auc"] is not None else 0.5
            out["matched"] = bool(_auc <= float(spec.get("adv_thresh", 0.65)))
            out["verdict"] = ("MATCHED (indistinguishable)" if _auc <= 0.6 else
                              "CLOSE" if _auc <= 0.7 else "DISTINGUISHABLE (domain gap)")
            msg = (f"[{worker}] MATH-MASTER adv-validate {out['label_B']} vs {out['label_A']}: "
                   f"adversarial-AUC={out['adversarial_auc']} → {out['verdict']} "
                   f"(FID={out['frechet_distance']} MMD={out['mmd_multivariate']} "
                   f"sliced-W={out['sliced_wasserstein']}). 0.5=matched, 1.0=separable.")
            self.log(msg, kind="verdict", recommendation="adversarial-AUC≈0.5 → the two feature sets are "
                     "distributionally indistinguishable (external now matches training); ≈1.0 → real domain gap.")
            return self.done(out, msg)
        # Mode A: generic — two dicts of arrays passed inline
        boxed = spec.get("boxed_dist"); comp = spec.get("comp_dist")
        boxed_percrop = comp_percrop = None
        # Mode B (biohub default): read competition_dist.json + boxed parquet → build both distributions
        if boxed is None or comp is None:
            cd = COMP / "config" / "_auto" / "competition_dist.json"
            comp_raw = _j.loads(cd.read_text()) if cd.exists() else {}
            comp = {k: [x for e in comp_raw.values() for x in e.get(k, [])] for k in
                    ("tracks_per_crop", "track_len", "z_std", "speed", "nn_dist", "cells_per_frame", "divisions_per_crop")}
            comp_percrop = [[e[k][i] for k in PERCROP] for e in comp_raw.values()
                            for i in range(min(len(e.get(k, [])) for k in PERCROP))] if comp_raw else []
            bp = spec.get("boxed_path", "results/flow_gt/flow_node_gt_matched.parquet")
            bpath = COMP / bp if not Path(bp).is_absolute() else Path(bp)
            boxed, boxed_percrop = self._boxed_dist(bpath, np) if bpath.exists() else ({}, [])
        thr = float(spec.get("ks_thresh", 0.34))
        per_col = compare_columns(boxed, comp, thr)
        best_fit = {c: (best_fit_distribution(v) or {}).get("best") for c, v in comp.items() if v and len(v) >= 8}
        for c, r in per_col.items():                            # attach the analytic law of each column
            r["comp_best_fit"] = best_fit.get(c)
        adv = spearman = frechet = mahal = mmd_mv = swd = kendall = emv = dcor = klg = None
        if boxed_percrop and comp_percrop and len(boxed_percrop) > 3 and len(comp_percrop) > 3:
            adv = adversarial_auc(boxed_percrop, comp_percrop)          # classifier two-sample test (CV↔LB gap)
            spearman = spearman_matrix_dist(boxed_percrop, comp_percrop, PERCROP)  # Spearman copula / cross-column dependence
            kendall = kendall_tau_matrix_dist(boxed_percrop, comp_percrop)   # Kendall-τ dependence (robust copula)
            frechet = frechet_distance(boxed_percrop, comp_percrop)     # FID-style multivariate Gaussian distance
            mahal = mahalanobis_dist(boxed_percrop, comp_percrop)       # between-mean Mahalanobis
            mmd_mv = mmd_multivariate(boxed_percrop, comp_percrop)      # kernel two-sample on vectors
            swd = sliced_wasserstein(boxed_percrop, comp_percrop)       # sliced-Wasserstein multivariate OT
            emv = energy_multivariate(boxed_percrop, comp_percrop)      # multivariate energy distance
            dcor = dcor_matrix_dist(boxed_percrop, comp_percrop)        # nonlinear dependence (dCor copula)
            klg = kl_gaussian_mv(boxed_percrop, comp_percrop)           # closed-form KL of fitted MV Gaussians
        mean_ks = round(float(np.mean([r["ks"] for r in per_col.values() if r["ks"] is not None])), 4) if per_col else None
        n_close = sum(1 for r in per_col.values() if r.get("close"))
        verdict = bool(per_col) and n_close == len(per_col) and (adv is None or adv <= 0.75)
        data = {"per_column": per_col, "mean_ks": mean_ks, "n_close": n_close, "n_cols": len(per_col),
                "adversarial_auc": adv, "spearman_copula_dist": spearman, "kendall_copula_dist": kendall,
                "frechet_distance": frechet, "mahalanobis": mahal, "mmd_multivariate": mmd_mv,
                "sliced_wasserstein": swd, "energy_multivariate": emv, "dcor_copula_dist": dcor, "kl_gaussian_mv": klg,
                "distributions_match": verdict}
        self.save_state(data)
        hdr = ("| column | KS | Wass₁ | energy | MMD² | Hellinger | TV | overlap | KS-p | Cohen-d | close |\n"
               "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|\n")
        body = "\n".join(f"| {c} | {r['ks']} | {r['wasserstein_norm']} | {r['energy']} | {r['mmd_rbf']} | "
                         f"{r['hellinger']} | {r['total_variation']} | {r['overlap']} | {r['ks_p']} | "
                         f"{r['cohens_d']} | {'✅' if r['close'] else '❌'} |" for c, r in per_col.items())
        msg = (f"[{worker}] **MATH-MASTER** · distribution match {'✅' if verdict else '❌'} · "
               f"mean-KS={mean_ks} · {n_close}/{len(per_col)} columns close\n{hdr}{body}\n"
               f"\n**joint (multivariate)**: adversarial-AUC={adv} (0.5=indistinguishable) · "
               f"Fréchet(FID)={frechet} · Mahalanobis={mahal} · MMD-mv={mmd_mv} · sliced-Wass={swd} · "
               f"Spearman-copula={spearman} · Kendall-copula={kendall}\n"
               f"_per-column also carries: Kuiper, Wass₂, CRPS, Sinkhorn, CvM, Anderson–Darling, KL, "
               f"Bhattacharyya, χ², PSI, Mann–Whitney-p, Levene-p, Epps–Singleton-p, Cliff's-δ, QQ-R², moments; "
               f"utilities: quantile/moment/power/parametric morph, KMM weights, best-fit, KDE/GMM/copula resample, "
               f"MMD & energy permutation tests._\n"
               f"KS→0 by Glivenko–Cantelli as the resample converges; adversarial-AUC is the CV↔LB-gap check.")
        try:                                                   # MATCH the 100%-confirmed GT protocol (paper-verify)
            from .paper_verify import label_facts
            imp = label_facts()["implications"]
            msg += (f"\n\n⚠️ **GT-BIAS CAVEAT (confirmed, eda §16):** {imp['cv_over_credits_easy']} So when judging "
                    f"a RECALL delta: use PER-EMBRYO/stage paired tests, weight/stratify by stage, and treat a CV "
                    f"gain on dense (S3–S4) crops as an UPPER bound — confirm on the Kaggle LB. {imp['div_term_thin']}")
        except Exception:  # noqa: BLE001
            pass
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(data, msg, to="leader")

    def _boxed_dist(self, bpath, np):
        import pandas as pd
        df = pd.read_parquet(bpath)
        has_tid = "track_id" in df.columns
        has_flow = all(c in df.columns for c in ("dz", "dy", "dx"))   # DENSE path: speed from the flow field (no tracks)
        tp, tl, zs, sp, nn, cf, dv, percrop = [], [], [], [], [], [], [], []
        for _, g in df.groupby("embryo"):
            gg = g[g["track_id"] >= 0] if has_tid else g
            if not len(gg): continue
            if has_tid:                                          # SPARSE path: track-based track_len + speed
                tp.append(gg["track_id"].nunique())
                for _, tr in gg.groupby("track_id"):
                    ts = sorted(tr["t"].unique()); tl.append(ts[-1] - ts[0] + 1)
                    P = tr.sort_values("t")[["z", "y", "x"]].to_numpy(float)
                    sp += [float(np.linalg.norm(P[i + 1] - P[i])) for i in range(len(P) - 1)]
            elif has_flow:                                       # DENSE path: speed = per-node flow magnitude (density-invariant)
                fm = np.linalg.norm(gg[["dz", "dy", "dx"]].to_numpy(float), axis=1)
                sp += list(fm[np.random.RandomState(0).choice(len(fm), min(len(fm), 3000), replace=False)])
            zsd = float(np.std(gg["z"].to_numpy(float))); zs.append(zsd)
            cf += list(gg.groupby("t").size().to_numpy())
            dvc = int(gg["is_division"].sum()) if "is_division" in gg.columns else 0; dv.append(dvc)
            frames = list(gg.groupby("t")); rng = np.random.RandomState(0)
            for _, fr in (frames if len(frames) <= 40 else [frames[i] for i in rng.choice(len(frames), 40, False)]):
                P = fr[["z", "y", "x"]].to_numpy(float)
                if len(P) > 150: P = P[rng.choice(len(P), 150, False)]   # cap O(n²) on dense frames
                if len(P) >= 2:
                    d = np.linalg.norm(P[:, None] - P[None], axis=-1); np.fill_diagonal(d, np.inf)
                    nn += list(d.min(1))
            percrop.append([tp[-1] if has_tid else len(gg), zsd, dvc, g["t"].nunique()])
        out = {"z_std": zs, "speed": sp, "nn_dist": nn, "cells_per_frame": cf, "divisions_per_crop": dv}
        if has_tid: out["tracks_per_crop"] = tp; out["track_len"] = tl
        return (out, percrop)


# ═══════════ ADDED 2026-07-14: paired tests / multiple-testing / classification metrics / calibration / power ═══════════
# The gaps tonight's rigor exposed — math-master was strong on distribution-DISTANCE but missing the PAIRED-COMPARISON,
# MULTIPLE-TESTING, CLASSIFIER-METRIC and CALIBRATION basics that a Kaggle grandmaster reaches for. All scipy/sklearn.
def wilcoxon_p(a, b):
    """Paired Wilcoxon SIGNED-RANK p (non-parametric paired t). Use for per-dataset before/after deltas."""
    from scipy import stats as _st
    a = _np().asarray(a, float); b = _np().asarray(b, float)
    if len(a) != len(b) or len(a) < 3 or _np().allclose(a, b): return 1.0
    try: return float(_st.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception: return 1.0

def sign_test_p(a, b):
    """Paired SIGN test p (robust to outliers; only counts direction of each pair)."""
    from scipy import stats as _st
    a = _np().asarray(a, float); b = _np().asarray(b, float)
    d = a - b; pos = int((d > 0).sum()); neg = int((d < 0).sum()); n = pos + neg
    if n == 0: return 1.0
    return float(_st.binomtest(min(pos, neg), n, 0.5).pvalue)

def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction → (rejected_mask, adjusted_pvals). USE whenever testing many hypotheses
    (per-dataset, per-feature) — tonight's 63-division / 71-dataset sweeps had NO correction."""
    p = _np().asarray(pvals, float); n = len(p); order = _np().argsort(p); ranks = _np().arange(1, n + 1)
    adj = _np().empty(n); adj[order] = _np().minimum.accumulate((p[order] * n / ranks)[::-1])[::-1]
    adj = _np().clip(adj, 0, 1)
    return (adj <= alpha).tolist(), adj.tolist()

def bonferroni_p(pvals, alpha=0.05):
    p = _np().asarray(pvals, float); adj = _np().clip(p * len(p), 0, 1)
    return (adj <= alpha).tolist(), adj.tolist()

def balanced_accuracy(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))

def matthews_corrcoef(y_true, y_pred):
    from sklearn.metrics import matthews_corrcoef as _m
    return float(_m(y_true, y_pred))

def cohen_kappa(y_true, y_pred):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred))

def roc_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    try: return float(roc_auc_score(y_true, y_score))
    except Exception: return float("nan")

def average_precision(y_true, y_score):
    from sklearn.metrics import average_precision_score
    try: return float(average_precision_score(y_true, y_score))
    except Exception: return float("nan")

def brier_score(y_true, y_prob):
    from sklearn.metrics import brier_score_loss
    try: return float(brier_score_loss(y_true, y_prob))
    except Exception: return float("nan")

def expected_calibration_error(y_true, y_prob, bins=10):
    """ECE — is the model's probability CALIBRATED? (is a 0.377 edge-prob really ~38% likely to be a true link?)"""
    yt = _np().asarray(y_true, float); yp = _np().asarray(y_prob, float)
    edges = _np().linspace(0, 1, bins + 1); ece = 0.0; n = len(yt)
    for i in range(bins):
        m = (yp > edges[i]) & (yp <= edges[i + 1])
        if m.sum() == 0: continue
        ece += (m.sum() / n) * abs(yp[m].mean() - yt[m].mean())
    return float(ece)

def bootstrap_ci(data, stat="mean", n_boot=2000, ci=0.95, seed=0):
    """Bootstrap CI for a statistic — quantify uncertainty on SMALL-SAMPLE numbers (e.g. div_J on few divisions)."""
    x = _np().asarray(data, float); rng = _np().random.RandomState(seed)
    fn = {"mean": _np().mean, "median": _np().median}.get(stat, _np().mean)
    bs = [fn(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    lo, hi = _np().quantile(bs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"stat": float(fn(x)), "lo": float(lo), "hi": float(hi), "ci": ci}

def paired_min_detectable_effect(n, sd_diff, alpha=0.05, power=0.8):
    """Minimum paired mean-difference detectable at given n/power (paired t). If your observed Δ < this, the
    comparison is UNDERPOWERED — flags golden's ~1-division evals as unable to detect a real division effect."""
    from scipy import stats as _st
    if n < 3: return float("inf")
    za = _st.norm.ppf(1 - alpha / 2); zb = _st.norm.ppf(power)
    return float((za + zb) * sd_diff / (n ** 0.5))

def paired_delta_report(before, after, alpha=0.05):
    """One-call paired comparison a grandmaster wants: mean Δ, Cliff's δ (effect size), Wilcoxon + sign p,
    bootstrap CI, and an UNDERPOWERED flag. The rigorous replacement for 'it scored higher, keep it'."""
    a = _np().asarray(before, float); b = _np().asarray(after, float)
    d = b - a; mde = paired_min_detectable_effect(len(d), float(_np().std(d) + 1e-12), alpha)
    ci = bootstrap_ci(d.tolist(), "mean")
    return {"n": int(len(d)), "mean_delta": float(_np().mean(d)), "cliffs_delta": round(float(cliffs_delta(b, a)), 4),
            "wilcoxon_p": round(wilcoxon_p(a, b), 5), "sign_p": round(sign_test_p(a, b), 5),
            "boot_ci": [round(ci["lo"], 5), round(ci["hi"], 5)],
            "underpowered": bool(abs(_np().mean(d)) < mde), "min_detectable": round(mde, 5),
            "significant": bool(ci["lo"] > 0 or ci["hi"] < 0)}


_AGENT = MathMaster()


def run(q, worker):
    return _AGENT.run(q, worker)
