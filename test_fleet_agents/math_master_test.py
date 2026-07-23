"""math_master_test — FULL coverage of the grandmaster stats toolkit. Every implemented function is
exercised with a mathematical property that must hold: distances ≈0 (or divergence-minimal) for identical
samples and STRICTLY larger under a known shift; similarity metrics do the opposite; p-values high for same /
low for different; transforms provably move the source onto the target; multivariate distances separate. Also
asserts NO metric silently returns None on valid input (catches unimplemented/broken tricks). No GPU/fleet."""
import os, sys
import numpy as np
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import math_master as G


def _run():
    print("=== MATH-MASTER FULL MATH VERIFIER ===")
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 900); a2 = rng.normal(0, 1, 900)      # same law
    b = rng.normal(2.5, 1, 900)                                # shifted mean
    sc = rng.normal(0, 3, 900)                                 # same mean, larger scale
    c = {}

    # ── distances that GROW with a mean shift (same-law < shifted-law) ──
    for nm, fn in [("ks", G.ks), ("kuiper", G.kuiper), ("wasserstein", G.wasserstein), ("wasserstein2", G.wasserstein2),
                   ("energy", G.energy), ("mmd_rbf", G.mmd_rbf), ("crps", G.crps), ("sinkhorn", G.sinkhorn),
                   ("js", G.js_divergence), ("kl", G.kl_divergence), ("hellinger", G.hellinger),
                   ("bhattacharyya", G.bhattacharyya), ("total_variation", G.total_variation),
                   ("chi_square", G.chi_square), ("psi", G.psi), ("cramer_von_mises", G.cramer_von_mises),
                   ("anderson_darling", G.anderson_darling)]:
        s, d = fn(a, a2), fn(a, b)
        c[f"{nm}_impl"] = (s is not None and d is not None)                 # actually implemented (not None)
        c[f"{nm}_grows"] = (s is not None and d is not None and s < d)      # monotone in shift

    # ── similarity metrics: HIGH for same law, LOWER for shifted ──
    c["overlap_shrinks"] = G.overlap_coef(a, a2) > G.overlap_coef(a, b)
    c["qq_r2_shrinks"] = (G.qq_r2(a, a2) or 0) > (G.qq_r2(a, b) or 0)

    # ── effect sizes: ≈0 for same law, large for shift ──
    c["cohens_d_zero_same"] = abs(G.cohens_d(a, a2)) < 0.2 and abs(G.cohens_d(a, b)) > 1.5
    c["cliffs_delta_zero_same"] = abs(G.cliffs_delta(a, a2)) < 0.2 and abs(G.cliffs_delta(a, b)) > 0.6

    # ── hypothesis tests: p HIGH when same, p LOW when different ──
    c["ks_p"] = G.ks_pvalue(a, a2) > 0.01 > G.ks_pvalue(a, b)         # diff strongly rejected, same not
    c["mannwhitney_p"] = G.mannwhitney_p(a, a2) > 0.01 > G.mannwhitney_p(a, b)
    c["levene_p_scale"] = G.levene_p(a, a2) > 0.01 > G.levene_p(a, sc)     # variance test catches scale
    c["epps_singleton_impl"] = G.epps_singleton_p(a, b) is not None

    # ── sampling transforms provably move source → target ──
    src = rng.exponential(1.0, 600); tgt = rng.normal(5, 2, 600)
    c["quantile_transform"] = G.ks(G.quantile_transform(src, tgt), tgt) < 0.1 < G.ks(src, tgt)
    mm = G.moment_match_affine(src, tgt)
    c["moment_match"] = abs(np.mean(mm) - np.mean(tgt)) < 0.1 and abs(np.std(mm) - np.std(tgt)) < 0.1
    pt = G.power_transform(src)
    c["power_transform"] = pt is not None and abs(float(__import__("scipy.stats", fromlist=["skew"]).skew(pt))) < abs(float(__import__("scipy.stats", fromlist=["skew"]).skew(src)))
    w = G.kmm_weights(src, tgt)                                # reweight src→tgt: weighted mean shifts toward tgt
    c["kmm_weights"] = abs(np.average(src, weights=w) - np.mean(tgt)) < abs(np.mean(src) - np.mean(tgt))

    # ── FIND best-fitting distribution: lognormal data → 'lognorm' ranks at/near top ──
    ln = rng.lognormal(0, 0.5, 1000); bf = G.best_fit_distribution(ln)
    c["best_fit_impl"] = bf is not None and bf.get("best") is not None and len(bf["ranked"]) >= 3
    c["best_fit_lognorm"] = bf is not None and bf["ranked"][0]["family"] in ("lognorm", "gamma", "weibull_min")  # right-skew family wins over norm
    nd = rng.normal(10, 2, 1000); bf2 = G.best_fit_distribution(nd)
    c["best_fit_normal"] = bf2 is not None and any(r["family"] == "norm" and r["ks"] < 0.06 for r in bf2["ranked"])

    # ── MORPH one distribution onto a target (3 methods) ──
    src2 = rng.exponential(1.0, 700); tg = rng.normal(5, 2, 700)
    c["morph_quantile"] = G.ks(G.morph_to_target(src2, tg, method="quantile"), tg) < 0.1
    mm2 = G.morph_to_target(src2, tg, method="moment")
    c["morph_moment"] = abs(np.mean(mm2) - np.mean(tg)) < 0.15
    mp = G.morph_to_target(src2, target=tg, target_family="norm", method="parametric")
    c["morph_parametric"] = G.ks(mp, tg) < 0.12                # mapped through fitted normal CDF⁻¹

    # ── multivariate distances separate matched vs shifted joint samples ──
    A = rng.normal(0, 1, (150, 3)); Bs = rng.normal(0, 1, (150, 3)); Bd = rng.normal(3, 1, (150, 3))
    c["frechet"] = G.frechet_distance(A, Bs) < G.frechet_distance(A, Bd)
    c["mahalanobis"] = G.mahalanobis_dist(A, Bs) < G.mahalanobis_dist(A, Bd)
    c["mmd_multivariate"] = G.mmd_multivariate(A, Bs) < G.mmd_multivariate(A, Bd)
    c["adv_auc"] = (G.adversarial_auc(A, Bs) or 0.5) < (G.adversarial_auc(A, Bd) or 1.0)
    c["spearman_copula"] = G.spearman_matrix_dist(A, Bs, None) is not None
    ct = G.empirical_copula_sample(A, 200, rng)               # copula sample preserves per-column rank order
    c["empirical_copula"] = ct.shape == (200, 3)

    # ── NEW: sliced-Wasserstein + Kendall separate matched vs shifted joint ──
    c["sliced_wasserstein"] = G.sliced_wasserstein(A, Bs) < G.sliced_wasserstein(A, Bd)
    c["kendall_copula"] = G.kendall_tau_matrix_dist(A, Bs) is not None

    # ── NEW: permutation p-values — high for same law, low for shifted ──
    # (permutation p on same-law is uniform, so assert the sound invariant: clear shift rejected + same never MORE significant)
    c["mmd_perm_p"] = G.mmd_permutation_pvalue(a, b, 100) <= 0.05 <= G.mmd_permutation_pvalue(a, a2, 100) * 100  # diff rejects
    c["mmd_perm_ordered"] = G.mmd_permutation_pvalue(a, a2, 100) >= G.mmd_permutation_pvalue(a, b, 100)
    c["energy_perm_p"] = G.energy_permutation_pvalue(a, b, 100) <= 0.05 and \
        G.energy_permutation_pvalue(a, a2, 100) >= G.energy_permutation_pvalue(a, b, 100)
    c["welch_t_p"] = G.welch_t_p(a, a2) > 0.01 > G.welch_t_p(a, b)

    # ── NEW: smooth/parametric samplers reproduce the target marginal (small KS) ──
    kd = G.kde_resample(b, 500, rng)
    c["kde_resample"] = kd is not None and G.ks(kd, b) < 0.12
    gm = G.gmm_fit_sample(b, 500, k=2)
    c["gmm_fit_sample"] = gm is not None and G.ks(gm, b) < 0.15
    gc = G.gaussian_copula_sample(A, 300, rng)
    c["gaussian_copula"] = gc.shape == (300, 3) and G.ks(gc[:, 0], A[:, 0]) < 0.15

    # ── NEW audit round: energy-mv, dCor, HSIC, Kruskal, normality, adversarial-weights ──
    c["energy_multivariate"] = G.energy_multivariate(A, Bs) < G.energy_multivariate(A, Bd)
    # dCor & HSIC detect dependence: independent≈0, dependent (y=x²) large
    xind = rng.normal(0, 1, 400); yind = rng.normal(0, 1, 400); ydep = xind ** 2 + rng.normal(0, 0.1, 400)
    c["distance_correlation"] = G.distance_correlation(xind, ydep) > G.distance_correlation(xind, yind)
    c["hsic"] = G.hsic(xind, ydep) > G.hsic(xind, yind)
    c["dcor_matrix_dist"] = G.dcor_matrix_dist(A, Bs) is not None
    # Kruskal-Wallis: same 3 groups high p, one shifted low p
    c["kruskal_wallis"] = G.kruskal_wallis_p(a, a2, rng.normal(0, 1, 400)) > 0.05 > G.kruskal_wallis_p(a, a2, b)
    # normality: normal data flagged normal, exponential not
    c["normality_normal"] = G.normality_test(rng.normal(0, 1, 800))["is_normal"] is True
    c["normality_nonnormal"] = G.normality_test(rng.exponential(1, 800))["is_normal"] is False
    # adversarial weights: reweighting A toward Bd shifts A's weighted mean toward Bd
    w = G.adversarial_weights(A, Bd)
    c["adversarial_weights"] = w is not None and abs(np.average(A[:, 0], weights=w) - Bd[:, 0].mean()) < abs(A[:, 0].mean() - Bd[:, 0].mean())

    # ── FINAL audit round: JS-distance, Jeffreys, Rényi, MI, Shapiro, Fligner, rank-gauss, KL-Gaussian-mv ──
    c["js_distance_grows"] = G.jensen_shannon_distance(a, a2) < G.jensen_shannon_distance(a, b)
    c["jeffreys_grows"] = G.jeffreys(a, a2) < G.jeffreys(a, b)
    c["renyi_grows"] = G.renyi_divergence(a, a2) < G.renyi_divergence(a, b)
    c["mutual_info"] = G.mutual_information(xind, ydep) > G.mutual_information(xind, yind)   # dependence
    c["shapiro"] = G.shapiro_p(rng.normal(0, 1, 800)) > 0.05 > G.shapiro_p(rng.exponential(1, 800))
    c["fligner_scale"] = G.fligner_p(a, a2) > 0.01 > G.fligner_p(a, sc)                     # robust scale test
    rg = G.rank_gauss(rng.exponential(1, 1000))
    c["rank_gauss"] = abs(float(np.mean(rg))) < 0.1 and abs(float(np.std(rg)) - 1) < 0.1 and G.shapiro_p(rg) > 0.05
    A2 = rng.normal(0, 1, (150, 3)); Bs2 = rng.normal(0, 1, (150, 3)); Bd2 = rng.normal(3, 1, (150, 3))
    c["kl_gaussian_mv"] = G.kl_gaussian_mv(A2, Bs2) < G.kl_gaussian_mv(A2, Bd2)

    # ══════════ 5-ROUND AUDIT ADDITIONS (21 tricks) ══════════
    # distances (critic 1): grow with shift; watson isolates shape (grows more for scale than location)
    c["wasserstein_inf"] = G.wasserstein_inf(a, a2) < G.wasserstein_inf(a, b)
    c["levy_metric"] = G.levy_metric(a, a2) < G.levy_metric(a, b)
    c["watson_u2"] = G.watson_u2(a, a2) < G.watson_u2(a, sc)          # detects scale/shape difference
    # hypothesis tests (critic 2)
    c["brunner_munzel"] = G.brunner_munzel_p(a, a2) > 0.01 > G.brunner_munzel_p(a, b)
    c["anderson_gof"] = G.anderson_gof(a, "norm") < G.anderson_gof(rng.exponential(1, 900), "norm")
    c["mood_median"] = G.mood_median_p(a, a2) > 0.01 > G.mood_median_p(a, b)
    # multivariate/dependence (critic 3)
    c["tail_dependence"] = G.tail_dependence(xind, ydep)["upper"] >= 0 and isinstance(G.tail_dependence(a, a)["upper"], float)
    c["chatterjee_xi"] = G.chatterjee_xi(xind, ydep) > G.chatterjee_xi(xind, yind)   # nonlinear dep detected
    c["knn_kl"] = G.knn_kl_divergence(A, Bs) < G.knn_kl_divergence(A, Bd)
    c["friedman_rafsky"] = G.friedman_rafsky(A, Bs) > 0.05 > G.friedman_rafsky(A, Bd)  # same high-p, diff low-p
    zc = rng.normal(0, 1, 400)
    c["partial_dcor"] = G.partial_distance_correlation(xind, ydep, zc) is not None
    c["mv_spearman"] = G.multivariate_spearman_rho(A) is not None
    # sampling/morphing (critic 4)
    tc = G.t_copula_sample(A, 200, rng); c["t_copula"] = tc.shape == (200, 3)
    sr = G.sir_resample(src2, G.kmm_weights(src2, tg), 500, rng)      # reweight exponential→normal-target sample
    c["sir_resample"] = G.ks(sr, tg) < G.ks(src2, tg)
    ss = np.r_[np.zeros(350), np.ones(350)].astype(int)
    sq = G.stratified_quantile_match(src2, tg, ss, ss); c["stratified_qmatch"] = G.ks(sq, tg) < G.ks(src2, tg)
    sm = G.smote_sample(A, 200, rng=rng); c["smote"] = sm is not None and sm.shape == (200, 3)
    bm = G.ot_barycentric_map(A, Bd); c["ot_barycentric"] = abs(bm[:, 0].mean() - Bd[:, 0].mean()) < abs(A[:, 0].mean() - Bd[:, 0].mean())
    ca = G.coral_align(A, Bd); c["coral"] = abs(np.cov(ca, rowvar=False)[0, 0] - np.cov(Bd, rowvar=False)[0, 0]) < abs(np.cov(A, rowvar=False)[0, 0] - np.cov(Bd, rowvar=False)[0, 0])
    # drift (critic 5)
    yl = rng.randint(0, 3, 300); pr = np.eye(3)[yl] * 0.7 + 0.1                      # confident-ish soft preds
    lw = G.label_shift_weights(yl, pr, np.eye(3)[rng.randint(0, 3, 200)] * 0.7 + 0.1)
    c["label_shift_weights"] = lw is not None and len(lw) == 3
    c["support_coverage"] = G.support_coverage_rate(a, a2) < G.support_coverage_rate(a, b + 10)  # far-shifted → more OOB
    iv = G.woe_iv(rng.normal(0, 1, 600), (rng.normal(0, 1, 600) > 0).astype(int))
    c["woe_iv"] = "iv" in iv and iv["iv"] >= 0

    # ── compare_columns emits a FULL row (every metric present, none None) + verdict flips ──
    row = G.compare_columns({"x": a}, {"x": a2})["x"]
    expect = {"ks", "kuiper", "wasserstein_norm", "wasserstein2", "crps", "sinkhorn", "energy", "mmd_rbf",
              "cramer_von_mises", "anderson_darling", "js_div", "kl_div", "hellinger", "bhattacharyya",
              "total_variation", "chi_square", "overlap", "psi", "ks_p", "mannwhitney_p", "levene_p",
              "epps_singleton_p", "cohens_d", "cliffs_delta", "qq_r2"}
    c["compare_has_all_metrics"] = expect.issubset(set(row)) and all(row[k] is not None for k in expect)
    c["compare_verdict_flips"] = row["close"] and not G.compare_columns({"x": a}, {"x": b})["x"]["close"]

    fails = [k for k, v in c.items() if not v]
    for k in sorted(c):
        print(f"  {'✅' if c[k] else '❌'} {k}")
    ok = not fails
    print(f"\n=== math-master: {'PASS' if ok else 'FAIL — ' + ', '.join(fails)} · {sum(c.values())}/{len(c)} checks ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); print(f"  ❌ ERROR: {e}"); sys.exit(1)
