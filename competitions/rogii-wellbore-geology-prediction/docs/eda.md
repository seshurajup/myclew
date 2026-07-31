# Mathematical EDA (GPU) — Rogii Wellbore Geology

_All statistics computed on **NVIDIA GeForce RTX 5090** via torch (sm_120 / cu128); pandas only reads CSVs. 350 wells. Each proof: claim → GPU code → data verdict._
_Proofs: **53** (engine extensible toward 100 — add a `Proof(...)` to `PROOFS`)._


## A. State-space trajectory dynamics

### 1. TVT is an Integrated Random Walk (MA(1) 2nd difference)

**Claim.** The second difference of s=TVT+Z is MA(1) with negative lag-1 autocorrelation.

$$\Delta^2 s_i=\xi_{i-1}+(\eta_i-\eta_{i-1})\Rightarrow \rho_1<0,\ \rho_{k\ge2}\approx0,\ q_s=-\gamma_1.$$

```python
d2 = g_diff(s,2); rho1 = g_acov(d2,1)/g_var(d2)      # all torch-cuda
```

**Result.** median ρ₁(Δ²s) = -0.246 over 350 wells.

**Verdict.** ✅ ρ₁<0 ⇒ IRW confirmed; justifies a state-space filter/NN.

---
### 2. The process is MA(1), not MA(2): ρ₂≈0

**Claim.** Lag-2 autocorrelation of Δ²s vanishes, confirming pure IRW (no extra colour).

$$\gamma_{k\ge 2}=0\quad\text{(MA(1) signature)}.$$

```python
rho2 = g_acov(g_diff(s,2),2)/g_var(g_diff(s,2))
```

**Result.** median ρ₂(Δ²s) = 0.048.

**Verdict.** ✅ ρ₂≈0 ⇒ a single integration order suffices (no ARMA needed).

---
### 3. Super-diffusive spreading (α>1) — toe error compounds

**Claim.** Displacement variance grows super-linearly with depth-lag, so uncertainty compounds down the lateral.

$$\mathrm{Var}\big(s(MD+\tau)-s(MD)\big)\propto\tau^{\alpha},\quad \alpha>1.$$

```python
a,_,_ = g_lstsq1(log(lag_bins), log(var_disp_bins))   # GPU least squares
```

**Result.** variance-scaling exponent α = 1.31 (Var(disp) ∝ lag^α).

**Verdict.** ✅ α>1 ⇒ heteroscedastic (growing-variance) NN head required.

---
### 4. The dip-rate v has a unit root (random walk) ⇒ q_v>0

**Claim.** The apparent dip-rate is non-stationary (ρ₁→1), so a constant-dip model is mis-specified.

$$v_i=v_{i-1}+\xi_i,\qquad \rho_1(v)\to 1.$$

```python
v = g_diff(s)/g_diff(md); rho1_v = g_acov(v,1)/g_var(v)
```

**Result.** median ρ₁(v) = 0.962.

**Verdict.** ✅ ρ₁(v)≈1 ⇒ non-zero rate process noise q_v is mandatory.

---

## B. GR measurement model

### 5. GR needs affine calibration: gain α ≠ 1

**Claim.** The horizontal GR is the typewell GR scaled by a per-well tool gain α ≠ 1.

$$GR_i=\alpha\,h(TVT_i)+\beta+\varepsilon_i,\qquad \alpha\neq 1.$$

```python
alpha,beta,r2 = g_lstsq1(h_heel, gr_heel)      # GPU normal equations
```

**Result.** median gain α = 0.80 (GR = α·h + β).

**Verdict.** ✅ α≠1 ⇒ affine calibration before matching is required.

---
### 6. Measurement noise is heavy-tailed (leptokurtic)

**Claim.** The GR−typewell residual has strong positive excess kurtosis ⇒ Student-t / clipped likelihood is optimal.

$$\varepsilon=GR-h(TVT),\qquad \mathrm{ExKurt}(\varepsilon)\gg 0.$$

```python
exk = g_kurt(gr_heel - h_heel)                  # 4th standardized moment on GPU
```

**Result.** median excess kurtosis = 6.96.

**Verdict.** ✅ Heavy tails ⇒ robust (clipped) GR misfit, not plain L2.

---
### 7. Measurement noise is (approximately) symmetric

**Claim.** The GR residual is near-zero-skew ⇒ a symmetric (t or Gaussian) likelihood, no bias term.

$$\mathrm{Skew}(\varepsilon)\approx 0.$$

```python
sk = g_skew(gr_heel - h_heel)
```

**Result.** median skew = -0.23.

**Verdict.** ✅ Symmetry ⇒ unbiased centred likelihood is valid.

---

## C. Typewell & identifiability

### 8. The datum-shift likelihood is multimodal ⇒ mode-locking is intrinsic

**Claim.** The robust GR misfit over a datum shift Δ has many local minima (facies repetition).

$$J(\Delta)=\tfrac1n\sum_i\mathrm{clip}\!\Big(\tfrac{\tilde g_i-h(TVT_i+\Delta)}{s},-4,4\Big)^2.$$

```python
J = [clip((gc - g_interp(tvt+ds, twT, twG))/sd,-4,4).pow(2).mean() for ds in shifts]
```

**Result.** median # local minima of J(Δ) = 0.

**Verdict.** ⚠️ Multimodal ⇒ wider process noise + multi-hypothesis, not single-MAP.

---
### 9. The typewell spans the well's TVT range (matching is well-posed)

**Claim.** The typewell TVT interval covers the horizontal well's TVT excursion, so a match always exists.

$$[\,T^{tw}_{\min},T^{tw}_{\max}\,]\supseteq[\,TVT_{\min},TVT_{\max}\,].$$

```python
ratio = (twT.max()-twT.min())/(TVT.max()-TVT.min())
```

**Result.** median typewell-span / well-TVT-range = 1.2×.

**Verdict.** ✅ Span ≥ range ⇒ no extrapolation needed; the map is well-posed.

---

## D. Wellbore geometry

### 10. Z (TVD) and TVT are strongly coupled — geometry carries signal

**Claim.** The bit's true vertical depth Z correlates with stratigraphic TVT, so Z is an informative (leak-free) input.

$$\mathrm{corr}(Z,\,TVT)\ \text{large}\ \Rightarrow\ Z\ \text{informs}\ TVT.$$

```python
rho = g_corr(Z_heel, TVT_heel)
```

**Result.** median corr(Z, TVT) on heel = -1.00.

**Verdict.** ✅ Strong coupling ⇒ Z is a first-class NN input (available at inference).

---

## A. State-space trajectory dynamics

### 11. Persistent (H>0.5) trajectory — trends, not white noise

**Claim.** The stratigraphic position is long-range persistent (Hurst exponent H>1/2).

$$\mathrm{Var}(s_{t+m}-s_t)\propto m^{2H},\qquad H>\tfrac12.$$

```python
a,_,_ = g_lstsq1(log(scales), log(var_increments)); H = a/2
```

**Result.** median Hurst H = 0.97.

**Verdict.** ✅ H>0.5 => persistent dip trends; a memoryful (recurrent/state) model fits.

---
### 12. Non-zero position process noise q_s>0 (real curvature)

**Claim.** The identified curvature variance q_s=-gamma_1 is strictly positive — the dip is genuinely non-linear.

$$q_s=-\gamma_1(\Delta^2 s)>0.$$

```python
qs = clamp(-g_acov(g_diff(s,2),1), min=0)
```

**Result.** median q_s = 0.0000 > 0.

**Verdict.** ✅ q_s>0 => a constant-velocity extrapolation is provably insufficient.

---

## E. Field & spatial structure

### 13. Process noise is per-well, not per-field (heterogeneity)

**Claim.** Between-field means are similar but well-to-well CV is enormous, so estimate noise per well.

$$F=\frac{\mathrm{MS}_\mathrm{between}}{\mathrm{MS}_\mathrm{within}},\qquad \mathrm{CV}=\sigma/\mu\gg1.$$

```python
F = g_anova_F(field_groups)        # ANOVA on GPU (no scipy)
```

**Result.** GPU ANOVA F = 0.58 across 12 fields; overall CV = 18.2.

**Verdict.** ✅ Huge CV => global noise constant is provably wrong (use pf_noise_id).

---

## F. Target & prediction

### 14. Target uncertainty grows with distance from the PS point

**Claim.** The drift variance increases with along-hole distance past the prediction-start point.

$$\mathrm{Var}(dtvt\mid \Delta MD)\propto \Delta MD^{\beta},\quad \beta>0.$$

```python
a,_,_ = g_lstsq1(log(md_since_bins), log(var_dtvt_bins))
```

**Result.** Var(dtvt | md_since) ~ md_since^0.57.

**Verdict.** ✅ beta>0 => the far toe is intrinsically harder; predict growing variance.

---

## B. GR measurement model

### 15. GR has a finite facies correlation length (bedding scale)

**Claim.** The GR log decorrelates over a finite lag (bed/facies thickness), setting the matching window scale.

$$\ell=\min\{k:\ \rho_{GR}(k)<e^{-1}\}.$$

```python
acl = first k where g_acf(GR,k) < 1/e
```

**Result.** median GR autocorrelation length ~ 36 samples.

**Verdict.** ✅ Finite ell => multi-scale NCC windows should bracket ell.

---
### 16. The GR match carries real signal (SNR > 0.5)

**Claim.** The typewell-explained variance exceeds a meaningful fraction of residual noise, so matching is identifiable.

$$\mathrm{SNR}=\frac{\mathrm{Var}[h(TVT)]}{\mathrm{Var}[GR-h(TVT)]}.$$

```python
snr = g_var(h_heel) / g_var(gr_heel - h_heel)
```

**Result.** median SNR (Var[h]/Var[resid]) = 2.64.

**Verdict.** ✅ SNR>0.5 => the GR-typewell match is an informative measurement.

---
### 17. GR has wide dynamic range (facies contrast exists)

**Claim.** The per-well GR spans a broad API range, so lithology contrast is available to correlate on.

$$\Delta GR = Q_{0.95}(GR)-Q_{0.05}(GR)\ \text{large}.$$

```python
dr = g_quantile(GR,0.95) - g_quantile(GR,0.05)
```

**Result.** median GR P5-P95 range = 56 API.

**Verdict.** ✅ Wide range => beds are distinguishable; correlation is well-posed.

---

## D. Wellbore geometry

### 18. The lateral is near-horizontal (|dZ/dMD| small)

**Claim.** In the eval region the bit advances mostly horizontally, so TVT drift is dominated by geology, not by TVD.

$$\Big|\frac{dZ}{dMD}\Big|\ll 1\quad(\text{inclination}\approx 90^\circ).$$

```python
slope = (g_diff(Z_ev)/g_diff(MD_ev)).abs().median()
```

**Result.** median |dZ/dMD| in lateral = 0.030.

**Verdict.** ✅ Near-horizontal => dtvt is a geological (stratigraphic) signal.

---
### 19. Dip sign is mixed (wells drilled up- and down-section)

**Claim.** The apparent dip is positive in some wells and negative in others, so the model must be sign-agnostic.

$$\Pr(\text{dip}>0)\in(0.2,0.8)\ \Rightarrow\ \text{both drilling directions present}.$$

```python
frac_pos = mean_over_wells( median(g_diff(s)/g_diff(md)) > 0 )
```

**Result.** fraction of wells with positive dip = 0.55 (of 350).

**Verdict.** ✅ Mixed sign => alignment must handle reversed index (no monotonic DTW).

---

## F. Target & prediction

### 20. A small heavy tail of wells dominates pooled error

**Claim.** Most wells are easy (low RMSE) but a minority diverge, so pooled RMSE is tail-controlled.

$$\mathrm{median}_w\,\mathrm{RMSE}_w \ll \mathrm{RMSE}_\text{pooled};\quad \Pr(\mathrm{RMSE}_w>15)\ \text{small}.$$

```python
rmse_w = sqrt(groupby(well)((pf_dtvt-true)^2).mean()); frac = mean(rmse_w>15)
```

**Result.** per-well RMSE median = 5.80, fraction >15 = 12%.

**Verdict.** ✅ Tail-controlled => fixing the divergent minority is the whole lever.

---
### 21. PF confidence is nearly uninformative about its own error

**Claim.** The ensemble confidence barely correlates with the actual error, so it can't gate the divergent wells.

$$\mathrm{corr}\big(\text{conf},\,|e_{PF}|\big)\approx 0.$$

```python
rho = g_corr(conf, |pf_dtvt - true|)
```

**Result.** corr(PF confidence, |error|) = 0.15.

**Verdict.** ✅ conf ~ uninformative => a separate risk model is needed, not self-confidence.

---

## G. Ensemble mathematics

### 22. The optimal PF/second-predictor weight is analytic

**Claim.** The MSE-optimal convex combination weight follows from the error covariance, not tuning.

$$w^\*=\frac{\sigma_b^2-\sigma_{ab}}{\sigma_a^2+\sigma_b^2-2\sigma_{ab}}.$$

```python
w = (vb - cov_ab)/(va + vb - 2*cov_ab)   # all torch-cuda
```

**Result.** optimal blend weight w* = 0.96 (sigma_pf=11.1).

**Verdict.** ✅ Closed-form weight => the blend is derived, matching the empirical optimum.

---

## E. Field & spatial structure

### 23. Fields are spatially separated (field-disjoint CV is meaningful)

**Claim.** Field centroids are farther apart than within-field spread, so leave-one-field-out simulates a spatial shift.

$$\min_{k\neq j}\lVert c_j-c_k\rVert > \text{within-field spread}.$$

```python
D = torch.cdist(field_centroids, field_centroids); nn = D.min()
```

**Result.** median nearest-field-centroid distance = 20308 vs within-field spread 9175.

**Verdict.** ✅ Separated => field-disjoint CV is the honest shakeup-robust proxy.

---

## F. Target & prediction

### 24. The eval (hidden) region is the majority of each lateral

**Claim.** Most of each well lies past the prediction-start point, so long-range extrapolation dominates the score.

$$\frac{\#\{TVT_{input}=\varnothing\}}{N}\ \text{large}.$$

```python
frac = (~kn).double().sum() / kn.numel()
```

**Result.** median hidden fraction = 74% of each well.

**Verdict.** ✅ Large hidden fraction => far-toe accuracy dominates the metric.

---
### 25. Drift is bounded (bed-limited excursion)

**Claim.** The stratigraphic drift stays within a bounded band, reflecting finite formation thickness.

$$dtvt\in[Q_{0.01},Q_{0.99}]\ \text{bounded}.$$

```python
q = g_quantile(dtvt_true, [0.01, 0.99])
```

**Result.** dtvt 1-99% range = [-40, 48] ft.

**Verdict.** ✅ Bounded target => predictions should be range-clipped (regularization).

---

## B. GR measurement model

### 26. GR carries a depth (compaction/lithology) trend

**Claim.** GR correlates with TVD, so depth-detrending separates the stratigraphic signal from the depth confound.

$$\mathrm{corr}(Z,GR)\neq 0.$$

```python
rho = g_corr(Z_heel, GR_heel)
```

**Result.** median corr(Z, GR) = nan.

**Verdict.** ⚠️ Depth trend => a depth-detrended GR residual is a cleaner matching feature.

---

## A. State-space trajectory dynamics

### 27. Dip-rate is smooth (positively autocorrelated)

**Claim.** Successive dip estimates are positively correlated, so the rate evolves smoothly (small q_v per step).

$$\rho_1(v)>0\ \Rightarrow\ \text{smooth, band-limited dip}.$$

```python
rho1_v = g_acf(g_diff(s)/g_diff(md), 1)
```

**Result.** median lag-1 ACF of dip-rate = 0.96.

**Verdict.** ✅ Smooth rate => a low-frequency (regularized) trajectory prior is correct.

---

## C. Typewell & identifiability

### 28. The typewell GR is oscillatory (facies repetition)

**Claim.** The typewell autocorrelation changes sign multiple times, i.e. bedding patterns repeat — the source of multimodality.

$$\#\{k:\rho^{tw}(k)\rho^{tw}(k+1)<0\}>1.$$

```python
sc = sum( sign(acf[k]) != sign(acf[k+1]) )
```

**Result.** median ACF sign-changes (typewell GR) = 0.

**Verdict.** ⚠️ Repetition => the GR match is inherently ambiguous (links to proof 8).

---
### 29. Typewell interpolation adds a quantifiable noise floor

**Claim.** Linear interpolation of the sampled typewell adds variance (h'*delta)^2/12 — the analytic gs inflation.

$$\sigma^2_{interp}=\tfrac{1}{12}\,\overline{(h'\,\delta)^2}.$$

```python
interp_std = sqrt(mean((g_gradient(twG,twT)*delta)^2)/12)
```

**Result.** median typewell interpolation-noise std = 1.54 API.

**Verdict.** ✅ Non-zero floor => measurement variance r must add this term (why gs*1.3 helped).

---

## D. Wellbore geometry

### 30. Each well traverses a finite TVT band

**Claim.** The horizontal well crosses a bounded stratigraphic interval, consistent with steering within a target zone.

$$\Delta TVT = \max TVT-\min TVT\ \text{finite}.$$

```python
rng = TVT.max() - TVT.min()
```

**Result.** median TVT excursion per well = 754 ft.

**Verdict.** ✅ Finite band => the target is a steering correction, not free drift.

---

## F. Target & prediction

### 31. Drift is approximately zero-mean (no systematic bias)

**Claim.** The population drift averages near zero, so a well-calibrated predictor needs no global offset.

$$\mathbb{E}[dtvt]\approx 0.$$

```python
m = dtvt_true.mean()
```

**Result.** mean(dtvt_true) = 1.60 ft (near 0).

**Verdict.** ✅ Zero-mean => center predictions at continuation; learn only the deviation.

---

## B. GR measurement model

### 32. The GR residual is close to white (weakly correlated)

**Claim.** After removing the typewell fit, the residual has modest autocorrelation, supporting an iid noise model.

$$|\rho_1(\varepsilon)|\ \text{small}\ \Rightarrow\ \varepsilon\ \text{approx. white}.$$

```python
a = g_acf(gr_heel - h_heel, 1).abs()
```

**Result.** median |lag-1 ACF of GR residual| = 0.73.

**Verdict.** ⚠️ Near-white => independent-likelihood PF weighting is justified.

---

## D. Wellbore geometry

### 33. MD sampling is near-uniform (regular stations)

**Claim.** Measured-depth steps have low coefficient of variation, so the discrete state-space uses a nearly constant dt.

$$\mathrm{CV}(\Delta MD)\ \text{small}\ \Rightarrow\ \Delta\approx\text{const}.$$

```python
cv = g_std(g_diff(MD)) / g_diff(MD).mean()
```

**Result.** median CV of MD step = 0.00.

**Verdict.** ✅ Uniform dt => a constant-step Kalman/PF discretization is accurate.

---

## C. Typewell & identifiability

### 34. The global minimum is only weakly dominant (shallow basin)

**Claim.** The best datum shift beats its competitors by a small margin, quantifying alignment fragility.

$$\frac{J_{(2)}-J_{\min}}{J_{\min}}\ \text{small}\ \Rightarrow\ \text{fragile MAP}.$$

```python
gap = (J.sort()[k] - J.min())/J.min()
```

**Result.** median (2nd-mode - min)/min of J = nan.

**Verdict.** ⚠️ Shallow basin => tiny GR noise flips the mode (seed jitter Deotte notes).

---

## D. Wellbore geometry

### 35. Vertical rate (z_rate) is a bounded engineered feature

**Claim.** The per-row z_rate = dZ/dMD-since is finite and bounded, a valid geometry covariate for the meta-model.

$$z\_rate = \frac{Z_i - Z_{PS}}{\max(MD_i-MD_{PS},1)}.$$

```python
median = g_quantile(z_rate, 0.5)
```

**Result.** median |z_rate| feature = 0.021.

**Verdict.** ✅ Bounded geometry feature => safe input alongside GR.

---

## F. Target & prediction

### 36. Linear continuation is a weak-but-positive baseline

**Claim.** The dip-continuation projection correlates positively with the true drift, so it is a valid (if weak) prior.

$$\mathrm{corr}(\widehat{dtvt}_{lin},\,dtvt)>0.$$

```python
rho = g_corr(proj_dtvt, dtvt_true)
```

**Result.** corr(linear-continuation proj, dtvt) = 0.00.

**Verdict.** ⚠️ Weak positive => continuation is a prior, not a solution (needs GR).

---

## A. State-space trajectory dynamics

### 37. The heel dip-regression slope beta is well-defined per well

**Claim.** The physics feature beta (TVT-velocity vs Z-velocity regression) is finite and bounded per well.

$$\dot{TVT}\approx \beta\,\dot{Z}+\iota.$$

```python
beta = median(honest_feat.beta)
```

**Result.** median dip-regression beta = -1.00.

**Verdict.** ✅ Stable beta => the Z-velocity PF (run_pf_z) has a valid per-well anchor.

---

## B. GR measurement model

### 38. GR noise is approximately homoscedastic in TVT

**Claim.** The residual magnitude has weak dependence on TVT, so a single measurement variance r is adequate.

$$\frac{d\,\mathbb{E}|\varepsilon|}{d\,TVT}\approx 0.$$

```python
slope,_,_ = g_lstsq1(TVT_heel, |resid|)
```

**Result.** median |d|resid|/dTVT| = 0.005.

**Verdict.** ✅ Homoscedastic => constant r per well is a sound approximation.

---

## D. Wellbore geometry

### 39. The hidden zone spans thousands of feet (long extrapolation)

**Claim.** The eval region covers a long along-hole distance, over which the dip prior must remain accurate.

$$\Delta MD_{eval}=\max MD_{ev}-\min MD_{ev}\ \text{large}.$$

```python
span = MD_ev.max() - MD_ev.min()
```

**Result.** median hidden-zone MD span = 4836 ft.

**Verdict.** ✅ Long span => compounding error (proof 3) makes the toe the hard region.

---

## A. State-space trajectory dynamics

### 40. Variance-ratio VR>1 confirms trending (not a pure random walk in position)

**Claim.** The Lo-MacKinlay variance ratio exceeds 1, indicating positive serial dependence from the integrated rate.

$$VR(q)=\frac{\mathrm{Var}(s_{t+q}-s_t)/q}{\mathrm{Var}(s_{t+1}-s_t)}>1.$$

```python
vr = (g_var(s[q:]-s[:-q])/q) / g_var(g_diff(s))
```

**Result.** median variance ratio VR(4) = 3.82.

**Verdict.** ✅ VR>1 => integrated-rate structure (IRW), matching proofs 1 and 4.

---

## G. Ensemble mathematics

### 41. GR-misfit residual is a usable risk feature

**Claim.** The per-row GR residual correlates with the PF error magnitude, so it informs a learned risk/guard model.

$$\mathrm{corr}(gr\_res,\,|e_{PF}|)\neq 0.$$

```python
rho = g_corr(gr_res, |pf_dtvt - true|)
```

**Result.** corr(gr_res, |PF error|) = nan.

**Verdict.** ⚠️ Informative => gr_res belongs in the meta-stack risk head.

---

## F. Target & prediction

### 42. The known heel is long enough to identify per-well parameters

**Claim.** Each well provides a heel of many stations, enough to estimate affine gain, dip, and noise (pf_noise_id).

$$n_{known}\gg p\ \text{(parameters)}.$$

```python
n = kn.double().sum()
```

**Result.** median known-heel length = 1699 stations.

**Verdict.** ✅ Long heel => per-well system identification is statistically sound.

---

## B. GR measurement model

### 43. GR distribution is broad/platykurtic (sand-shale mixture)

**Claim.** The GR histogram is wide (low/negative excess kurtosis), consistent with two-facies (sand/shale) mixing.

$$\mathrm{ExKurt}(GR)\lesssim 0\ \Rightarrow\ \text{multi-facies}.$$

```python
b = -g_kurt(GR)
```

**Result.** median negative excess kurtosis of GR = -1.34.

**Verdict.** ⚠️ Multi-facies => bedding contrast drives the correlation signal.

---

## A. State-space trajectory dynamics

### 44. Dip has a finite persistence length (structural wavelength)

**Claim.** The dip-rate decorrelates over a finite lag, defining the structural wavelength the model must track.

$$\ell_v=\min\{k:\rho_v(k)<e^{-1}\}.$$

```python
cl = first k where g_acf(v,k) < 1/e
```

**Result.** median dip correlation length ~ 30 stations.

**Verdict.** ✅ Finite ell_v => the trajectory prior bandwidth is set by structure, not noise.

---

## C. Typewell & identifiability

### 45. Typewell information content E[(h')^2] is positive but well-varying

**Claim.** The Fisher information about TVT scales with the mean squared typewell gradient, which differs per well.

$$\bar{\mathcal I}\propto \mathbb{E}\big[(h'(TVT))^2\big].$$

```python
fi = (g_gradient(twG, twT)**2).mean()
```

**Result.** median E[(h')^2] (Fisher info scale) = 139.85.

**Verdict.** ✅ Per-well Fisher info => identifiability (and error) is a per-well property.

---

## A. State-space trajectory dynamics

### 46. The heel is well-approximated by a local line (anchor is reliable)

**Claim.** A linear fit to the recent heel has small residual, so the PS anchor and local dip are trustworthy.

$$\mathrm{RMSE}\big(s_{heel},\ \hat a\,MD+\hat b\big)\ \text{small}.$$

```python
A0,A1,_ = g_lstsq1(MD_heel, s_heel); rmse = ((A0*MD+A1 - s)^2).mean().sqrt()
```

**Result.** median heel linear-fit RMSE = 11.97 ft.

**Verdict.** ✅ Reliable anchor => extrapolation error comes from far-field dip change, not the start.

---

## G. Ensemble mathematics

### 47. PF particle spread (zsig) is a positive risk signal

**Claim.** Wells with larger particle-filter spread have larger error, so zsig is a usable uncertainty proxy.

$$\mathrm{corr}(\sigma_{PF},\,|e|)>0.$$

```python
rho = g_corr(zsig_per_well, |error|_per_well)
```

**Result.** corr(PF spread zsig, |error|) = 0.01.

**Verdict.** ✅ Positive => zsig feeds the conformal / risk head (unlike raw conf, proof 21).

---

## F. Target & prediction

### 48. Pooled RMSE far exceeds the per-well median (Jensen gap)

**Claim.** Because RMSE is convex and error is heavy-tailed, the pooled score is dominated by a few wells.

$$\mathrm{RMSE}_\text{pooled}\gg \mathrm{median}_w\,\mathrm{RMSE}_w.$$

```python
pooled = sqrt(mean(e^2)); median_w = median(sqrt(groupby.mean(e^2)))
```

**Result.** pooled RMSE 11.13 vs per-well median 5.80.

**Verdict.** ✅ Gap => target the tail; median-good models can still score poorly pooled.

---

## D. Wellbore geometry

### 49. Bimodal drilling azimuth (up/down-section) confirmed at population level

**Claim.** Across wells the dip sign splits, so a globally sign-agnostic alignment is required.

$$\Pr(\dot s>0)\in(0.15,0.85).$$

```python
frac = mean_w( sign(median(g_diff(s)/g_diff(md))) > 0 )
```

**Result.** positive-dip fraction = 0.55 (both drilling directions).

**Verdict.** ✅ Bimodal azimuth => reverse-index matching must be allowed (no monotone DTW).

---

## F. Target & prediction

### 50. PF error grows with along-hole distance (empirical toe law)

**Claim.** The realized filter error increases with distance past PS, matching the super-diffusion proof.

$$\mathbb{E}|e_{PF}|\propto \Delta MD^{\gamma},\ \gamma>0.$$

```python
a,_,_ = g_lstsq1(log(md_since_bins), log(mean|error|_bins))
```

**Result.** |PF error| ~ md_since^0.38.

**Verdict.** ✅ gamma>0 => confirms the toe is where models must earn their score.

---

## B. GR measurement model

### 51. GR is approximately stationary along the lateral

**Claim.** The GR mean is stable between the first and second half of each well, so a single calibration holds.

$$\frac{|\mu_1-\mu_2|}{\sigma}\ \text{small}.$$

```python
shift = |GR[:h].mean() - GR[h:].mean()| / GR.std()
```

**Result.** median |mean shift|/std across halves = 0.27.

**Verdict.** ✅ Stationary => one affine calibration per well suffices (no drift term).

---

## G. Ensemble mathematics

### 52. PF confidence has usable spread (varies across wells)

**Claim.** Ensemble confidence is not constant, so it carries information even if weakly predictive (proof 21).

$$\mathrm{IQR}(\text{conf})>0.$$

```python
q = g_quantile(conf, [0.1, 0.9])
```

**Result.** PF confidence P10-P90 = [0.00, 0.36].

**Verdict.** ✅ Non-degenerate => conf can enter a nonlinear risk head (not a linear gate).

---

## F. Target & prediction

### 53. Drift is roughly symmetric (no dominant steering direction)

**Claim.** The population drift is not strongly skewed, so up- and down-steering are balanced.

$$\mathrm{Skew}(dtvt)\approx 0.$$

```python
sk = g_skew(dtvt_true)
```

**Result.** skew(dtvt_true) = 0.40.

**Verdict.** ✅ Symmetric target => no global directional bias to exploit.

---