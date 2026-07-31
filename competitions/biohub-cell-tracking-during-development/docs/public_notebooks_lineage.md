# Biohub Cell Tracking — Public Notebook Lineage (recipe evolution)

Competition: `biohub-cell-tracking-during-development`. Snapshot: **2026-07-18** (IST).
Built from the Kaggle CLI only (`kernels list`, `kernels pull`, `competitions leaderboard` — official API, no scraping).

**How to read scores.** The `kernels list` API returns ref/title/author/votes but **not** a per-notebook score. Score per row is resolved by, in order of preference: (a) the LB best-submission score for that author's team (`competitions leaderboard`, cross-referenced by `teamSlug`), (b) the score embedded in the title/markdown, (c) the notebook's own claimed provenance. Confidence is stated per row. Where a team's LB best is a *different* (exploit) submission than the honest notebook, that is called out explicitly.

**The two clusters (LB, full 1,359 teams).**
- **86 teams at *exactly* 0.950** — the **metric-hack cluster** (fabricated off-volume divisions, see EXPLOIT row). Unmistakable identical-score signature.
- **110 teams at *exactly* 0.903** — the **honest public ceiling** (pilkwang/praxel 0.902–0.903 lineage and its forks).
- A handful above 0.950 (TWEAK 0.979, Matt Goldfield 0.975, codebeforework/pipi14ramu 0.971, 0.955×2, 0.954) = enhanced/tuned exploits. **Kevin 0.968** is the long-standing lone honest-ish outlier (+0.058 over the honest cluster; no public notebook).
- **Monday re-score wipes the fabricated-division cluster** → every "0.950" (and the >0.950 exploit variants) is expected to collapse back toward the honest ~0.903 base they were built on.

---

## Ranked table (distinct recipes)

**Post-patch (re-score) estimates now filled** — see the dedicated section [Post-patch re-score estimates](#post-patch-re-score-estimates) below. Rule (from the organizers' statement "submissions that did not actively exploit will not be affected" + the exploit arithmetic): **honest → unchanged; exploit → drops to its real edge base (−0.1 fabricated division).** These are principled ESTIMATES pending the official Monday numbers, not the official values.

| # | Notebook (link) | Author | Score | Source (conf.) | Δ over prior best | What changed (grounded in cells) | Honest / EXPLOIT | Post-Monday |
|---|---|---|---|---|---|---|---|---|
| 1 | [yusuketogashi/lb897-baseline](https://kaggle.com/code/yusuketogashi/lb897-baseline) | uskt (Y. Togashi) | ~0.897 | title+MD (med) | baseline anchor | Learned TemporalUNet3D detector (local-maxima peaks, τ) + node-transformer edges + ILP (`w_e=-edge_prob`, app/disapp/div costs) + 2-pass Hungarian **motion relink** (tight 6.0µm / relaxed 10.0µm, velocity λ=0.5) + 1-frame gap-close + line-fit smooth + safe divisions. This variant adds a per-dataset short-track A/B (min7 global, restore min6 on the smallest dataset). | Honest | |
| 2 | [pilkwang/biohub-cell-tracking-blend-preprocessings](https://kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings) | Pilkwang Kim | 0.903 | LB best (high) | **+~0.006** (0.897→0.903) | **D4 detection TTA**: 8 logit volumes inverse-aligned + averaged → one shared node set; **shared-point edge TTA**: transformer run once per each of 8 XY-D4 views, raw link logits averaged (w=0.125 each) *before* sigmoid+ILP. Downstream graph rules unchanged. Motion cost `C=d_motion+0.05·d_raw−1.0·p_ij`. | Honest | |
| 3 | [yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction](https://kaggle.com/code/yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction) | Yaroslav K. | 0.900 | LB best (high) | −0.003 vs pilkwang (parallel repro) | Canonical reproducible UNet+ILP pipeline + **logistic edge-policy veto**: removes a learned edge only if `p_ψ(e)<τ=0.35` (features incl. 7µm local density ρ₇); asymmetric (never adds), protects out-degree-2 (divisions), skips synthetic edges. | Honest | |
| 4 | [abhijithneilabraham/solution](https://kaggle.com/code/abhijithneilabraham/solution) | neilan | 0.900 | LB best (high) | ≈ parallel (0.900) | 300-epoch temporal-graph backbone + **DeepCenter add-only repair gate**: an independent center model scores *only newly proposed* gap nodes / safe-division daughters (τ_gap soft 0.06), can reject an added repair but never deletes backbone nodes/edges. min6. | Honest | |
| 5 | [praxel/biohub-0-902-motion-division-calibration](https://kaggle.com/code/praxel/biohub-0-902-motion-division-calibration) | Praxel | 0.902→0.903 | title 0.902 / LB best 0.903 (high) | **+0 to +0.001**, becomes the canonical anchor | The widely-forked "0.902 calibration": DET 0.970, motion bonus 1.0, min-track 6, **safe-division** parent 4.66µm / sister 8.5µm / existing-child 7.65µm, frame/global add caps 0.0076/0.00375. Key finding: **disabled the inert DeepCenter veto** (diagnostics showed it gated *zero* nodes) and removed its private dependency. | Honest | |
| 6a | [beicicc/biohub-exp058-center-gap-span-7-75-public](https://kaggle.com/code/beicicc/biohub-exp058-center-gap-span-7-75-public) | Kun Zhang | 0.903 (team best) | LB best (med) | micro-probe (~0) | On the anchor: DeepCenter gap confirmation begins at **7.75µm** span (gaps 7.75–12µm kept only if center prob ≥0.20; <7.75µm auto-kept). `MOTION_RELINK_LEARNED_BONUS=0.75`. | Honest | |
| 6b | [beicicc/biohub-exp071-linefit-weight-0-81-public](https://kaggle.com/code/beicicc/biohub-exp071-linefit-weight-0-81-public) | Kun Zhang | 0.903 (team best) | LB best (med) | micro-probe (~0) | Only line-fit smoothing weight **0.80→0.81**; topology fixed. | Honest | |
| 6c | [beicicc/biohub-exp056-division-prior09](https://kaggle.com/code/beicicc/biohub-exp056-division-prior09) | Kun Zhang | 0.903 (team best) | LB best (med) | micro-probe (~0) | Only ILP **division prior 1.0→0.9** (conservative extra splits); D4 TTA + 400-epoch graph fixed. | Honest | |
| 6d | [beicicc/biohub-exp073-gap-5-8-public](https://kaggle.com/code/beicicc/biohub-exp073-gap-5-8-public) | Kun Zhang | 0.903 | LB best (high) | **the 0.903 "gap 5.8" base** | The reference "gap-5.8µm-per-step" mask recipe that the later honest deltas (chiranjith, arnav) fork from. Config = the praxel anchor with the 5.8µm gap-step mask. | Honest | |
| 6e | [beicicc/biohub-exp084-det-0-96875-gap-5-8-public](https://kaggle.com/code/beicicc/biohub-exp084-det-0-96875-gap-5-8-public) | Kun Zhang | 0.903 | LB best (high) | detection-recall probe on 6d | **Only** `BIOHUB_DET_THRESHOLD 0.9700→0.96875` (lower peak threshold → more candidate detections / higher node recall) on the gap-5.8 base. | Honest | |
| 7 | [yusuketogashi/biohub-another-approach](https://kaggle.com/code/yusuketogashi/biohub-another-approach) | uskt | 0.903 | title "Biohub 090/106" + LB (high) | precision micro-refine on 0.903 | **Local-density adaptive gap radius** (density gain 0.040, step cap 0.125µm) that expands the gap-close radius only where local density supports it, + a **weakest-adaptive-gap micro-prune** that deletes exactly the single globally-weakest synthetic adaptive-only gap pair (Hungarian ambiguity tie-break). One extra inference-free pass. | Honest | |
| 8 | [chiranjithdharma/replace-midpoint-insertion-with-weighted-interpola](https://kaggle.com/code/chiranjithdharma/replace-midpoint-insertion-with-weighted-interpola) | Chiranjith | honest nb on 0.903 base (team LB best = 0.950 exploit) | code (high) | **the adoptable Δ** — motion-aware gap fill | Replaces the **pure geometric midpoint** for gap-close synthetic nodes with a **weighted blend of the geometric split and a motion prediction** built from each endpoint's own local velocity (source's real predecessor / target's real successor): `GAP_MIDPOINT_MOTION_WEIGHT=0.5`, safety clamp `GAP_MIDPOINT_MOTION_MAX_DEVIATION_UM=4.0`. `refine_synthetic_midpoint()`. | Honest (author's *team* best is a separate 0.950 exploit) | |
| 9 | [arnavsalkade/biohub-exp073-gap58-edge-consensus-v1](https://kaggle.com/code/arnavsalkade/biohub-exp073-gap58-edge-consensus-v1) | Arnav Salkade | 0.903 | code + LB (high) | edge-precision Δ on 0.903 base | On the gap-5.8 base, adds a **cross-architecture edge-error consensus** (HGB/XGBoost, `apply_edge_error_consensus.py`, artifact `biohub-edge-error-consensus-v1`) that flags/removes learned edges a second model strongly disagrees with — "leakage-free", no test-aligned labels. | Honest | |
| **X** | [outwrest/metric-hack-minimal-baseline-tta-2gpu](https://kaggle.com/code/outwrest/metric-hack-minimal-baseline-tta-2gpu) | outwrest | **0.950** | LB best + code (high) | **+~0.047 FABRICATED** | `augment_dataset()`: adds an **off-volume hub node at t=−1000, (z,y,x)=(−10000,−10000,−10000)** linked to up to `MAX_COMPONENTS=1400` real track roots, then `FORKS=5` synthetic `divider→(child, continuation)` triplets at negative times. Pure **division-jaccard exploit** — no real detection/linking change. | **EXPLOIT — wiped Monday** | |
| X-forks | navazshfathi/biohub-best-score-0-950; biohack44/biohub-another-approach-v2 (Emre Cirak); kaiwalyaatulraut/biohub-competition-solution; +83 more at exactly 0.950 | various | 0.950 | LB best (high) | copy of exploit X | Near-identical forks of the fabricated-division hack (86 teams total sit at exactly 0.950). Not re-pulled individually. | **EXPLOIT — wiped Monday** | |
| dup | llccqq624/biohub-exp001…005; romanrozen/biohub-best-score; ron506/exp011-013; taopy2/bch-005/006; pawanmali/*; enddl22/* | various | 0.900–0.903 | LB/title (med) | reproductions | Near-duplicate reproductions of praxel-0.902 / abhijith / pilkwang-blend / density-motion-gated. Noted, not re-pulled. | Honest (repro) | |

---

## Post-patch re-score estimates

Estimates under the **live patched metric** (Monday re-scores officially; these are principled, not official). Method: honest submissions keep their score (organizers' statement); exploit submissions lose the fabricated +0.1 division term and fall to their real edge base (they also dropped real forks → ~0 genuine division credit).

| Notebook / cluster | Pre-patch (shown) | Post-patch (est.) | Basis |
|---|---|---|---|
| uskt lb897-baseline | 0.897 | **~0.897** | honest — unaffected |
| pilkwang blend (D4 TTA) | 0.903 | **~0.903** | honest — unaffected (real safe-div forks still count) |
| praxel 0.902 anchor + all beicicc/yaroslav/abhijith/yusuke/arnav/chiranjith | 0.900–0.903 | **~0.90–0.903** | honest — unaffected |
| **outwrest metric-hack + 86 forks @ 0.950** | 0.950 | **~0.85** | EXPLOIT — minimal edge base ~0.85 + 0.1 fake division → fake forks become FP |
| **"3rd place" 0.971 (self-confessed hack)** | 0.971 | **~0.871** | EXPLOIT — edge base 0.871 + 0.1 fake |
| top gamed 0.968–0.979 (TWEAK, Goldfield, Kevin…) | 0.968–0.979 | **~0.87–0.88** | EXPLOIT — strong-ish edge base + 0.1 fake |
| **OUR v4 (on board)** | 0.880 | **0.880** | honest — unaffected |
| **OUR v6 (ready / scoring)** | — | **~0.90 (est.)** | honest — reproduces the 0.903 recipe |

**Net after Monday:** the ~0.95+ field collapses to ~0.85–0.88 (their real edge). The honest ~0.90–0.903 notebooks stay. Our honest v6 (~0.90) lands **in the honest top band, above the re-scored exploiters.**

---

## Recipe-evolution narrative (each step's improvement over the last)

1. **Base — learned UNet+edge-transformer+ILP (~0.897, uskt lb897).** TemporalUNet3D center-peak detector → node-transformer adjacent-frame edges → tracksdata ILP (`w_e=−edge_prob`) → heavy but deterministic post-proc: 2-pass Hungarian motion-relink (6/10µm, velocity λ=0.5), 1-frame gap-close, line-fit smoothing, conservative safe-divisions, short-track filter. Everything after this point keeps the **model weights frozen** and only edits post-processing. This is the whole story of the honest cluster: *graph construction, not the detector, moves the score.*

2. **+ D4 TTA on detection AND edges → 0.903 (pilkwang blend, +~0.006).** Average 8 inverse-aligned detection-logit volumes into one shared node set; run the edge transformer on each of 8 XY-D4 views and average link logits before sigmoid. Training-free; the single biggest honest jump in the lineage.

3. **Parallel reproductions settle the ~0.90 plateau (yaroslav V4 0.900, abhijith 0.900).** Two independent "safety-rail" ideas: yaroslav's **logistic edge-policy veto** (τ=0.35, remove-only, division-protected) and abhijith's **DeepCenter add-only repair gate** (validate only *added* gap/division nodes). Both land at 0.900 — neither beats the TTA blend, and both later get switched off as inert.

4. **The canonical 0.902/0.903 anchor (praxel).** Freezes DET=0.970, motion-bonus 1.0, min-track 6, safe-division 4.66/8.5/7.65µm, and — critically — **removes the DeepCenter veto after proving it gated zero nodes.** This clean, dependency-light notebook becomes the fork base for the entire honest field (110 teams at 0.903).

5. **beicicc micro-probes (Kun Zhang, all ~0.903).** Systematic one-knob sweeps on the anchor: gap-confirm span 7.75µm (exp058), line-fit 0.81 (exp071), division prior 0.9 (exp056), the **gap-5.8µm-per-step** base (exp073), and **DET 0.970→0.96875** for a touch more recall (exp084). Individually flat, but they map the local sensitivity of every knob.

6. **Precision/coverage refinements on the 0.903 base (still honest):**
   - **yusuke "another approach"** — density-adaptive gap radius + delete-the-single-weakest-synthetic-gap micro-prune.
   - **chiranjith weighted interpolation** — the standout honest idea: gap-fill nodes are placed by **blending the geometric midpoint with a local-velocity motion prediction** instead of a naive midpoint.
   - **arnav edge-consensus** — a second-architecture (HGB/XGBoost) edge-error model vetoes disagreed-with learned edges.
   These are the *genuine* frontier of honest public work; each is a small, measurable post-proc delta on 0.903.

7. **The exploit fork (outwrest → 86 teams at 0.950).** Orthogonal to everything above: fabricate divisions with an off-volume, negative-time hub + 5 fake fork triplets to inflate the division-jaccard term by ~+0.047. Purely a metric attack; **expected to be wiped in Monday's re-score.** Many honest authors (chiranjith, kaiwalya, navazsh, Emre) *also* submitted this, so their *team* LB best shows 0.950 while their honest notebook is 0.903.

---

## Adoptable-for-us (honest levers we have NOT tried), ranked by likely gain

Our honest **v4 = 0.880** (baseline 0.877 + `gap_fill` **+0.003**). `gap_fill` already covers the public **gap-closing/linear-midpoint interpolation** family — so plain gap-close is *not* new headroom for us. The levers below are the public honest improvements we do **not** yet have.

**Top 3 to try first:**

1. **Lower detection peak threshold toward 0.96875 (beicicc exp084) — and sweep below.**
   *Why it may beat +0.003:* our own memory says **node recall is THE lever** (`adjJ ≈ node_rec²·edge_prec`); recall enters *squared*. A lower peak threshold recovers the dim 44b6 cells our detector currently drops. This is a zero-cost config sweep that hits the highest-leverage term in the metric — a plausible bigger win than any post-proc gap edit. (Caveat: watch edge-precision on 6bba; sweep per-embryo.)

2. **Weighted motion-blended gap interpolation (chiranjith) to replace our linear midpoint.**
   *Why it may beat +0.003:* our `gap_fill` inserts nodes at the geometric midpoint; the metric matches centroids at **7µm** in physical units. Placing the synthetic node on the *actual motion path* (blend geometric split with each endpoint's local velocity, weight 0.5, clamp 4µm) moves inserted nodes inside the match radius of the true cell — converting "inserted-but-unmatched" gap nodes into **matched** ones. Same node count, strictly better placement. This is a drop-in upgrade to the exact component we already gained +0.003 from.

3. **D4 detection + shared-point edge TTA (pilkwang blend).**
   *Why it may beat +0.003:* this training-free 8-view averaging is what lifted the public base **0.897→0.903 (+0.006)**. We have not applied D4 TTA to our detector logits or edge head. It directly raises node recall (averaged detection) and edge precision (averaged link logits) at once — the two terms of our metric.

**Secondary (try after the top 3):**
- **Density-adaptive gap radius + weakest-gap micro-prune (yusuke)** — expand gap radius only where local density supports it, then delete the single weakest synthetic gap; a precision guard against the FP-edge trap our notes warn about.
- **Cross-architecture edge-error consensus (arnav)** — an HGB/XGBoost veto on disagreed learned edges to lift edge precision.

**Do NOT adopt:** the outwrest/navazsh **fabricated-division metric hack** — it is the wiped-Monday cluster and is exactly the class of exploit our ledger already retracted (`det0.9`/off-volume forks).

---

## STACKED-LEVER outcomes (2026-07-19, golden-12 canonical / patched metric, per-embryo, LOEO-honest) — EXP_308–310

Every honest lever from the lineage was CV-gated on the real 0.9162 chiranjith base (NOT a weak
baseline). Kept only both-embryo Δadj>0. **The edge-consensus is the one lever that beats 0.903.**

| Running stack | combined | 44b6 | 6bba | verdict |
|---|---|---|---|---|
| base = pilkwang/praxel + EXP_156 (mtl10/gap5.5) + **chiranjith** motion-blend | 0.9162 | 0.8885 | 0.9231 | KEEP (base) |
| + our `gap_fill` post-ILP module (max_k=3) | 0.9175 | 0.8884 | 0.9248 | borderline (44b6 −0.0001; v6 ships it) |
| **+ arnav EDGE-CONSENSUS (XGBoost, LOEO, remove-only, div-protected, τ=0.05)** | **0.9188** | **0.8915** | **0.9255** | **KEEP — the differentiator** |

**Dropped (measured, fail both-embryo gate):**
- beicicc micro-tunes — linefit 0.81 (flat 0.9162), gap 5.8 (−0.0007), gap 5.6/linefit 0.79 (neg), div-prior/det0.96875 (N/A on frozen pred / recall-saturated). All flat/neg.
- our gap_fill `GAP_CLOSE_MAX_GAP` 2/3 — flat (public gap-close already covers it; the separate `gap_fill_graph` module is what adds the +0.0013).
- gap-2 recovery — −0.0015/−0.0011 BOTH neg.
- yusuke density-adaptive gap radius (`gap_fill_graph density_adaptive=True`) — 0.9176 < 0.9188 fixed. Node recall saturated → any gap-radius tweak is flat/neg.

**Edge-consensus (arnav) — built from scratch, leakage-free.** GT is sparse (~50 GT edges/dataset);
only `pred_valid` edges (both endpoints matched to a GT node) score. Oracle ceiling (remove ALL valid
FP): 44b6 **+0.073**, 6bba **+0.041** — the FP edges concentrate in the dense datasets (fp=88/126/36).
A second architecture (**XGBoost**, 17 geometry+local-competition+motion-consistency+edge_prob features)
trained on one embryo's GT-matched edges flags edges the transformer included that it is >95% confident
are false; **remove-only, division-protected** (source out-deg<2). LOEO (train 44b6→apply 6bba and
vice-versa) is **both-embryo positive** (44b6 +0.0030, 6bba +0.0024), **seed-stable**, safe over the
whole τ∈[0.05,0.08] region. It captures a small honest slice of the FP ceiling (edge precision is the
binding constraint; node recall is saturated ~0.985). Shipped as a version-portable `ec_model.json` in
the v7 kernel.

**HONEST verdict vs 0.903.** Base 0.9162 canonical ≈ LB 0.903 (EXP_250 calibration ~1:1, golden
slightly over-credits). v7 = **0.9188** golden-12-canonical → the edge-consensus is a genuine,
both-embryo, LOEO-validated gain of **+0.0026** over the base, so v7 **marginally BEATS the 0.903 honest
ceiling on honest CV**. Because golden-12 over-credits precision tuning, the real-LB gain is likely
**+0.001–0.002** (v7 ≈ LB ~0.904–0.905): a small, real improvement — NOT a decisive leap. Edge-consensus
is our real differentiator; everything else in the lineage is flat on our pipeline.

Ship: `seshurajup/biohub-v7-stack` (2×T4, offline) = base engine + edge-consensus + gap_fill.

## MEASURED outcomes (2026-07-18, golden-12 canonical / official patched metric, per-embryo) — EXP_304–307

The honest chain **reproduces** on our CV. The reproduction infra was already local: `pilk_post.py`
ports praxel cell-11 (reads all `BIOHUB_*` knobs), `score_golden12_official.py` scores with the
byte-identical official metric. The "0.877→0.90" gap = **pool5 re-detect + full graph post-proc**,
NOT a hidden artifact. (The scorer's default cached blobs are the STALE `pool_kernel_um=3.0`
over-detect, raw 0.8527; the honest base is a fresh `pool5` re-detect.)

| Stage | combined adjJ | 44b6 | 6bba | node_rec |
|---|---|---|---|---|
| Bare pilk (stale pool3 cache), raw | 0.8527 | — | — | 0.9873 |
| Fresh pool5 re-detect (det0.99), verbatim = **honest BASE** | 0.9075 | 0.8571 | 0.9203 | 0.9952 |
| + praxel post-proc (mtl6/gap6/safe-div/relink/linefit) | 0.9096 | 0.8891 | 0.9147 | 0.9877 |
| + EXP_156 post-proc (mtl10/gap5.5) = **best anchor** (≡EXP_156 0.9161) | 0.9155 | 0.8885 | 0.9222 | 0.9846 |
| **+ chiranjith weighted motion-blend gap interp = BEST HONEST CV** | **0.9162** | 0.8885 | 0.9231 | 0.9848 |

**Adopted:** (2) chiranjith weighted motion-blend gap interpolation — now in
`pilk_post.motion_blend_midpoint` (`BIOHUB_GAP_MIDPOINT_MOTION_WEIGHT=0.5`,
`_MAX_DEVIATION_UM=4.0`). +0.0007 combined, **both-embryo ≥ 0** (44b6 flat, 6bba +0.0009),
4/12 datasets move all-positive (Wilcoxon p=0.125). Small but strictly never-hurts, training-free.

**Rejected (measured, both-embryo negative):** (1) det-threshold lower — det0.99 (0.9075) > det0.98
(0.9047), recall already saturated ~0.995 → extra candidates only add FP; det0.96875 not pursued.
(3a) D4 8-view **detection** TTA — 0.9075→**0.9015** (−0.006 both embryos); it pushed 44b6 node_recall
to 0.9987 (near-perfect) yet combined FELL because edge precision dropped → **recall is saturated and
non-binding; edge PRECISION binds.** Confirms EXP_303.

**Still open (the only genuinely-untested honest lever):** (3b) pilkwang **shared-point EDGE TTA**
(average the transformer's link logits over 8 XY-D4 views before sigmoid) and (4) arnav
edge-consensus — both target **edge precision**, the binding constraint, but neither is implemented in
our local predict (edge-TTA needs a predict-script change + GPU re-predict; edge-consensus needs a
trained HGB/XGB artifact).

**Reach 0.903?** Calibration EXP_250: praxel golden-12-canonical **0.9202 ↔ LB 0.902** (≈1:1, golden
slightly over). Our best **0.9162** golden-12-canonical → **reproduces the honest public ~0.90–0.903
ceiling** and sits just under the praxel-0.902 reproduction; it does **not demonstrably exceed 0.903**.
Golden-12 is embryo-leaky (over-credits); 44b6 (the lever) = 0.8885 vs 6bba 0.9231. v6 recipe =
fresh pool5 re-detect (det0.99) + mtl10/gap5.5 post-proc + chiranjith knob ON.

---

### Provenance
- Notebook list: `kaggle kernels list --competition biohub-cell-tracking-during-development --sort-by scoreDescending --page-size 50 --csv`.
- Scores: `kaggle competitions leaderboard biohub-cell-tracking-during-development --download` (1,359 teams; best-submission-per-team only — raw attempt history is not exposed by the API), cross-referenced by `teamSlug` to notebook authors.
- Distinct recipes pulled to `docs/public_nb_lineage/<author__slug>/` and read cell-by-cell; near-duplicate forks noted but not re-pulled.
- Honest ceiling = **0.903** (110 teams). Metric-hack cluster = **0.950** (86 teams), expected wiped Monday.
