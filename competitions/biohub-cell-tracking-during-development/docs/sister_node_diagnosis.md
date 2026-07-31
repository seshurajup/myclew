# Sister-node diagnosis — where division daughters are lost, and does it cost edge_jaccard?

**Question.** At a GT division a parent splits into 2 daughters ("sisters"). Our runs measure
`div_J ≈ 0`. WHERE are the sisters lost — detection, linking, or metric geometry — and does missing
them cost the BIG term (`edge_jaccard`, weight 0.9) or only the capped term (`division_jaccard`, weight 0.1)?

**Data / method (measured, not modelled).**
- GT `.geff` = founder-lineage tracks (sparse). Scored with the byte-identical patched official metric
  (`research/official_repo/src/tracking_cellmot/{metrics,division_metrics}.py`), scale z=1.625/y=x=0.40625, 7µm gate.
- GT-side division + sister-separation scan over **all 199** train datasets (both embryos).
- Pred-side attribution on the ready **pilkwang unet_transformer split_0** predictions — all **71 held-out 44b6**
  datasets (covers 100% of 44b6 GT). 6bba has no preds → GT-side only.
- Scripts: `scratchpad/gt_scan.py`, `scratchpad/pred_scan.py`. Nothing retrained.

---

## 1. Division rarity

| embryo | datasets | GT nodes | GT edges | divisions | div / dataset | fork edges (2·D) | **fork edges / all edges** |
|--------|---------:|---------:|---------:|----------:|--------------:|-----------------:|---------------------------:|
| 44b6   | 71  | 20,197  | 19,826  | **26**  | 0.37 | 52  | **0.26 %** |
| 6bba   | 128 | 113,121 | 109,057 | **125** | 0.98 | 250 | **0.23 %** |

Divisions are **extremely rare**: ≈0.4–1 per dataset, and the birth edges they create are only
**~0.25 % of all GT edges** in both embryos. This is the structural ceiling on any edge_jaccard gain
from divisions before we even look at predictions.

## 2. Attribution — the 26 GT divisions in 44b6 (pilkwang preds)

Pilkwang on 44b6: **edge_J = 0.874**, **div_J = 0.077** (div TP=2, FN=24, FP=0).

| stage | count | of 26 |
|-------|------:|------:|
| **Detected** — both sisters have a pred node within 7µm | 21 | 81 % |
| — only one sister detected | 5 | 19 % |
| — neither detected | 0 | 0 % |
| Parent detected | 26 | 100 % |
| **Credited** (official division TP) | 2 | 8 % |

Attribution of the **24 lost divisions**:

| failure mode | count | share of losses |
|--------------|------:|----------------:|
| **Detection-miss** (a sister never detected) | 5 | 21 % |
| **Linking-miss** (both sisters detected, but graph never forks) | **19** | **79 %** |
| **Metric-geometry** (fork formed but rejected by scorer) | 0 | 0 % |

At the fork-edge level (52 birth edges): 25 TP, 5 FN-from-detection, 22 FN-from-linking.

**The dominant failure is LINKING, not detection.** In 19/24 lost divisions both daughters are
detected as distinct nodes, but the tracker links the parent to only ONE daughter (treats the
division as a continuation) and drops the parent→other-sister edge. Zero divisions are lost to
metric geometry — the patched scorer is not the bottleneck.

## 3. THE DECISIVE NUMBER — edge_jaccard cost of missed sisters

Only **27 of the 1,495** edge false-negatives (1.8 %) are division fork edges. The other 98 % are
ordinary lineage-continuity edges with nothing to do with divisions.

| scenario | edge_J | Δ edge_J |
|----------|-------:|---------:|
| pilkwang as-is | 0.8740 | — |
| **all 52 fork edges recovered (perfect division detect+link)** | 0.8753 | **+0.0013** |
| all 1,495 edge FN recovered (perfect everything) | 0.9453 | +0.0713 |

**Fixing sisters perfectly moves edge_jaccard by +0.0013 — an order of magnitude below the 0.005
materiality bar, and that is the optimistic bound** (actually forming forks adds pred edges, which can
raise FP, so the real Δ is ≤ 0.0013). The edge_J headroom that exists (→0.945) lives almost entirely
in **ordinary track continuity/recall, NOT in divisions.**

Where sisters DO pay off is the capped term: crediting all 26 divisions would take div_J 0.077→1.0,
i.e. +0.092 on the combined score via the 0.1 weight — the same locked division oracle we already found.

**Verdict: sister-node recovery is a CAPPED lever (helps only the 0.1 div_J term), not an edge_J lever.**

## 4. Resolution or linking? — sister separation vs the gate/kernel

| embryo | median sep | mean | p25–p75 | min | ≤7µm | ≤3.25µm |
|--------|-----------:|-----:|--------:|----:|-----:|--------:|
| 44b6   | **8.98 µm** | 8.84 | 7.19–10.46 | 2.87 | 23 % | 3.8 % |
| 6bba   | **11.47 µm** | 10.88 | 8.33–13.01 | 3.72 | 13 % | 0 % |

Sisters at birth are typically **farther apart than the 7µm match gate** (median 9–11µm). In pixels the
median is ~22 xy-pixels (0.406µm/px) / ~5–7 z-planes (1.625µm/px) — trivially resolvable by any 3–5px
pooling/NMS kernel. Only 0–4 % of sisters sit within ~8 xy-pixels. This is confirmed empirically:
81 % of divisions have **both** sisters detected. **Sisters are not a detection-resolution problem —
they are well separated and mostly detected. The failure is that the linker won't fork.**

## 5. Recommendation

**Do NOT invest in sister detection / detector resolution as an edge_J lever.** The diagnosis is decisive
on all three axes:

1. **Rarity** — division fork edges are ~0.25 % of all edges.
2. **Edge cost** — recovering every sister/fork edge is worth **+0.0013 edge_J** (< 0.005, negligible).
3. **Mechanism** — sisters are already detected (81 %); the loss is 79 % **linking**, 0 % geometry, only
   21 % detection — and even that detection slice is capped by rarity.

This **CONFIRMS the "division lever exhausted" finding from a new (detection/linking) angle** and closes
the door the task asked about: **there is NO hidden edge_J angle via missed sisters.** The only reward
sisters can return is through the 0.1-weighted div_J (the known locked ≈+0.07 oracle), and unlocking it
is a **linker/branching problem** (the tracker must emit a parent→2-daughter fork when both daughters are
present), not a detector-resolution or division-oversampling problem. Given div-recovery post-proc / ILP /
temporal-retrain already failed net-negative, and the honest ceiling here is the same capped +0.07,
**the productive lever remains ordinary node-recall + linking continuity (the 98 % of edge FN, →0.945 edge_J),
which drives the 0.9-weighted term.** Sister-specific work is not worth prioritising.
