# Stage 1 Baseline: Pilkwang FIXED Detections (Golden-12 Pilot)

## Summary

**Golden-12 Pilot Evaluation** (rank-faithful proxy per memory; full LOEO → Stage 2)

- **Golden-12 (pilot)**: **0.8527**
- **Context**: 12-dataset rank-faithful proxy (validated earlier as ~+0.02 under LB)
- **Note**: Full LOEO fold evaluation (199 datasets) deferred to Stage 2 full-model lock-in

---

## Metric Anatomy: Where the Score Is

### Golden-12 Baseline

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| R_node (detection recall) | 0.9896 | Node matching accuracy (frozen at pilkwang) |
| R_edge (edge recall) | 0.9183 | Edge recall (both endpoints matched + linked) |
| Q_link (link quality \| matched) | 0.9377 | Linking efficiency given detected nodes |
| edge_P (edge precision) | 0.9408 | Edge precision (FP-edge flooding) |
| count_ratio (t_pred / estN) | 1.2370 | Over/under-prediction ratio |
| count_penalty | 0.9811 | Realized count penalty factor |
| div_J (division jaccard) | 0.0000 | Division event accuracy (rare, 0.1 weight) |
| **official_score** | **0.8527** | **adj_edge_jaccard + 0.1 × division_jaccard** |

---

## L11 Lesson: Point-Substrate Linker Constraint

**Key Finding from Thread-2:** Pilkwang's post-processing pipeline (pilk_post: node-fixing + relinking + division-filtering)
establishes the Stage-1 baseline. Our Thread-2 experiments (Trackastra pretrained linker on FIXED detections)
failed because the learned linker does NOT fit the point-substrate model (L10 closure).

**Implications for Stage 2-9:**
- **Detection (R_node):** FIXED at pilkwang frozen values. Detector work OFF the table (L8 gate).
- **Linking (R_edge/Q_link):** Learned linker path closed (point-substrate mismatch). Stage 2+ explores post-processing variants:
  - Edge-length thresholding + gap-recovery
  - Local motion smoothness + centroid refinement
  - Point-cloud-aware re-linking (if constraints permit)
- **Division (div_J):** Weak signal (0.1 weight). Focus downstream first.

**Credibility Gate:** pilkwang golden-12 ≈ 0.8708; thread-2 Trackastra drop (-0.183) closes learner linker bet.
Stage 2 post-proc variants prioritize RoI-aware edge features + missing-edge recovery.

---

## Process

- **Model**: pilkwang unet_transformer (split_0, GEFFS → final-format .geff)
- **Evaluation Set**: golden-12 (12 embryos, rank-faithful proxy)
- **Scorer**: official_scorer (adj_edge_jaccard + 0.1 × division_jaccard, Hungarian ≤7μm)
- **Anatomy**: metric_anatomy (R_node/R_edge/Q_link/count/div decomposition)
- **Split**: fleet_loeo_mini.json (embryo-disjoint, frozen; full LOEO → Stage 2)

---

*Stage 1 Pilot:* pilkwang baseline on golden-12 validates decomposition tool + unblocks Stage 2 architecture.
*Stage 2+:* full LOEO GEFFS generation + post-processing variant screening.
