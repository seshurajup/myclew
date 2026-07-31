# Biohub Cell-Tracking During Development

Track every cell nucleus in a developing zebrafish embryo in 3-D over time (light-sheet).
Predict a tracking graph (nodes = cells per frame, edges = same cell across frames, divisions).

## Metric
Official = adjusted edge-Jaccard + 0.1 * division-Jaccard. Node match = centroid distance within 7 µm.
Public leader: pilkwang LB 0.890 (learned graph + gap recovery + full-frame center-detector fusion).

## Local baseline
Pilkwang pipeline reproduces at golden-12 (his leak-free split_0 fold) = 0.8700 (no fusion) ↔ LB 0.885;
+ his best.pt fusion → +0.0068 recall ↔ LB 0.890. Beat it via a (1,2,2) higher-res detector + richer augs.
