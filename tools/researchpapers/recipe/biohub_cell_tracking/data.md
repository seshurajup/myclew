# Data

Root: /home/seshu/kaggle/2026/biohub-cell-tracking-during-development/input/biohub-cell-tracking-during-development/train/
- 199 embryos, two groups: 44b6 (71, biologically late/dense) and 6bba (128, early/sparse).
- Each embryo: {stem}.zarr (image, shape T,Z,Y,X; Z=64,Y=256,X=256) + {stem}.geff (GT tracking graph).
- geff node = one ANNOTATED cell at one frame (props t,z,y,x); edges = source_id->target_id.
- Labels are SPARSE (~4% of real cells); true count = geff attr estimated_number_of_nodes (estN).
- True density 38..1015 cells/frame; voxel µm (z,y,x)=(1.625,0.40625,0.40625) -> (1,4,4) downsample = isotropic.
- CV: golden-12 = pilkwang split_0 held-out fold (splits_ft.json). Official metric via research/official_repo/scripts/evaluate.py.
  Note: golden-12 over-credits dense detectors (density-blind on sparse labels) — trust for divisions/linking, use recall-proxy for detection.
