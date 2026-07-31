# Video Pack — grounded in the top solutions of 5 real VIDEO competitions

Mined via the fleet's `gm-writeup-mine` agent (nvidia-kaggle bearer / KGAT token — token worked, no
WebSearch fallback needed). Raw writeups saved under `docs/gm_writeups/<slug>/rank*.md`. This doc distills the
recurring VIDEO techniques with per-writeup provenance, then states the dedup verdict and what the pack builds.

Competitions mined (most-recent first), top-5 each (25 writeups total):
- **deepfake-detection-challenge** (2020) — per-frame / 3D-CNN video classification of face crops
- **dfl-bundesliga-data-shootout** (2022) — temporal event (action) spotting in soccer broadcast
- **nfl-player-contact-detection** (2023) — per-step contact classification from multi-view video + tracking
- **nfl-impact-detection** (2020) — helmet detection + impact (event) classification over frames
- **youtube8m-2019** (2019) — temporal-segment classification from pre-extracted frame features

---

## Recurring VIDEO techniques (with provenance)

### A. Frame / clip sampling
- **Uniform / fixed-N per clip** — deepfake 1st (Selim Seferbekov): "I used 32 frames for each video"
  [rank1 selim]; deepfake 12th: 1 or 10 random frames per video [rank12].
- **Every-nth-frame (stride)** — The Medics 5th (deepfake): "every nth frame (we settled on 10) is passed
  through MTCNN" [rank5]; DFL Team Hydrogen 1st: "predicted only every second frame" [rank1]; NFL-contact
  Dmytro 3rd: "every second frame for input", Convnext-pico "skip 2 frames" [rank3].
- **Dense-around-event (non-uniform)** — NFL-contact nvnn 1st: PP model samples 18 frames
  `{-44,-37,-30,-24,-18,-13,-8,-4,-2,0,2,4,8,13,18,24,30,37}`, "enables the model to observe more frames close
  to the estimated frame"; PG model 23 frames out to ±54 [rank1]. NFL-impact I3D 5th: "a frame with impact and
  four frames before it and three after it" [rank5].
- **Random-with-jitter / positive-negative window sampling** — DFL camaro 3rd: "Half pos and half neg. Pos
  means at least 1 event in the label and neg means randomly sampled" [rank3]; DFL kmat 2nd: "extracted only
  1 sec frames randomly from labeled intervals" [rank2]; NFL-impact I3D 5th hard-negative mining of windows
  [rank5].
- **Variable-length handling (cyclic pad / trim), contiguous clips** — The Medics 5th (deepfake): "faces if
  contiguous over at least 30 frames, up to a maximum of 100" [rank5].
- **Long-sequence fine-tune** — DFL camaro 3rd: "train with a relatively short clip like 32 frames, I fine
  tune with a long sequence like 128 frames" [rank3].

### B. Per-frame CNN embedding → TEMPORAL aggregation (image backbone → video model)
- **Mean / confident averaging pool** — deepfake 1st `confident_strategy` (mean of confident-only frames)
  [rank1]; deepfake ntechlab 3rd confidence-weighted averaging "works like attention" [rank3].
- **LSTM / GRU over frames** — NFL-contact m-t-s-s 5th: "CNN + LSTM" [rank5]; DFL camaro 3rd: 1D-UNet head
  "some variants have LSTM and GRU at the end" [rank3]; youtube8m 2nd: GRU frame-level model [rank2].
- **Temporal attention / transformer over frames** — NFL-contact Dmytro 3rd: "transformer decoder to combine
  tracking features and video activations", self-attention over players/steps [rank3]; youtube8m 1st:
  segment/group LSTM + transformer segment models [rank1].
- **NetVLAD / NeXtVLAD / DBOF / NetFV pooling** (the youtube8m temporal-pooling family) — youtube8m 2nd:
  "Mix-[NeXtVLAD, LightNetVLAD, EarlyNetVLAD, GatedDBOF, SoftDBOF, NetFV, GRU]" [rank2]; 3rd: MixNeXtVLAD
  with online distillation [rank3]; 1st: NetVLAD_LF / netFV_LF segment models [rank1].
- **TSM (Temporal Shift Module)** — DFL camaro 3rd: "EfficientNet with TSM" at end of each block [rank3];
  NFL-impact Dmytro 1st: "instead of 3d convolution I used the Temporal Shift Module … little overhead …
  reuse any ImageNet pretrained 2d model", shifted by 2-3 frames in last blocks to mimic dilation [rank1];
  NFL-contact Dmytro 3rd: 2D + TSM (convnext pico / resnet34) [rank3].
- **1D-conv over time / 1D-UNet head** — DFL camaro 3rd: "head is simple 1D UNet" [rank3]; DFL kmat 2nd:
  "1D CNN to extract features from ball path / 1D CNN to predict event" [rank2].
- **2.5D (2D backbone + 3D conv in last block) & full 3D-CNN** — DFL Team Hydrogen 1st: 2.5D+3D, "3D conv
  layers only in the last block" [rank1]; NFL-contact nvnn 1st: irCSN action-recognition net [rank1];
  NFL-contact Dmytro 3rd: "2d imagenet models + 3d Conv layer (credits Team Hydrogen)", X-CLIP, Video Swin
  [rank3]; deepfake The Medics 5th: I3D / 3D-ResNet34 / MC3 / R2+1D 3D-CNNs [rank5]; deepfake ntechlab 3rd:
  "added a 3d convolution to each block of EfficientNet" (7-frame sequence) [rank3]; NFL-impact I3D 5th /
  Happy New Impact 2nd (3D conv replacing first 2D conv in inverted-residual blocks) [rank5, rank2].

### C. Frame-stacking as channels (cheap 2.5D)
- DFL Team Hydrogen 1st: "stack three neighboring frames {14,15,16} as the channels of the input" [rank1];
  DFL ohkawa3 5th: "11 channels total (±5 frames before and after)" [rank4].

### D. Motion features (frame-difference / optical-flow proxy)
- **Frame difference as extra input channel** — DFL ohkawa3 5th: "Difference from previous frame and apply
  absolute … explicitly extract video features" [rank4]; DFL kmat 2nd: detector input has "difference between
  the current frame and previous frame … and next frame", "drastically improved accuracy in crowded scene"
  [rank2].
- **Optical flow (RAFT / OpenCV) for global camera motion + tracking** — DFL kmat 2nd: RAFT self-supervised
  optical flow for global camera motion [rank2]; NFL-impact Dmytro 1st: "optical flow between surrounding
  frames … to track helmets and estimate velocity", MotionSqueeze (flow + TSM) [rank1].
- **Motion-compensated crops** — NFL-impact Dmytro 1st: shift frames by current box velocity so the helmet
  stays centered, "acceleration is important for classification" [rank1].

### E. Clip / window construction + overlap
- **Overlapping windows, drop-edge center-crop** — DFL camaro 3rd: "model predicts 64 frames … drop edges,
  take only center 48" [rank3]; NFL-contact Dmytro 3rd: "input interval of 11/16 steps run with offset of 5
  steps to predict over overlapped intervals … all predictions averaged" [rank3].

### F. ROI crop pipeline (detector → crop → per-frame model)
- Face crop: deepfake MTCNN [rank1,rank5] / BlazeFace [rank12]. Helmet crop: NFL YoloV5 → 128×128 crops
  [nfl-impact rank1, rank2; nfl-contact rank1 "crop around contact area 10× helmet size"]. Ball crop: DFL
  CenterNet detector → crop near ball [rank2].

### G. Multi-view + temporal TTA
- NFL two-view (endzone+sideline), random-swap augmentation [nfl-contact rank1]; NFL-impact 5th: two-side-view
  post-processing, scale-TTA [0.75,1.0,1.25,1.5] + horizontal flip [rank5]; deepfake flip-half-frames TTA
  [rank3].

### H. Sequence-of-detections → event (post-processing)
- Peak detection (max-pool) + gaussian smoothing + NMS — DFL kmat 2nd [rank2], DFL camaro 3rd NMS [rank3];
  neighbor-filter over time — youtube8m 1st "size 3 kernel … over the time dimension" [rank1]; XGB over
  neighboring steps — NFL-contact nvnn 1st [rank1], m-t-s-s 5th GBDT stage [rank5].

---

## Dedup verdict — what the fleet ALREADY covers (referenced, NOT rebuilt)

| Mined technique | Already covered by | Verdict |
|---|---|---|
| Sequence-of-detections → action **segments** (thr + min-len + merge-gap) | `temporal-segment-decoder` (forecast_sports_pack) | REFERENCE — do not rebuild |
| Peak/gaussian **smoothing** of per-frame scores over time | `audio-crop-tta`'s `neighbor_smooth` + `heatmap-peak-decoder` | REFERENCE |
| Pad-aware **mean/max/attention pooling** over a variable-length sequence | `masked-sequence-pool` (masked_sequence_pack) | REFERENCE for the mask-invariant path; EXTEND with the learnable video aggregators it lacks |
| Overlapping-window **inference TTA** / multi-scale / flip | `multi-tta`, `snapshot-average`, `wbf-fusion` | REFERENCE |
| ROI detector (face/helmet/ball) → crop | Detection & Tracking pack (`detector-select`, `saliency-detect`, `gaussian-heatmap-encoder`) + `geometric-spatial-augmentor` | REFERENCE |
| Sparse **keyframe** selection to save runtime (detector-budget) | `keyframe` (CORE) — different intent (T4 detection budget, not training clip sampling) | REFERENCE, distinct |
| Trajectory / position forecasting | `trajectory-forecaster` (heavy_runnable2_pack) | REFERENCE, distinct |
| 3D-CNN / nnU-Net volumetric | `volumetric-patch-inference`, `nnunet-segmentation-runner` | REFERENCE (weights not shipped) |

## The genuine VIDEO gap → what THIS pack builds (3 agents)

The fleet had **no video-specific frame/clip SAMPLING** and **no learnable TEMPORAL AGGREGATION of per-frame
embeddings** (TSM / 1D-temporal-conv / GRU / content-attention that turn an image backbone into a video model),
and **no motion-feature** primitive. `masked-sequence-pool` only does mask-invariant mean/max/attention over a
padded sequence — it is not the TSM / temporal-conv / GRU family and knows nothing about frame sampling or
motion. So:

- **`video-frame-sampler`** — sample T frame indices from a length-`n_frames` clip: `uniform` (evenly spaced,
  TSN-style), `stride` (start+stride·k, cyclic-wrap), `dense` (non-uniform, densest around an event index — the
  NFL `{-44…0…37}` schedule), `random` (segment-binned with jitter, TSN train-time). Handles clips SHORTER
  than T via cyclic pad and clips longer via subsampling; always returns T indices in `[0, n_frames)`. Ships a
  `gather_frames` helper. The foundation for any video-training dataloader. (Techniques A.)
- **`video-temporal-aggregator`** — per-frame embeddings `[B,T,D]` → clip vector `[B,D']` via
  `mean` / `max` / `attention` (content/energy-weighted, parameter-free — captures sparse-in-time signal) /
  `tconv` (1D conv over time) / `tsm` (temporal-shift then pool) / `gru`. This is the piece that turns an image
  backbone into a video model. References `masked-sequence-pool` for the mask-invariant mean/max/attention path;
  adds the TSM / temporal-conv / GRU aggregators the fleet lacked. (Techniques B.)
- **`video-motion-features`** — frame-difference (abs, 1st/2nd order), temporal-gradient (central difference =
  optical-flow proxy along time), a brightness-constancy flow-magnitude proxy, and a `motion_channels` stacker
  that appends motion maps as extra CNN input channels. Non-zero where motion is, ~zero on a static clip — the
  deepfake/sports motion cue (Techniques C, D).

Pure torch/numpy, GPU-first (every op on CUDA when available; CPU fallback only when no CUDA). No numpy/torch
version touched. Heavy pretrained video backbones (I3D/CSN/X-CLIP/Video-Swin/NetVLAD weights) are NOT shipped —
`video-temporal-aggregator` builds untrained aggregation heads on top of whatever frame embeddings you pass;
it does not fabricate pretrained 3D-CNN weights. Data-wise test: `test_fleet_agents/video_pack_test.py`.
