# Audio modality pack — grounded in top solutions of 5 real audio competitions

**Mined** (2026-07-16) via the fleet's `gm-writeup-mine` agent → nvidia-kaggle bearer API
(`fetch_leaderboard_writeups` → `fetch_writeup`, KGAT token in `.env` + `~/.kaggle/access_token`),
top-5 solutions each saved under `docs/gm_writeups/<slug>/rank{1..5}_*.md`. This is the fleet
self-improving loop: mine → extract recurring techniques → DEDUP against existing fleet → build only
what is genuinely audio-specific and missing → register + test.

Source: **bearer API (real writeups)**, not WebSearch — every technique below is traced to a specific
placed writeup file. (WebSearch/WebFetch fallback was NOT needed; the token worked.)

## Competitions mined (most recent first)

| slug | task | metric | top solns saved |
| --- | --- | --- | --- |
| birdclef-2024 | weakly-labelled bird call ID (soundscape) | macro ROC-AUC | rank1..5 |
| birdclef-2023 | weakly-labelled bird call ID | padded cmAP | rank1..5 |
| birdclef-2021 | bird call ID (soundscape 5s) | row-wise F1 | rank1..5 |
| bengaliai-speech | Bengali ASR (out-of-distribution) | WER | rank1..5 |
| freesound-audio-tagging-2019 | multi-label audio tagging (noisy+curated) | lwlrap | rank1..6 |

## Recurring techniques (with provenance)

### 1. Log-mel spectrogram front-end — UNIVERSAL (all 5 comps)
The core representation. Waveform → STFT → mel filterbank → power → dB, then per-instance normalize.
- birdclef-2024 rank1 (team kefir): `n_fft=1024, hop_length=500, n_mels=128, fmin=40, fmax=15000, power=2` → `1×128×640` for a 10s clip.
- birdclef-2024 rank2 (adsr): `n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=16000`, resized 3-ch mel image; diversity from varying `n_mels∈{64,128}`, `hop∈{512,1024}`, image size.
- birdclef-2021 rank3 (shiro): "ResNet34 with log-mel, 128 mels. Log-mel is converted from power to dB **after all augmentations applied**. Thereafter normalized by the mean and std of each data."
- freesound-2019 rank6 (miguel pinto): `n_dft=1024, sr=44100, n_mels=64, power=2, return_decibel=True`; "the two axes of the log-mel feature have different physical meanings" → treat freq/time axes asymmetrically.
- freesound-2019 rank1 (baikulov): **linear frequency map (−1..1) concatenated as a 2nd channel** so the (position-invariant) 2D conv knows the frequency of each row — "no less than 0.005 CV gain". → the `freq_channel` option.
- Everyone computes mels **on GPU** (torchaudio / kapre) with mixed precision to kill the CPU bottleneck (birdclef-2021 rank2 vialactea; freesound rank1).

### 2. SpecAugment (time + freq masking) — recurring
- birdclef-2023 rank1 (volodymyr): "Spec Aug: Freq max-length 10 / max-lines 3 / p 0.3; Time max-length 20 / max-lines 3 / p 0.3". Multiple masked bands per axis, probabilistic.
- birdclef-2023 rank2 (griffith): `FrequencyMasking`, `TimeMasking` in the aug list.
- freesound-2019 rank1/rank3: SpecAugment + Mixup as the two main augs.
- birdclef-2024 rank1 calls it "XY masking" (their #1 single-step gain after sigmoid inference).

### 3. Waveform-domain augmentation — recurring
- birdclef-2023 rank2 (griffith), the canonical list: `GaussianNoise, PinkNoise, Gain, NoiseInjection, BackgroundNoise (nocall/rainforest/ESC), PitchShift, TimeShift, OR-Mixup on waveforms, Mixup on spectrograms`.
- bengaliai rank4/others: `AddBackgroundNoise (MUSAN, SNR 3-30 dB), AddGaussianNoise (amp 0.005-0.015), PitchShift(±4 semitones)`; resample 16k→8k→16k as aug.
- birdclef-2024 rank2 (adsr): pseudo-label test clips mixed in as **background-noise augmentation** with a random amplitude factor `10**uniform(ampExpMin,ampExpMax)` — closes the Xeno-Canto→soundscape domain gap.
- birdclef-2023 rank1: `BackgroundNoise` with Zenodo nocall + `RandomFiltering` (a simplified random equalizer).

### 4. Mixup with OR-rule label mixing (multi-label) — recurring
- freesound-2019 rank1 (baikulov): "modified MixUp… **OR rule for mixing labels** (a mix of two sounds still lets you hear both). Weighted targets were worse." ~0.05 CV gain with the aug set.
- birdclef-2024 rank3 (NVBird): "additive mixup: primary labels are the **max** of the two audios' primary labels."
- birdclef-2023 rank1: "Mixup: simply **OR** Mixup with p=0.5"; rank2: OR-mixup on waveforms + mixup on spectrograms (both).
→ audio mixup ≠ generic Beta-weighted image mixup: it happens in the **time (waveform) domain** and mixes multi-label targets with max/OR.

### 5. Fixed-window crop training + multi-window TTA on long clips — UNIVERSAL (BirdCLEF standard)
- Train on a fixed crop: 5s (bc24 rank2/adsr, bc21), 10s (bc24 rank1 = two adjacent 5s with averaged labels), 30s reshaped to 6×5s (bc21 rank2 vialactea), 20s (bc21 rank5).
- Inference = slide a fixed window over the long clip → **aggregate windows**:
  - bc24 rank1 (kefir): chunk-averaging + ensemble via `min()` reduction (lowers uncertain preds); score jumped 0.688→ with min-ensemble.
  - bc24 rank2 (adsr) / Cornell 3rd: **neighbor-window smoothing** — each 5s window += 0.5×(prev+next).
  - bc24 rank3 (NVBird): smoothing convolution with kernel `[0.1, 0.2, 0.4, 0.2, 0.1]` over time (+0.01).
  - birdclef-2023 rank1: "take **max prob** over time across the whole sample"; temperature mean `(pred**2).mean()**0.5`.
  - freesound rank6/rank3: **padding TTA** — zero-pad both sides by varying lengths, average (emphasizes clip start/end); "prediction using full-length audio scores better than slicing TTA".
→ Generic `multi-tta` (image flips/rotations) does NOT do audio sliding-window crop aggregation. This is the gap.

### 6. Pretrained backbones on a mel image — UNIVERSAL
timm CNNs on 1- or 3-channel mel spectrograms: `efficientnet_b0` (bc24 rank1/rank2 the workhorse — fast on CPU/T4), `regnety_008` (bc24 rank1), `eca_nfnet_l0`, `convnext_small/convnextv2_tiny` (bc23 rank1), `seresnext26t`, `tf_efficientnetv2_s`, `resnet34d` (bc23 rank2), `efficientvit_b0/b1` (bc24 rank3 NVBird — chosen for CPU speed), `resnet18/34` (freesound). Waveform models (wav2vec2/XLS-R, AVES, Whisper) dominate ASR (bengaliai) but are heavy. Inference optimized with ONNX/OpenVINO, parallel mel precompute, mels cached in RAM.

### 7. SED attention pooling (framewise → clipwise) — recurring, ALREADY IN FLEET
- birdclef-2023 rank1/rank2/rank3 all use SED architecture: per-frame logits + attention over time → clip prediction; bc23 rank1 blends `attention_probs*0.75 + max_timewise*0.25`. bc21 rank2 reshapes 30s→6×5s then pools time+freq.
→ Covered by existing `sed-attention-pool` (weakly-supervised attention pool + learnable GeM freq pool). **NOT rebuilt** — referenced.

### 8. Class-balanced sampling (long tail) — recurring, ALREADY IN FLEET
- birdclef-2023 rank1: `sample_weights = (value_counts/sum) ** (-0.5)` (SUPER IMPORTANT). NVBird upsamples low-freq classes to ≥10.
→ Covered by existing `class-balance-sampler` / `imbalance_sampler_pack` (tempered `count^power`, power=-0.5 = this exact recipe). **NOT rebuilt** — referenced.

### 9. Loss choices — mixed signal (kept in existing packs)
- BCE / focal / average(BCE, focal) for multi-label (bc24 rank2, bc23 rank1); CE-for-train-sigmoid-for-infer for the near-multiclass case (bc24 rank1). Rank losses (LSEP) beat BCE on the rank metric lwlrap (freesound rank1, +0.015). Secondary-label loss **masking** (NVBird, +0.01). Label smoothing for noisy weak labels (bc21 rank2).
→ focal / label-smoothing already in `train-tricks`; secondary-label handling is a data recipe, not a new agent.

## Dedup verdict → what to BUILD (genuinely audio-specific, missing)

| technique | existing fleet coverage | action |
| --- | --- | --- |
| log-mel front-end | none | **BUILD `audio-melspec-fe`** |
| SpecAugment + waveform aug + OR-mixup-in-time | `train-tricks` mixup is batch/image only; no SpecAugment, no waveform aug | **BUILD `audio-augment`** |
| sliding-window crop + multi-window TTA on long clips | `multi-tta` = image flips/rot; no audio crop aggregation | **BUILD `audio-crop-tta`** |
| mel→CNN classifier wrapper (timm, freq-channel, window-reshape) | generic arch agents don't glue mel→CNN | **BUILD `audio-backbone`** (thin; escalate-clean if timm/PANNs weights absent) |
| framewise→clipwise attention pool (SED) | `sed-attention-pool` | reference, do NOT rebuild |
| class-balanced sampling `count^-0.5` | `class-balance-sampler`, `imbalance_sampler_pack` | reference, do NOT rebuild |
| mixup/focal/label-smoothing (batch) | `train-tricks` | reference for batch mixup; audio-augment adds the waveform/OR variant |

## Agents built (fleet_agents/audio_pack.py, pure torch/numpy, GPU-first)

- **audio-melspec-fe** — waveform → log-mel `[n_mels, T]`. torchaudio if present, else pure-`torch.stft` + a numpy/torch mel filterbank (HTK/Slaney), configurable `n_mels/n_fft/hop/fmin/fmax`, `power`, `to_db` (power→dB, top_db clamp), per-instance normalize, optional `freq_channel` (linear −1..1 second channel, freesound rank1). Runs on CUDA.
- **audio-augment** — SpecAugment (`freq_mask`+`time_mask`, multi-band, probabilistic; bc23 rank1 params as defaults) on a `[.,n_mels,T]` spec + waveform aug (gaussian noise, pink noise, gain, background-mix, OR-mixup-in-time with max-label). Shape-preserving, DataLoader-safe (no in-place graph ops, deterministic under seed).
- **audio-crop-tta** — `fixed_crop` (random crop for train / center for eval) + `sliding_windows` (hop-strided fixed windows over a long clip) + `aggregate` (mean/max/`min`/temperature-mean over windows) + `neighbor_smooth` (the `[0.1,0.2,0.4,0.2,0.1]` / 0.5-neighbor conv). The BirdCLEF long-clip inference standard that generic `multi-tta` lacks.
- **audio-backbone** — builds a mel→CNN classifier: timm EfficientNet on 1- or 3-channel mel (present here) else a small pure-torch CNN; optional window-reshape head (30s→6×5s pool, bc21 rank2). Optional PANNs/wav2vec2 interface **escalates cleanly** if heavy weights are absent (they are not shipped).

## Registration
- `fleet_agents/__init__.py`: import alias `audio_pack as _audio`, SEED tuples (thread `M`), HANDLERS entries.
- New coverage pack **"Audio"** in `coverage_audit.py` PACKS = {audio-melspec-fe, audio-augment, audio-crop-tta, audio-backbone}.
- `agent_routing.py` `DOMAIN_MODALITIES["Audio"] = ("audio", "multimodal")` → routes for audio + multimodal comps; 0 untagged, UNCLASSIFIED == [].

## Honesty notes
- torchaudio and librosa are **absent** in `research/cellmot_venv`; the front-end therefore uses pure-`torch.stft` + a hand-built mel filterbank (the GPU-first path anyway). No numpy/torch version was changed.
- PANNs / wav2vec2 / AVES pretrained weights are **not shipped**; `audio-backbone` escalates cleanly rather than faking them. timm IS present so the mel→EfficientNet path is real.
