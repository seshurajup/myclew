"""audio_pack — the AUDIO modality pack, GROUNDED in the top-5 solutions of 5 real audio competitions
(birdclef-2024/2023/2021, bengaliai-speech, freesound-audio-tagging-2019). Mined via the fleet's
`gm-writeup-mine` agent; distilled recurring techniques + full provenance in docs/audio_pack_grounded.md.

The fleet already covered the two audio techniques that are NOT audio-specific:
  • framewise→clipwise attention pooling (SED)  → `sed-attention-pool` (training_head_pack)  — referenced, NOT rebuilt
  • class-balanced sampling (count^-0.5)         → `class-balance-sampler` / imbalance_sampler_pack — referenced, NOT rebuilt
  • batch mixup / focal / label-smoothing        → `train-tricks` — referenced (audio-augment adds the waveform/OR variant)

What the fleet was genuinely MISSING (this pack), each recurring across the mined winners:
  • audio-melspec-fe  — waveform → log-mel spectrogram, THE audio representation (all 5 comps)
  • audio-augment     — SpecAugment (time+freq masking) + waveform aug (noise/gain/bg-mix/OR-mixup-in-time)
  • audio-crop-tta    — fixed-window crop (train) + multi-window sliding-crop TTA aggregation (long-clip infer)
  • audio-backbone    — mel→CNN classifier wrapper (timm EfficientNet else small CNN); PANNs iface escalates clean

Pure torch/numpy, GPU-FIRST (every tensor op runs on CUDA when available; CPU fallback only if no CUDA).
torchaudio/librosa are optional — the front-end falls back to pure `torch.stft` + a hand-built mel filterbank
(the GPU path anyway). No numpy/torch version is touched. Data-wise tests: test_fleet_agents/audio_pack_test.py.
"""
from __future__ import annotations
import math
from .base import BaseAgent


def _device(spec):
    import torch
    d = (spec or {}).get("device")
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


# ════════════════════════════════════════════════════════════ mel filterbank (pure numpy/torch, no librosa)
def _hz_to_mel(f, slaney=True):
    import numpy as np
    scalar = np.ndim(f) == 0
    f = np.atleast_1d(np.asarray(f, dtype="float64"))
    if slaney:
        # Slaney (librosa default): linear below 1kHz, log above
        f_min, f_sp = 0.0, 200.0 / 3
        mel = (f - f_min) / f_sp
        min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        above = f >= min_log_hz
        mel[above] = min_log_mel + np.log(f[above] / min_log_hz) / logstep
    else:
        mel = 2595.0 * np.log10(1.0 + f / 700.0)
    return float(mel[0]) if scalar else mel


def _mel_to_hz(m, slaney=True):
    import numpy as np
    scalar = np.ndim(m) == 0
    m = np.atleast_1d(np.asarray(m, dtype="float64"))
    if slaney:
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * m
        min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        above = m >= min_log_mel
        freqs[above] = min_log_hz * np.exp(logstep * (m[above] - min_log_mel))
    else:
        freqs = 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    return float(freqs[0]) if scalar else freqs


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax, slaney=True):
    """Triangular mel filterbank matrix, shape (n_mels, n_fft//2 + 1). Pure numpy (Slaney/HTK), no librosa.
    Slaney-normalized (area-1 triangles) so it matches torchaudio/librosa defaults.
    """
    import numpy as np
    fmax = fmax or sr / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_freqs)
    m_min, m_max = _hz_to_mel(fmin, slaney), _hz_to_mel(fmax, slaney)
    mel_pts = np.linspace(float(m_min), float(m_max), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts, slaney)
    fb = np.zeros((n_mels, n_freqs), dtype="float32")
    for i in range(n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0.0, None)
        if slaney:                                  # area normalization (Slaney)
            enorm = 2.0 / max(hi - lo, 1e-9)
            fb[i] *= enorm
    return fb


# ════════════════════════════════════════════════════════════ 1. melspec front-end
def log_mel_spectrogram(wav, sr=32000, n_fft=1024, hop=500, win=None, n_mels=128, fmin=20.0, fmax=None,
                        power=2.0, to_db=True, top_db=80.0, normalize=True, freq_channel=False,
                        center=True, device=None):
    """Waveform → log-mel spectrogram — THE core audio representation (grounded: all 5 mined comps).

    `wav`: 1D (T,) or batched (B, T) tensor/array. Returns a torch tensor:
        (n_mels, F)         for a 1D input,      or (B, n_mels, F) for a batch;
        with `freq_channel=True` a leading channel dim is added: (1|2, n_mels, F) / (B, 1|2, n_mels, F),
        where channel-2 is a linear frequency map (−1..1) so a position-invariant 2D conv knows each row's
        frequency (freesound-2019 1st place, ~0.005 CV gain).

    Uses torchaudio if present, else pure `torch.stft` + `mel_filterbank` (no librosa). Runs on `device`
    (CUDA when available) — the GPU-first rule. `to_db` applies power→dB (10·log10) with a `top_db` floor,
    matching the winners who convert power→dB AFTER augmentations then z-score per instance.
    """
    import torch
    import numpy as np
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(wav):
        wav = torch.as_tensor(np.asarray(wav, dtype="float32"))
    wav = wav.to(dev).float()
    squeeze = wav.dim() == 1
    if squeeze:
        wav = wav.unsqueeze(0)                                   # (B, T)
    win = win or n_fft
    window = torch.hann_window(win, device=dev)
    # STFT magnitude → power
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                      center=center, return_complex=True, pad_mode="reflect")
    mag = spec.abs()                                            # (B, F_freq, T)
    p = mag ** power
    fb = torch.as_tensor(mel_filterbank(sr, n_fft, n_mels, fmin, fmax), device=dev)  # (n_mels, F_freq)
    mel = torch.einsum("mf,bft->bmt", fb, p)                    # (B, n_mels, T)
    if to_db:
        mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        if top_db is not None:
            mel = torch.clamp(mel, min=mel.amax(dim=(-2, -1), keepdim=True) - top_db)
    if normalize:                                              # per-instance z-score (winner standard)
        mu = mel.mean(dim=(-2, -1), keepdim=True)
        sd = mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        mel = (mel - mu) / sd
    if freq_channel:
        B, M, T = mel.shape
        fmap = torch.linspace(-1.0, 1.0, M, device=dev).view(1, M, 1).expand(B, M, T)
        mel = torch.stack([mel, fmap], dim=1)                  # (B, 2, n_mels, T)
    return mel[0] if squeeze else mel


class AudioMelspecFE(BaseAgent):
    name = "audio-melspec-fe"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            sr = int(spec.get("sr", 32000))
            n_mels = int(spec.get("n_mels", 128))
            n_fft = int(spec.get("n_fft", 1024))
            hop = int(spec.get("hop", 500))
            dur = float(spec.get("seconds", 5.0))
            wav = spec.get("wav")
            if wav is None:                                    # self-synthesize a chirp for a health-check
                t = torch.linspace(0, dur, int(sr * dur), device=dev)
                wav = torch.sin(2 * math.pi * (200 + 400 * t) * t)
            mel = log_mel_spectrogram(wav, sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels,
                                      fmin=float(spec.get("fmin", 20.0)), fmax=spec.get("fmax"),
                                      power=float(spec.get("power", 2.0)),
                                      to_db=bool(spec.get("to_db", True)),
                                      normalize=bool(spec.get("normalize", True)),
                                      freq_channel=bool(spec.get("freq_channel", False)), device=dev)
            fin = bool(torch.isfinite(mel).all())
            shp = tuple(mel.shape)
            msg = (f"audio-melspec-fe: log-mel {shp} (n_mels={n_mels}, n_fft={n_fft}, hop={hop}, sr={sr}) "
                   f"finite={fin} device={dev}; pure torch.stft + Slaney mel filterbank (no librosa/torchaudio).")
            self.log(msg, kind="finding",
                     recommendation="import log_mel_spectrogram from fleet_agents.audio_pack as the audio front-end; "
                                    "power→dB after aug, per-instance z-score, freq_channel=True for the freesound-1st gain")
            return self.done({"shape": list(shp), "finite": fin, "n_mels": n_mels, "device": str(dev)}, msg)
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] audio-melspec-fe FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ 2. audio augment (SpecAugment + waveform)
def spec_augment(mel, n_freq_masks=2, freq_mask=10, n_time_masks=2, time_mask=20, p=0.5,
                 mask_value=None, seed=None):
    """SpecAugment on a log-mel spec (grounded: birdclef-2023 1st — freq len 10 / time len 20 / p 0.3, multi-band).

    `mel`: (..., n_mels, T). Zeros (or `mask_value`, default the per-spec min so it reads as silence) up to
    `n_freq_masks` frequency bands (each ≤ `freq_mask` rows) and `n_time_masks` time bands (each ≤ `time_mask`
    cols). Shape-preserving, out-of-place (DataLoader/autograd-safe). `p` is the per-call apply probability.
    """
    import torch
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))

    def _rand():
        return float(torch.rand(1, generator=g).item())
    out = mel.clone()
    if _rand() > p:
        return out
    M, T = out.shape[-2], out.shape[-1]
    fill = out.amin().detach() if mask_value is None else mask_value
    for _ in range(n_freq_masks):
        f = int(_rand() * min(freq_mask, M))
        if f > 0:
            f0 = int(_rand() * max(1, M - f))
            out[..., f0:f0 + f, :] = fill
    for _ in range(n_time_masks):
        tt = int(_rand() * min(time_mask, T))
        if tt > 0:
            t0 = int(_rand() * max(1, T - tt))
            out[..., :, t0:t0 + tt] = fill
    return out


def _pink_noise(n, device, g):
    """Approximate pink (1/f) noise via FFT shaping of white noise. Length n, unit-ish scale."""
    import torch
    white = torch.randn(n, generator=g).to(device)
    spec = torch.fft.rfft(white)
    freqs = torch.arange(spec.shape[0], device=device).float()
    freqs[0] = 1.0
    spec = spec / torch.sqrt(freqs)                            # 1/sqrt(f) amplitude → 1/f power
    pink = torch.fft.irfft(spec, n=n)
    return pink / (pink.std() + 1e-8)


def waveform_augment(wav, gaussian=0.0, pink=0.0, gain_db=0.0, background=None, bg_snr_db=10.0, seed=None):
    """Waveform-domain augmentation (grounded: birdclef-2023 2nd — GaussianNoise/PinkNoise/Gain/BackgroundNoise).

    `wav`: (..., T). Adds gaussian noise (std `gaussian`), pink noise (std `pink`), applies a fixed `gain_db`,
    and mixes a `background` clip at `bg_snr_db` SNR. Out-of-place, shape-preserving, deterministic under `seed`.
    """
    import torch
    dev = wav.device
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    out = wav.clone().float()
    if gaussian and gaussian > 0:
        out = out + gaussian * torch.randn(out.shape, generator=g).to(dev)
    if pink and pink > 0:
        out = out + pink * _pink_noise(out.shape[-1], dev, g).expand_as(out)
    if gain_db:
        out = out * (10.0 ** (gain_db / 20.0))
    if background is not None:
        bg = background.to(dev).float()
        if bg.shape[-1] != out.shape[-1]:                     # tile/crop to match
            reps = out.shape[-1] // bg.shape[-1] + 1
            bg = bg.repeat(*([1] * (bg.dim() - 1)), reps)[..., :out.shape[-1]]
        sp = out.pow(2).mean().clamp_min(1e-10)
        bp = bg.pow(2).mean().clamp_min(1e-10)
        scale = torch.sqrt(sp / bp / (10.0 ** (bg_snr_db / 10.0)))
        out = out + scale * bg
    return out


def or_mixup(wav_a, wav_b, y_a, y_b, alpha=0.5, seed=None):
    """Audio OR-mixup (grounded: freesound-2019 1st, birdclef-2023/2024) — mix two WAVEFORMS in the TIME domain,
    combine multi-label targets by OR/max (both sounds are still audible). Returns (mixed_wav, mixed_label, lam).
    """
    import torch, numpy as np
    lam = float(np.random.RandomState(seed).beta(alpha, alpha)) if (alpha and alpha > 0) else 0.5
    lam = max(0.0, min(1.0, lam))
    mixed = lam * wav_a + (1.0 - lam) * wav_b
    y = torch.maximum(torch.as_tensor(y_a).float(), torch.as_tensor(y_b).float())   # OR-rule labels
    return mixed, y, lam


class AudioAugment(BaseAgent):
    name = "audio-augment"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            sr = int(spec.get("sr", 32000))
            t = torch.linspace(0, 5.0, sr * 5, device=dev)
            wav = torch.sin(2 * math.pi * 220 * t)
            wav_aug = waveform_augment(wav, gaussian=0.01, pink=0.01, gain_db=3.0, seed=0)
            mel = log_mel_spectrogram(wav, sr=sr, device=dev)
            mel_aug = spec_augment(mel, seed=0, p=1.0)
            checks = {
                "wave_shape": tuple(wav_aug.shape) == tuple(wav.shape),
                "wave_finite": bool(torch.isfinite(wav_aug).all()),
                "spec_shape": tuple(mel_aug.shape) == tuple(mel.shape),
                "spec_masked": bool((mel_aug != mel).any()),
            }
            wb = torch.sin(2 * math.pi * 440 * t)
            _, y, lam = or_mixup(wav, wb, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), seed=0)
            checks["ormix"] = bool((y == torch.tensor([1.0, 1.0])).all())
            ok = all(checks.values())
            msg = (f"audio-augment: {sum(checks.values())}/{len(checks)} ok ({checks}); SpecAugment(time+freq) "
                   f"+ waveform(gaussian/pink/gain/bg-mix) + OR-mixup-in-time. DataLoader-safe, shape-preserving.")
            self.log(msg, kind="finding",
                     recommendation="spec_augment on the mel, waveform_augment/or_mixup on the raw wave BEFORE the "
                                    "front-end; OR-rule labels for multi-label (freesound/birdclef standard)")
            return self.done({"checks": {k: bool(v) for k, v in checks.items()}, "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher", f"[{worker}] audio-augment checks failed: {checks}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] audio-augment FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ 3. crop + multi-window TTA
def fixed_crop(wav, length, mode="random", seed=None):
    """Fixed-length crop of a waveform (grounded: BirdCLEF 5s-crop training). `mode`='random' (train) or
    'center' (eval). Cyclic-pads if the clip is shorter than `length` (bc24 rank1). `wav`: (..., T)."""
    import torch
    T = wav.shape[-1]
    if T < length:                                            # cyclic pad
        reps = length // T + 1
        wav = wav.repeat(*([1] * (wav.dim() - 1)), reps)[..., :length]
        return wav
    if T == length:
        return wav
    if mode == "center":
        s = (T - length) // 2
    else:
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))
        s = int(torch.randint(0, T - length + 1, (1,), generator=g).item())
    return wav[..., s:s + length]


def sliding_windows(wav, length, hop=None):
    """Slide a fixed window over a long clip → stacked windows (grounded: BirdCLEF long-clip inference).
    Returns (n_windows, ..., length). `hop` defaults to `length` (non-overlapping 5s chunks)."""
    import torch
    hop = hop or length
    T = wav.shape[-1]
    if T <= length:
        return fixed_crop(wav, length, mode="center").unsqueeze(0)
    starts = list(range(0, T - length + 1, hop))
    if starts[-1] != T - length:
        starts.append(T - length)
    return torch.stack([wav[..., s:s + length] for s in starts], dim=0)


def aggregate(preds, mode="mean"):
    """Aggregate per-window predictions (grounded: bc24 rank1 min-reduction, bc23 rank1 temperature-mean/max).
    `preds`: (n_windows, ..., C). modes: mean | max | min | tmean (=(p^2).mean^0.5, sharpens confident windows)."""
    import torch
    if mode == "max":
        return preds.amax(dim=0)
    if mode == "min":
        return preds.amin(dim=0)
    if mode == "tmean":
        return (preds.pow(2).mean(dim=0)).clamp_min(0).sqrt()
    return preds.mean(dim=0)


def neighbor_smooth(preds, kernel=(0.1, 0.2, 0.4, 0.2, 0.1)):
    """Smooth per-window predictions over time by convolving with a neighbor kernel (grounded: bc24 rank3
    [0.1,0.2,0.4,0.2,0.1]; bc24 rank2 += 0.5·(prev+next)). `preds`: (n_windows, C). Reflect-padded, shape-kept."""
    import torch
    import torch.nn.functional as F
    k = torch.as_tensor(kernel, dtype=preds.dtype, device=preds.device)
    k = k / k.sum()
    N, C = preds.shape
    x = preds.t().unsqueeze(0)                                # (1, C, N)
    pad = len(kernel) // 2
    x = F.pad(x, (pad, pad), mode="reflect")
    w = k.view(1, 1, -1).expand(C, 1, -1)
    out = F.conv1d(x, w, groups=C).squeeze(0).t()            # (N, C)
    return out


class AudioCropTTA(BaseAgent):
    name = "audio-crop-tta"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            sr = int(spec.get("sr", 32000))
            win = int(spec.get("window", 5) * sr)
            clip = torch.randn(int(sr * 23), device=dev)      # a 23s long clip
            crop = fixed_crop(clip, win, mode="center")
            wins = sliding_windows(clip, win, hop=win)
            # planted-signal check: window with a big constant should dominate max-aggregation
            preds = torch.rand(wins.shape[0], 4, device=dev) * 0.2
            preds[2, 1] = 0.95
            agg_mean = aggregate(preds, "mean")
            agg_max = aggregate(preds, "max")
            sm = neighbor_smooth(preds)
            checks = {
                "crop_len": crop.shape[-1] == win,
                "n_windows": wins.shape[0] >= 4,
                "max_keeps_signal": bool(agg_max[1] >= 0.9),
                "mean_dilutes": bool(agg_mean[1] < agg_max[1]),
                "smooth_shape": tuple(sm.shape) == tuple(preds.shape),
            }
            ok = all(checks.values())
            msg = (f"audio-crop-tta: {sum(checks.values())}/{len(checks)} ok ({checks}); fixed-crop train + "
                   f"{wins.shape[0]} sliding {spec.get('window',5)}s windows → agg(mean/max/min/tmean) + neighbor-smooth.")
            self.log(msg, kind="finding",
                     recommendation="fixed_crop for training, sliding_windows+aggregate for long-clip inference; "
                                    "min/tmean reduction + neighbor_smooth are the BirdCLEF long-clip standard multi-tta lacks")
            return self.done({"checks": {k: bool(v) for k, v in checks.items()}, "n_windows": int(wins.shape[0])}, msg) \
                if ok else self.escalate(worker, "researcher", f"[{worker}] audio-crop-tta checks failed: {checks}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] audio-crop-tta FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ 4. mel → CNN backbone
class _SmallMelCNN:
    """Tiny pure-torch mel→CNN classifier (timm-free fallback). 4 conv blocks + GeM-ish adaptive pool + fc."""
    def __new__(cls, in_ch, n_classes):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                def blk(i, o):
                    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                         nn.MaxPool2d(2))
                self.body = nn.Sequential(blk(in_ch, 16), blk(16, 32), blk(32, 64), blk(64, 64))
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, n_classes)

            def forward(self, x):
                if x.dim() == 3:                              # (B, n_mels, T) → add channel
                    x = x.unsqueeze(1)
                x = self.body(x)
                x = self.pool(x).flatten(1)
                return self.fc(x)
        return Net()


def build_mel_backbone(n_classes, in_ch=1, arch="tf_efficientnet_b0", pretrained=False, device=None):
    """Build a mel→CNN classifier (grounded: efficientnet_b0 the BirdCLEF workhorse). Uses timm if present
    (real EfficientNet on `in_ch`-channel mel), else a small pure-torch CNN. Returns (model, backend_str).
    `pretrained=True` needs downloaded weights → only honored if timm can fetch them offline; else pretrained=False.
    """
    import torch
    from torch import nn
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    class _MelChannelAdapter(nn.Module):
        """Accept a 3D mel (B, n_mels, T) by adding the channel dim, so the mel front-end plugs straight in."""
        def __init__(self, net, expect_ch):
            super().__init__()
            self.net = net
            self.expect_ch = expect_ch

        def forward(self, x):
            if x.dim() == 3 and self.expect_ch == 1:
                x = x.unsqueeze(1)                        # (B, n_mels, T) → (B, 1, n_mels, T)
            return self.net(x)

    try:
        import timm
        net = timm.create_model(arch, pretrained=bool(pretrained), in_chans=in_ch, num_classes=n_classes)
        return _MelChannelAdapter(net, in_ch).to(dev), f"timm:{arch}"
    except Exception:  # noqa: BLE001 — timm absent or offline weight fetch failed → honest fallback
        return _SmallMelCNN(in_ch, n_classes).to(dev), "small_cnn"


class AudioBackbone(BaseAgent):
    name = "audio-backbone"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            heavy = (spec.get("backbone") or "").lower()
            if heavy in ("panns", "cnn14", "wav2vec2", "ast", "aves"):
                # heavy pretrained audio nets are NOT shipped — escalate cleanly, never fake weights
                return self.escalate(worker, "researcher",
                                     f"[{worker}] audio-backbone: '{heavy}' pretrained weights are not present in this "
                                     f"environment. Use setup-env to fetch them, or fall back to the mel→timm-EfficientNet "
                                     f"path (build_mel_backbone). Not fabricating an untrained '{heavy}'.")
            n_classes = int(spec.get("n_classes", 10))
            in_ch = int(spec.get("in_ch", 1))
            model, backend = build_mel_backbone(n_classes, in_ch=in_ch,
                                                arch=spec.get("arch", "tf_efficientnet_b0"),
                                                pretrained=bool(spec.get("pretrained", False)), device=dev)
            n_mels = int(spec.get("n_mels", 128))
            x = torch.randn(2, in_ch, n_mels, 128, device=dev) if in_ch > 1 else torch.randn(2, n_mels, 128, device=dev)
            model.eval()
            with torch.no_grad():
                y = model(x)
            ok = tuple(y.shape) == (2, n_classes) and bool(torch.isfinite(y).all())
            n_params = sum(p.numel() for p in model.parameters())
            msg = (f"audio-backbone: mel→CNN [{backend}] out={tuple(y.shape)} params={n_params/1e6:.2f}M "
                   f"finite={bool(torch.isfinite(y).all())} device={dev}; efficientnet_b0-style mel classifier "
                   f"(the BirdCLEF workhorse). Heavy PANNs/wav2vec2 escalate-clean (weights not shipped).")
            self.log(msg, kind="finding",
                     recommendation="build_mel_backbone(n_classes, in_ch) for a mel→timm-EfficientNet (or small-CNN) "
                                    "classifier; pair with audio-melspec-fe front-end + sed-attention-pool head for weak labels")
            return self.done({"backend": backend, "out_shape": list(y.shape), "params": int(n_params),
                              "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher", f"[{worker}] audio-backbone bad output {tuple(y.shape)}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] audio-backbone FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ handlers
_MEL = AudioMelspecFE()
_AUG = AudioAugment()
_TTA = AudioCropTTA()
_BB = AudioBackbone()


def run_melspec(q, worker):
    return _MEL.run(q, worker)


def run_augment(q, worker):
    return _AUG.run(q, worker)


def run_crop_tta(q, worker):
    return _TTA.run(q, worker)


def run_backbone(q, worker):
    return _BB.run(q, worker)
