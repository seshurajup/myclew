"""audio_pack_test — DATA-WISE, offline, deterministic (BLAS-pinned) verifier for the AUDIO pack.

Synthesizes deterministic sine/chirp waveforms (no files, no network) and asserts the ground-truth behaviour
of each audio agent's underlying function, plus that each raw handler returns a valid (status,data,to,msg)
contract on an EMPTY spec (the fleet smoke contract). Exit 0 iff all checks pass.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

import math
import torch

from fleet_agents import audio_pack as A

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
_fails = []


def check(name, cond):
    print(("  ok  " if cond else "  FAIL") + "  " + name)
    if not cond:
        _fails.append(name)


def _chirp(seconds=5.0, f0=200.0, f1=1200.0):
    t = torch.linspace(0, seconds, int(SR * seconds), device=DEV)
    return torch.sin(2 * math.pi * (f0 + (f1 - f0) * t / seconds) * t)


# ── 1. melspec front-end: right shape, finite, dB-scaled ─────────────────────────────────────────────
wav = _chirp()
mel = A.log_mel_spectrogram(wav, sr=SR, n_fft=1024, hop=500, n_mels=128, fmin=20.0, fmax=8000.0,
                            power=2.0, to_db=True, normalize=False, device=DEV)
check("melspec shape [n_mels, T]", mel.dim() == 2 and mel.shape[0] == 128)
check("melspec finite", bool(torch.isfinite(mel).all()))
# with to_db the values span a dB range (>~30 dB dynamic) and are NOT all >=0 like raw power would trend
check("melspec db-scaled (wide dynamic range)", float(mel.max() - mel.min()) > 20.0)
# normalize → zero mean, unit std
mel_n = A.log_mel_spectrogram(wav, sr=SR, n_mels=64, normalize=True, device=DEV)
check("melspec normalized ~zero-mean unit-std", abs(float(mel_n.mean())) < 1e-3 and abs(float(mel_n.std()) - 1.0) < 0.05)
# batched
melb = A.log_mel_spectrogram(torch.stack([wav, wav]), sr=SR, n_mels=64, device=DEV)
check("melspec batched -> (B, n_mels, T)", melb.dim() == 3 and melb.shape[0] == 2)
# freq_channel adds a 2nd channel that is a linear -1..1 map
melf = A.log_mel_spectrogram(wav, sr=SR, n_mels=64, freq_channel=True, device=DEV)
check("melspec freq_channel -> 2 channels", melf.shape[0] == 2)
check("melspec freq_channel is linear -1..1", abs(float(melf[1].min()) + 1.0) < 1e-4 and abs(float(melf[1].max()) - 1.0) < 1e-4)

# ── 2. SpecAugment zeroes expected masked bands, preserves shape ─────────────────────────────────────
spec = A.log_mel_spectrogram(wav, sr=SR, n_mels=128, normalize=True, device=DEV)
aug = A.spec_augment(spec, n_freq_masks=2, freq_mask=15, n_time_masks=2, time_mask=25, p=1.0, mask_value=0.0, seed=7)
check("specaug preserves shape", aug.shape == spec.shape)
diff = (aug != spec)
check("specaug actually masked something", bool(diff.any()))
# masked positions must equal the fill value (0.0); and there must be full rows/cols masked (band structure)
masked_rows = (aug == 0.0).all(dim=1).sum().item()
masked_cols = (aug == 0.0).all(dim=0).sum().item()
check("specaug masked >=1 full freq band (row)", masked_rows >= 1)
check("specaug masked >=1 full time band (col)", masked_cols >= 1)
# p=0 → no-op
check("specaug p=0 is identity", bool((A.spec_augment(spec, p=0.0, seed=1) == spec).all()))

# ── 3. waveform augment + OR-mixup are shape-preserving / DataLoader-safe ────────────────────────────
wa = A.waveform_augment(wav, gaussian=0.01, pink=0.01, gain_db=3.0, seed=0)
check("waveform_augment shape-preserving", wa.shape == wav.shape)
check("waveform_augment finite", bool(torch.isfinite(wa).all()))
check("waveform_augment changed the signal", bool((wa != wav).any()))
# background mix at SNR
bg = torch.randn(SR, device=DEV)
wb = A.waveform_augment(wav, background=bg, bg_snr_db=5.0, seed=0)
check("waveform bg-mix shape-preserving", wb.shape == wav.shape and bool(torch.isfinite(wb).all()))
# OR-mixup: multi-label targets combine by OR/max
_, y, lam = A.or_mixup(wav, _chirp(f0=800, f1=800), torch.tensor([1.0, 0.0, 0.0]),
                       torch.tensor([0.0, 1.0, 0.0]), seed=3)
check("or_mixup OR-labels", bool((y == torch.tensor([1.0, 1.0, 0.0])).all()))
check("or_mixup lam in [0,1]", 0.0 <= lam <= 1.0)

# ── 4. crop + multi-window TTA: N windows -> one vector; aggregation keeps a planted signal ──────────
win = 3 * SR
crop_r = A.fixed_crop(wav, win, mode="random", seed=1)
crop_c = A.fixed_crop(wav, win, mode="center")
check("fixed_crop random length", crop_r.shape[-1] == win)
check("fixed_crop center length", crop_c.shape[-1] == win)
short = wav[: SR]                                   # 1s clip, shorter than 3s window → cyclic pad
check("fixed_crop cyclic-pads short clip", A.fixed_crop(short, win).shape[-1] == win)
clip = torch.randn(int(SR * 20), device=DEV)         # 20s long clip
wins = A.sliding_windows(clip, win, hop=win)
check("sliding_windows -> (N, win)", wins.dim() == 2 and wins.shape[1] == win and wins.shape[0] >= 6)
# plant a strong signal in one window for class 1; max-agg must keep it, mean must dilute it → one vector
preds = torch.rand(wins.shape[0], 5, device=DEV) * 0.1
preds[3, 1] = 0.97
amean, amax, amin, atmean = (A.aggregate(preds, m) for m in ("mean", "max", "min", "tmean"))
check("aggregate mean -> single vector (C,)", amean.shape == (5,))
check("aggregate max keeps planted signal", float(amax[1]) >= 0.95)
check("aggregate mean dilutes vs max", float(amean[1]) < float(amax[1]))
check("aggregate min <= mean <= max", float(amin[1]) <= float(amean[1]) <= float(amax[1]))
check("aggregate tmean sharpens (>= mean)", float(atmean[1]) >= float(amean[1]) - 1e-4)
sm = A.neighbor_smooth(preds)
check("neighbor_smooth preserves shape", sm.shape == preds.shape)
check("neighbor_smooth finite", bool(torch.isfinite(sm).all()))

# ── 5. backbone builds + forwards; small-CNN fallback always available ───────────────────────────────
model, backend = A.build_mel_backbone(7, in_ch=1, pretrained=False, device=DEV)
model.eval()
with torch.no_grad():
    out = model(torch.randn(2, 128, 128, device=DEV))
check("backbone out shape (B, n_classes)", tuple(out.shape) == (2, 7))
check("backbone out finite", bool(torch.isfinite(out).all()))
print("  backend used:", backend)

# ── 6. every raw handler returns a valid contract on EMPTY spec (fleet smoke contract) ───────────────
VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}
for h in (A.run_melspec, A.run_augment, A.run_crop_tta, A.run_backbone):
    r = h({"question": "test", "spec": {}}, "unit")
    check(f"handler {h.__name__} valid contract", isinstance(r, tuple) and len(r) == 4 and r[0] in VALID)
# heavy pretrained request must escalate cleanly, NOT fabricate weights
r = A.run_backbone({"question": "t", "spec": {"backbone": "panns"}}, "unit")
check("audio-backbone PANNs escalates clean", r[0] == "escalated")

print()
if _fails:
    print("FAILURES:", _fails)
    sys.exit(1)
print("ALL AUDIO PACK CHECKS PASSED")
sys.exit(0)
