"""audio-train — an end-to-end MULTI-LABEL audio-classification TRAINER that COMPOSES the fleet's audio
PRIMITIVES into a real, spec-driven training run. Built for BirdCLEF-2026 (234 species, one sigmoid per
species, multi-label) and reusable for ANY soundscape/tagging audio competition.

Grounded ONLY in the fleet's prev-year BirdCLEF knowledge (docs/audio_pack_grounded.md +
docs/gm_writeups/birdclef-{2021,2023,2024,2025}): the recipe below is the recurring winner pipeline —
log-mel front-end (bc24 r1 n_fft=1024/hop=500/n_mels=128), 5s fixed-crop training + cyclic-pad short clips,
SpecAugment + waveform aug + OR-mixup-in-time (bc23 r1/r2, freesound r1), class-balanced sampling count^-0.5
(bc25 2nd / bc23 r1 "SUPER IMPORTANT"), a timm EfficientNet-b0 mel classifier (the bc24 workhorse), BCE
multi-label loss, EMA weights, and the crucial validation choice: score on the domain-matched soundscape
5s windows (train_soundscapes_labels.csv) with the OFFICIAL macro-ROC-AUC that SKIPS empty classes — that
number tracks the LB, not the focal-clip CV.

COMPOSES (does not reimplement):
  • audio_pack.log_mel_spectrogram   — mel front-end (bf16-friendly, pure torch.stft)
  • audio_pack.spec_augment / waveform_augment / or_mixup — augmentation
  • audio_pack.fixed_crop            — 5s random crop (train) / cyclic-pad short clips
  • audio_pack.build_mel_backbone    — mel→timm-EfficientNet-b0 (else small pure-torch CNN)
  • imbalance_sampler_pack.sample_weights — class-balance-sampler (count^-0.5) on primary_label
  • train_tricks_pack.ModelEMA / mixup — EMA eval weights, OR-combine mixup labels
  • hardware_tune.load_config        — bf16 autocast / tf32 / matmul precision for THIS box

CV (predicts LB + defends the private board):
  • PRIMARY (LB-proxy)  = train_soundscapes_labels.csv — domain-matched 5s windows → best_val_auc.
  • SECONDARY (leak-safe) = GROUPED-BY-AUTHOR K-fold of the focal train recordings (col 'author'), so the
    same author never straddles train/val (BirdCLEF's classic leak). spec['cv']=True builds + SAVES the fold
    assignment (out_dir/folds.json) so it is reproducible and adversarial-val can quantify the domain gap.
  Both AUCs are reported. Grouped fold builder is local (the fleet's split-build is embryo-specific).

SMALL-FIRST → SCALE ladder (prove signal cheap before spending 5090-hours):
    tiny   : spec{classes:<=10, per_class:20, seconds:5, epochs:1}     — smoke, val AUC must COMPUTE + move
    small  : spec{classes:<=50, per_class:50, epochs:3}                — signal check on a species subset
    full   : spec{} (all 234 species, all recordings, epochs:15+, cv:True} — commit GPU-hours
  Knobs: spec['classes'] (subset of species / None=all), spec['per_class'] (cap N recordings/primary_label),
  spec['seconds'] (crop length), spec['epochs'], spec['limit'] (hard cap on #recordings), spec['dry_run'].

Checkpoint (out_dir/best.pt, Kaggle-ready — audio-infer consumes it directly):
    {state_dict(EMA eval weights), classes[list], melspec_cfg, sr, seconds, arch, in_ch}
Returns {best_val_auc, focal_val_auc, epochs, ckpt_path, curve, classes, folds}.

Pure torch/numpy + soundfile for .ogg I/O + scipy.signal.resample_poly (else linear) → 32 kHz. GPU-first
(bf16 autocast from hardware_tune). Data-wise test: test_fleet_agents/audio_train_test.py (synth .ogg).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .base import BaseAgent
from . import audio_pack as A
from . import imbalance_sampler_pack as IMB
from . import train_tricks_pack as TT
from . import hardware_tune as HW

SR = 32000  # BirdCLEF standard sample rate


# ════════════════════════════════════════════════════════════ audio I/O (soundfile + resample → 32 kHz)
def resample_to(wav, sr, target_sr=SR):
    """Resample a 1-D numpy waveform sr→target_sr. scipy.signal.resample_poly (polyphase) if present, else a
    pure-numpy linear interpolation (honest fallback). No torch/numpy version is touched."""
    import numpy as np
    if sr == target_sr:
        return np.asarray(wav, dtype="float32")
    try:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), int(target_sr))
        return resample_poly(np.asarray(wav, dtype="float64"), target_sr // g, sr // g).astype("float32")
    except Exception:  # noqa: BLE001 — linear-interp fallback (documented as lower-fidelity)
        n_out = int(round(len(wav) * target_sr / sr))
        if n_out <= 1:
            return np.asarray(wav, dtype="float32")
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, np.asarray(wav, dtype="float64")).astype("float32")


import os as _os
_AUDIO_CACHE = _os.environ.get("AUDIO_CACHE_DIR")   # if set, FULL-file decodes are cached as .npy (huge speedup:
                                                    # decode+resample each .ogg ONCE, reuse every epoch)


def _cache_path(path):
    import hashlib
    h = hashlib.md5(str(path).encode()).hexdigest()
    return Path(_AUDIO_CACHE) / f"{h[:2]}" / f"{h}.npy"


def load_audio(path, target_sr=SR, start=None, stop=None):
    """Load an .ogg/.wav/.flac via soundfile → mono float32 at target_sr. `start`/`stop` (seconds) read only a
    sub-range (used for soundscape 5s windows) in NATIVE frames, then resample. Returns a 1-D numpy array.
    Full-file reads (training crops) are cached to AUDIO_CACHE_DIR as .npy so epochs 2..N skip .ogg decode."""
    import numpy as np
    import soundfile as sf
    if start is None and _AUDIO_CACHE:                       # cache only full-file decodes
        cp = _cache_path(path)
        if cp.exists():
            try:
                return np.load(cp)
            except Exception:  # noqa: BLE001
                pass
    info = sf.info(str(path))
    file_sr = info.samplerate
    if start is not None:
        s0 = max(0, int(round(start * file_sr)))
        s1 = int(round(stop * file_sr)) if stop is not None else None
        data, _ = sf.read(str(path), start=s0, stop=s1, dtype="float32", always_2d=False)
    else:
        data, _ = sf.read(str(path), dtype="float32", always_2d=False)
    data = np.asarray(data, dtype="float32")
    if data.ndim > 1:                       # stereo → mono
        data = data.mean(axis=1)
    if data.size == 0:
        data = np.zeros(1, dtype="float32")
    out = resample_to(data, file_sr, target_sr)
    if start is None and _AUDIO_CACHE:                       # write cache for next epoch
        try:
            cp = _cache_path(path); cp.parent.mkdir(parents=True, exist_ok=True)
            np.save(cp, out)
        except Exception:  # noqa: BLE001
            pass
    return out


# ════════════════════════════════════════════════════════════ label / class helpers
def _parse_secondary(val):
    """train.csv secondary_labels is a stringified python list (e.g. "['x','y']") or a real list; parse robustly."""
    import ast
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(v) for v in val]
    s = str(val).strip()
    if not s or s in ("[]", "nan", "None"):
        return []
    try:
        out = ast.literal_eval(s)
        return [str(v) for v in out] if isinstance(out, (list, tuple)) else [str(out)]
    except Exception:  # noqa: BLE001
        return [t.strip().strip("'\"") for t in s.strip("[]").split(",") if t.strip()]


def multihot(primary, secondary, class_to_idx):
    """Multi-hot target over the class list: primary + every secondary label that is a scored class."""
    import numpy as np
    y = np.zeros(len(class_to_idx), dtype="float32")
    if primary in class_to_idx:
        y[class_to_idx[primary]] = 1.0
    for s in secondary:
        if s in class_to_idx:
            y[class_to_idx[s]] = 1.0
    return y


def derive_classes(df, sample_submission=None, taxonomy=None, subset=None):
    """The scored class list, in a STABLE order. Priority: sample_submission columns (minus row_id) →
    taxonomy 'primary_label' → sorted unique primary_label in train.csv. `subset` restricts to a species subset
    (small-first ladder)."""
    classes = None
    if sample_submission is not None:
        classes = [c for c in list(sample_submission.columns) if c.lower() not in ("row_id", "filename")]
    elif taxonomy is not None and "primary_label" in taxonomy.columns:
        classes = sorted(taxonomy["primary_label"].astype(str).unique().tolist())
    else:
        classes = sorted(df["primary_label"].astype(str).unique().tolist())
    if subset:
        subset = set(str(s) for s in subset)
        classes = [c for c in classes if c in subset]
    return classes


# ════════════════════════════════════════════════════════════ grouped-by-author K-fold (leak-safe focal CV)
def group_author_kfold(df, n_folds=5, seed=0, group_col="author", label_col="primary_label"):
    """Assign each focal recording a fold so the same AUTHOR never straddles train/val (BirdCLEF's classic
    leak). StratifiedGroupKFold on (label, group) if sklearn present, else a deterministic hash-of-group
    round-robin. Returns an int fold array aligned to df.index order."""
    import numpy as np
    groups = df[group_col].astype(str).fillna("_na").values if group_col in df.columns else \
        np.array([f"row{i}" for i in range(len(df))])
    y = df[label_col].astype(str).values
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold = np.full(len(df), -1, dtype=int)
        for f, (_, val_idx) in enumerate(skf.split(np.zeros(len(df)), y, groups)):
            fold[val_idx] = f
        if (fold < 0).any():                          # sklearn can drop rows if a group is huge; patch them
            fold[fold < 0] = 0
        return fold
    except Exception:  # noqa: BLE001 — deterministic hash round-robin over unique groups
        uniq = sorted(set(groups.tolist()))
        rng = np.random.RandomState(seed); rng.shuffle(uniq)
        gmap = {g: i % n_folds for i, g in enumerate(uniq)}
        return np.array([gmap[g] for g in groups], dtype=int)


# ════════════════════════════════════════════════════════════ official metric (macro ROC-AUC, skip empties)
def macro_roc_auc(y_true, y_pred, return_n=False):
    """OFFICIAL BirdCLEF metric: macro-averaged ROC-AUC over classes, SKIPPING classes with no positive (or no
    negative) in y_true. y_true/y_pred: (N, C). Uses sklearn.roc_auc_score per class. Returns nan if no class
    is scorable."""
    import numpy as np
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:  # noqa: BLE001
        return (float("nan"), 0) if return_n else float("nan")
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        pos = col.sum()
        if pos == 0 or pos == len(col):               # skip: no positive OR no negative (unscorable)
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except Exception:  # noqa: BLE001
            continue
    val = float(np.mean(aucs)) if aucs else float("nan")
    return (val, len(aucs)) if return_n else val


# ════════════════════════════════════════════════════════════ dataset
class _FocalDataset:
    """Torch Dataset: focal recording → random 5s crop waveform (cyclic-pad short) + multi-hot target. Returns
    RAW waveforms; melspec/aug run batched on-GPU in the loop (the winner GPU-mel path)."""
    def __new__(cls, df, audio_dir, class_to_idx, seconds, sr, train=True, seed=0):
        import numpy as np
        import torch
        from torch.utils.data import Dataset

        rows = df.reset_index(drop=True)
        length = int(seconds * sr)

        class DS(Dataset):
            def __len__(self):
                return len(rows)

            def __getitem__(self, i):
                r = rows.iloc[i]
                path = Path(audio_dir) / str(r["filename"])
                try:
                    wav = load_audio(path, target_sr=sr)
                except Exception:  # noqa: BLE001 — unreadable clip → silence, never crash the loader
                    wav = np.zeros(length, dtype="float32")
                wav = torch.as_tensor(wav, dtype=torch.float32)
                mode = "random" if train else "center"
                wav = A.fixed_crop(wav, length, mode=mode, seed=(seed + i) if train else None)
                y = multihot(str(r["primary_label"]), _parse_secondary(r.get("secondary_labels")), class_to_idx)
                return wav, torch.as_tensor(y, dtype=torch.float32)
        return DS()


# ════════════════════════════════════════════════════════════ soundscape validation set (LB-proxy)
def _parse_time(v, default=0.0):
    """Soundscape start/end may be float seconds OR 'HH:MM:SS' timestamps (BirdCLEF-2026). Return seconds."""
    if v is None:
        return default
    s = str(v).strip()
    if ":" in s:
        try:
            parts = [float(p) for p in s.split(":")]
            while len(parts) < 3:
                parts = [0.0] + parts
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        except Exception:  # noqa: BLE001
            return default
    try:
        return float(s)
    except Exception:  # noqa: BLE001
        return default


def _split_labels(v):
    """Soundscape label cell may be multi-label: 'a;b;c' (or comma). Return the list of species codes."""
    if v is None:
        return []
    return [x for x in str(v).replace(";", " ").replace(",", " ").split() if x and x.lower() != "nocall"]


def _load_soundscape_val(sdf, sdir, class_to_idx, seconds, sr, limit=None):
    """Load labeled 5s soundscape windows → (X_wavs list, Y_true multihot). Each labeled window = one row.
    Handles HH:MM:SS start/end and ';'-separated multi-label cells (BirdCLEF-2026 soundscape format)."""
    import numpy as np
    import torch
    rows = sdf if limit is None else sdf.head(int(limit))
    wavs, ys = [], []
    length = int(seconds * sr)
    for _, r in rows.iterrows():
        fn = str(r["filename"])
        path = Path(sdir) / fn
        if not path.exists() and not fn.endswith(".ogg"):
            path = Path(sdir) / (fn + ".ogg")
        start = _parse_time(r.get("start"), 0.0)
        end = _parse_time(r.get("end"), start + seconds)
        try:
            wav = load_audio(path, target_sr=sr, start=start, stop=end)
        except Exception:  # noqa: BLE001
            wav = np.zeros(length, dtype="float32")
        wav = A.fixed_crop(torch.as_tensor(wav, dtype=torch.float32), length, mode="center")
        wavs.append(wav)
        labs = _split_labels(r.get("primary_label"))
        prim = labs[0] if labs else "nocall"
        ys.append(multihot(prim, labs[1:], class_to_idx))
    if not wavs:
        return None, None
    return torch.stack(wavs), np.stack(ys)


# ════════════════════════════════════════════════════════════ the trainer
def train(spec, log=lambda *a, **k: None):
    """Run a training campaign from `spec`. Returns a results dict. All heavy deps imported lazily so the
    module import stays light for the fleet registry."""
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import pandas as pd

    dev = spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    hw = HW.load_config()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(hw.get("amp_dtype"), None)
    use_amp = bool(amp_dtype) and dev == "cuda"
    if hw.get("allow_tf32") and dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed = int(spec.get("seed", 0))
    torch.manual_seed(seed); np.random.seed(seed)
    seconds = float(spec.get("seconds", 5.0))
    sr = int(spec.get("sr", SR))
    epochs = int(spec.get("epochs", 1 if spec.get("dry_run") else 10))
    bs = int(spec.get("batch_size", 8 if spec.get("dry_run") else 32))
    n_mels = int(spec.get("n_mels", 128))
    n_fft = int(spec.get("n_fft", 1024))
    hop = int(spec.get("hop", 500))
    fmin = float(spec.get("fmin", 40.0)); fmax = spec.get("fmax", 15000.0)
    arch = spec.get("arch", "tf_efficientnet_b0")
    out_dir = Path(spec.get("out_dir", "config/_auto/audio_train")); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    df = pd.read_csv(spec["train_csv"])
    if "filename" not in df.columns:
        raise ValueError("train_csv must have a 'filename' column")
    sample_sub = pd.read_csv(spec["sample_submission"]) if spec.get("sample_submission") else None
    taxonomy = pd.read_csv(spec["taxonomy"]) if spec.get("taxonomy") else None
    classes = spec.get("class_list") or derive_classes(df, sample_sub, taxonomy, subset=spec.get("classes"))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # small-first subset ladder: restrict to the class subset, then cap recordings/class, then a hard limit
    if spec.get("classes"):
        df = df[df["primary_label"].astype(str).isin(set(classes))].reset_index(drop=True)
    if spec.get("per_class"):
        pc = int(spec["per_class"])
        df = pd.concat([g.sample(n=min(len(g), pc), random_state=seed)
                        for _, g in df.groupby("primary_label")], ignore_index=True)
    if spec.get("limit"):
        df = df.head(int(spec["limit"])).reset_index(drop=True)

    # ---- CV: grouped-by-author focal split (secondary, leak-safe) ----
    folds_info = None
    focal_val_df = None
    if spec.get("cv"):
        n_folds = int(spec.get("n_folds", 5)); fold = int(spec.get("fold", 0))
        fold_arr = group_author_kfold(df, n_folds=n_folds, seed=seed)
        df = df.assign(_fold=fold_arr)
        focal_val_df = df[df["_fold"] == fold].reset_index(drop=True)
        train_df = df[df["_fold"] != fold].reset_index(drop=True)
        folds_info = {"n_folds": n_folds, "held_fold": fold, "group_col": "author",
                      "assignment": {str(r["filename"]): int(r["_fold"]) for _, r in df.iterrows()}}
        (out_dir / "folds.json").write_text(json.dumps(folds_info, indent=1))
        if len(train_df) == 0:                        # tiny data → don't starve training
            train_df, focal_val_df = df.reset_index(drop=True), None
    else:
        train_df = df.reset_index(drop=True)

    # ---- sampler: class-balance count^-0.5 on primary_label ----
    prim = train_df["primary_label"].astype(str).values
    weights = IMB.sample_weights(prim, power=float(spec.get("sampler_power", -0.5)))
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(train_df),
                                    replacement=True)
    ds = _FocalDataset(train_df, spec["audio_dir"], class_to_idx, seconds, sr, train=True, seed=seed)
    _nw = int(spec.get("num_workers", 0))
    loader = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=_nw, drop_last=False,
                        pin_memory=(dev == "cuda"),
                        persistent_workers=(_nw > 0), prefetch_factor=(4 if _nw > 0 else None))

    # ---- model / optim / EMA ----
    _pre = bool(spec.get("pretrained", False))  # local training has internet → ImageNet-pretrained is the big lever
    model, backend = A.build_mel_backbone(len(classes), in_ch=1, arch=arch, pretrained=_pre, device=dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(spec.get("lr", 1e-3)),
                            weight_decay=float(spec.get("weight_decay", 1e-4)))
    steps = max(1, epochs * max(1, len(loader)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ema = TT.ModelEMA(model, decay=float(spec.get("ema_decay", 0.999)), warmup=10, device=dev)
    label_smooth = float(spec.get("label_smoothing", 0.0))
    focal = bool(spec.get("focal_loss", False))
    mixup_p = float(spec.get("mixup_p", 0.5))
    mixup_alpha = float(spec.get("mixup_alpha", 0.5))
    specaug_p = float(spec.get("specaug_p", 0.5))
    wave_noise = float(spec.get("wave_noise", 0.0 if spec.get("dry_run") else 0.005))
    max_steps = spec.get("max_steps")

    def mel_of(wav):
        m = A.log_mel_spectrogram(wav, sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels, fmin=fmin, fmax=fmax,
                                  to_db=True, normalize=True, device=dev)
        return m                                       # (B, n_mels, T)

    def loss_fn(logits, target):
        if focal:
            return TT.binary_focal_loss(logits, target, alpha=0.25, gamma=2.0, reduction="mean")
        if label_smooth > 0:
            target = target * (1.0 - label_smooth) + 0.5 * label_smooth
        return nn.functional.binary_cross_entropy_with_logits(logits, target)

    curve = []
    best = {"val_auc": float("-inf"), "epoch": -1}
    gstep = 0
    for ep in range(epochs):
        model.train(); ep_losses = []
        for wav, y in loader:
            wav = wav.to(dev).float(); y = y.to(dev).float()
            if wave_noise > 0:                         # waveform-domain aug (before front-end)
                wav = A.waveform_augment(wav, gaussian=wave_noise, seed=seed + gstep)
            if mixup_p > 0 and float(torch.rand(1).item()) < mixup_p and wav.size(0) > 1:
                perm = torch.randperm(wav.size(0), device=dev)
                lam = float(np.random.RandomState(seed + gstep).beta(mixup_alpha, mixup_alpha))
                wav = lam * wav + (1.0 - lam) * wav[perm]
                y = torch.maximum(y, y[perm])          # OR-rule multi-label mixing (grounded)
            mel = mel_of(wav)
            if specaug_p > 0:                          # SpecAugment per-sample (diverse masks)
                mel = torch.stack([A.spec_augment(mel[b], p=specaug_p, seed=seed + gstep + b)
                                   for b in range(mel.size(0))])
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(mel); loss = loss_fn(logits, y)
                loss.backward()
            else:
                logits = model(mel); loss = loss_fn(logits, y); loss.backward()
            opt.step(); sched.step(); ema.update(model)
            ep_losses.append(float(loss.detach().cpu()))
            gstep += 1
            if max_steps and gstep >= int(max_steps):
                break

        # ---- validation (EMA weights) ----
        eval_model, _ = A.build_mel_backbone(len(classes), in_ch=1, arch=arch, pretrained=False, device=dev)  # weights come from EMA state_dict below
        eval_model.load_state_dict(ema.module.state_dict()); eval_model.eval()
        val_auc = _validate_soundscape(spec, eval_model, class_to_idx, seconds, sr, n_fft, hop, n_mels,
                                       fmin, fmax, dev, mel_of)
        focal_auc = _validate_focal(focal_val_df, spec, eval_model, class_to_idx, seconds, sr, dev, mel_of,
                                    bs) if focal_val_df is not None and len(focal_val_df) else float("nan")
        score = val_auc if not (isinstance(val_auc, float) and math.isnan(val_auc)) else focal_auc
        row = {"epoch": ep, "train_loss": float(np.mean(ep_losses)) if ep_losses else float("nan"),
               "val_auc": val_auc, "focal_val_auc": focal_auc}
        curve.append(row)
        log(f"  [audio-train] epoch {ep}: loss={row['train_loss']:.4f} val_auc={val_auc} focal={focal_auc}")
        if isinstance(score, float) and not math.isnan(score) and score > best["val_auc"]:
            best = {"val_auc": score, "epoch": ep}
            ckpt = {"state_dict": eval_model.state_dict(), "classes": classes,
                    "melspec_cfg": {"sr": sr, "n_fft": n_fft, "hop": hop, "n_mels": n_mels,
                                    "fmin": fmin, "fmax": fmax, "to_db": True, "normalize": True},
                    "seconds": seconds, "arch": arch, "in_ch": 1, "backend": backend}
            torch.save(ckpt, out_dir / "best.pt")

    if best["epoch"] < 0:                              # never got a scorable val → still save last for infer
        torch.save({"state_dict": ema.module.state_dict(), "classes": classes,
                    "melspec_cfg": {"sr": sr, "n_fft": n_fft, "hop": hop, "n_mels": n_mels,
                                    "fmin": fmin, "fmax": fmax, "to_db": True, "normalize": True},
                    "seconds": seconds, "arch": arch, "in_ch": 1, "backend": backend}, out_dir / "best.pt")

    return {"best_val_auc": None if best["val_auc"] == float("-inf") else best["val_auc"],
            "focal_val_auc": curve[-1]["focal_val_auc"] if curve else float("nan"),
            "best_epoch": best["epoch"], "epochs": epochs, "ckpt_path": str(out_dir / "best.pt"),
            "curve": curve, "n_classes": len(classes), "n_train": len(train_df),
            "backend": backend, "device": str(dev), "amp": hw.get("amp_dtype", "off"),
            "folds": str(out_dir / "folds.json") if folds_info else None}


def _predict(model, wavs, mel_of, dev, bs=32):
    """Batched sigmoid predictions for a stack of waveforms → (N, C) numpy probs."""
    import numpy as np
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(wavs), bs):
            batch = wavs[i:i + bs].to(dev).float()
            mel = mel_of(batch)
            logits = model(mel)
            out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype="float32")


def _validate_soundscape(spec, model, class_to_idx, seconds, sr, n_fft, hop, n_mels, fmin, fmax, dev, mel_of):
    """PRIMARY (LB-proxy) validation on train_soundscapes_labels.csv."""
    import pandas as pd
    if not (spec.get("soundscape_csv") and spec.get("soundscape_dir")):
        return float("nan")
    sdf = pd.read_csv(spec["soundscape_csv"])
    X, Ytrue = _load_soundscape_val(sdf, spec["soundscape_dir"], class_to_idx, seconds, sr,
                                    limit=spec.get("val_limit"))
    if X is None:
        return float("nan")
    P = _predict(model, X, mel_of, dev)
    return macro_roc_auc(Ytrue, P)


def _validate_focal(focal_df, spec, model, class_to_idx, seconds, sr, dev, mel_of, bs):
    """SECONDARY leak-safe validation on the held-out author-grouped focal recordings."""
    import numpy as np
    import torch
    ds = _FocalDataset(focal_df, spec["audio_dir"], class_to_idx, seconds, sr, train=False)
    wavs, ys = [], []
    for i in range(len(ds)):
        w, y = ds[i]; wavs.append(w); ys.append(y.numpy())
    if not wavs:
        return float("nan")
    P = _predict(model, torch.stack(wavs), mel_of, dev, bs=bs)
    return macro_roc_auc(np.stack(ys), P)


# ════════════════════════════════════════════════════════════ agent
class AudioTrain(BaseAgent):
    name = "audio-train"
    thread = "M"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        # empty / underspecified spec → escalate CLEAN (never train on nothing; keeps the smoke gate honest)
        need = [k for k in ("train_csv", "audio_dir") if not spec.get(k)]
        if need:
            return self.escalate(worker, "leader",
                                 f"[{worker}] audio-train needs spec keys {need} (train_csv + audio_dir, plus "
                                 f"sample_submission/taxonomy for the class list and soundscape_csv/dir for the "
                                 f"LB-proxy CV). Ladder: classes/per_class/seconds/epochs for small-first. Not training on empty spec.")
        if BaseAgentGPUHold():
            return self.escalate(worker, "leader",
                                 f"[{worker}] audio-train: GPU training is HELD (gpu_train_hold.flag). "
                                 f"Clear the hold (set 5090 power cap first) before a full run.")
        try:
            res = train(spec, log=lambda m: self.post(worker, "leader", m, routine=True, kind="finding"))
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] audio-train FAILED ({type(e).__name__}: {str(e)[:200]})")
        bv = res.get("best_val_auc"); fv = res.get("focal_val_auc")
        msg = (f"[{worker}] **AUDIO-TRAIN** [{res['backend']}] {res['n_classes']} classes, {res['n_train']} recs, "
               f"{res['epochs']} ep (amp={res['amp']}) → best_val_auc(soundscape LB-proxy)={bv} "
               f"focal_val_auc(author-grouped)={fv}; ckpt {res['ckpt_path']}")
        self.log(msg, kind="verdict",
                 recommendation="soundscape best_val_auc is the LB tracker; feed ckpt to audio-infer for the CPU "
                                "submission notebook. Grow the ladder (classes/per_class/epochs) once val AUC moves.")
        if isinstance(bv, (int, float)):
            self.record(change="audio-train (compressed EfficientNet-b0 mel classifier)", cv=bv,
                        description=f"{res['n_classes']}-class multi-label, soundscape macro-AUC",
                        script="fleet_agents/audio_train.py")
        return self.done(res, msg, to="leader")


def BaseAgentGPUHold():
    try:
        from .base import gpu_train_held
        return gpu_train_held()
    except Exception:  # noqa: BLE001
        return False


_AGENT = AudioTrain()


def run(q, worker):
    return _AGENT.run(q, worker)
