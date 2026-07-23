"""kaggle-submit — the REUSABLE offline-notebook (code-competition) submission pipeline. Repeats on every new
best model, so it is a real fleet agent, not a one-off: given {ckpt, competition, dataset_slug, message,
sample_submission, inference} it (1) PACKAGES the checkpoint as a private Kaggle dataset (create or version),
(2) GENERATES a fully SELF-CONTAINED offline inference notebook (.ipynb) that inlines the mel front-end + model
rebuild + sliding-window inference + submission writing (NO import of our fleet package on Kaggle), (3) PUSHES
the kernel with the model dataset + competition attached (CPU, no internet), (4) SUBMITS the kernel output, and
(5) READS BACK the public AND private LB score from `kaggle competitions submissions`.

Design notes / honesty gates:
  • The Kaggle CLI is called ONLY through `_run_cli`, so the data-wise test can monkeypatch it and NEVER hits
    the real Kaggle API. Score parsing is a PURE function (`parse_submissions`) tested directly.
  • The generated notebook DISCOVERS input paths by content (sample_submission for column order, test_soundscapes
    dir, the mounted checkpoint) — it does not hardcode absolute Kaggle paths beyond /kaggle/input/<competition>/.
    It reads INPUT_ROOT/WORKING_DIR env overrides so the EXACT notebook body runs locally in the dry-run.
  • Fallback: if test_soundscapes is empty at commit time (it is until scoring), the notebook still writes a
    header-correct submission (copied from sample_submission) so the notebook commits clean.
  • `inference='audio'` reproduces fleet_agents.audio_pack/audio_infer math inline. Other modalities escalate
    (never fabricate an untested body).

Spec: ckpt, competition, dataset_slug, message, [sample_submission, inference='audio', batch_size, threads,
smooth, kernel_slug, ckpt_name, wait, submit, dry_run_files]. Data-wise test: test_fleet_agents/kaggle_submit_test.py.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .base import BaseAgent, COMP

KAGGLE_BIN = os.environ.get("KAGGLE_BIN", "/home/seshu/miniconda3/envs/llm/bin/kaggle")


# ════════════════════════════════════════════════════════════ CLI indirection (mocked in tests)
def _run_cli(args, timeout=1800, cwd=None):
    """Run the Kaggle CLI. ISOLATED so the data-wise test monkeypatches it — the test never touches real Kaggle.
    Returns (returncode, stdout, stderr)."""
    cmd = [KAGGLE_BIN] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


# ════════════════════════════════════════════════════════════ 1. the self-contained AUDIO notebook body
# Static python body of the offline notebook. A `CONFIG = {...}` dict is prepended at generation time; this body
# reads it. Reproduces fleet_agents.audio_pack math inline (mel filterbank + log-mel + timm backbone) so the
# notebook has NO dependency on our fleet package on Kaggle. INPUT_ROOT/WORKING_DIR env overrides let the exact
# same body run in the local dry-run.
_AUDIO_BODY = r'''
import os, glob, math, time, shutil
import numpy as np
import pandas as pd

# guarded imports — verify the Kaggle image has them (torch/timm/soundfile are pre-installed offline)
import torch
import soundfile as sf
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
WORKING_DIR = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")
COMP_DIR = os.path.join(INPUT_ROOT, CONFIG["competition"])
DEVICE = "cpu"
torch.set_num_threads(int(CONFIG.get("threads", 4)))


# ---- discover input paths BY CONTENT (no hardcoded absolute paths beyond /kaggle/input/<comp>/) ----
def _first_glob(*patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None

def find_sample_submission():
    return _first_glob(os.path.join(COMP_DIR, "**", "sample_submission.csv"),
                       os.path.join(INPUT_ROOT, "**", "sample_submission.csv"))

def find_test_dir():
    for pat in (os.path.join(COMP_DIR, "**", "test_soundscapes"),
                os.path.join(INPUT_ROOT, "**", "test_soundscapes")):
        for p in sorted(glob.glob(pat, recursive=True)):
            if os.path.isdir(p):
                return p
    return os.path.join(COMP_DIR, "test_soundscapes")

def find_ckpt():
    name = CONFIG.get("ckpt_name", "best.pt")
    return _first_glob(os.path.join(INPUT_ROOT, "**", name),
                       os.path.join(INPUT_ROOT, "**", "*.pt"))


# ---- mel filterbank + log-mel spectrogram (pure numpy/torch, Slaney; identical to fleet audio_pack) ----
def _hz_to_mel(f):
    f = np.atleast_1d(np.asarray(f, dtype="float64"))
    f_sp = 200.0 / 3
    mel = f / f_sp
    min_log_hz, min_log_mel = 1000.0, 1000.0 / f_sp
    logstep = math.log(6.4) / 27.0
    above = f >= min_log_hz
    mel[above] = min_log_mel + np.log(f[above] / min_log_hz) / logstep
    return mel

def _mel_to_hz(m):
    m = np.atleast_1d(np.asarray(m, dtype="float64"))
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_mel = 1000.0 / f_sp
    logstep = math.log(6.4) / 27.0
    above = m >= min_log_mel
    freqs[above] = 1000.0 * np.exp(logstep * (m[above] - min_log_mel))
    return freqs

def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    fmax = fmax or sr / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_freqs)
    m_min, m_max = _hz_to_mel(fmin)[0], _hz_to_mel(fmax)[0]
    mel_pts = np.linspace(float(m_min), float(m_max), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_freqs), dtype="float32")
    for i in range(n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0.0, None)
        enorm = 2.0 / max(hi - lo, 1e-9)
        fb[i] *= enorm
    return fb

def log_mel_spectrogram(wav, sr, n_fft, hop, n_mels, fmin, fmax, to_db=True, top_db=80.0, normalize=True):
    if not torch.is_tensor(wav):
        wav = torch.as_tensor(np.asarray(wav, dtype="float32"))
    wav = wav.to(DEVICE).float()
    squeeze = wav.dim() == 1
    if squeeze:
        wav = wav.unsqueeze(0)
    window = torch.hann_window(n_fft, device=DEVICE)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window,
                      center=True, return_complex=True, pad_mode="reflect")
    p = spec.abs() ** 2.0
    fb = torch.as_tensor(mel_filterbank(sr, n_fft, n_mels, fmin, fmax), device=DEVICE)
    mel = torch.einsum("mf,bft->bmt", fb, p)
    if to_db:
        mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        mel = torch.clamp(mel, min=mel.amax(dim=(-2, -1), keepdim=True) - top_db)
    if normalize:
        mu = mel.mean(dim=(-2, -1), keepdim=True)
        sd = mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        mel = (mel - mu) / sd
    return mel[0] if squeeze else mel


# ---- model rebuild (timm EfficientNet on 1-ch mel via a channel adapter; identical to fleet build_mel_backbone) ----
class _MelChannelAdapter(torch.nn.Module):
    def __init__(self, net, expect_ch):
        super().__init__()
        self.net = net
        self.expect_ch = expect_ch
    def forward(self, x):
        if x.dim() == 3 and self.expect_ch == 1:
            x = x.unsqueeze(1)
        return self.net(x)

class _SmallMelCNN(torch.nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        from torch import nn
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.body = nn.Sequential(blk(in_ch, 16), blk(16, 32), blk(32, 64), blk(64, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n_classes)
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.body(x)
        return self.fc(self.pool(x).flatten(1))

def build_model(n_classes, in_ch, arch):
    if _HAS_TIMM:
        try:
            net = timm.create_model(arch, pretrained=False, in_chans=in_ch, num_classes=n_classes)
            return _MelChannelAdapter(net, in_ch)
        except Exception:
            pass
    return _SmallMelCNN(in_ch, n_classes)


# ---- audio loading + sliding windows ----
def resample_to(wav, sr, target_sr):
    if sr == target_sr:
        return np.asarray(wav, dtype="float32")
    try:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), int(target_sr))
        return resample_poly(np.asarray(wav, dtype="float64"), target_sr // g, sr // g).astype("float32")
    except Exception:
        n_out = int(round(len(wav) * target_sr / sr))
        if n_out <= 1:
            return np.asarray(wav, dtype="float32")
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, np.asarray(wav, dtype="float64")).astype("float32")

def load_audio(path, target_sr):
    info = sf.info(str(path))
    data, _ = sf.read(str(path), dtype="float32", always_2d=False)
    data = np.asarray(data, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.size == 0:
        data = np.zeros(1, dtype="float32")
    return resample_to(data, info.samplerate, target_sr)

def sliding_windows(wav, length):
    T = wav.shape[-1]
    if T <= length:
        reps = length // max(T, 1) + 1
        wav = np.tile(wav, reps)[:length]
        return wav[None, :]
    starts = list(range(0, T - length + 1, length))
    if starts[-1] != T - length:
        starts.append(T - length)
    return np.stack([wav[s:s + length] for s in starts], axis=0)

def neighbor_smooth(P, kernel=(0.1, 0.2, 0.4, 0.2, 0.1)):
    import torch.nn.functional as F
    t = torch.as_tensor(P, dtype=torch.float32)
    k = torch.as_tensor(kernel, dtype=torch.float32); k = k / k.sum()
    N, C = t.shape
    x = t.t().unsqueeze(0)
    pad = len(kernel) // 2
    x = F.pad(x, (pad, pad), mode="reflect")
    w = k.view(1, 1, -1).expand(C, 1, -1)
    return F.conv1d(x, w, groups=C).squeeze(0).t().numpy()


# ════════════════════════════════════════════════════════════ main
def main():
    sample_path = find_sample_submission()
    if sample_path is None:
        raise FileNotFoundError("sample_submission.csv not found under /kaggle/input")
    sample = pd.read_csv(sample_path)
    sub_cols = [c for c in sample.columns if str(c).lower() != "row_id"]
    col_idx = {str(c): j for j, c in enumerate(sub_cols)}

    os.makedirs(WORKING_DIR, exist_ok=True)
    out_path = os.path.join(WORKING_DIR, "submission.csv")

    test_dir = find_test_dir()
    exts = (".ogg", ".wav", ".flac")
    files = []
    if os.path.isdir(test_dir):
        files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir)
                        if f.lower().endswith(exts)])
    if CONFIG.get("limit"):
        files = files[:int(CONFIG["limit"])]

    # FALLBACK: no test audio at commit time → write a header-correct submission from sample_submission
    if not files:
        sample.to_csv(out_path, index=False)
        print("no test audio found -> wrote header-correct fallback submission:", out_path, sample.shape)
        return

    ckpt_path = find_ckpt()
    if ckpt_path is None:
        raise FileNotFoundError("checkpoint (*.pt) not found under /kaggle/input")
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    classes = [str(c) for c in ck["classes"]]
    cfg = dict(ck["melspec_cfg"])
    seconds = float(ck.get("seconds", 5.0)); sr = int(cfg["sr"])
    in_ch = int(ck.get("in_ch", 1)); arch = ck.get("arch", "tf_efficientnet_b0")
    model = build_model(len(classes), in_ch, arch)
    model.load_state_dict(ck["state_dict"]); model.eval()
    model_pos = {c: i for i, c in enumerate(classes)}
    length = int(round(seconds * sr))
    batch_size = int(CONFIG.get("batch_size", 16))
    smooth = bool(CONFIG.get("smooth", False))
    fill = float(CONFIG.get("fill", 0.0))

    rows, row_ids = [], []
    t0 = time.time(); n_windows = 0
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            wav = load_audio(path, sr)
        except Exception:
            continue
        wins = sliding_windows(wav, length)
        probs = []
        with torch.no_grad():
            for i in range(0, wins.shape[0], batch_size):
                batch = torch.as_tensor(wins[i:i + batch_size], dtype=torch.float32)
                mel = log_mel_spectrogram(batch, sr=sr, n_fft=int(cfg["n_fft"]), hop=int(cfg["hop"]),
                                          n_mels=int(cfg["n_mels"]), fmin=float(cfg["fmin"]),
                                          fmax=cfg.get("fmax"), to_db=bool(cfg.get("to_db", True)),
                                          normalize=bool(cfg.get("normalize", True)))
                logits = model(mel)
                probs.append(torch.sigmoid(logits).float().cpu().numpy())
        P = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(classes)), dtype="float32")
        if smooth and P.shape[0] >= 3:
            P = neighbor_smooth(P)
        n_windows += P.shape[0]
        for w in range(P.shape[0]):
            end_sec = int(round((w + 1) * seconds))
            row = np.full(len(sub_cols), fill, dtype="float32")
            for c, mi in model_pos.items():
                j = col_idx.get(c)
                if j is not None:
                    row[j] = P[w, mi]
            rows.append(row); row_ids.append(f"{stem}_{end_sec}")

    arr = np.clip(np.stack(rows), 0.0, 1.0) if rows else np.zeros((0, len(sub_cols)), dtype="float32")
    sub = pd.DataFrame(arr, columns=sub_cols)
    sub.insert(0, "row_id", row_ids)
    sub = sub[["row_id"] + sub_cols]
    sub.to_csv(out_path, index=False)
    dt = time.time() - t0
    wps = (n_windows / dt) if dt > 0 else 0.0
    print(f"wrote {out_path}: {len(sub)} rows / {len(files)} files / {n_windows} windows "
          f"@ {wps:.1f} win/s (CPU); {len(sub.columns)} cols")


main()
'''


def build_notebook_source(spec) -> str:
    """Return the python source of the self-contained offline notebook for `inference` modality. Prepends a
    CONFIG dict (json literal) so the static body needs no .format() over its f-strings/dict braces."""
    inference = spec.get("inference", "audio")
    if inference != "audio":
        raise ValueError(f"kaggle-submit: inference='{inference}' has no self-contained notebook body yet "
                         f"(only 'audio' is implemented; add a body before submitting it).")
    cfg = {
        "competition": spec["competition"],
        "ckpt_name": spec.get("ckpt_name", "best.pt"),
        "batch_size": int(spec.get("batch_size", 16)),
        "threads": int(spec.get("threads", 4)),
        "smooth": bool(spec.get("smooth", False)),
        "fill": float(spec.get("fill", 0.0)),
        "limit": spec.get("limit"),
    }
    return "CONFIG = " + repr(cfg) + "\n" + _AUDIO_BODY


def write_notebook(source: str, path) -> str:
    """Wrap the python source into a single-code-cell Jupyter notebook (.ipynb) and write it."""
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(nb, indent=1))
    return str(p)


# ════════════════════════════════════════════════════════════ 2. dataset packaging
def package_dataset(ckpt, dataset_slug, workdir, log=lambda *a: None):
    """Stage best.pt + metadata.json under `workdir` and `kaggle datasets create` (or `version` if it exists).
    Returns a dict {slug, action, ok, stdout}."""
    import torch
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    dst = workdir / Path(ckpt).name
    shutil.copy2(ckpt, dst)
    # tiny metadata sidecar (classes + melspec_cfg) so the dataset is self-describing
    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        meta = {"classes": [str(c) for c in ck.get("classes", [])], "melspec_cfg": ck.get("melspec_cfg"),
                "seconds": ck.get("seconds"), "arch": ck.get("arch"), "in_ch": ck.get("in_ch")}
        (workdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    except Exception:  # noqa: BLE001
        pass
    title = dataset_slug.split("/")[-1].replace("-", " ")[:50]
    (workdir / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": dataset_slug,
                                                               "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    # does it already exist? → version, else create
    rc, out, err = _run_cli(["datasets", "status", dataset_slug], timeout=120)
    exists = rc == 0 and "error" not in (out + err).lower() and "not found" not in (out + err).lower()
    if exists:
        rc, out, err = _run_cli(["datasets", "version", "-p", str(workdir), "-m",
                                 "update checkpoint", "-r", "zip"], timeout=1800, cwd=str(workdir))
        action = "version"
    else:
        rc, out, err = _run_cli(["datasets", "create", "-p", str(workdir), "-r", "zip"], timeout=1800, cwd=str(workdir))
        action = "create"
    ok = rc == 0
    log(f"dataset {action} {dataset_slug}: rc={rc} {(out or err)[:200]}")
    return {"slug": dataset_slug, "action": action, "ok": ok, "stdout": (out or "") + (err or "")}


# ════════════════════════════════════════════════════════════ 3. kernel push
def push_kernel(nb_path, kernel_slug, competition, dataset_slug, workdir, log=lambda *a: None):
    """Stage the notebook + kernel-metadata.json (CPU, no internet, competition + dataset attached) and push.
    Returns {slug, ok, stdout}."""
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    nb_name = Path(nb_path).name
    dst = workdir / nb_name
    if Path(nb_path).resolve() != dst.resolve():
        shutil.copy2(nb_path, dst)
    meta = {
        "id": kernel_slug,
        "title": kernel_slug.split("/")[-1],
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [dataset_slug],
        "competition_sources": [competition],
        "kernel_sources": [],
    }
    (workdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    rc, out, err = _run_cli(["kernels", "push", "-p", str(workdir)], timeout=1800, cwd=str(workdir))
    ok = rc == 0
    log(f"kernel push {kernel_slug}: rc={rc} {(out or err)[:200]}")
    return {"slug": kernel_slug, "ok": ok, "stdout": (out or "") + (err or "")}


def wait_for_kernel(kernel_slug, timeout=5400, poll=30, log=lambda *a: None):
    """Poll `kaggle kernels status` until the run completes/errors. Returns {status, ok}."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rc, out, err = _run_cli(["kernels", "status", kernel_slug], timeout=120)
        blob = (out + err).lower()
        if "complete" in blob:
            return {"status": "complete", "ok": True}
        if "error" in blob or "cancelacknowledged" in blob or "cancelrequested" in blob:
            return {"status": "error", "ok": False, "stdout": out + err}
        log(f"kernel {kernel_slug} status: {out.strip()[:120]}")
        time.sleep(poll)
    return {"status": "timeout", "ok": False}


# ════════════════════════════════════════════════════════════ 4. submit + read back score
def submit(competition, kernel_slug, message, submission_file="submission.csv", version=None,
           file_path=None, log=lambda *a: None):
    """Submit to a code competition (kernel output) or a plain file. Returns {ok, stdout}."""
    if file_path:
        args = ["competitions", "submit", competition, "-f", str(file_path), "-m", message]
    else:
        args = ["competitions", "submit", competition, "-k", kernel_slug, "-f", submission_file, "-m", message]
        if version:
            args += ["-v", str(version)]
    rc, out, err = _run_cli(args, timeout=1800)
    blob = (out or "") + (err or "")
    daily_limit = "maximum" in blob.lower() and "submission" in blob.lower()
    ok = rc == 0 and not daily_limit
    log(f"submit {competition}: rc={rc} daily_limit={daily_limit} {blob[:200]}")
    return {"ok": ok, "daily_limit": daily_limit, "stdout": blob}


def parse_submissions(csv_text, message=None):
    """PURE parser: `kaggle competitions submissions --format csv` → the newest (or `message`-matching) row's
    public & private score. Returns {public, private, submission_id, description, status}. Tested directly."""
    import csv as _csv
    import io
    lines = [ln for ln in csv_text.splitlines() if ln.strip() and "Next Page Token" not in ln]
    if not lines:
        return {"public": None, "private": None, "submission_id": None}
    reader = list(_csv.DictReader(io.StringIO("\n".join(lines))))
    if not reader:
        return {"public": None, "private": None, "submission_id": None}

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _get(row, *keys):
        for k in row:
            kl = k.lower().replace(" ", "").replace("_", "")
            for want in keys:
                if kl == want:
                    return row[k]
        return None

    chosen = None
    if message:
        for row in reader:
            if (_get(row, "description") or "") == message:
                chosen = row; break
    chosen = chosen or reader[0]                       # CLI lists newest first
    return {
        "public": _num(_get(chosen, "publicscore")),
        "private": _num(_get(chosen, "privatescore")),
        "submission_id": _get(chosen, "ref", "submissionid", "fileref", "filename", "date"),
        "description": _get(chosen, "description"),
        "status": _get(chosen, "status"),
    }


def read_scores(competition, message=None, log=lambda *a: None):
    """Fetch submissions and parse OUR submission's public+private score. Returns the parse dict (+ raw)."""
    rc, out, err = _run_cli(["competitions", "submissions", competition, "--format", "csv"], timeout=300)
    parsed = parse_submissions(out or "", message=message)
    parsed["rc"] = rc
    log(f"scores {competition}: {parsed.get('public')}/{parsed.get('private')}")
    return parsed


# ════════════════════════════════════════════════════════════ local dry-run of the EXACT notebook body
def dry_run(spec, sample_submission, test_dir, ckpt, files=3, workdir=None):
    """Execute the GENERATED notebook source locally against a stand-in test dir (INPUT_ROOT/WORKING_DIR env
    overrides) → assert it writes a valid submission. Returns {ok, out_path, shape, min, max}."""
    import tempfile
    import pandas as pd
    workdir = Path(workdir or tempfile.mkdtemp(prefix="ksubmit_dry_"))
    inroot = workdir / "input"; comp_dir = inroot / spec["competition"]; comp_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = inroot / "model"; ds_dir.mkdir(parents=True, exist_ok=True)
    working = workdir / "working"; working.mkdir(parents=True, exist_ok=True)
    # stage sample_submission + a stand-in test_soundscapes + the checkpoint
    shutil.copy2(sample_submission, comp_dir / "sample_submission.csv")
    ss = comp_dir / "test_soundscapes"; ss.mkdir(exist_ok=True)
    src_files = sorted([p for p in Path(test_dir).iterdir() if p.suffix.lower() in (".ogg", ".wav", ".flac")])[:files]
    for p in src_files:
        try:
            os.symlink(p, ss / p.name)
        except Exception:  # noqa: BLE001
            shutil.copy2(p, ss / p.name)
    shutil.copy2(ckpt, ds_dir / spec.get("ckpt_name", "best.pt"))

    dspec = dict(spec); dspec["limit"] = files
    source = build_notebook_source(dspec)
    env = dict(os.environ)
    env["KAGGLE_INPUT_ROOT"] = str(inroot); env["KAGGLE_WORKING_DIR"] = str(working)
    g = {"__name__": "__ksubmit_dry__"}
    old = {k: os.environ.get(k) for k in ("KAGGLE_INPUT_ROOT", "KAGGLE_WORKING_DIR")}
    os.environ.update({"KAGGLE_INPUT_ROOT": str(inroot), "KAGGLE_WORKING_DIR": str(working)})
    try:
        exec(compile(source, "<notebook>", "exec"), g, g)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    out_path = working / "submission.csv"
    if not out_path.exists():
        return {"ok": False, "reason": "no submission.csv written"}
    df = pd.read_csv(out_path)
    sample = pd.read_csv(sample_submission)
    cols_ok = list(df.columns) == list(sample.columns)
    body = df[[c for c in df.columns if c != "row_id"]].values
    rng_ok = (len(body) == 0) or (float(body.min()) >= 0.0 and float(body.max()) <= 1.0)
    return {"ok": bool(cols_ok and rng_ok and len(df) > 0), "out_path": str(out_path),
            "shape": list(df.shape), "columns": len(df.columns), "cols_ok": cols_ok, "range_ok": rng_ok,
            "min": float(body.min()) if len(body) else None, "max": float(body.max()) if len(body) else None}


# ════════════════════════════════════════════════════════════ orchestration
def submit_pipeline(spec, log=lambda *a: None):
    """Full pipeline: package dataset → generate+write notebook → (dry-run) → push kernel → wait → submit →
    read scores. Each stage is guarded; `spec['submit']=False` stops before the live submit (dry only)."""
    import tempfile
    comp = spec["competition"]; slug = spec["dataset_slug"]; msg = spec["message"]
    kernel_slug = spec.get("kernel_slug") or f"{slug.split('/')[0]}/{comp}-agents-infer"
    workroot = Path(spec.get("workdir") or tempfile.mkdtemp(prefix="ksubmit_"))
    res = {"competition": comp, "dataset_slug": slug, "kernel_slug": kernel_slug}

    # notebook (always generated + written; path returned so it can be pushed manually if a stage is blocked)
    source = build_notebook_source(spec)
    nb_path = write_notebook(source, workroot / "kernel" / f"{comp}-agents-infer.ipynb")
    res["notebook"] = nb_path

    # optional local dry-run (proves the body before any network call)
    if spec.get("dry_run") and spec.get("sample_submission") and spec.get("test_dir"):
        res["dry_run"] = dry_run(spec, spec["sample_submission"], spec["test_dir"], spec["ckpt"],
                                 files=int(spec.get("dry_run_files", 3)), workdir=workroot / "dry")

    # 1. dataset
    res["dataset"] = package_dataset(spec["ckpt"], slug, workroot / "dataset", log=log)
    if not res["dataset"]["ok"]:
        res["blocked"] = "dataset packaging failed"; return res
    # 2. kernel push
    res["push"] = push_kernel(nb_path, kernel_slug, comp, slug, workroot / "kernel", log=log)
    if not res["push"]["ok"]:
        res["blocked"] = "kernel push failed"; return res
    # 3. wait for the run
    if spec.get("wait", True):
        res["kernel_run"] = wait_for_kernel(kernel_slug, log=log)
    # 4. submit
    if spec.get("submit", True):
        res["submit"] = submit(comp, kernel_slug, msg, submission_file=spec.get("submission_file", "submission.csv"),
                               version=spec.get("version"), log=log)
        if res["submit"].get("daily_limit"):
            res["blocked"] = "daily submission limit reached"
        # 5. read scores (late subs show both public + private)
        time.sleep(int(spec.get("score_wait", 10)))
        sc = read_scores(comp, message=msg, log=log)
        res["public"] = sc.get("public"); res["private"] = sc.get("private")
        res["submission_id"] = sc.get("submission_id")
    return res


# ════════════════════════════════════════════════════════════ agent
class KaggleSubmit(BaseAgent):
    name = "kaggle-submit"
    thread = "M"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        need = [k for k in ("ckpt", "competition", "dataset_slug", "message") if not spec.get(k)]
        if need:
            return self.escalate(worker, "leader",
                                 f"[{worker}] kaggle-submit needs spec keys {need} (ckpt=best model, competition "
                                 f"slug, dataset_slug=user/name to mount, message). This packages a model → offline "
                                 f"notebook → pushes → submits → reads PUBLIC+PRIVATE LB; not faking a score on empty spec.")
        try:
            res = submit_pipeline(spec, log=lambda m: self.post(worker, "leader", m, routine=True, kind="finding"))
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher",
                                 f"[{worker}] kaggle-submit FAILED ({type(e).__name__}: {str(e)[:200]})")
        if res.get("blocked"):
            msg = (f"[{worker}] **KAGGLE-SUBMIT BLOCKED** at '{res['blocked']}' for {res['competition']} — notebook "
                   f"ready at {res.get('notebook')}, dataset={res['dataset_slug']}. NOT fabricating a score.")
            self.log(msg, kind="reason", recommendation="resolve the block (auth/rate-limit/notebook rules) and re-run")
            return self.done(res, msg, to="leader")
        pub, prv = res.get("public"), res.get("private")
        msg = (f"[{worker}] **KAGGLE-SUBMIT** {res['competition']} → PUBLIC={pub} PRIVATE={prv} "
               f"(sub {res.get('submission_id')}); dataset {res['dataset']['action']} {res['dataset_slug']}, "
               f"kernel {res['kernel_slug']}. Offline CPU notebook, self-contained.")
        self.log(msg, kind="verdict",
                 recommendation="reusable on every new best model: pass {ckpt, competition, dataset_slug, message}")
        return self.done(res, msg, to="leader")


_AGENT = KaggleSubmit()


def run(q, worker):
    return _AGENT.run(q, worker)
