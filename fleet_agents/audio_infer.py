"""audio-infer — CPU sliding-window inference → Kaggle-ready submission.csv. This is the BODY of the offline
BirdCLEF-2026 notebook (2×T4/CPU, ≤90 min, no network): load an audio-train checkpoint, read every test
soundscape .ogg, split it into consecutive 5s windows, melspec + batch-predict each, optional window-mean TTA
+ neighbor smoothing, and write one probability row per 5s window in the EXACT sample_submission column order.

Grounded ONLY in prev-year BirdCLEF knowledge (docs/audio_pack_grounded.md): the long-clip inference standard
is slide-a-fixed-window → per-window sigmoid → aggregate; row_id = f"{soundscape_stem}_{end_seconds}" and one
column per species (bc21/bc23/bc24). Neighbor smoothing [0.1,0.2,0.4,0.2,0.1] (bc24 r3) is the recurring
post-process. CPU-fast + import-light so it drops straight into the Kaggle notebook.

COMPOSES:
  • audio_train.load_audio            — soundfile .ogg → mono 32 kHz (resample_poly / linear)
  • audio_pack.log_mel_spectrogram    — mel front-end (same cfg as training, read from the checkpoint)
  • audio_pack.sliding_windows        — consecutive 5s windows over the clip
  • audio_pack.neighbor_smooth        — optional temporal smoothing of the per-window preds
  • audio_pack.build_mel_backbone     — rebuild the EfficientNet-b0 mel classifier to load the state_dict

Columns come from sample_submission.csv (the ONLY source of truth for order); species the checkpoint did not
train (small-first subset) are filled with spec['fill'] (default 0.0). Reports windows/sec so the 90-min
budget can be projected (Kaggle hidden test = 199×100 = 19,900 windows-scale).

Spec: ckpt, test_dir (soundscapes), sample_submission, out (submission.csv path), [batch_size, fill, smooth,
threads, limit]. Data-wise test: test_fleet_agents/audio_infer_test.py.
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import BaseAgent
from . import audio_pack as A
from . import audio_train as AT


def load_checkpoint(ckpt_path, device="cpu"):
    """Load an audio-train checkpoint on CPU and rebuild the model. Returns (model, ckpt_dict)."""
    import torch
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    classes = ck["classes"]
    model, _ = A.build_mel_backbone(len(classes), in_ch=int(ck.get("in_ch", 1)),
                                    arch=ck.get("arch", "tf_efficientnet_b0"), pretrained=False, device=device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def _list_soundscapes(test_dir):
    exts = (".ogg", ".wav", ".flac")
    return sorted([p for p in Path(test_dir).rglob("*") if p.suffix.lower() in exts])


def predict_soundscape(model, wav, cfg, seconds, sr, device="cpu", batch_size=16, smooth=False):
    """Slide 5s windows over one soundscape waveform → per-window sigmoid probs (n_windows, C). Optional
    neighbor smoothing. `wav`: 1-D tensor at `sr`."""
    import torch
    length = int(seconds * sr)
    wins = A.sliding_windows(wav, length, hop=length)          # (n_windows, length)
    probs = []
    with torch.no_grad():
        for i in range(0, wins.size(0), batch_size):
            batch = wins[i:i + batch_size].to(device).float()
            mel = A.log_mel_spectrogram(batch, sr=sr, n_fft=cfg["n_fft"], hop=cfg["hop"], n_mels=cfg["n_mels"],
                                        fmin=cfg["fmin"], fmax=cfg.get("fmax"), to_db=cfg.get("to_db", True),
                                        normalize=cfg.get("normalize", True), device=device)
            logits = model(mel)
            probs.append(torch.sigmoid(logits).float().cpu())
    P = torch.cat(probs, dim=0) if probs else torch.zeros((0, len(cfg.get("classes", []))))
    if smooth and P.size(0) >= 3:
        P = A.neighbor_smooth(P)
    return P


def build_submission(spec, log=lambda *a, **k: None):
    """Run CPU inference over test_dir and write submission.csv. Returns a stats dict (rows, windows/sec…)."""
    import numpy as np
    import pandas as pd
    import torch

    device = "cpu"                                             # the Kaggle offline notebook runs this on CPU
    threads = int(spec.get("threads", 0))
    if threads > 0:
        torch.set_num_threads(threads)
    model, ck = load_checkpoint(spec["ckpt"], device=device)
    classes = ck["classes"]
    cfg = dict(ck["melspec_cfg"]); cfg["classes"] = classes
    seconds = float(ck.get("seconds", 5.0)); sr = int(cfg["sr"])
    batch_size = int(spec.get("batch_size", 16))
    smooth = bool(spec.get("smooth", False))
    fill = float(spec.get("fill", 0.0))

    sample = pd.read_csv(spec["sample_submission"])
    sub_cols = [c for c in sample.columns if c.lower() != "row_id"]   # species columns, in sample order
    col_idx = {c: j for j, c in enumerate(sub_cols)}
    model_class_pos = {c: i for i, c in enumerate(classes)}           # model output index per species

    files = _list_soundscapes(spec["test_dir"])
    if spec.get("limit"):
        files = files[:int(spec["limit"])]

    rows, row_ids = [], []
    t0 = time.time(); n_windows = 0
    for path in files:
        stem = path.stem
        try:
            wav_np = AT.load_audio(path, target_sr=sr)
        except Exception:  # noqa: BLE001
            continue
        wav = torch.as_tensor(wav_np, dtype=torch.float32)
        P = predict_soundscape(model, wav, cfg, seconds, sr, device=device, batch_size=batch_size, smooth=smooth)
        P = P.numpy()
        n_windows += P.shape[0]
        for w in range(P.shape[0]):
            end_sec = int(round((w + 1) * seconds))
            row = np.full(len(sub_cols), fill, dtype="float32")
            for c, mi in model_class_pos.items():             # scatter model probs into sample-sub columns
                if c in col_idx:
                    row[col_idx[c]] = P[w, mi]
            rows.append(row); row_ids.append(f"{stem}_{end_sec}")

    elapsed = time.time() - t0
    out = Path(spec.get("out", "submission.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        arr = np.clip(np.stack(rows), 0.0, 1.0)
        sub = pd.DataFrame(arr, columns=sub_cols)
    else:                                                     # no test audio → valid empty-body submission
        sub = pd.DataFrame(columns=sub_cols)
    sub.insert(0, "row_id", row_ids)
    sub = sub[["row_id"] + sub_cols]                          # EXACT sample_submission column order
    sub.to_csv(out, index=False)

    wps = (n_windows / elapsed) if elapsed > 0 else 0.0
    proj_19900 = (19900 / wps / 60.0) if wps > 0 else None    # project the 19,900-window hidden test to minutes
    return {"rows": len(row_ids), "n_files": len(files), "n_windows": n_windows,
            "windows_per_sec": round(wps, 2), "proj_minutes_19900": None if proj_19900 is None else round(proj_19900, 2),
            "columns": len(sub.columns), "out": str(out), "n_model_classes": len(classes)}


# ════════════════════════════════════════════════════════════ agent
class AudioInfer(BaseAgent):
    name = "audio-infer"
    thread = "M"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        need = [k for k in ("ckpt", "test_dir", "sample_submission") if not spec.get(k)]
        if need:
            return self.escalate(worker, "leader",
                                 f"[{worker}] audio-infer needs spec keys {need} (ckpt from audio-train + test_dir "
                                 f"of soundscape .ogg + sample_submission for the column order). This IS the offline "
                                 f"CPU notebook body; not fabricating a submission on empty spec.")
        try:
            res = build_submission(spec, log=lambda m: self.post(worker, "leader", m, routine=True, kind="finding"))
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] audio-infer FAILED ({type(e).__name__}: {str(e)[:200]})")
        pm = res["proj_minutes_19900"]
        msg = (f"[{worker}] **AUDIO-INFER** → {res['rows']} rows / {res['n_files']} soundscapes / {res['n_windows']} "
               f"windows @ {res['windows_per_sec']} win/s (CPU); projected {pm} min for the 19,900-window hidden test "
               f"(budget 90). {res['columns']} cols → {res['out']}")
        self.log(msg, kind="verdict",
                 recommendation="submission.csv matches sample_submission columns exactly; row_id=stem_endsec. "
                                "If proj_minutes>90, compress the backbone / raise batch_size / drop smoothing.")
        return self.done(res, msg, to="leader")


_AGENT = AudioInfer()


def run(q, worker):
    return _AGENT.run(q, worker)
