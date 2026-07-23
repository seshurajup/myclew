"""audio_train_test — DATA-WISE, offline, deterministic (BLAS-pinned) verifier for the audio-train agent.

Synthesizes a handful of short sine/chirp .ogg files (written with soundfile), a tiny train.csv (primary +
secondary labels, an 'author' column for the leak-safe grouped CV), a tiny soundscape-label csv, and a
sample_submission header with a few species. Runs audio-train in the TINY ladder rung (few classes, few
recordings/class, 1-2 epochs) and asserts: training loss is finite, the soundscape (LB-proxy) val AUC and the
author-grouped focal val AUC both COMPUTE in [0,1], a Kaggle-ready checkpoint is saved, and the fold json is
written. Also asserts the empty-spec handler escalates cleanly. Exit 0 iff all checks pass.
"""
import os
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

import math
import numpy as np
import pandas as pd
import soundfile as sf

from fleet_agents import audio_train as AT

SR = 32000
_fails = []


def check(name, cond):
    print(("  ok  " if cond else "  FAIL") + "  " + name)
    if not cond:
        _fails.append(name)


def _tone(seconds, freq, sr=SR, chirp=False):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype="float64")
    f = freq + (freq * 0.5) * t / seconds if chirp else freq
    return (0.5 * np.sin(2 * math.pi * f * t)).astype("float32")


SPECIES = ["sp_a", "sp_b", "sp_c"]                       # tiny 3-class problem
FREQ = {"sp_a": 300.0, "sp_b": 900.0, "sp_c": 2000.0}   # each species = a distinct pitch (real signal)

tmp = tempfile.mkdtemp(prefix="audiotrain_")
audio_dir = os.path.join(tmp, "train_audio")
ss_dir = os.path.join(tmp, "soundscapes")
os.makedirs(audio_dir); os.makedirs(ss_dir)

# ── focal train clips: 4 recordings/species, 2 authors/species (grouped CV needs >1 group/class) ──────
rows = []
for sp in SPECIES:
    for r in range(4):
        fn = f"{sp}/{sp}_{r}.ogg"
        os.makedirs(os.path.join(audio_dir, sp), exist_ok=True)
        wav = _tone(3.0 + 0.5 * r, FREQ[sp], chirp=(r % 2 == 0))
        sf.write(os.path.join(audio_dir, fn), wav, SR, format="OGG", subtype="VORBIS")
        sec = ["sp_c"] if (sp == "sp_a" and r == 0) else []     # one multi-label row (secondary)
        rows.append({"filename": fn, "primary_label": sp, "secondary_labels": str(sec),
                     "author": f"{sp}_author{r % 2}"})
train_csv = os.path.join(tmp, "train.csv"); pd.DataFrame(rows).to_csv(train_csv, index=False)

# ── sample_submission header (row_id + one col per species) ───────────────────────────────────────────
sample_sub = os.path.join(tmp, "sample_submission.csv")
pd.DataFrame(columns=["row_id"] + SPECIES).to_csv(sample_sub, index=False)

# ── soundscape val: 2 long clips, labeled 5s windows (domain-matched LB-proxy) ────────────────────────
srows = []
for k, sp in enumerate(["sp_a", "sp_b"]):
    fn = f"soundscape_{k}.ogg"
    wav = _tone(15.0, FREQ[sp])                           # 15s → three 5s windows
    sf.write(os.path.join(ss_dir, fn), wav, SR, format="OGG", subtype="VORBIS")
    for w in range(3):
        srows.append({"filename": fn, "start": w * 5.0, "end": (w + 1) * 5.0, "primary_label": sp})
ss_csv = os.path.join(tmp, "train_soundscapes_labels.csv"); pd.DataFrame(srows).to_csv(ss_csv, index=False)

# ── run audio-train in the TINY ladder rung, WITH cv (author-grouped focal fold) ──────────────────────
out_dir = os.path.join(tmp, "out")
spec = {"train_csv": train_csv, "audio_dir": audio_dir, "sample_submission": sample_sub,
        "soundscape_csv": ss_csv, "soundscape_dir": ss_dir, "out_dir": out_dir,
        "epochs": 2, "batch_size": 4, "seconds": 5.0, "per_class": 4, "cv": True, "n_folds": 2, "fold": 0,
        "lr": 2e-3, "device": "cpu", "seed": 0, "wave_noise": 0.0}
res = AT.train(spec, log=print)

check("checkpoint saved", os.path.exists(res["ckpt_path"]))
check("folds.json written", res["folds"] is not None and os.path.exists(res["folds"]))
loss0 = res["curve"][0]["train_loss"]
check("train loss finite", isinstance(loss0, float) and math.isfinite(loss0))
bv = res["best_val_auc"]
check("soundscape (LB-proxy) val AUC in [0,1]", isinstance(bv, float) and 0.0 <= bv <= 1.0)
fv = res["curve"][-1]["focal_val_auc"]
check("author-grouped focal val AUC in [0,1]", isinstance(fv, float) and 0.0 <= fv <= 1.0)
check("n_classes == 3", res["n_classes"] == 3)

# checkpoint is Kaggle-ready: has classes + melspec cfg + seconds
import torch
ck = torch.load(res["ckpt_path"], map_location="cpu", weights_only=False)
check("ckpt has classes", ck.get("classes") == SPECIES)
check("ckpt has melspec_cfg", isinstance(ck.get("melspec_cfg"), dict) and "n_mels" in ck["melspec_cfg"])

# ── ladder unit checks: subset of classes + per_class actually restrict the problem ───────────────────
sub_classes = AT.derive_classes(pd.read_csv(train_csv), pd.read_csv(sample_sub), subset=["sp_a", "sp_b"])
check("class subset restricts to 2", sub_classes == ["sp_a", "sp_b"])
fold_arr = AT.group_author_kfold(pd.read_csv(train_csv), n_folds=2, seed=0)
check("grouped kfold assigns a fold to every row", len(fold_arr) == len(rows) and set(fold_arr.tolist()) <= {0, 1})

# ── official metric: skips empty classes, perfect separation → AUC 1.0 ────────────────────────────────
yt = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")   # col2 all-zero → skipped
yp = np.array([[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.8, 0.2, 0.5], [0.2, 0.8, 0.5]], dtype="float32")
auc, n = AT.macro_roc_auc(yt, yp, return_n=True)
check("macro_roc_auc perfect=1.0", abs(auc - 1.0) < 1e-6)
check("macro_roc_auc skipped the empty class (n_scored==2)", n == 2)

# ── empty-spec handler escalates cleanly (fleet smoke contract) ───────────────────────────────────────
VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}
r = AT.run({"question": "t", "spec": {}}, "unit")
check("empty-spec valid contract", isinstance(r, tuple) and len(r) == 4 and r[0] in VALID)
check("empty-spec escalates clean", r[0] == "escalated")

print()
if _fails:
    print("FAILURES:", _fails); sys.exit(1)
print("ALL AUDIO-TRAIN CHECKS PASSED"); sys.exit(0)
