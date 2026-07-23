"""audio_infer_test — DATA-WISE, offline, deterministic (BLAS-pinned) verifier for the audio-infer agent.

Builds a tiny audio-train checkpoint (real EfficientNet-b0/small-CNN over a 3-class head), synthesizes a test
soundscape .ogg + a sample_submission header, runs audio-infer on CPU, and asserts submission.csv: has the
EXACT sample_submission columns (row_id first), one row per 5s window, row_id = f"{stem}_{end_seconds}", and
all probabilities in [0,1]. Also checks a subset-trained checkpoint fills absent species with the fill value,
and that the empty-spec handler escalates cleanly. Exit 0 iff all checks pass.
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
import torch

from fleet_agents import audio_infer as AI
from fleet_agents import audio_pack as A

SR = 32000
_fails = []


def check(name, cond):
    print(("  ok  " if cond else "  FAIL") + "  " + name)
    if not cond:
        _fails.append(name)


def _tone(seconds, freq, sr=SR):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype="float64")
    return (0.5 * np.sin(2 * math.pi * freq * t)).astype("float32")


tmp = tempfile.mkdtemp(prefix="audioinfer_")
test_dir = os.path.join(tmp, "test_soundscapes"); os.makedirs(test_dir)

# ── a 3-species sample_submission header (the ONLY source of column order) ────────────────────────────
SPECIES = ["sp_a", "sp_b", "sp_c"]
sample_sub = os.path.join(tmp, "sample_submission.csv")
pd.DataFrame(columns=["row_id"] + SPECIES).to_csv(sample_sub, index=False)


def _make_ckpt(classes, path):
    model, _ = A.build_mel_backbone(len(classes), in_ch=1, pretrained=False, device="cpu")
    ck = {"state_dict": model.state_dict(), "classes": classes,
          "melspec_cfg": {"sr": SR, "n_fft": 1024, "hop": 500, "n_mels": 128, "fmin": 40.0, "fmax": 15000.0,
                          "to_db": True, "normalize": True},
          "seconds": 5.0, "arch": "tf_efficientnet_b0", "in_ch": 1, "backend": "test"}
    torch.save(ck, path)


# ── a 23s test soundscape → sliding 5s windows: ceil gives 5 windows (0-5,5-10,10-15,15-20,18-23) ─────
ss_path = os.path.join(test_dir, "soundscape_X.ogg")
sf.write(ss_path, _tone(23.0, 500.0), SR, format="OGG", subtype="VORBIS")

# ── full-class checkpoint ─────────────────────────────────────────────────────────────────────────────
ckpt_full = os.path.join(tmp, "best.pt"); _make_ckpt(SPECIES, ckpt_full)
out_csv = os.path.join(tmp, "submission.csv")
spec = {"ckpt": ckpt_full, "test_dir": test_dir, "sample_submission": sample_sub, "out": out_csv,
        "batch_size": 4, "smooth": True}
res = AI.build_submission(spec, log=print)

check("submission.csv written", os.path.exists(out_csv))
sub = pd.read_csv(out_csv)
expect_cols = ["row_id"] + SPECIES
check("columns match sample_submission exactly", list(sub.columns) == expect_cols)
check("one row per 5s window (>=4 for 23s clip)", len(sub) >= 4)
# expected number of windows for a 23s clip with 5s non-overlapping + a final tail window
n_expected = res["n_windows"]
check("rows == n_windows reported", len(sub) == n_expected)
# row_id format = stem_endseconds
check("row_id = stem_endsec", sub["row_id"].iloc[0] == "soundscape_X_5")
probs = sub[SPECIES].values
check("all probs in [0,1]", float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0)
check("windows/sec reported", isinstance(res["windows_per_sec"], float) and res["windows_per_sec"] > 0)
check("90-min budget projected", res["proj_minutes_19900"] is not None)

# ── subset-trained checkpoint (small-first): absent species filled with `fill` ────────────────────────
ckpt_sub = os.path.join(tmp, "best_sub.pt"); _make_ckpt(["sp_a"], ckpt_sub)   # only trained sp_a
out_sub = os.path.join(tmp, "submission_sub.csv")
AI.build_submission({"ckpt": ckpt_sub, "test_dir": test_dir, "sample_submission": sample_sub,
                     "out": out_sub, "fill": 0.0}, log=print)
sub2 = pd.read_csv(out_sub)
check("subset ckpt keeps sample columns", list(sub2.columns) == expect_cols)
check("absent species filled with 0.0", float(sub2["sp_b"].abs().max()) == 0.0 and float(sub2["sp_c"].abs().max()) == 0.0)
check("trained species non-trivial", float(sub2["sp_a"].max()) > 0.0)

# ── empty-spec handler escalates cleanly ──────────────────────────────────────────────────────────────
VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}
r = AI.run({"question": "t", "spec": {}}, "unit")
check("empty-spec valid contract", isinstance(r, tuple) and len(r) == 4 and r[0] in VALID)
check("empty-spec escalates clean", r[0] == "escalated")

print()
if _fails:
    print("FAILURES:", _fails); sys.exit(1)
print("ALL AUDIO-INFER CHECKS PASSED"); sys.exit(0)
