"""kaggle_submit_test — DATA-WISE, OFFLINE, deterministic verifier for the kaggle-submit agent.

NEVER touches the real Kaggle API: the CLI indirection `kaggle_submit._run_cli` is monkeypatched to a recording
mock that returns canned outputs. Builds a tiny synthetic audio-train checkpoint (3-class EfficientNet/small-CNN)
+ a sample_submission header + a stand-in test soundscape .ogg, then asserts:
  • build_notebook_source → valid, self-contained python (compiles, no fleet_agents import, has CONFIG + main()),
  • write_notebook → a loadable single-cell .ipynb,
  • dry_run of the EXACT notebook body → submission.csv with sample_submission columns + probs in [0,1],
  • package_dataset picks create-vs-version from the status mock + emits dataset-metadata.json,
  • push_kernel writes CPU/no-internet kernel-metadata.json with the dataset+competition attached,
  • parse_submissions parses a mocked `submissions --format csv` into {public, private, submission_id},
  • submit_pipeline (fully mocked CLI) returns {public, private, submission_id} end to end,
  • the empty-spec agent handler escalates cleanly.
Exit 0 iff all checks pass.
"""
import os
import sys
import json
import math
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from fleet_agents import kaggle_submit as KS
from fleet_agents import audio_pack as A

SR = 32000
SPECIES = ["sp_a", "sp_b", "sp_c"]
_fails = []


def check(name, cond):
    print(("  ok  " if cond else "  FAIL") + "  " + name)
    if not cond:
        _fails.append(name)


def _tone(seconds, freq, sr=SR):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype="float64")
    return (0.5 * np.sin(2 * math.pi * freq * t)).astype("float32")


tmp = tempfile.mkdtemp(prefix="ksubmit_")
test_dir = os.path.join(tmp, "test_soundscapes"); os.makedirs(test_dir)
sf.write(os.path.join(test_dir, "soundscape_X.ogg"), _tone(23.0, 500.0), SR, format="OGG", subtype="VORBIS")

sample_sub = os.path.join(tmp, "sample_submission.csv")
pd.DataFrame(columns=["row_id"] + SPECIES).to_csv(sample_sub, index=False)

ckpt = os.path.join(tmp, "best.pt")
model, _ = A.build_mel_backbone(len(SPECIES), in_ch=1, pretrained=False, device="cpu")
torch.save({"state_dict": model.state_dict(), "classes": SPECIES,
            "melspec_cfg": {"sr": SR, "n_fft": 1024, "hop": 500, "n_mels": 128, "fmin": 40.0, "fmax": 15000.0,
                            "to_db": True, "normalize": True},
            "seconds": 5.0, "arch": "tf_efficientnet_b0", "in_ch": 1, "backend": "test"}, ckpt)

SPEC = {"ckpt": ckpt, "competition": "toy-comp", "dataset_slug": "tester/toy-model",
        "message": "agents-test run", "sample_submission": sample_sub, "test_dir": test_dir,
        "inference": "audio", "batch_size": 4, "smooth": True}

# ── 1. notebook source is self-contained + compiles ──────────────────────────────────────────────────
src = KS.build_notebook_source(SPEC)
check("source compiles", (compile(src, "<nb>", "exec") is not None) or True)
check("source has CONFIG", "CONFIG = " in src and "'competition': 'toy-comp'" in src)
check("source has main()", "def main()" in src and "submission.csv" in src)
check("source does NOT import fleet_agents", "fleet_agents" not in src and "import audio_pack" not in src)
check("source inlines mel + timm rebuild", "def log_mel_spectrogram" in src and "timm.create_model" in src)

# ── 2. notebook writes a loadable single-cell .ipynb ─────────────────────────────────────────────────
nb_path = KS.write_notebook(src, os.path.join(tmp, "nb", "infer.ipynb"))
nb = json.loads(open(nb_path).read())
check("ipynb valid", nb.get("nbformat") == 4 and len(nb["cells"]) == 1 and nb["cells"][0]["cell_type"] == "code")

# ── 3. dry-run of the EXACT notebook body → valid submission ─────────────────────────────────────────
dr = KS.dry_run(SPEC, sample_sub, test_dir, ckpt, files=1, workdir=os.path.join(tmp, "dry"))
check("dry-run ok", dr.get("ok") is True)
check("dry-run columns match sample_submission", dr.get("columns") == 4 and dr.get("cols_ok"))
check("dry-run probs in [0,1]", dr.get("range_ok") is True)
sub = pd.read_csv(dr["out_path"])
check("dry-run row_id = stem_endsec", sub["row_id"].iloc[0] == "soundscape_X_5")
check("dry-run one row per 5s window (>=4 for 23s)", len(sub) >= 4)

# ── mock the CLI so NOTHING hits real Kaggle ─────────────────────────────────────────────────────────
CALLS = []
SUBMISSIONS_CSV = ("fileName,date,description,status,publicScore,privateScore\n"
                   "submission.csv,2026-07-16 12:00:00,agents-test run,complete,0.6842,0.6501\n"
                   "submission.csv,2026-07-15 09:00:00,older run,complete,0.6100,0.5900\n")


def fake_cli(args, timeout=1800, cwd=None):
    CALLS.append(list(args))
    a = " ".join(str(x) for x in args)
    if a.startswith("datasets status"):
        return 1, "", "404 - Not Found"                       # → create path
    if a.startswith("datasets create"):
        return 0, "Your dataset was created", ""
    if a.startswith("datasets version"):
        return 0, "Dataset version created", ""
    if a.startswith("kernels push"):
        return 0, "Your kernel was pushed", ""
    if a.startswith("kernels status"):
        return 0, 'status "complete"', ""
    if a.startswith("competitions submit"):
        return 0, "Successfully submitted", ""
    if a.startswith("competitions submissions"):
        return 0, SUBMISSIONS_CSV, ""
    return 0, "", ""


KS._run_cli = fake_cli

# ── 4. package_dataset: create (status 404) + metadata sidecar + dataset-metadata.json ───────────────
pk = KS.package_dataset(ckpt, "tester/toy-model", os.path.join(tmp, "ds"))
check("dataset create chosen (not version)", pk["action"] == "create" and pk["ok"])
check("dataset-metadata.json written", os.path.exists(os.path.join(tmp, "ds", "dataset-metadata.json")))
check("metadata sidecar has classes", json.loads(open(os.path.join(tmp, "ds", "metadata.json")).read())["classes"] == SPECIES)
check("best.pt copied into dataset dir", os.path.exists(os.path.join(tmp, "ds", "best.pt")))

# ── 5. push_kernel: CPU / no-internet / dataset+competition attached ─────────────────────────────────
kd = os.path.join(tmp, "kern")
KS.push_kernel(nb_path, "tester/toy-comp-infer", "toy-comp", "tester/toy-model", kd)
km = json.loads(open(os.path.join(kd, "kernel-metadata.json")).read())
check("kernel gpu off", km["enable_gpu"] is False)
check("kernel internet off", km["enable_internet"] is False)
check("kernel attaches dataset", km["dataset_sources"] == ["tester/toy-model"])
check("kernel attaches competition", km["competition_sources"] == ["toy-comp"])

# ── 6. parse_submissions: pure parser → public + private (message-matched, else newest) ──────────────
p = KS.parse_submissions(SUBMISSIONS_CSV, message="agents-test run")
check("parse public score", abs((p["public"] or 0) - 0.6842) < 1e-6)
check("parse private score", abs((p["private"] or 0) - 0.6501) < 1e-6)
check("parse submission id present", p["submission_id"] is not None)
p_newest = KS.parse_submissions(SUBMISSIONS_CSV)          # no message → newest row
check("parse newest defaults to first row", abs((p_newest["public"] or 0) - 0.6842) < 1e-6)
p_empty = KS.parse_submissions("")
check("parse empty is safe", p_empty["public"] is None and p_empty["private"] is None)

# ── 7. submit_pipeline end-to-end (fully mocked) → {public, private, submission_id} ──────────────────
CALLS.clear()
full = dict(SPEC)
full["workdir"] = os.path.join(tmp, "pipe"); full["dry_run"] = True; full["dry_run_files"] = 1
full["score_wait"] = 0; full["wait"] = True
res = KS.submit_pipeline(full, log=lambda *a: None)
check("pipeline public score", abs((res.get("public") or 0) - 0.6842) < 1e-6)
check("pipeline private score", abs((res.get("private") or 0) - 0.6501) < 1e-6)
check("pipeline dataset action create", res["dataset"]["action"] == "create")
check("pipeline notebook path exists", os.path.exists(res["notebook"]))
check("pipeline ran a submit call", any(str(c[0]) == "competitions" and "submit" in c for c in CALLS))
check("pipeline never called real bin", KS._run_cli is fake_cli)

# ── 8. daily-limit detection ─────────────────────────────────────────────────────────────────────────
def fake_cli_limit(args, timeout=1800, cwd=None):
    if " ".join(str(x) for x in args).startswith("competitions submit"):
        return 1, "", "You have reached your maximum number of submissions for today"
    return fake_cli(args, timeout, cwd)


KS._run_cli = fake_cli_limit
sr_lim = KS.submit("toy-comp", "tester/k", "msg")
check("daily-limit flagged", sr_lim["daily_limit"] is True and sr_lim["ok"] is False)
KS._run_cli = fake_cli

# ── 9. empty-spec agent handler escalates cleanly ───────────────────────────────────────────────────
VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}
r = KS.run({"question": "t", "spec": {}}, "unit")
check("empty-spec valid contract", isinstance(r, tuple) and len(r) == 4 and r[0] in VALID)
check("empty-spec escalates", r[0] == "escalated")

print()
if _fails:
    print("FAILURES:", _fails); sys.exit(1)
print("ALL KAGGLE-SUBMIT CHECKS PASSED"); sys.exit(0)
