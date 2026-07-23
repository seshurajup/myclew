"""CV-build adapter — REUSES the competition's own src.cv (embryo-disjoint) via its venv.

We don't re-implement embryo-disjoint CV; we call src.cv.kfold_embryo through research/cellmot_venv
(where src's deps live), then write the split JSON in the existing format (list of {train,test}).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
CV_PY = COMP / "research" / "cellmot_venv" / "bin" / "python"
FINETUNE = COMP / "learning" / "ensemble_work" / "finetune"

# runs inside cellmot_venv (cwd=COMP): reuse src.cv, print folds as JSON
_CODE = r'''
import glob, json, os, sys
from src.cv import kfold_embryo, group_by_embryo
k = int(sys.argv[1]); mini = int(sys.argv[2]); mini_test = int(sys.argv[3])
TRAIN = "input/biohub-cell-tracking-during-development/train"
def _norm(x):
    x = str(x).strip()
    for suf in (".zarr.geff", ".geff", ".zarr"):  # strip malformed file-suffixed ids (e.g. "<ds>.zarr")
        if x.endswith(suf): x = x[:-len(suf)]
    return x
raw = set()
for f in glob.glob("learning/ensemble_work/finetune/splits_*.json"):
    try:
        for fold in json.load(open(f)): raw |= set(fold.get("train", ())) | set(fold.get("test", ()))
    except Exception: pass
# normalize suffixes, dedup, and DROP phantom ids that do not resolve to a real train geff
ids = sorted({n for n in map(_norm, raw) if os.path.isdir(os.path.join(TRAIN, n + ".geff"))})
def stride(items, n):
    items = sorted(items)
    if n <= 0 or n >= len(items): return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]
folds = [{"train": stride(tr, mini) if mini else sorted(tr),
          "test": stride(va, mini_test) if mini_test else sorted(va)}
         for tr, va in kfold_embryo(ids, k=k)]
print(json.dumps({"pool": len(ids), "embryos": sorted(group_by_embryo(ids)), "folds": folds}))
'''


def handle(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    k = int(spec.get("k", 2))
    mini = int(spec.get("mini_per_fold", 0))
    mini_test = int(spec.get("mini_test", 0))  # stride TEST too → fast to score, not just train
    timeout = max(1, int(spec.get("timeout", 120)))  # timeout: src.cv subprocess wall-clock cap (s)
    if not CV_PY.exists():
        return ("failed", {"error": "cellmot_venv missing"}, "all",
                f"[{worker}] CV build FAILED: competition venv {CV_PY} not found.")
    try:
        r = subprocess.run([str(CV_PY), "-c", _CODE, str(k), str(mini), str(mini_test)],
                           cwd=str(COMP), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ("failed", {"error": f"src.cv timed out after {timeout}s"}, "all",
                f"[{worker}] CV build FAILED: src.cv timed out after {timeout}s.")
    if r.returncode != 0:
        return ("failed", {"stderr": r.stderr[-300:]}, "all",
                f"[{worker}] CV build FAILED (src.cv): {r.stderr.strip()[-160:]}")
    out_lines = r.stdout.strip().splitlines()
    try:
        d = json.loads(out_lines[-1]) if out_lines else {}
    except (ValueError, json.JSONDecodeError):
        return ("failed", {"error": "unparseable src.cv output"}, "all",
                f"[{worker}] CV build FAILED: src.cv output not JSON ({(out_lines[-1] if out_lines else '')[:120]}).")
    if "folds" not in d:
        return ("failed", {"error": "no folds in cv output"}, "all",
                f"[{worker}] CV build FAILED: src.cv produced no folds ({str(d)[:120]}).")
    folds = d["folds"]
    emb = lambda s: s.split("_")[0]  # noqa: E731
    ok = all(not ({emb(x) for x in f["test"]} & {emb(x) for x in f["train"]}) for f in folds)
    out = str(FINETUNE / spec.get("out", "fleet_loeo_mini.json"))
    # CHURN-GUARD: fleet_loeo_mini.json is the FROZEN primary CV (full-LOEO 128/71, re-locked 2026-07-05).
    # A cv-build call that omits mini_per_fold/out defaults to full-regen INTO this filename and silently
    # clobbers the frozen split (mtime churn). Refuse unless spec.allow_overwrite_frozen is explicitly set.
    _frozen = os.path.abspath(str(FINETUNE / "fleet_loeo_mini.json"))
    if os.path.abspath(out) == _frozen and not spec.get("allow_overwrite_frozen"):
        return ("failed",
                {"error": "refused overwrite of FROZEN fleet_loeo_mini.json; set spec.out to a different "
                          "filename, or spec.allow_overwrite_frozen=true to intentionally re-lock."},
                "all",
                f"[{worker}] CV build REFUSED: fleet_loeo_mini.json is the FROZEN primary CV (churn-guard). "
                f"Use spec.out=<other> for a screen split, or allow_overwrite_frozen=true to re-lock.")
    if ok:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(folds, open(out, "w"), indent=1)
    desc = ", ".join(f"fold{i}: train{len(f['train'])}/test{len(f['test'])}({emb(f['test'][0])})"
                     for i, f in enumerate(folds))
    return ("done" if ok else "failed",
            {"out": out if ok else None, "pool": d["pool"], "embryos": d["embryos"], "embryo_disjoint": ok},
            "all",
            f"[{worker}] CV BUILT via src.cv.kfold_embryo (embryo-disjoint, pool={d['pool']}): {desc}. Split: {out}")
