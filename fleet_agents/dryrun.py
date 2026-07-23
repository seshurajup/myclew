"""Dry-run agent — the researcher's dry-run RESPONSIBILITY, done deterministically.

Validates an experiment config is runnable WITHOUT consuming training time: config parses, and its
declared trainer / splits / data_dir / pythonpath paths resolve. No GPU. Produces the
'dry-run passed / ready for queue' signal the trainer role waits for — now from Python, not Claude.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
CV_PY = COMP / "research" / "cellmot_venv" / "bin" / "python"

_CODE = r'''
import sys, os, json, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
paths = cfg.get("paths", {}) or {}
probs = []
for key in ("trainer", "splits", "data_dir"):
    v = paths.get(key)
    if v and not os.path.exists(v):
        probs.append(f"{key} path missing: {v}")
for pp in (paths.get("pythonpath") or []):
    if not os.path.exists(pp):
        probs.append(f"pythonpath missing: {pp}")
print(json.dumps({"name": cfg.get("name", "?"), "probs": probs,
                  "trainer": paths.get("trainer"), "splits": paths.get("splits")}))
'''


def validate(config: str):
    """Return (ok: bool, notes: list[str]). Wiring check via the competition venv (has yaml)."""
    p = COMP / config
    if not p.exists():
        return False, [f"config {config} missing"]
    if not CV_PY.exists():
        return False, ["cellmot_venv missing — cannot parse config"]
    r = subprocess.run([str(CV_PY), "-c", _CODE, str(p)], cwd=str(COMP),
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return False, [f"config parse/validate failed: {r.stderr.strip()[-160:]}"]
    d = json.loads(r.stdout.strip().splitlines()[-1])
    if d.get("probs"):
        return False, d["probs"]
    return True, [f"wiring OK (trainer={Path(d['trainer']).name if d.get('trainer') else '?'}, "
                  f"splits={Path(d['splits']).name if d.get('splits') else '?'})"]
