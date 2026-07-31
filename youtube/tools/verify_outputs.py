"""Output-truth gate: every `>>>` result a video shows must come from really running its code.

Authenticity is the whole promise of the channel — "watch the code typed live and RUN". Before
this gate, outputs.json was hand-authored, and two videos shipped values the code does not
produce (078-math-random showed the wrong seeded random values while the narration promised
"you get the same random numbers every run"; 072-pytest claimed 5 passed when pytest reports 6).

Rules for an outputs.json event:
  • `"text"` lines are matched literally against the real stdout+stderr, with `...` acting as a
    wildcard for elided output (e.g. "3.14159..." matches the full float).
  • a line wrapped in (parentheses) is a descriptive aside, not a claim — not checked.
    Prefer putting those in `"caption"` instead.
  • `"volatile": true` marks genuinely run-dependent values (timings, today's date, unseeded
    random). The literal check is skipped, but the code must still run clean.

`spec.json` may set `"run"` to override the default command, e.g.
    "run": ["python", "-m", "pytest", "-q", "code.py"]   (for the testing playlist)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_RUN = [sys.executable, "code.py"]
SKIP_RUN = re.compile(r"\binput\(|sys\.argv")   # interactive / CLI demos can't be run headless


def _claim_re(claim: str) -> re.Pattern:
    """'3.14159...' -> regex matching '3.141592653589793'. '...' is the only metacharacter."""
    return re.compile(".*".join(re.escape(p) for p in claim.split("...")), re.S)


def verify_outputs(d, timeout=30):
    d = Path(d)
    problems = []
    code = (d / "code.py").read_text()
    spec = json.loads((d / "spec.json").read_text())
    outs = json.loads((d / "outputs.json").read_text())

    if spec.get("run"):
        cmd = [sys.executable if c == "python" else c for c in spec["run"]]
    elif SKIP_RUN.search(code):
        return problems                      # interactive demo: nothing to execute headlessly
    else:
        cmd = DEFAULT_RUN

    try:
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [f"code.py did not finish in {timeout}s"]
    except Exception as e:                                        # noqa: BLE001
        return [f"could not run code.py: {e}"]

    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().split("\n")[-1][:120]
        return [f"code exits {r.returncode}: {tail}"]

    real = r.stdout + r.stderr                                    # logging/unittest use stderr
    for i, ev in enumerate(outs):
        if ev.get("volatile") or not ev.get("text"):
            continue
        for line in ev["text"].replace("\\n", "\n").split("\n"):
            line = line.strip()
            if line.startswith(">>>"):
                line = line[3:].strip()
            if not line or line == "output":
                continue
            if line.startswith("(") and line.endswith(")"):
                continue                                          # descriptive aside
            if not _claim_re(line).search(real):
                problems.append(
                    f"output[{i}] shows {line!r} but running the code never produces it "
                    f"(mark the event \"volatile\": true if it genuinely varies per run)")
    return problems


if __name__ == "__main__":
    probs = verify_outputs(sys.argv[1])
    for p in probs:
        print("  -", p)
    sys.exit(1 if probs else 0)
