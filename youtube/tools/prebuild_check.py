# shared pre-build critic: returns list of problems (empty = OK to build)
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retention_rules import check_retention
from verify_outputs import verify_outputs

def prebuild_check(d):
    d = Path(d)
    problems = []
    # OUTPUT TRUTH: the channel's promise is "typed live and RUN" — every `>>>` result shown must
    # come from actually executing the code. Two videos once shipped fabricated values.
    problems += verify_outputs(d)
    # retention rules (README §3.9): promise hook, clean cold open, next-video + CTA tail.
    # These gate the BUILD, not just review — a label hook ships as the title AND the thumbnail.
    problems += check_retention(json.loads((d/"spec.json").read_text()),
                                json.loads((d/"transcript.json").read_text()))
    code = (d/"code.py").read_text()
    nlines = len(code.rstrip("\n").split("\n"))
    t = json.loads((d/"transcript.json").read_text())
    outs = json.loads((d/"outputs.json").read_text())
    if len(t) < 8:
        problems.append(f"only {len(t)} segments (<8)")
    prev = 0
    for i, s in enumerate(t):
        ul = s.get("until_line")
        if ul is None:
            if i != len(t)-1: problems.append(f"seg{i}: until_line missing")
            continue
        if not (1 <= ul <= nlines): problems.append(f"seg{i}: until_line {ul} out of 1..{nlines}")
        if ul < prev: problems.append(f"seg{i}: until_line {ul} BACKWARD from {prev}")
        prev = max(prev, ul)
    if max((s.get("until_line") or 0) for s in t) < nlines - 2:
        problems.append("code won't finish typing")
    for o in outs:
        al = o.get("after_line")
        if al is not None and not (1 <= al <= nlines):
            problems.append(f"output after_line {al} out of 1..{nlines}")
    if "human" in json.dumps(t).lower():
        problems.append("contains 'human'")
    return problems
