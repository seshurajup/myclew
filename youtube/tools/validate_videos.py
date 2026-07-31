import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retention_rules import check_retention

YT = Path("/home/seshu/kaggle/2026/youtube")
PLAYLISTS = ["01-python-basics","02-python-functions","03-python-loops-iteration",
             "04-python-oop","05-python-advanced","06-python-testing-tools","07-python-libraries"]

def check(d):
    problems = []
    code = (d/"code.py").read_text()
    nlines = len(code.rstrip("\n").split("\n"))
    t = json.loads((d/"transcript.json").read_text())
    outs = json.loads((d/"outputs.json").read_text())
    # 1. segment count for 60-120s
    if len(t) < 8:
        problems.append(f"only {len(t)} segments (<8, likely <60s)")
    # 2. until_line valid + non-decreasing (narration must follow code top->bottom)
    prev = 0
    for i, s in enumerate(t):
        ul = s.get("until_line")
        if ul is None:
            if i != len(t)-1:
                problems.append(f"seg{i}: until_line missing (only last segment may omit it)")
            continue
        if not (1 <= ul <= nlines):
            problems.append(f"seg{i}: until_line {ul} out of range 1..{nlines}")
        if ul < prev:
            problems.append(f"seg{i}: until_line {ul} goes BACKWARD from {prev} (code-highlight desync)")
        prev = max(prev, ul)
    # 3. last segment should reach near end of code (all code typed)
    last_ul = max((s.get("until_line") or 0) for s in t)
    if last_ul < nlines - 2:
        problems.append(f"last until_line {last_ul} << code end {nlines} (code won't finish typing)")
    # 4. outputs after_line valid
    for o in outs:
        al = o.get("after_line")
        if al is not None and not (1 <= al <= nlines):
            problems.append(f"output after_line {al} out of range 1..{nlines}")
    # 5. no 'human'
    if "human" in json.dumps(t).lower():
        problems.append("contains the word 'human'")
    return problems

total, bad = 0, 0
for pl in PLAYLISTS:
    base = YT/pl
    if not base.exists(): continue
    for d in sorted(base.iterdir()):
        if not (d.is_dir() and (d/"code.py").exists()): continue
        total += 1
        probs = check(d)
        if probs:
            bad += 1
            print(f"FAIL {pl}/{d.name}:")
            for p in probs: print(f"    - {p}")
print(f"\n{total} videos checked, {bad} with problems, {total-bad} clean")
sys.exit(1 if bad else 0)
