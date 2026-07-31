import json
from pathlib import Path
YT = Path("/home/seshu/kaggle/2026/youtube")
PLAYLISTS = ["01-python-basics","02-python-functions","03-python-loops-iteration",
             "04-python-oop","05-python-advanced","06-python-testing-tools","07-python-libraries"]

def fix(d):
    code = (d/"code.py").read_text()
    nlines = len(code.rstrip("\n").split("\n"))
    t = json.loads((d/"transcript.json").read_text())
    # 1. clamp every until_line to [1, nlines]; treat missing as nlines
    uls = []
    for s in t:
        ul = s.get("until_line")
        ul = nlines if ul is None else max(1, min(int(ul), nlines))
        uls.append(ul)
    # 2. force the LAST segment to reach the end so all code types
    uls[-1] = nlines
    # 3. enforce non-decreasing by LOWERING earlier highs (typing never runs ahead of voice)
    for i in range(len(uls)-2, -1, -1):
        uls[i] = min(uls[i], uls[i+1])
    # 4. never let an early segment collapse below 1
    for s, ul in zip(t, uls):
        s["until_line"] = ul
    (d/"transcript.json").write_text(json.dumps(t, indent=2, ensure_ascii=False))
    # 5. clamp outputs after_line
    outs = json.loads((d/"outputs.json").read_text())
    changed = False
    for o in outs:
        al = o.get("after_line")
        if al is not None and not (1 <= al <= nlines):
            o["after_line"] = max(1, min(int(al), nlines)); changed = True
    if changed:
        (d/"outputs.json").write_text(json.dumps(outs, indent=2, ensure_ascii=False))

for pl in PLAYLISTS:
    base = YT/pl
    if not base.exists(): continue
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d/"code.py").exists():
            fix(d)
print("auto-fix applied to all videos")
