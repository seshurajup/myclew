# Semantic critic: flags segments whose narration references a code symbol/operator that
# appears in the code AFTER the segment's until_line (talking about code not yet shown) —
# the 003-style "right order, wrong line" desync mechanical checks miss.
import json, re, sys
from pathlib import Path
YT = Path("/home/seshu/kaggle/2026/youtube")
PLAYLISTS = ["01-python-basics","02-python-functions","03-python-loops-iteration",
             "04-python-oop","05-python-advanced","06-python-testing-tools","07-python-libraries"]

# narration phrase -> code token that MUST be on/before the pinned line
OPS = [
    (r"\bmodulo\b|\bremainder\b|percent sign", "%"),
    (r"floor division|integer division|double slash", "//"),
    (r"\bpower\b|double star|raises? (a )?number|cubed|squared via", "**"),
]
def first_lines(code):
    return code.rstrip("\n").split("\n")

def symbol_last_line(lines, token):
    hits = [i+1 for i,l in enumerate(lines) if token in l]
    return hits

def critic(d):
    problems = []
    lines = first_lines((d/"code.py").read_text())
    t = json.loads((d/"transcript.json").read_text())
    # method names actually called in code: .name(
    methods = {}
    for i,l in enumerate(lines):
        for m in re.findall(r"\.(\w+)\(", l):          # method call
            methods.setdefault(m, []).append(i+1)
        for m in re.findall(r"\bdef\s+(\w+)", l):       # method/function DEFINITION counts too
            methods.setdefault(m, []).append(i+1)
    ul_prev = 0
    for si, s in enumerate(t):
        ul = s.get("until_line") or len(lines)
        text = s["text"].lower()
        covered = set(range(1, ul+1))
        # operator checks
        for pat, tok in OPS:
            if re.search(pat, text):
                occ = symbol_last_line(lines, tok)
                if occ and not any(o in covered for o in occ):
                    problems.append(f"seg{si}: mentions '{tok}' but pinned line {ul} doesn't cover it (code at {occ})")
        # method checks: narration names a method that's called in code
        for m, mlines in methods.items():
            if len(m) >= 4 and re.search(rf"\b{m}\b", text):
                if not any(ml in covered for ml in mlines):
                    problems.append(f"seg{si}: mentions '.{m}()' but pinned line {ul} doesn't cover it (code at {mlines})")
        ul_prev = ul
    return problems

total=bad=0
for pl in PLAYLISTS:
    base = YT/pl
    if not base.exists(): continue
    for d in sorted(base.iterdir()):
        if not (d.is_dir() and (d/"code.py").exists()): continue
        total+=1
        p = critic(d)
        if p:
            bad+=1
            print(f"REVIEW {pl}/{d.name}:")
            for x in p: print("    -",x)
print(f"\n{total} checked, {bad} flagged for semantic review")
