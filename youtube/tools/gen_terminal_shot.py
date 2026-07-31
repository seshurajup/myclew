"""Render a REAL terminal capture for each video by actually executing its code.

Why this exists
---------------
The channel's promise is "watch it typed & run", but every output pane was a hand-authored
`>>> ...` text box — 244 of them, 0 images — and two of them were provably wrong. Reviewers
looking for authentic coding content want to see the actual command executing; a feed of 81
identical text boxes reads as templated.

This runs each video's code for real and paints the captured session into a terminal-window PNG
(title bar, `$ python code.py` prompt, true stdout/stderr, exit status). The image lands in
`<video>/artifacts/terminal.png` and is attached as the video's closing output event, so the last
thing a viewer sees is the genuine run — and it looks different in every video, because the
content is different.

Usage:
    python tools/gen_terminal_shot.py                 # all 81
    python tools/gen_terminal_shot.py 04-python-oop   # one playlist
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

YT = Path(__file__).resolve().parent.parent
PLAYLISTS = ["01-python-basics", "02-python-functions", "03-python-loops-iteration",
             "04-python-oop", "05-python-advanced", "06-python-testing-tools", "07-python-libraries"]

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# terminal palette — deliberately NOT the editor palette, so the run reads as a separate surface
BG, CHROME, BORDER = "#0d1117", "#161b22", "#30363d"
FG, DIM = "#c9d1d9", "#8b949e"
GREEN, RED, PROMPT = "#7ee787", "#ff7b72", "#79c0ff"

W, PAD, LH, FS = 1180, 34, 42, 30
MAX_LINES = 16           # keep the image short enough for the 320px output zone


def run_code(d: Path, timeout=30):
    spec = json.loads((d / "spec.json").read_text())
    cmd = spec.get("run") or ["python", "code.py"]
    real = [sys.executable if c == "python" else c for c in cmd]
    r = subprocess.run(real, cwd=d, capture_output=True, text=True, timeout=timeout)
    # display the file under the name shown on the editor tab, which is what a viewer would type
    display = spec.get("title") or "code.py"
    shown = " ".join(display if c == "code.py" else c for c in cmd)
    return shown, display, r


def wrap(s, cols):
    out = []
    for line in s.split("\n"):
        while len(line) > cols:
            out.append(line[:cols])
            line = line[cols:]
        out.append(line)
    return out


def render(d: Path) -> Path | None:
    try:
        shown, display, r = run_code(d)
    except Exception as e:                                        # noqa: BLE001
        print(f"  ! {d.name}: cannot run ({e})")
        return None

    f = ImageFont.truetype(MONO, FS)
    fb = ImageFont.truetype(MONO_B, FS)
    cols = (W - 2 * PAD) // int(f.getlength("M"))

    body = [("prompt", f"$ {shown}")]
    for line in wrap((r.stdout or "").rstrip("\n"), cols):
        body.append(("out", line))
    for line in wrap((r.stderr or "").rstrip("\n"), cols):
        if line.strip():
            body.append(("err", line))
    if len(body) > MAX_LINES:
        body = body[:MAX_LINES - 1] + [("dim", f"… {len(body) - MAX_LINES + 1} more lines")]
    body.append(("exit", f"exit {r.returncode}"))

    h = 78 + PAD + len(body) * LH + PAD
    img = Image.new("RGB", (W, h), BG)
    dr = ImageDraw.Draw(img)
    # window chrome
    dr.rectangle([0, 0, W, 78], fill=CHROME)
    dr.line([0, 78, W, 78], fill=BORDER, width=2)
    for i, c in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        dr.ellipse([PAD + i * 40, 30, PAD + i * 40 + 20, 50], fill=c)
    dr.text((PAD + 148, 24), display, font=f, fill=DIM)

    y = 78 + PAD
    for kind, line in body:
        if kind == "prompt":
            dr.text((PAD, y), "$ ", font=fb, fill=PROMPT)
            dr.text((PAD + f.getlength("$ "), y), line[2:], font=fb, fill=FG)
        elif kind == "err":
            dr.text((PAD, y), line, font=f, fill=RED)
        elif kind == "dim":
            dr.text((PAD, y), line, font=f, fill=DIM)
        elif kind == "exit":
            dr.text((PAD, y), line, font=f, fill=GREEN if r.returncode == 0 else RED)
        else:
            dr.text((PAD, y), line, font=f, fill=FG)
        y += LH

    art = d / "artifacts"
    art.mkdir(exist_ok=True)
    out = art / "terminal.png"
    img.save(out)
    return out


def attach(d: Path, png: Path):
    """Add/refresh the closing terminal event in outputs.json (idempotent)."""
    p = d / "outputs.json"
    outs = json.loads(p.read_text())
    nlines = len((d / "code.py").read_text().rstrip("\n").split("\n"))
    outs = [o for o in outs if o.get("image") != "artifacts/terminal.png"]
    # no caption: the prompt line already shows the command, so any label is redundant
    outs.append({"after_line": nlines, "image": "artifacts/terminal.png"})
    p.write_text(json.dumps(outs, indent=2, ensure_ascii=False) + "\n")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    dirs = []
    for pl in PLAYLISTS:
        if only and pl != only:
            continue
        dirs += [d for d in sorted((YT / pl).iterdir())
                 if d.is_dir() and (d / "code.py").exists()]
    made = 0
    for d in dirs:
        png = render(d)
        if png:
            attach(d, png)
            made += 1
            print(f"  ✓ {d.parent.name}/{d.name}")
    print(f"\n{made}/{len(dirs)} terminal captures generated from real execution")


if __name__ == "__main__":
    main()
