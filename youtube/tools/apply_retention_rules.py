"""One-shot migration: bring existing video folders up to the retention rules in README §3.9.

Rules applied (idempotent — safe to re-run):
  1. HOOK      — replace label-style hooks with the promise-style ones in hooks.tsv (spec.json).
  2. COLD OPEN — strip paralinguistic tags from the FIRST transcript segment (they burn second 1).
  3. TAIL      — append a closing segment: next-video pointer + one engagement CTA, so the
                 series binges and viewers have something to do besides swipe.

Usage:  python tools/apply_retention_rules.py <hooks.tsv>   [--dry-run]
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retention_rules import TAG_RE, is_tail  # noqa: E402

YT = Path(__file__).resolve().parent.parent

# Each SERIES is an independent binge chain: "up next" threads through the playlists WITHIN a
# series, and the final video of each series loops the viewer back to its own start. The seven
# python-* playlists are one numbered curriculum (001-081), so they form a single chain.
SERIES = [
    (["01-python-basics", "02-python-functions", "03-python-loops-iteration", "04-python-oop",
      "05-python-advanced", "06-python-testing-tools", "07-python-libraries"],
     "And that's the whole Python series, all eighty-one videos. "
     "Start back at number one and binge it. Comment what I should build next."),
]

# rotated so a playlist binge never hears the same closing twice in a row
NEXT_TMPL = [
    "Next video: {topic}.",
    "Up next, {topic}.",
    "In the next one we take on {topic}.",
    "{topic} is next, video number {nn}.",
    "Video {nn} tackles {topic}.",
]
CTA_TMPL = [
    "Comment the topic you want after that.",
    "Follow so you don't miss it.",
    "Save this one for when you need it.",
    "Tell me in the comments if this clicked.",
    "Comment PYTHON if you want the cheat sheet.",
    "Which one do you already use? Comment below.",
]
TOPICS = dict(l.split("\t", 1) for l in
              (Path(__file__).resolve().parent / "topics.tsv").read_text().strip().split("\n"))


def topic_of(slug: str) -> str:
    """Spoken name of a video, e.g. 003-numbers-math -> 'numbers and math'. Falls back to the
    slug so a newly added video still builds (the gate will flag a missing topics.tsv row)."""
    return TOPICS.get(slug, slug.split("-", 1)[1].replace("-", " "))


def series_dirs():
    """[(ordered video dirs, final-tail text)] — one entry per independent binge chain."""
    out = []
    for playlists, final in SERIES:
        dirs = []
        for pl in playlists:
            base = YT / pl
            if not base.exists():
                continue
            dirs += [d for d in sorted(base.iterdir())
                     if d.is_dir() and (d / "code.py").exists()]
        if dirs:
            out.append((dirs, final))
    return out


def main():
    hooks_tsv = Path(sys.argv[1])
    dry = "--dry-run" in sys.argv
    hooks = dict(l.split("\t", 1) for l in hooks_tsv.read_text().strip().split("\n"))
    changed = {"hook": 0, "coldopen": 0, "tail": 0}
    total = 0

    for dirs, final_tail in series_dirs():
      total += len(dirs)
      for i, d in enumerate(dirs):
        key = f"{d.parent.name}/{d.name}"
        spec_p, tr_p = d / "spec.json", d / "transcript.json"
        spec = json.loads(spec_p.read_text())
        tr = json.loads(tr_p.read_text())

        # 1. hook
        new_hook = hooks.get(key)
        if new_hook and spec.get("hook") != new_hook:
            spec["hook"] = new_hook
            changed["hook"] += 1
            if not dry:
                spec_p.write_text(json.dumps(spec, indent=2) + "\n")

        # 2. cold open: no vocal tag in the opening line
        first = tr[0]["text"]
        stripped = " ".join(TAG_RE.sub(" ", first).split())
        stripped = stripped[0].upper() + stripped[1:] if stripped else stripped
        if stripped != first:
            tr[0]["text"] = stripped
            changed["coldopen"] += 1

        # 3. tail: next-video pointer + CTA (replace ours if already present, never stack)
        nlines = len((d / "code.py").read_text().rstrip("\n").split("\n"))
        if tr and is_tail(tr[-1]["text"]):
            tr.pop()
        if i + 1 < len(dirs):
            nxt = dirs[i + 1]
            nn = int(nxt.name.split("-")[0])
            nxt_txt = NEXT_TMPL[i % len(NEXT_TMPL)].format(topic=topic_of(nxt.name), nn=nn)
            text = f"{nxt_txt} {CTA_TMPL[i % len(CTA_TMPL)]}"
        else:
            text = final_tail
        tr.append({"text": text, "until_line": nlines})
        changed["tail"] += 1

        if not dry:
            tr_p.write_text(json.dumps(tr, indent=2, ensure_ascii=False) + "\n")

    print(f"{total} videos | hooks rewritten {changed['hook']} | "
          f"cold opens cleaned {changed['coldopen']} | tails set {changed['tail']}"
          + (" (DRY RUN, nothing written)" if dry else ""))


if __name__ == "__main__":
    main()
