"""Single source of truth for the Shorts retention rules (README §3.9).

Imported by the pre-build gate (tools/prebuild_check.py), the repo-wide auditor
(tools/validate_videos.py) and the one-shot migration (tools/apply_retention_rules.py)
so the three can never disagree about what a compliant video looks like.

Each rule returns a list of problem strings; empty list = compliant.
"""
import re

TAG_RE = re.compile(r"\[[a-z ]+\]")

# --- rule 1: the hook ---------------------------------------------------------
# The hook is three things at once: the title card (= the Shorts thumbnail), the
# YouTube title, and the first line of the description. A noun label ("Lists:
# ordered collections") names a topic; it does not give anyone a reason not to swipe.
LABEL_RE = re.compile(r"^[\w &,+\-*/%.()'\"]{1,30}:")   # "Topic: description" shape
# A promise needs one of: an imperative verb, a second-person address, a question, or a
# curiosity noun (a named bug/gotcha/trap the viewer might be living with). This is a curated
# allowlist on purpose — if a new hook fails it, either open with a listed word or add yours
# here deliberately, rather than shipping another bare topic label.
PROMISE_RE = re.compile(
    r"\b(you|your|stop|never|why|what|how|which|when|make|build|write|turn|replace|delete|"
    r"catch|find|kill|master|reuse|force|prove|sort|loop|read|move|test|cache|grab|chain|"
    r"store|pass|teach|talk|swap|roll|meet|understand|upgrade|unpack|look|assign|raise|"
    r"yield|divide|tally|remove|produce|accept|generate|guarantee|change|count|group|"
    r"does|is|are|do|can|will|use|the one|three|four)\b"
    r"|\b(bug|gotcha|trap|mistake|secret|trick|difference|reason)\b"
    r"|\?", re.I)

# --- rule 3: the tail ---------------------------------------------------------
NEXT_RE = re.compile(r"\b(next video|up next|next one|is next|video \w+ tackles|next\b)", re.I)
# The LAST video of a series has nothing to point forward to, so it must instead send the viewer
# back to the start of the series — same job (keep them on the channel), different direction.
SERIES_END_RE = re.compile(r"\b(start back at|that's the whole|that's the|work through them all|"
                           r"binge it)\b", re.I)
CTA_RE = re.compile(r"\b(comment|follow|save this|subscribe|tell me)\b", re.I)


def is_tail(text: str) -> bool:
    """True if this segment is a generated closing tail (forward pointer or series closer + CTA)."""
    return bool((NEXT_RE.search(text) or SERIES_END_RE.search(text)) and CTA_RE.search(text))


def check_hook(hook: str):
    p = []
    h = (hook or "").strip()
    if not h:
        return ["hook missing"]
    if LABEL_RE.match(h):
        p.append(f"hook is a noun label, not a promise: {h!r} (drop the 'Topic:' prefix)")
    if len(h.split()) < 4:
        p.append(f"hook only {len(h.split())} words: {h!r} (needs >=4 to make a promise)")
    if not PROMISE_RE.search(h):
        p.append(f"hook has no verb/second-person/question: {h!r}")
    if len(h) > 60:
        p.append(f"hook {len(h)} chars (>60 wraps badly on the title card / truncates in the feed)")
    return p


def check_cold_open(transcript):
    """Second 1 decides the swipe — no vocal tag may sit in front of the first word."""
    if not transcript:
        return ["empty transcript"]
    first = transcript[0]["text"]
    tags = TAG_RE.findall(first)
    if tags:
        return [f"first segment opens with vocal tag(s) {tags} — burns the hook second"]
    return []


def check_tail(transcript):
    """Last segment must point at the next video AND ask for one interaction."""
    if not transcript:
        return ["empty transcript"]
    last = transcript[-1]["text"]
    p = []
    if not (NEXT_RE.search(last) or SERIES_END_RE.search(last)):
        p.append("last segment has no next-video pointer and no series closer (never binges)")
    if not CTA_RE.search(last):
        p.append("last segment has no engagement CTA (comment/follow/save)")
    return p


def check_retention(spec, transcript):
    """All retention rules at once."""
    return (check_hook(spec.get("hook", ""))
            + check_cold_open(transcript)
            + check_tail(transcript))
