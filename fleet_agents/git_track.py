"""git-track — commit the current code state and return the commit hash, so EVERY experiment maps to an exact
git commit (user 2026-07-12: "git init the competition folder; map each experiment on the journal + add a
commit-hash column"). Also commits research/official_repo (the training pipeline — its OWN git repo) when it has
changes, so a training experiment's code (INCLUDING the augmentations) is fully captured. Refreshes the ledger's
git-hash cache so subsequent experiment rows/decisions carry the new hash. Pure-ish (only shells out to git)."""
from __future__ import annotations
import subprocess
from .base import BaseAgent, COMP

OFFICIAL = COMP / "research" / "official_repo"
_TRAILER = "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"


def _git(args, cwd, timeout=90):
    """`timeout`: per-git-command wall-clock cap in seconds (default 90)."""
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        timeout = 90
    try:
        r = subprocess.run(["git", "-C", str(cwd)] + args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except Exception as e:  # noqa: BLE001
        return 1, "", f"{type(e).__name__}: {str(e)[:80]}"


def _head(cwd, timeout=90):
    return _git(["rev-parse", "--short", "HEAD"], cwd, timeout)[1] or "nogit"


def summarize_changes(cwd):
    """Auto-write a change summary from `git status` — counts by area + a few notable files, so every commit
    has a meaningful 'what changed' line even without a hand-written message (user 2026-07-12)."""
    from collections import Counter
    out = _git(["status", "--porcelain"], cwd)[1]
    if not out:
        return "", []
    files, tags = [], Counter()
    for ln in out.splitlines():
        st, f = ln[:2].strip(), ln[3:].strip()
        if " -> " in f:                                        # rename
            f = f.split(" -> ")[-1]
        if not f:
            continue
        files.append((st or "M", f))
        parts = f.split("/")
        area = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
        tags[area] += 1
    head = f"{len(files)} files: " + ", ".join(f"{a} ({n})" for a, n in tags.most_common(6))
    detail = [f"{st} {f}" for st, f in files[:40]]
    return head, detail


def commit_repo(cwd, message, timeout=90):
    """Stage + commit a repo if it has changes; return {committed, hash, clean, err}. No-op (clean) if nothing
    changed → returns the current HEAD so the experiment still maps to a hash. `timeout`: per-git-op cap (s)."""
    if not (cwd / ".git").exists():
        return {"committed": False, "hash": "nogit", "clean": True, "err": "not a git repo"}
    _git(["add", "-A"], cwd, timeout)
    dirty = _git(["status", "--porcelain"], cwd, timeout)[1]
    if not dirty:
        return {"committed": False, "hash": _head(cwd, timeout), "clean": True, "err": ""}
    code, _out, err = _git(["commit", "-m", message], cwd, timeout)
    return {"committed": code == 0, "hash": _head(cwd, timeout), "clean": False, "err": "" if code == 0 else err}


class GitTrack(BaseAgent):
    name = "git-track"
    thread = "V"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q) or {}
        msg = (spec.get("message") or spec.get("tag") or "experiment snapshot").strip()
        try:
            gtimeout = int(spec.get("timeout", 90))            # timeout: per-git-command cap in seconds
        except (TypeError, ValueError):
            gtimeout = 90
        # AUTO change-summary per repo → the commit body always says WHAT changed (user 2026-07-12).
        p_head, p_detail = summarize_changes(COMP)
        parent_msg = f"{msg}\n\n{p_head}\n" + "\n".join(p_detail) + f"\n\n{_TRAILER}" if p_head else f"{msg}\n\n{_TRAILER}"
        res = {"parent": commit_repo(COMP, parent_msg, gtimeout), "parent_summary": p_head}
        # no_official: skip committing the official_repo (e.g. when only the parent repo changed).
        if OFFICIAL.exists() and not spec.get("no_official"):   # the training pipeline (augs) — its own repo
            o_head, o_detail = summarize_changes(OFFICIAL)
            off_msg = f"{msg}\n\n{o_head}\n" + "\n".join(o_detail) + f"\n\n{_TRAILER}" if o_head else f"{msg}\n\n{_TRAILER}"
            res["official_repo"] = commit_repo(OFFICIAL, off_msg, gtimeout)
            res["official_summary"] = o_head
        try:
            from . import ledger
            newhash = ledger.refresh_git_hash()                # later ledger rows/decisions get the new hash
        except Exception as e:  # noqa: BLE001 — ledger unavailable → still report the commit hash we made
            newhash = res["parent"].get("hash", "nogit")
            _ = e
        parts = [f"parent={res['parent']['hash']}"
                 + ("" if res['parent']['clean'] else " (new commit)")]
        if "official_repo" in res:
            parts.append(f"official_repo={res['official_repo']['hash']}"
                         + ("" if res['official_repo']['clean'] else " (new commit)"))
        chg = res.get("parent_summary") or res.get("official_summary") or "no changes"
        summary = f"git-track: {'; '.join(parts)} → ledger stamps '{newhash}' · changed: {chg}"
        self.log(summary, kind="verdict",
                 recommendation="every ledger row + decision now carries this commit hash; the :7788 journal "
                                "maps each experiment → its exact code state. Commit BEFORE launching a run.")
        return self.done({"git_hash": newhash, **res}, summary)


_AGENT = GitTrack()


def run(q, worker):
    return _AGENT.run(q, worker)
