"""github-solution-mine — winners publish their actual CODE on GitHub; this agent harvests it. It extracts
the repo links already present in the mined writeups (docs/gm_writeups/), lists each repo's tree via the
GitHub API (gh, authed), and fetches the KEY ML modules (train/model/loss/dataset/infer/config/augment/
ensemble) so their reusable code can be distilled into fleet agents — not re-derived from prose.

Light by design: uses `gh api` (tree + contents) instead of full clones. Injectable runner (`_gh`) so the
parse/index logic is data-wise tested OFFLINE. Output: docs/gm_writeups/_github_index.json = per-repo key
files, feeding trick-extractor/component-graft to add the missing modules.
"""
from __future__ import annotations
import base64
import json
import os
import re
import subprocess
from pathlib import Path
from .base import BaseAgent, COMP

GH = os.environ.get("GH_BIN", "gh")
WRITEUPS = COMP / "docs" / "gm_writeups"
INDEX = WRITEUPS / "_github_index.json"

# reusable ML modules worth extracting (winning code lives in these)
KEY_RX = re.compile(r"(train|model|models|loss|losses|dataset|datamodule|infer|inference|predict|pipeline|"
                    r"augment|aug|ensemble|blend|config|cfg|solver|postprocess|post_process|feature)", re.I)
_REPO_RX = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")


def _gh(args, timeout=45):
    """Run gh; return stdout ('' on failure). Injectable for offline tests."""
    try:
        r = subprocess.run([GH, *args], capture_output=True, text=True, timeout=timeout, env=dict(os.environ))
        return r.stdout if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def repos_from_writeups(writeups_dir=None):
    """Unique (owner, repo) pairs linked in the writeup markdown. Skips obvious library deps."""
    d = Path(writeups_dir or WRITEUPS)
    SKIP = {"albumentations", "sam2", "diffvg", "flagembedding", "autoawq", "anthropic-cookbook",
            "schedule_free", "python-blosc2", "arnie", "cvat", "glue-factory", "slowfast"}
    seen = {}
    for md in d.rglob("*.md"):
        for owner, repo in _REPO_RX.findall(md.read_text(errors="replace")):
            repo = repo.replace(".git", "")
            if repo.lower() in SKIP:
                continue
            seen[(owner, repo)] = seen.get((owner, repo), 0) + 1
    return sorted(seen)


def repo_tree(owner, repo, gh_timeout=None):
    """List file paths in a repo (default branch, recursive) via the GitHub API.
    gh_timeout: per gh-call wall-clock cap in seconds (None → the _gh default)."""
    kw = {} if gh_timeout is None else {"timeout": int(gh_timeout)}
    meta = _gh(["api", f"repos/{owner}/{repo}"], **kw)
    branch = "main"
    try:
        branch = json.loads(meta).get("default_branch", "main")
    except Exception:  # noqa: BLE001
        pass
    out = _gh(["api", f"repos/{owner}/{repo}/git/trees/{branch}?recursive=1"], **kw)
    try:
        tree = json.loads(out).get("tree", [])
        return [t["path"] for t in tree if t.get("type") == "blob"]
    except Exception:  # noqa: BLE001
        return []


def key_files(paths, limit=12):
    """The reusable ML .py files (+ README) from a repo tree, prioritized."""
    py = [p for p in paths if p.endswith(".py") and KEY_RX.search(os.path.basename(p))]
    readme = [p for p in paths if os.path.basename(p).lower().startswith("readme")]
    return (readme[:1] + py)[:limit]


def fetch_file(owner, repo, path, gh_timeout=None):
    """Fetch a file's text via the contents API (base64-decoded).
    gh_timeout: per gh-call wall-clock cap in seconds (None → the _gh default)."""
    kw = {} if gh_timeout is None else {"timeout": int(gh_timeout)}
    out = _gh(["api", f"repos/{owner}/{repo}/contents/{path}"], **kw)
    try:
        c = json.loads(out).get("content", "")
        return base64.b64decode(c).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


def mine(repos, out_dir=None, fetch=True, per_repo=8, gh_timeout=None):
    """Index repos → {owner/repo: {n_files, key_files, saved}}. Saves key files under out_dir for distillation.
    gh_timeout: per gh-call wall-clock cap in seconds passed to every API call (None → the _gh default)."""
    out_dir = Path(out_dir or (WRITEUPS / "_github"))
    index = {}
    for owner, repo in (repos or []):
        paths = repo_tree(owner, repo, gh_timeout=gh_timeout)
        kf = key_files(paths, limit=per_repo)
        saved = []
        if fetch and kf:
            rd = out_dir / f"{owner}__{repo}"; rd.mkdir(parents=True, exist_ok=True)
            for p in kf:
                txt = fetch_file(owner, repo, p, gh_timeout=gh_timeout)
                if txt:
                    fn = rd / p.replace("/", "__")
                    fn.write_text(txt); saved.append(str(fn))
        index[f"{owner}/{repo}"] = {"n_files": len(paths), "key_files": kf, "saved": len(saved)}
    return index


class GithubSolutionMine(BaseAgent):
    name = "github-solution-mine"
    thread = "R"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        repos = spec.get("repos")
        if repos:
            repos = [tuple(r.split("/")) if isinstance(r, str) else tuple(r) for r in repos]
        else:
            repos = repos_from_writeups(spec.get("writeups_dir"))
        limit = int(spec.get("limit", 0))
        if limit:
            repos = repos[:limit]
        idx = mine(repos, out_dir=spec.get("out_dir"), fetch=bool(spec.get("fetch", True)),
                   per_repo=int(spec.get("per_repo", 8)), gh_timeout=spec.get("gh_timeout"))
        INDEX.write_text(json.dumps(idx, indent=2))
        got = sum(1 for v in idx.values() if v["saved"] > 0)
        tot = sum(v["saved"] for v in idx.values())
        msg = (f"github-solution-mine: indexed {len(idx)} winner repos, fetched {tot} key ML modules from "
               f"{got} repos → {INDEX.name}. Feed to trick-extractor/component-graft to add missing modules.")
        self.log(msg, kind="finding", recommendation="distill saved modules into reusable agents (loss/model/aug/postproc)")
        return self.done({"index": idx, "repos": len(idx), "modules_fetched": tot}, msg)


_AGENT = GithubSolutionMine()


def run(q, worker):
    return _AGENT.run(q, worker)
