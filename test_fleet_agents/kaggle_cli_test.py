"""Verifier for the SHARED kaggle_cli layer. Offline: the CLI is stubbed, no network, no credentials.

This layer exists because 11 agents each had a private CLI wrapper, and that duplication hid a real bug for
an entire CLI release: `kaggle competitions pages <slug> ...` always fails on 2.2.4 (pages now needs a
subcommand), so every onboarding silently ran with EMPTY page text. These checks pin the call shapes and the
two parsing traps found against the live API.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "researchpapers"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fleet_agents import kaggle_cli as K  # noqa: E402


def _run():
    checks, calls = {}, []

    def fake(args, timeout=60, strict=False):
        calls.append(list(args))
        a = " ".join(args)
        if "pages list --content" in a or ("pages" in a and "--content" in a):
            return "Using competition: x\nname content\n---- ----\nEvaluation  Bradley-Terry win/loss ladder"
        if "pages" in a and "list" in a:
            return "Using competition: x\nname\nEvaluation\nrules\nTimeline"
        if "leaderboard" in a:
            # the CLI prepends a page-token line BEFORE the header, and names can contain commas
            return ('Next Page Token = ABC\n'
                    'teamId,teamName,submissionDate,score\n'
                    '1,"Smith, J",2026-07-30,1215.2\n2,Other,2026-07-29,1100.0\n')
        if "submissions" in a:
            return "ref,fileName,date,description,status,publicScore,privateScore\n1,s.csv,d,,COMPLETE,0.908,\n"
        if "files" in a:
            return "name,size,creationDate\nAGENTS.md,13057,x\nREADME.md,21917,x\n"
        return ""

    K.run, real = fake, K.run
    try:
        # PAGES: must set the default competition first, then use `pages list` — never `pages <slug>`
        names = K.page_names("kaggriculture")
        checks["page_names returns the page list"] = names == ["Evaluation", "rules", "Timeline"]
        checks["sets the default competition first"] = any(
            c[:3] == ["config", "set", "-n"] for c in calls)
        checks["never passes the slug as the pages command"] = not any(
            c[:2] == ["competitions", "pages"] and len(c) > 2 and c[2] == "kaggriculture" for c in calls)
        checks["uses the `pages list` subcommand"] = any(
            c[:3] == ["competitions", "pages", "list"] for c in calls)

        pages = K.all_pages("kaggriculture")
        checks["all_pages fetches every page"] = len(pages) == 3

        # LEADERBOARD: page-token preamble skipped, comma-in-name preserved
        lb = K.leaderboard("x", top=2)
        checks["leaderboard skips the page-token preamble"] = lb and lb[0].get("teamName") == "Smith, J"
        checks["and does not shred names containing commas"] = lb[0]["score"] == "1215.2"
        checks["top= limits rows"] = len(K.leaderboard("x", top=1)) == 1

        subs = K.submissions("x")
        checks["submissions parse publicScore"] = subs and subs[0]["publicScore"] == "0.908"
        checks["files returns the manifest"] = K.files("x") == ["AGENTS.md", "README.md"]

        # a CLI failure is a RESULT, not a crash
        K.run = lambda *a, **k: ""
        checks["empty CLI output yields [] not an exception"] = K.leaderboard("x") == []
    finally:
        K.run = real

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  {sum(1 for v in checks.values() if v)}/{len(checks)} passed")
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
