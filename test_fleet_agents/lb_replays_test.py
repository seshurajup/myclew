"""Data-wise verifier for lb-replays (agent-ladder episode mining). Offline: no network, no credentials.

The value of this agent is that it tells the truth about what Kaggle will and will not give you. Measured:
`ListEpisodes` accepts ONLY {submissionId} — {teamId}, {competitionId} and {competitionId, teamId} all
return HTTP 400 — and the leaderboard row exposes teamId/teamName/score/submissionDate and no submissionId.
So other teams' episodes are reachable only via Meta-Kaggle. A silent empty corpus would look exactly like
"no episodes played yet", so the agent must NAME the constraint instead.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "researchpapers"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fleet_agents import lb_replays as R  # noqa: E402


def _run():
    checks = {}
    checks["agent exposes run(q, worker)"] = callable(getattr(R, "run", None))
    checks["registered under the fleet name"] = getattr(R._AGENT, "name", "") == "lb-replays"
    for fn in ("competition_id", "top_teams", "submission_episodes", "our_submission_ids",
               "metakaggle_episodes", "download_episode"):
        checks[f"exposes {fn}()"] = callable(getattr(R, fn, None))

    # the episode endpoint must be called with submissionId — pin the shape that actually works
    import inspect
    src = inspect.getsource(R.submission_episodes)
    checks["ListEpisodes is called with submissionId"] = '"submissionId"' in src
    checks["the 400-returning shapes are documented"] = "400" in src

    # Meta-Kaggle path must fail LOUDLY when the dataset is absent
    try:
        R.metakaggle_episodes(1, [2], "/nonexistent-dir", 1)
        checks["missing Meta-Kaggle dir raises"] = False
    except FileNotFoundError as e:
        checks["missing Meta-Kaggle dir raises"] = "meta-kaggle" in str(e).lower()
    except Exception:  # noqa: BLE001
        checks["missing Meta-Kaggle dir raises"] = False

    # episodes are immutable: an already-downloaded replay must be skipped, not refetched
    src_dl = inspect.getsource(R.download_episode)
    checks["existing replays are skipped (immutable episodes)"] = "exists()" in src_dl

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  {sum(1 for v in checks.values() if v)}/{len(checks)} passed")
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
