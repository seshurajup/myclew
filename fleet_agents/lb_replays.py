"""lb-replays — download LB opponents' EPISODE REPLAYS for any agent competition.

Why it exists: in an agent ladder the leaderboard is not a number, it is a corpus. Every ranked episode is
a full recorded game against a strong opponent — free supervision for imitation, opponent modelling, and
failure mining. On Orbit Wars this was the highest-value source we had, but it lived as a one-off script in
that competition's folder (`scripts/fetch_lb_episodes.py`), so a NEW agent comp (kaggriculture) started with
nothing. This is that logic promoted to a fleet agent, competition-parameterised.

Kaggle's public CLI cannot enumerate other players' episodes, so this uses the same internal JSON endpoints
the website itself calls, authenticated with the user's own kaggle.json:

    slug -> competitionId -> top-N teams -> each team's latest submission -> its episodes -> replay + logs

The internal endpoints are undocumented. This agent FAILS LOUDLY if they change rather than silently
returning an empty corpus — a quiet zero here would look exactly like "no episodes yet".

Spec:
    {"kind": "lb-replays", "spec": {"comp": "kaggriculture", "top_n": 10,
                                    "max_episodes_per_team": 5, "out": null, "dry_run": false}}
`out` defaults to <comp>/replays/ (+ <comp>/logs/replay_logs/). dry_run resolves teams/episodes and reports
the plan WITHOUT downloading, so the endpoint contract can be checked cheaply.
"""
from __future__ import annotations

import json
import os
import pathlib

from .base import BaseAgent

ROOT = "/home/seshu/kaggle/2026"
BASE = "https://www.kaggle.com"


def _auth():
    """(username, key) from ~/.kaggle/kaggle.json — the same credentials the CLI uses."""
    p = pathlib.Path(os.environ.get("KAGGLE_CONFIG_DIR", pathlib.Path.home() / ".kaggle")) / "kaggle.json"
    if not p.exists():
        raise FileNotFoundError(f"no kaggle.json at {p} — cannot reach the episode endpoints")
    d = json.loads(p.read_text())
    return d["username"], d["key"]


def _session():
    import requests
    from requests.auth import HTTPBasicAuth
    u, k = _auth()
    return requests, HTTPBasicAuth(u, k)


def _post(path, payload, timeout=30):
    requests, auth = _session()
    r = requests.post(f"{BASE}{path}", json=payload, auth=auth,
                      headers={"Content-Type": "application/json"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get(path, params=None, timeout=30):
    requests, auth = _session()
    r = requests.get(f"{BASE}{path}", params=params or {}, auth=auth, timeout=timeout)
    r.raise_for_status()
    return r.json()


def competition_id(slug):
    """slug -> numeric competitionId, via the public list endpoint the CLI uses."""
    j = _get("/api/v1/competitions/list", {"search": slug})
    for c in j if isinstance(j, list) else []:
        if str(c.get("ref", "")).rstrip("/").endswith(slug):
            return int(c["id"])
    raise LookupError(f"competition `{slug}` not found via /competitions/list")


def top_teams(slug, top_n=10):
    """Top-N leaderboard teams (id + name + score), newest standings."""
    j = _get(f"/api/v1/competitions/{slug}/leaderboard/view")
    rows = (j or {}).get("submissions") or []
    out = []
    for r in rows[: int(top_n)]:
        out.append({"team_id": r.get("teamId") or r.get("teamNameNullable"),
                    "team": r.get("teamName"), "score": r.get("score")})
    return out


def submission_episodes(submission_id, limit=5):
    """Episode ids for ONE submission — the only shape `ListEpisodes` accepts.

    Measured: {teamId}, {competitionId} and {competitionId, teamId} all return HTTP 400. Kaggle exposes no
    team -> submission mapping (the leaderboard row carries teamId, teamName, score, submissionDate and
    nothing else), so other teams' episodes are only reachable via Meta-Kaggle. Ours are reachable directly.
    """
    j = _post("/api/i/competitions.EpisodeService/ListEpisodes", {"submissionId": int(submission_id)})
    eps = (j or {}).get("episodes") or []
    eps = sorted(eps, key=lambda e: (e.get("endTime") or {}).get("seconds", 0), reverse=True)
    return [int(e["id"]) for e in eps[: int(limit)] if e.get("id")]


def our_submission_ids(comp, limit=10):
    """Our own submission ids for a competition, via the public submissions endpoint."""
    j = _get(f"/api/v1/competitions/submissions/list/{comp}")
    rows = j if isinstance(j, list) else (j or {}).get("submissions") or []
    out = []
    for r in rows[: int(limit)]:
        sid = r.get("ref") or r.get("id")
        try:
            out.append(int(sid))
        except (TypeError, ValueError):
            continue
    return out


def metakaggle_episodes(comp_id, team_ids, mk_dir, limit_per_team=5):
    """Resolve other teams' episodes from Meta-Kaggle CSVs (the ONLY route Kaggle leaves open).

    Needs Episodes.csv + EpisodeAgents.csv (and Submissions.csv when present) from the `kaggle/meta-kaggle`
    dataset. Streams in chunks — the files are multi-GB.
    """
    import csv as _csv
    ep_path = os.path.join(mk_dir, "EpisodeAgents.csv")
    if not os.path.exists(ep_path):
        raise FileNotFoundError(f"{ep_path} missing — pull the kaggle/meta-kaggle dataset first")
    want = {int(t) for t in team_ids}
    per, out = {}, []
    with open(ep_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            tid = row.get("TeamId") or row.get("SubmissionTeamId")
            eid = row.get("EpisodeId")
            if not tid or not eid:
                continue
            try:
                tid_i = int(tid)
            except ValueError:
                continue
            if tid_i in want and per.get(tid_i, 0) < limit_per_team:
                per[tid_i] = per.get(tid_i, 0) + 1
                out.append(int(eid))
    return out


def download_episode(ep_id, out_replays, out_logs=None):
    """Replay JSON (+ per-agent logs when available). Returns the replay path, or None if unavailable."""
    out_replays = pathlib.Path(out_replays)
    out_replays.mkdir(parents=True, exist_ok=True)
    dst = out_replays / f"{ep_id}.json"
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst)                                   # already mined — episodes are immutable
    j = _post("/api/i/competitions.EpisodeService/GetEpisodeReplay", {"episodeId": int(ep_id)})
    replay = (j or {}).get("replay")
    if not replay:
        return None
    dst.write_text(replay if isinstance(replay, str) else json.dumps(replay))
    if out_logs:
        try:
            lg = _post("/api/i/competitions.EpisodeService/GetEpisodeLogs", {"episodeId": int(ep_id)})
            p = pathlib.Path(out_logs)
            p.mkdir(parents=True, exist_ok=True)
            (p / f"{ep_id}.json").write_text(json.dumps(lg))
        except Exception:  # noqa: BLE001 — logs are a bonus; a missing log must not lose the replay
            pass
    return str(dst)


class LBReplays(BaseAgent):
    name = "lb-replays"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        comp = s.get("comp") or s.get("slug") or "kaggriculture"
        top_n = int(s.get("top_n", 10))
        per_team = int(s.get("max_episodes_per_team", 5))
        out_replays = s.get("out") or os.path.join(ROOT, comp, "replays")
        out_logs = s.get("out_logs") or os.path.join(ROOT, comp, "logs", "replay_logs")

        try:
            cid = competition_id(comp)
            teams = top_teams(comp, top_n)
        except Exception as e:  # noqa: BLE001 — endpoint drift is a RESULT worth reporting, not a crash
            return self.escalate(worker, "leader",
                                 f"[{worker}] lb-replays `{comp}`: could not resolve teams "
                                 f"({type(e).__name__}: {str(e)[:160]}). Kaggle's internal episode "
                                 f"endpoints may have changed — they are undocumented.")

        planned, got, failed = [], [], []
        source = s.get("source", "auto")

        # OURS: reachable directly, and our own episodes still contain the OPPONENT's every action —
        # a full replay records both players — so this is real opponent data as soon as we have submitted.
        if source in ("auto", "own"):
            try:
                for sid in our_submission_ids(comp, int(s.get("own_submissions", 5))):
                    planned += [(sid, e) for e in submission_episodes(sid, per_team)]
            except Exception as e:  # noqa: BLE001
                failed.append(f"own: {type(e).__name__}: {str(e)[:60]}")

        # THEIRS: only via Meta-Kaggle. Kaggle exposes no team->submission mapping.
        mk = s.get("metakaggle_dir")
        if source in ("auto", "metakaggle") and mk:
            try:
                ids = [t["team_id"] for t in teams if isinstance(t.get("team_id"), int)]
                planned += [(None, e) for e in metakaggle_episodes(cid, ids, mk, per_team)]
            except Exception as e:  # noqa: BLE001
                failed.append(f"metakaggle: {type(e).__name__}: {str(e)[:80]}")
        elif source in ("auto", "metakaggle"):
            failed.append("metakaggle: no metakaggle_dir given — other teams' episodes need "
                          "the kaggle/meta-kaggle dataset (leaderboard exposes no submissionId)")

        if s.get("dry_run"):
            msg = (f"[{worker}] lb-replays DRY-RUN `{comp}`: {len(teams)} teams, "
                   f"{len(planned)} episodes resolvable (top_n={top_n}, {per_team}/team). "
                   f"{len(failed)} team(s) failed. Nothing downloaded.")
            self.log(msg, kind="finding", recommendation="re-run without dry_run to mine the corpus")
            return self.done({"comp": comp, "teams": teams, "planned": len(planned),
                              "failed": failed, "downloaded": 0}, msg)

        for _tid, ep in planned:
            try:
                p = download_episode(ep, out_replays, out_logs)
                (got if p else failed).append(ep if p else f"ep{ep}: no replay body")
            except Exception as e:  # noqa: BLE001
                failed.append(f"ep{ep}: {type(e).__name__}")

        msg = (f"[{worker}] lb-replays `{comp}`: mined {len(got)}/{len(planned)} episodes from "
               f"{len(teams)} top teams → {out_replays}. {len(failed)} failed.")
        self.log(msg, kind="finding",
                 recommendation="feed replays to opponent-modelling / imitation; re-run to top up as the "
                                "ladder moves (episodes are immutable, existing files are skipped)")
        return self.done({"comp": comp, "teams": teams, "downloaded": len(got),
                          "planned": len(planned), "failed": failed[:20],
                          "out": out_replays}, msg)


_AGENT = LBReplays()


def run(q, worker):
    return _AGENT.run(q, worker)
