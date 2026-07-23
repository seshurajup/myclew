"""lb-sync — periodically SNAPSHOT the competition leaderboard (official Kaggle API) and ESTIMATE activity:
how long since each LB member last submitted, how many are still active, and how the top is moving. Run every
~3h by a cron (user 2026-07-12: "sync the competition every 3 hours to estimate how long since LB members last
submitted"). Uses the OFFICIAL `kaggle competitions leaderboard --show --csv` (teamId,teamName,submissionDate,
score) — no scraping. Snapshots append to docs/lb_history.jsonl so we can watch movement over time.

Timestamps from Kaggle are UTC; we report IST (UTC+5:30) per the user rule.
"""
from __future__ import annotations
from .base import BaseAgent, COMP

HIST = "docs/lb_history.jsonl"
IST_OFFSET_H = 5.5


def _parse_dt(s):
    """'2026-07-11 00:25:45.093000' (UTC) → epoch hours, or None."""
    import datetime
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.datetime.strptime(str(s).strip(), fmt).replace(tzinfo=datetime.timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def _analyze(rows, now):
    """PURE activity analysis (data-wise tested). rows=[{teamName,submissionDate,score}], now=aware datetime.
    Returns per-competition activity: leader, top score, N teams, hours since leader/median last submission,
    active counts in 24/72h. 'submissionDate' is each team's BEST-submission time = last time they improved."""
    import datetime
    parsed = []
    for r in rows:
        dt = _parse_dt(r.get("submissionDate"))
        try:
            sc = float(r.get("score"))
        except (TypeError, ValueError):
            sc = None
        if dt is not None and sc is not None:
            age_h = (now - dt).total_seconds() / 3600.0
            parsed.append({"team": r.get("teamName", "?"), "score": sc, "age_h": age_h, "dt": dt})
    if not parsed:
        return {"n_teams": 0}
    parsed.sort(key=lambda d: -d["score"])
    ages = sorted(d["age_h"] for d in parsed)
    med = ages[len(ages) // 2]
    leader = parsed[0]
    return {
        "n_teams": len(parsed),
        "top_score": round(leader["score"], 4),
        "leader": leader["team"],
        "leader_last_sub_age_h": round(leader["age_h"], 1),
        "median_last_sub_age_h": round(med, 1),
        "active_24h": sum(1 for d in parsed if d["age_h"] <= 24),
        "active_72h": sum(1 for d in parsed if d["age_h"] <= 72),
        "top10_active_24h": sum(1 for d in parsed[:10] if d["age_h"] <= 24),
        "freshest_age_h": round(ages[0], 1),
    }


class LbSync(BaseAgent):
    name = "lb-sync"
    thread = "S"
    kind = "finding"

    def _fetch(self, competition, timeout=60):
        """Pull the official LB CSV; ANY CLI/network failure → [] (caller handles empty, never crashes)."""
        import subprocess, shutil, csv, io
        kaggle = shutil.which("kaggle") or "/home/seshu/miniconda3/envs/llm/bin/kaggle"
        try:
            r = subprocess.run([kaggle, "competitions", "leaderboard", competition, "--show", "--csv"],
                               capture_output=True, text=True, timeout=max(1, int(timeout)))
            lines = [ln for ln in r.stdout.splitlines() if not ln.startswith("Next Page Token")]
            return list(csv.DictReader(io.StringIO("\n".join(lines))))
        except Exception:  # noqa: BLE001
            return []

    def _team_rows(self, rows, now):
        """Per-team detail rows (rank-sorted) for the PG lb_team table + the :7777 LB page."""
        teams = []
        for r in rows:
            dt = _parse_dt(r.get("submissionDate"))
            try:
                sc = float(r.get("score"))
            except (TypeError, ValueError):
                sc = None
            if dt is not None and sc is not None:
                teams.append({"team": r.get("teamName"), "team_id": r.get("teamId"), "score": sc,
                              "submission_utc": dt.isoformat(),
                              "age_h": round((now - dt).total_seconds() / 3600.0, 1)})
        teams.sort(key=lambda t: -t["score"])
        return teams

    def _snapshot(self, competition, analysis, now, rows=None):
        import json, os
        path = COMP / HIST
        os.makedirs(path.parent, exist_ok=True)
        rec = {"competition": competition, "utc": now.isoformat(), **analysis}
        with open(path, "a") as f:                              # JSONL backup
            f.write(json.dumps(rec) + "\n")
        try:                                                    # PG = queryable store (powers the LB page)
            from . import db
            db.insert_lb(competition, now.isoformat(), analysis, self._team_rows(rows or [], now))
        except Exception:  # noqa: BLE001
            pass
        return str(path)

    def run(self, q, worker):
        import datetime
        spec = self.spec(q)
        competition = spec.get("competition") or COMP.name
        now = spec.get("now")                                 # inject for tests; else real UTC now
        now = datetime.datetime.fromisoformat(now) if isinstance(now, str) else \
            datetime.datetime.now(datetime.timezone.utc)
        rows = spec.get("rows") or self._fetch(competition, spec.get("timeout", 60))  # injected rows (test/offline)
        # OPTIONAL top_n: analyse only the top-N teams (default all). stale_ok: on no rows return a clean
        # done (a stale/failed pull is not an error to escalate) instead of pinging the researcher.
        if spec.get("top_n"):
            try:
                rows = rows[:int(spec["top_n"])]
            except Exception:  # noqa: BLE001
                pass
        a = _analyze(rows, now)
        if not a.get("n_teams"):
            if spec.get("stale_ok"):
                return self.done({"analysis": a, "history": None, "stale": True},
                                 f"lb-sync: no LB rows for {competition} (stale_ok — treated as no-op).")
            return self.escalate(worker, "researcher", f"lb-sync: no LB rows for {competition}")
        path = self._snapshot(competition, a, now, rows=rows)
        ist = (now + datetime.timedelta(hours=IST_OFFSET_H)).strftime("%Y-%m-%d %H:%M IST")
        summary = (f"LB-sync {competition} @ {ist}: top={a['top_score']} ({a['leader']}), {a['n_teams']} teams; "
                   f"leader last submitted {a['leader_last_sub_age_h']}h ago, median {a['median_last_sub_age_h']}h; "
                   f"active(24h)={a['active_24h']} top10-active(24h)={a['top10_active_24h']}. → {path}")
        self.log(summary, kind="finding",
                 recommendation="watch top10_active_24h + leader recency: high = hot race (submit conservatively), "
                                "cold = plateau. Re-run every ~3h via cron; movement is in docs/lb_history.jsonl.")
        return self.done({"analysis": a, "history": path}, summary)


_AGENT = LbSync()


def run(q, worker):
    return _AGENT.run(q, worker)
