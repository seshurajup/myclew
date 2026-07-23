"""lb_sync_test — DATA-WISE verifier of the LB activity analysis (_analyze), no network. Synthetic LB with
known submission ages → asserts leader/top-score, recency, and active-window counts are computed correctly.
"""
import os, sys, datetime
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import lb_sync as L


def _run():
    print("=== LB-SYNC ANALYSIS VERIFIER ===")
    now = datetime.datetime(2026, 7, 12, 0, 0, 0, tzinfo=datetime.timezone.utc)
    rows = [
        {"teamName": "Kevin", "submissionDate": "2026-07-11 00:00:00.000000", "score": "0.968"},  # 24h ago
        {"teamName": "Rahul", "submissionDate": "2026-07-05 00:00:00.000000", "score": "0.910"},  # 168h ago
        {"teamName": "Fresh", "submissionDate": "2026-07-11 22:00:00.000000", "score": "0.900"},  # 2h ago
        {"teamName": "bad",   "submissionDate": "not-a-date",                 "score": "0.5"},     # dropped
    ]
    a = L._analyze(rows, now)
    checks = {
        "n_teams_valid_only": a["n_teams"] == 3,                       # bad-date row dropped
        "top_score_and_leader": a["top_score"] == 0.968 and a["leader"] == "Kevin",
        "leader_recency": abs(a["leader_last_sub_age_h"] - 24.0) < 0.2,
        "active_24h_counts": a["active_24h"] == 2,                     # Kevin(24h) + Fresh(2h)
        "freshest": abs(a["freshest_age_h"] - 2.0) < 0.2,
        "parse_dt_ok": L._parse_dt("2026-07-11 00:00:00.000000") is not None,
        "parse_dt_bad": L._parse_dt("nope") is None,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("    analysis=", {k: a[k] for k in ("top_score", "leader", "leader_last_sub_age_h", "active_24h")})
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
