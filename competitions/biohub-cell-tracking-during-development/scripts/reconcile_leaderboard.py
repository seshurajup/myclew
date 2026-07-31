"""Canonical-vs-proxy golden-12 leaderboard reconciliation (leader mandate 2026-07-09).

For each named post-proc config, score golden-12 with BOTH:
  * CANONICAL  = scripts/score_golden12_official.py  (tracking_cellmot.metrics — the real metric)
  * PROXY      = scripts/score_postproc_golden12.py   (src.metric.official_score — self-declared proxy)
apples-to-apples (identical BIOHUB_* env → identical post-proc, two metrics). Writes one JSON artifact
and prints a ranking table so we can see whether the PROXY RANKING holds under the canonical metric and
which configs (if any) beat the CANONICAL bar. Every canonical score is the number the ledger will gate on.

Usage:
  research/cellmot_venv/bin/python scripts/reconcile_leaderboard.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
PY = str(COMP / "research" / "cellmot_venv" / "bin" / "python")
CANON = str(COMP / "scripts" / "score_golden12_official.py")
PROXY = str(COMP / "scripts" / "score_postproc_golden12.py")
OUT = COMP / "tools" / "researchpapers" / "output" / "reconcile" / "leaderboard_reconcile.json"

FILT = {"BIOHUB_OUTPUT_FILTER_SHORT_TRACKS": "1"}   # required for the min_track_len axis to fire

# name -> (proxy-claimed golden_cv, env overrides). The env is the FULL post-proc knob set.
CONFIGS = {
    "anchor_EXP147_gap4_mtl4":  (0.8735, {**FILT, "BIOHUB_GAP_CLOSE_UM": "4.0", "BIOHUB_OUTPUT_MIN_TRACK_LEN": "4"}),
    "anchor_v13_gap6_mtl4":     (0.8735, {**FILT, "BIOHUB_GAP_CLOSE_UM": "6.0", "BIOHUB_OUTPUT_MIN_TRACK_LEN": "4"}),
    "mtl10_gap6.0":             (0.8837, {**FILT, "BIOHUB_GAP_CLOSE_UM": "6.0", "BIOHUB_OUTPUT_MIN_TRACK_LEN": "10"}),
    "mtl10_gap5.5":             (0.8846, {**FILT, "BIOHUB_GAP_CLOSE_UM": "5.5", "BIOHUB_OUTPUT_MIN_TRACK_LEN": "10"}),
}

# fleet fullconfig "running_best" (proxy 0.8803) — load its exact 53-key env if dumped
_FC = Path("/tmp/claude-1001/fullconfig_best_env.json")
if _FC.exists():
    CONFIGS["fullconfig_best_0.8803seed"] = (0.8803, {**json.loads(_FC.read_text())})


def _run(script, env, canonical):
    run_env = dict(os.environ)
    run_env.update(env)
    r = subprocess.run([PY, script], capture_output=True, text=True, timeout=900, cwd=str(COMP), env=run_env)
    lines = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    if not lines:
        return None
    d = json.loads(lines[-1])
    return d


def main():
    rows = []
    for name, (proxy_claim, env) in CONFIGS.items():
        print(f"[scoring] {name} ...", file=sys.stderr, flush=True)
        canon = _run(CANON, env, True)
        proxy = _run(PROXY, env, False)
        rows.append({
            "config": name,
            "proxy_claimed_golden_cv": proxy_claim,
            "proxy_measured": (proxy or {}).get("score"),
            "canonical": (canon or {}).get("score"),
            "canon_adj_edge": (canon or {}).get("adj_edge_jaccard"),
            "canon_edge_jaccard": (canon or {}).get("edge_jaccard"),
            "canon_node_recall": (canon or {}).get("node_recall"),
            "canon_div_jaccard": (canon or {}).get("division_jaccard"),
            "canon_minus_proxy": (round((canon or {}).get("score", float("nan")) - (proxy or {}).get("score", float("nan")), 4)
                                  if canon and proxy else None),
            "env": env,
        })

    canon_rank = sorted([r for r in rows if r["canonical"] is not None], key=lambda r: -r["canonical"])
    proxy_rank = sorted([r for r in rows if r["proxy_measured"] is not None], key=lambda r: -r["proxy_measured"])
    ranking_holds = [r["config"] for r in canon_rank] == [r["config"] for r in proxy_rank]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "eval_set": "golden_12", "metric_canonical": "tracking_cellmot.metrics (official_repo)",
        "metric_proxy": "src.metric.official_score (self-declared proxy)",
        "ranking_holds_canon_vs_proxy": ranking_holds,
        "canonical_ranking": [r["config"] for r in canon_rank],
        "proxy_ranking": [r["config"] for r in proxy_rank],
        "rows": rows,
    }, indent=2))

    print(f"\n{'config':<30} {'proxy_meas':>10} {'canonical':>10} {'Δc-p':>8} {'node_rec':>9}")
    for r in canon_rank:
        print(f"{r['config']:<30} {str(r['proxy_measured']):>10} {str(r['canonical']):>10} "
              f"{str(r['canon_minus_proxy']):>8} {str(r['canon_node_recall']):>9}")
    print(f"\nRANKING HOLDS (canon vs proxy): {ranking_holds}")
    print(f"artifact -> {OUT}")


if __name__ == "__main__":
    main()
