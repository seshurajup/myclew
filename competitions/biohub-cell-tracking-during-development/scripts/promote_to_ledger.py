"""THE canonical promotion gate — the ONLY sanctioned path to record a golden-12 🏆 in the ledger.

Runs scripts/score_golden12_official.py (canonical tracking_cellmot.metrics, NOT the src.metric proxy),
writes its JSON artifact, then calls ledger.record with verify_json=<that artifact>. Because the ledger
provenance gate requires the claimed cv to appear in a canonical-marked artifact, a PROXY score can no
longer record a winning row — it structurally cannot produce this artifact.

Post-proc params come from BIOHUB_* env (read by pilk_post), e.g. set BIOHUB_OUTPUT_FILTER_SHORT_TRACKS=1
BIOHUB_OUTPUT_MIN_TRACK_LEN=10 BIOHUB_GAP_CLOSE_UM=6.0 before invoking.

Usage:
  BIOHUB_OUTPUT_FILTER_SHORT_TRACKS=1 BIOHUB_OUTPUT_MIN_TRACK_LEN=10 BIOHUB_GAP_CLOSE_UM=6.0 \
  research/cellmot_venv/bin/python scripts/promote_to_ledger.py \
      --change mtl10_gap60 --parent EXP_147 --observation "min_track_len=10 canonical peak"
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP))
PY = str(COMP / "research" / "cellmot_venv" / "bin" / "python")
SCORER = str(COMP / "scripts" / "score_golden12_official.py")
ART_DIR = COMP / "tools" / "researchpapers" / "output" / "reconcile"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--change", required=True, help="ledger KEY for this config")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--observation", default="")
    ap.add_argument("--train-set", default="golden12")
    ap.add_argument("--stage", type=int, default=8)
    ap.add_argument("--dry", action="store_true", help="score only, do NOT record")
    args = ap.parse_args()

    ART_DIR.mkdir(parents=True, exist_ok=True)
    art = ART_DIR / f"promote_{args.change}.json"
    print(f"[promote] canonical scoring golden-12 for '{args.change}' ...", file=sys.stderr)
    r = subprocess.run([PY, SCORER, "--tag", args.change], capture_output=True, text=True,
                       timeout=1200, cwd=str(COMP), env=dict(os.environ))
    lines = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    if not lines:
        print(f"[promote] scorer produced no JSON. stderr tail:\n{r.stderr[-800:]}", file=sys.stderr)
        sys.exit(1)
    result = json.loads(lines[-1])
    art.write_text(json.dumps(result, indent=2))
    cv = result["score"]
    env_note = (f"gap={os.environ.get('BIOHUB_GAP_CLOSE_UM','6.0')} "
                f"mtl={os.environ.get('BIOHUB_OUTPUT_MIN_TRACK_LEN','4')} "
                f"filt={os.environ.get('BIOHUB_OUTPUT_FILTER_SHORT_TRACKS','0')}")
    print(json.dumps({"change": args.change, "canonical_cv": cv, "artifact": str(art),
                      "node_recall": result.get("node_recall"), "env": env_note}, indent=2))

    if args.dry:
        print("[promote] --dry: not recording.", file=sys.stderr)
        return

    from fleet_agents import ledger
    row = ledger.record(
        change=args.change,
        script=f"BIOHUB_* env; {PY} scripts/score_golden12_official.py --tag {args.change}",
        cv=cv, train_set=args.train_set, parent=args.parent, stage=args.stage,
        observation=(args.observation + f" [{env_note}]").strip(),
        description=f"canonical golden-12 official = {cv}",
        verify_json=str(art),
    )
    print(f"[promote] recorded {row.get('exp')} cv={cv} (verify_json={art})", file=sys.stderr)


if __name__ == "__main__":
    main()
