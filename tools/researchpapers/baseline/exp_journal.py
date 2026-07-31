#!/usr/bin/env python
"""Experiment journal — scaffold pre-registered EXP docs and AUTO-FILL their results.

Ownership (per docs/experiments/RESULTS_WRITEBACK_CONTRACT.md): researcher owns this tool + the
template + the field schema; trainer's post-job lane just calls `fill` after each score run.

Two modes:
  new  — scaffold docs/experiments/EXP-<id>.md from EXP-TEMPLATE.md (write the HYPOTHESIS first):
           python baseline/exp_journal.py new --id EXP-001 --title "rot90 on/off" --class aug \
                  --package baseline/brackets/screen_v3.yml
  fill — read the score.json sidecars whose exp_id matches and rewrite the AUTOFILL block of
         EXP-<id>.md with a results table (idempotent — re-scoring updates, never duplicates):
           python baseline/exp_journal.py fill --id EXP-001
           python baseline/exp_journal.py fill --all

SOURCE = `output/scores/*.json` sidecars from baseline/score_v1.py (deterministic, git-versionable,
no MLflow-uptime dependency). Each carries: exp_id, run_name, fidelity(mini|golden12), eval_split,
official_score, micro_adjJ, golden_cv, mean_node_recall, mean_count_ratio, div_tp_total, status.
(score_v1.py ALSO logs MLflow tags exp_id/fidelity/eval_split for the hub UI + run URLs.)

Schema decisions (answers to the contract's §6):
  Q1 keying   = tag/sidecar `exp_id` (passed to score_v1 via --exp-id; declared in the EXP doc).
  Q2 columns  = single `official adjJ` (headline) + `golden_cv`; micro omitted (≡ official when div=0).
  Q3 fidelity = ONE official column + a `fidelity` column (mini|golden-12); mini rows are REAL official
                numbers (never "proxy-only"); golden-12 rows are the final judge, sorted first.
  Q4 pruned   = derived: a config with mini row(s) but NO golden-12 row was DROPPED between rungs
                (mini-official keep-top-half); status column reflects DONE/DROPPED/KILLED.
  Q5 source   = score.json SIDECAR (primary); MLflow tags available as secondary/linkage.
"""
import argparse
import json
import re
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[1]
EXP_DIR = WORKDIR / "docs" / "experiments"
TEMPLATE = EXP_DIR / "EXP-TEMPLATE.md"
SCORES = WORKDIR / "output" / "scores"


def _fnum(x, nd=4):
    try:
        f = float(x)
        return "—" if f != f else f"{f:.{nd}f}"   # NaN -> em dash
    except (TypeError, ValueError):
        return "—"


def load_sidecars(exp_id):
    out = []
    for p in sorted(SCORES.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            d = json.loads(p.read_text().replace("NaN", "null"))  # tolerate 1-embryo NaN
        if exp_id is None or d.get("exp_id") == exp_id:
            d["_file"] = p.name
            out.append(d)
    return out


def results_table(rows) -> str:
    if not rows:
        return "_(no scored runs yet — run `python baseline/exp_journal.py fill --id <ID>` after scoring)_"
    # golden-12 rows first (final judge), then mini screens; dedup by run_name (latest sidecar wins)
    by_run = {}
    for d in sorted(rows, key=lambda d: d.get("_file", "")):
        by_run[d.get("run_name", d.get("_file"))] = d
    ordered = sorted(by_run.values(), key=lambda d: (d.get("fidelity") != "golden12", d.get("run_name", "")))
    head = ("| run | fidelity | official adjJ | golden_cv | recall | count× | div_tp | status |\n"
            "|---|---|---|---|---|---|---|---|")
    lines = [head]
    for d in ordered:
        fid = "**golden-12**" if d.get("fidelity") == "golden12" else f"mini·{d.get('eval_split','?')}"
        lines.append(
            f"| {d.get('run_name','?')} | {fid} | **{_fnum(d.get('official_score'))}** | "
            f"{_fnum(d.get('golden_cv'))} | {_fnum(d.get('mean_node_recall'), 3)} | "
            f"{_fnum(d.get('mean_count_ratio'), 2)} | {d.get('div_tp_total', '—')} | "
            f"{d.get('status', '?')} |")
    return "\n".join(lines)


def do_fill(exp_id):
    doc = EXP_DIR / f"{exp_id}.md"
    if not doc.exists():
        raise FileNotFoundError(f"{doc} — scaffold first: exp_journal.py new --id {exp_id} --title ...")
    text = doc.read_text()
    start, end = f"<!-- AUTOFILL:{exp_id}:START -->", f"<!-- AUTOFILL:{exp_id}:END -->"
    if start not in text or end not in text:
        raise ValueError(f"{doc} missing AUTOFILL markers for {exp_id}")
    rows = load_sidecars(exp_id)
    new = re.sub(re.escape(start) + r".*?" + re.escape(end),
                 f"{start}\n{results_table(rows)}\n{end}", text, flags=re.DOTALL)
    doc.write_text(new)
    print(f"[exp_journal] filled {doc} — {len(rows)} scored run(s)")


def do_new(exp_id, title, klass, package, date):
    doc = EXP_DIR / f"{exp_id}.md"
    if doc.exists():
        raise FileExistsError(f"{doc} exists — refusing to overwrite a pre-registered hypothesis")
    t = (TEMPLATE.read_text().replace("{ID}", exp_id).replace("{TITLE}", title)
         .replace("{CLASS}", klass or "TBD").replace("{PACKAGE}", package or "TBD")
         .replace("{DATE}", date or "TBD"))
    doc.write_text(t)
    print(f"[exp_journal] scaffolded {doc} — now WRITE THE HYPOTHESIS before running.")


def main():
    ap = argparse.ArgumentParser(description="experiment journal scaffold + results auto-fill")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--id", required=True); n.add_argument("--title", required=True)
    n.add_argument("--class", dest="klass", default=""); n.add_argument("--package", default="")
    n.add_argument("--date", default="")
    f = sub.add_parser("fill"); f.add_argument("--id"); f.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.cmd == "new":
        do_new(args.id, args.title, args.klass, args.package, args.date)
    elif args.cmd == "fill":
        if args.all:
            for doc in sorted(EXP_DIR.glob("EXP-*.md")):
                if doc.stem != "EXP-TEMPLATE":
                    try:
                        do_fill(doc.stem)
                    except Exception as e:  # noqa: BLE001
                        print(f"[exp_journal] skip {doc.stem}: {e}")
        elif args.id:
            do_fill(args.id)
        else:
            ap.error("fill needs --id EXP-00N or --all")


if __name__ == "__main__":
    main()
