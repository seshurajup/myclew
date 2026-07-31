"""Experiment ledger — grandmaster experiment-journal style (DrHB/icecube-journal + DrHB/rna-stanford).

Every experiment is ONE row:
  EXP | CV | LB | DESCRIPTION ("same as EXP_X but <one change>" + observation) | SCRIPT | TRN_SET
- CV and LB tracked SEPARATELY (the gap is the primary anti-overfit signal; higher CV better for us).
- Parent lineage ("same as EXP_X but ...") + a per-experiment observation/learning note.
- Failures logged too (bad / overfit / nan / crashed). Append-only JSONL + a rendered markdown table
  at docs/experiment_ledger.md (shown on the :7777 hub). This IS the full experiment history.
"""
from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
PROJECTS_BASE = COMP.parent                                 # /home/seshu/kaggle/2026 (all comp dirs)
# Biohub defaults (backward-compatible). The ACTIVE competition is resolved per write via RP_COMP so the
# same fleet code can journal to ANY competition — biohub by default, birdclef when RP_COMP=birdclef-2026.
LEDGER = COMP / "docs" / "experiment_ledger.jsonl"
TABLE = COMP / "docs" / "experiment_ledger.md"
DECISIONS = COMP / "docs" / "experiment_decisions.jsonl"   # the ANALYSIS / decision trail (the story)


def _active_slug() -> str:
    """The competition this write targets: env RP_COMP (e.g. 'birdclef-2026') else this comp (biohub)."""
    return (os.environ.get("RP_COMP") or "").strip() or COMP.name


def _comp_docs(slug: str | None = None) -> Path:
    slug = slug or _active_slug()
    return (COMP / "docs") if slug == COMP.name else (PROJECTS_BASE / slug / "docs")


def _ledger_path() -> Path:
    return _comp_docs() / "experiment_ledger.jsonl"


def _table_path() -> Path:
    return _comp_docs() / "experiment_ledger.md"


def _decisions_path() -> Path:
    return _comp_docs() / "experiment_decisions.jsonl"


def _pg_journal(rows) -> None:
    """Dual-write experiment rows to the active comp's Postgres (kaggle_<slug>). Best-effort, never raises."""
    try:
        from . import db
        db.upsert_journal(_active_slug(), rows)
    except Exception:  # noqa: BLE001
        pass


def _pg_decisions(rows) -> None:
    try:
        from . import db
        db.upsert_decisions(_active_slug(), rows)
    except Exception:  # noqa: BLE001
        pass

# Common threshold/hyperparam defaults that must NEVER be mistaken for a measured CV (the EXP_153 bug:
# a division THRESHOLD of 0.9 was recorded as cv=0.9). Recording one of these AS a score requires proof.
_THRESHOLD_SENTINELS = {0.5, 0.9, 0.95, 0.99}

# GIT COMMIT HASH per experiment (user 2026-07-12) — every ledger row/decision maps to the exact code state.
# Cached per-process; git-track agent calls refresh_git_hash() after committing so new rows get the new hash.
_GIT_HASH = None


def git_hash():
    """Current short git commit hash of the competition repo (or 'nogit' if uninitialised/unavailable)."""
    global _GIT_HASH
    if _GIT_HASH is None:
        try:
            import subprocess
            r = subprocess.run(["git", "-C", str(COMP), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=5)
            dirty = subprocess.run(["git", "-C", str(COMP), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            h = r.stdout.strip()
            _GIT_HASH = (h + ("-dirty" if dirty else "")) if h else "nogit"
        except Exception:  # noqa: BLE001
            _GIT_HASH = "nogit"
    return _GIT_HASH


def refresh_git_hash():
    """Force re-read of the git hash (call after a commit so later rows use the new hash)."""
    global _GIT_HASH
    _GIT_HASH = None
    return git_hash()


def _verify_cv(cv, verify_json):
    """PROVENANCE GATE — a numeric CV that claims a top/near-top score (>= current best) or equals a
    threshold sentinel MUST be backed by a real measured artifact (verify_json) that literally contains
    that number. Otherwise raise. This makes a fabricated 🏆 (like the EXP_153 cv=0.9 placeholder)
    structurally impossible: you cannot record a winning score without the scorer's own output file.
    Regressions / lower scores don't need proof (they can't fake a win)."""
    if not isinstance(cv, (int, float)) or isinstance(cv, bool) or cv != cv:
        return                                            # None / 'bad' / 'nan' status strings (and nan) are fine
    prev = [e.get("cv") for e in entries()
            if isinstance(e.get("cv"), (int, float)) and not isinstance(e.get("cv"), bool) and e.get("cv") == e.get("cv")]
    best = max(prev) if prev else 0.0
    claims_win = cv >= best - 1e-9                         # a new best or a tie — the exact thing worth faking
    if not (claims_win or cv in _THRESHOLD_SENTINELS):
        return                                            # a clear regression can't fabricate a win → allow
    if not verify_json:
        raise ValueError(
            f"record(): cv={cv} claims a top/near-top (or sentinel) score but carries NO measured "
            f"artifact. Pass verify_json=<path to the scorer's JSON output>. (guards fabricated scores)")
    if not os.path.exists(verify_json):
        raise ValueError(f"record(): verify_json '{verify_json}' does not exist — cannot confirm cv={cv}.")
    txt = Path(verify_json).read_text(errors="replace")
    nums = [float(n) for n in re.findall(r"-?\d+\.\d+", txt)]
    if not any(abs(n - float(cv)) <= 1e-3 for n in nums):
        raise ValueError(
            f"record(): cv={cv} was NOT found in the measured artifact '{verify_json}'. "
            f"Refusing to record a score the scorer never produced.")
    # CANONICAL-ONLY WIN GATE (2026-07-09) — BIOHUB-ONLY: a claimed biohub golden-12 WIN must be proven by the
    # canonical scorer (scripts/score_golden12_official.py → tracking_cellmot.metrics), NOT the src.metric PROXY.
    # This is specific to biohub's golden-12; other competitions prove a score by their own measured artifact
    # (e.g. a Kaggle-submission JSON with public/private) — the number-match above is sufficient for them.
    _comp = os.environ.get("RP_COMP", "biohub-cell-tracking-during-development")
    _is_biohub = ("biohub" in _comp)
    if _is_biohub and claims_win and "tracking_cellmot.metrics.canonical" not in txt:
        raise ValueError(
            f"record(): cv={cv} claims a WIN but verify_json '{verify_json}' is NOT a canonical-scorer "
            f"artifact (missing 'tracking_cellmot.metrics.canonical' marker). Winning golden-12 scores "
            f"must be proven by scripts/score_golden12_official.py, not the src.metric proxy.")


def decisions():
    _dec = _decisions_path()
    if not _dec.exists():
        return []
    out = []
    for ln in _dec.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except Exception:  # noqa: BLE001
            pass
    return out


def log(agent, summary, detail="", kind="finding", recommendation=None, run=None, ts=None):
    """ANY agent contributes to the journal's story (deduped). kind: finding | decision | verdict | data.
    summary = the short headline (always shown); detail = the long text (behind a 'more' toggle).
    Dedup: skip if this agent's last entry has the SAME summary+recommendation (no per-cycle spam)."""
    prev = decisions()
    key = (agent, summary, recommendation)
    for e in reversed(prev[-40:]):
        if (e.get("agent"), e.get("finding") or e.get("summary"), e.get("recommendation")) == key:
            return e   # already logged this exact contribution → don't repeat
    e = {"ts": ts or datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "agent": agent, "kind": kind, "run": run, "summary": summary, "finding": summary,
         "detail": detail, "recommendation": recommendation, "git_hash": git_hash()}
    _dec = _decisions_path()
    _dec.parent.mkdir(parents=True, exist_ok=True)
    with open(_dec, "a") as f:
        f.write(json.dumps(e) + "\n")
    _pg_decisions([e])                                     # dual-write to the active comp's Postgres
    _refresh_insights()                                    # decision trail feeds /insights → keep it live
    return e


def log_decision(agent, finding, recommendation, run=None, ts=None):
    """Back-compat wrapper for pre/post-analysis → routes into the general journal log."""
    return log(agent, summary=finding, kind="decision", recommendation=recommendation, run=run, ts=ts)


def entries():
    _led = _ledger_path()
    if not _led.exists():
        return []
    out = []
    for ln in _led.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except Exception:  # noqa: BLE001
            pass
    return out


def _next_id() -> str:
    return f"EXP_{len(entries()):02d}"


# status noise that must NEVER pollute the DESCRIPTION ("what we did") column
import re as _re
_STATUS_NOISE = _re.compile(r"dry-run|pending run|auto-submit|queued|green", _re.I)


def record(change, script, cv=None, lb=None, train_set="mini", parent=None,
           stage=None, kept=None, observation="", description=None, verify_json=None):
    """Append one experiment row (dedup on script+change) and re-render the table.

    change      : the KEY for this experiment (== MLflow run_name / weights/<method>/; used for backfill)
    description : WHAT WE DID in this experiment (the DESCRIPTION column). Falls back to `change`.
    script      : the train command — ALWAYS `bash start_train.sh config/X.yml` for config-driven runs.
    cv, lb      : full-metric CV and leaderboard scores (None=pending; 'bad'/'overfit'/'nan' allowed)
    train_set   : data scope ('mini' / 'loeo' / 'golden12' / 'full')
    parent      : EXP_id this is based on (→ "same as EXP_X but <what we did>")
    observation : a REAL learning (e.g. 'overfit at the end'); transient run-status is dropped.
    verify_json : path to the scorer's JSON output that PROVES `cv`. REQUIRED when cv claims a
                  top/near-top score — the provenance gate refuses fabricated winning scores.
    """
    _verify_cv(cv, verify_json)                           # PROVENANCE GATE (see _verify_cv) — fabricated wins raise
    for e in entries():  # dedup: one row per (script, change)
        if e.get("script") == script and e.get("change") == change:
            return e
    _led = _ledger_path()
    _led.parent.mkdir(parents=True, exist_ok=True)
    exp = _next_id()
    what = (description or change).strip()
    desc = f"same as {parent} but {what}" if parent else what
    # only a genuine learning goes into the description — never 'dry-run GREEN, pending run' etc.
    if observation and not _STATUS_NOISE.search(observation):
        desc += f" — {observation}"
    e = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "exp": exp, "cv": cv, "lb": lb, "desc": desc, "change": change, "parent": parent,
         "script": script, "trn_set": train_set, "stage": stage, "kept": kept,
         "observation": observation, "description": description, "git_hash": git_hash()}
    with open(_led, "a") as f:
        f.write(json.dumps(e) + "\n")
    _pg_journal([e])                                       # dual-write to the active comp's Postgres
    _write_table()
    return e


def set_scores(exp, cv=None, lb=None, kept=None, observation=None, public=None, private=None):
    """Backfill CV/LB/observation for an experiment after it is scored (metric/scorer agent).

    public/private : the Kaggle PUBLIC and PRIVATE leaderboard scores (sub-journal agent). Stored as their
                     own fields so BOTH land in the journal; `lb` defaults to the public score when unset."""
    es = entries()
    hit = False
    for e in es:
        if e.get("exp") == exp:
            if cv is not None:
                e["cv"] = cv
            if lb is not None:
                e["lb"] = lb
            if kept is not None:
                e["kept"] = kept
            if observation is not None:
                e["observation"] = observation
            if public is not None:
                e["public"] = public
                if e.get("lb") is None:                        # LB column surfaces the public score
                    e["lb"] = public
            if private is not None:
                e["private"] = private
            hit = True
    if hit:
        with open(_ledger_path(), "w") as f:
            for e in es:
                f.write(json.dumps(e) + "\n")
        _pg_journal(es)                                    # dual-write backfilled scores to Postgres
        _write_table()
    return hit


def _finite(v):
    """True only for a real finite number (rejects None / nan / inf / bool)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _fmt(s):
    if s is None:
        return "pending"
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return f"{s:.4f}" if _finite(s) else str(s)   # nan/inf print literally, never crash
    return str(s)


def _write_table():
    es = entries()
    body = "\n".join(
        f"| {e.get('exp')} | `{_fmt(e.get('cv'))}` | `{_fmt(e.get('lb'))}` | {e.get('desc', '')} "
        f"| `{e.get('script', '')}` | `{e.get('trn_set', '')}` |" for e in es)
    md = ("# biohub cell-tracking — experiment journal\n\n"
          "Every experiment, grandmaster-journal style (one change per row; CV & LB separate; "
          "failures kept; higher CV better). Modeled on DrHB/icecube-journal + DrHB/rna-stanford.\n\n"
          "| EXP | CV | LB | DESCRIPTION | SCRIPT | TRN_SET |\n"
          "| :-- | :-- | :-- | :-- | :-- | :-- |\n" + body + "\n")
    _tbl = _table_path()
    _tbl.parent.mkdir(parents=True, exist_ok=True)
    _tbl.write_text(md)
    _refresh_insights()                                    # keep /insights current on every scored write


def _refresh_insights():
    """Auto-regenerate docs/INSIGHTS.md from the ledger. Best-effort: never let a handoff-report
    refresh break the actual experiment record. Lazy import avoids the insights↔ledger cycle."""
    if os.environ.get("LEDGER_NO_INSIGHTS") == "1":        # opt-out for bulk backfills
        return
    try:
        from . import insights
        (_comp_docs() / "INSIGHTS.md").write_text(insights.build_md())
    except Exception:  # noqa: BLE001 — a stale insights page must never crash a record()
        pass


def summary() -> dict:
    es = entries()
    cvs = [e.get("cv") for e in es if _finite(e.get("cv"))]
    return {"n": len(es), "kept": sum(1 for e in es if e.get("kept")),
            # sort by str(): `stage` is free-form and the real ledger holds BOTH strings and ints, so a
            # bare sorted() raises TypeError('<' not supported between 'str' and 'int') and takes the whole
            # agent down on data it is supposed to summarise.
            "stages_touched": sorted({e.get("stage") for e in es if e.get("stage") is not None}, key=str),
            "best_cv": max(cvs) if cvs else None, "table": str(_table_path())}


def report(q, worker):
    """Fleet handler — post the ledger state (icecube/rna-stanford-style journal)."""
    s = summary()
    recent = entries()[-3:]
    tail = "; ".join(f"{e.get('exp')}:cv{_fmt(e.get('cv'))}/lb{_fmt(e.get('lb'))} {(e.get('change') or '')[:18]}"
                     for e in recent) or "(empty)"
    return ("done", s, "all",
            f"[{worker}] LEDGER ({s['n']} experiments, {s['kept']} kept, best_cv={_fmt(s['best_cv'])}): "
            f"table → docs/experiment_ledger.md (on :7777). Recent: {tail}")
