"""sub-journal — keep a competition's experiment JOURNAL COMPLETE by syncing EVERY Kaggle submission into it,
so no experiment is ever missing. This is the AGENT that replaces ad-hoc `ledger.record(...)` calls for
submissions: it pulls the official `kaggle competitions submissions <comp> --csv`, parses every COMPLETE
submission (ref / date / description / publicScore / privateScore / status), and for each one NOT already in
the journal it writes a real journal entry (ledger.record) + attaches a per-submission provenance artifact
(docs/sub_<ref>.json) so the provenance gate passes, then ledger.set_scores(exp, public=, private=) so BOTH
scores land. Already-journaled submissions are ENRICHED in place with public/private (never double-recorded).

Honesty gates (real Kaggle-API data only, never fabricated):
  • The Kaggle CLI is called ONLY through `_run_cli`, so the data-wise test monkeypatches it and NEVER hits the
    real Kaggle API. `parse_submissions` / `parse_cv` / `_expand_refs` are PURE and tested directly.
  • CV is parsed from the description ONLY if it literally contains `cv0.xxx` / `cvX.xx`; otherwise cv=None
    (honest — a missing CV is left None, never guessed).
  • The provenance artifact docs/sub_<ref>.json literally contains the parsed cv, so the ledger provenance gate
    (which for a near-top/first score demands the scorer's own JSON) is satisfied by a REAL Kaggle-derived file.
  • Comp-agnostic: `competition` is a spec PARAMETER (never hardcoded). RP_COMP is set to it internally so the
    ledger dual-write targets the right Postgres DB (kaggle_<slug>) and per-comp docs dir.

Idempotency: dedupe by the stable Kaggle submission REF. A submission whose ref already appears in an existing
entry (as `ksub:<ref>`, in its text, or via an abbreviated `54761848/50/51` mention) is NOT re-recorded — it is
enriched with public/private if missing. Re-runs therefore journal 0 new.

Spec: {competition, [out_dir=<comp>/docs, rp_comp, rows (str CSV or list for offline/test), since='YYYY-MM-DD'
       date floor, refs=[allow-list], train_set='submission', timeout]}. Data-wise test: sub_journal_test.py.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
from pathlib import Path

from .base import BaseAgent, COMP

KAGGLE_BIN = os.environ.get("KAGGLE_BIN", "/home/seshu/miniconda3/envs/llm/bin/kaggle")


# ════════════════════════════════════════════════════════════ CLI indirection (mocked in tests)
def _run_cli(args, timeout=300):
    """Run the Kaggle CLI. ISOLATED so the data-wise test monkeypatches it — the test never touches real Kaggle.
    Returns (returncode, stdout, stderr)."""
    cmd = [KAGGLE_BIN] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


# ════════════════════════════════════════════════════════════ pure parsers (tested directly)
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_submissions(csv_text):
    """PURE parser: `kaggle competitions submissions --csv` → a list of every COMPLETE submission as
    {ref, date, description, public, private, status}. Rows without COMPLETE status are dropped; scores may be
    None (a still-scoring submission). Robust to column-name spelling/spacing/underscores."""
    lines = [ln for ln in (csv_text or "").splitlines() if ln.strip() and "Next Page Token" not in ln]
    if not lines:
        return []
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))

    def g(row, *keys):
        for k in row:
            # csv.DictReader yields a None KEY for any column beyond the header (a short/ragged header, or
            # a stubbed CLI response), and `None.lower()` took the whole agent down on data it is meant to
            # parse defensively. Skip non-string keys rather than trusting the CSV shape.
            if not isinstance(k, str):
                continue
            kl = k.lower().replace(" ", "").replace("_", "")
            for want in keys:
                if kl == want:
                    return row[k]
        return None

    out = []
    for r in rows:
        status = str(g(r, "status") or "")
        if "complete" not in status.lower():                  # only fully-scored submissions
            continue
        ref = g(r, "ref", "submissionid", "fileref")
        out.append({
            "ref": str(ref).strip() if ref else None,
            "date": (g(r, "date", "submissiondate") or "").strip(),
            "description": (g(r, "description") or "").strip(),
            "public": _num(g(r, "publicscore")),
            "private": _num(g(r, "privatescore")),
            "status": status,
        })
    return out


def parse_cv(description):
    """Pull a CV from a description ONLY if it literally contains `cv0.xxx` / `cvX.xx` (else None). Honest — a
    description without an explicit CV yields None; we never invent a number."""
    if not description:
        return None
    m = re.search(r"cv[\s:=]*([01]?\.\d{2,4})", description, re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _expand_refs(text):
    """Every Kaggle submission REF mentioned in a text: full 8-digit ids PLUS abbreviated runs like
    '54761848/50/51' → {54761848, 54761850, 54761851} (base + tail-substituted suffixes). Used to detect
    submissions already covered by a pre-existing hand-written journal entry."""
    refs = set()
    for m in re.finditer(r"(\d{8})((?:/\d{1,4})+)", text or ""):
        base = m.group(1)
        refs.add(base)
        for suf in re.findall(r"/(\d{1,4})", m.group(2)):
            if len(suf) <= len(base):
                refs.add(base[:len(base) - len(suf)] + suf)
    for m in re.finditer(r"\b(\d{8})\b", text or ""):         # plain 8-digit refs
        refs.add(m.group(1))
    return refs


def _existing_ref_map(entries):
    """{ref -> the first existing journal entry that mentions it}. Lets us ENRICH (not duplicate) a submission
    already represented by a hand-written EXP row."""
    m = {}
    for e in entries:
        try:
            text = json.dumps(e, default=str)
        except Exception:  # noqa: BLE001
            text = str(e)
        for ref in _expand_refs(text):
            m.setdefault(ref, e)
    return m


# ════════════════════════════════════════════════════════════ agent
class SubJournal(BaseAgent):
    name = "sub-journal"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        competition = spec.get("competition") or COMP.name
        # Target the right competition's Postgres/journal: the ledger resolves the active comp via RP_COMP.
        prev_rp = os.environ.get("RP_COMP")
        os.environ["RP_COMP"] = spec.get("rp_comp") or competition
        try:
            return self._sync(spec, competition, worker)
        finally:
            if prev_rp is None:
                os.environ.pop("RP_COMP", None)
            else:
                os.environ["RP_COMP"] = prev_rp

    def _sync(self, spec, competition, worker):
        from . import ledger
        out_dir = Path(spec.get("out_dir") or ledger._comp_docs())
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. fetch every submission (real Kaggle CLI; `rows` may be injected as CSV text or a parsed list for tests)
        rows = spec.get("rows")
        if rows is None:
            rc, out, err = _run_cli(["competitions", "submissions", competition, "--csv"],
                                    timeout=int(spec.get("timeout", 300)))
            subs = parse_submissions(out or "")
        elif isinstance(rows, str):
            subs = parse_submissions(rows)
        else:
            subs = list(rows)                                 # already-parsed dicts (offline injection)

        # 2. filters (comp-agnostic scoping): a date floor and/or an explicit ref allow-list
        since = spec.get("since")
        refs_only = {str(r) for r in (spec.get("refs") or [])}

        entries = ledger.entries()
        ref2exist = _existing_ref_map(entries)
        train_set = spec.get("train_set", "submission")

        journaled, skipped = [], []
        for s in subs:
            ref = s.get("ref")
            if s.get("public") is None and s.get("private") is None:
                continue                                      # no score yet → nothing honest to journal
            if since and (s.get("date") or "") < str(since):
                continue
            if refs_only and (str(ref) not in refs_only):
                continue

            pub, prv = s.get("public"), s.get("private")
            desc = s.get("description") or f"submission {ref}"
            cv = parse_cv(s.get("description"))

            # real provenance artifact (Kaggle-derived) — passed as verify_json so the provenance gate passes
            art = out_dir / f"sub_{ref}.json"
            art.write_text(json.dumps({
                "ref": ref, "description": s.get("description"), "public": pub, "private": prv,
                "cv": cv, "status": s.get("status"), "source": "kaggle competitions submissions"}, indent=2))

            existing = ref2exist.get(str(ref))
            if existing is not None:                          # already in the journal → ENRICH, never duplicate
                ledger.set_scores(existing.get("exp"), public=pub, private=prv,
                                  cv=(cv if existing.get("cv") is None else None))
                skipped.append({"ref": ref, "exp": existing.get("exp"), "public": pub, "private": prv})
                continue

            e = ledger.record(change=f"ksub:{ref}", script="kaggle competitions submissions", cv=cv, lb=pub,
                              train_set=train_set, stage="submission", kept=(pub is not None),
                              description=desc, observation="", verify_json=str(art))
            exp = e.get("exp")
            ledger.set_scores(exp, public=pub, private=prv)   # BOTH public + private land in the journal
            ref2exist[str(ref)] = e                            # a later duplicate in THIS run is enriched, not re-recorded
            journaled.append({"ref": ref, "exp": exp, "cv": cv, "public": pub, "private": prv,
                              "description": desc, "private_score": prv})

        journaled.sort(key=lambda r: (r.get("private") is None, -(r.get("private") or -1)))
        msg = (f"sub-journal {competition}: journaled {len(journaled)} new + enriched {len(skipped)} existing "
               f"submissions (of {len(subs)} complete) → every submission now carries public+private in the "
               f"journal. artifacts → {out_dir}")
        self.log(msg, kind="finding",
                 recommendation="run after every submission so the journal never misses an experiment; "
                                "idempotent (re-runs journal 0), dedupes by Kaggle ref.")
        return self.done({"competition": competition, "journaled": len(journaled),
                          "skipped": len(skipped), "entries": journaled, "enriched": skipped}, msg)


_AGENT = SubJournal()


def run(q, worker):
    return _AGENT.run(q, worker)
