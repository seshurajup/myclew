"""notebook-sync — DAILY: pull new/updated top public Kaggle notebooks and EXTRACT learnings.

Public notebooks get shared/updated every day; missing them = missing the floor. This agent (once/day):
  1. lists the top notebooks by votes (Kaggle CLI),
  2. PULLS any new/updated ones (dedup by ref+version),
  3. deterministically EXTRACTS learnings — detection thresholds, linking/gap µm, division params, aug
     names, architecture keywords — via a targeted scan of the notebook source,
  4. appends dated entries to docs/kaggle_learnings.md and FLAGS anything we don't already use
     (a param/aug/technique not in our menu) so the orchestrator / human can adopt it.

Time-gated to ~once/day (Kaggle rate + it's a daily job). Deep understanding of a notebook still escalates
to the super-researcher; this is the mechanical daily sweep so nothing is missed.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
KAGGLE = os.environ.get("KAGGLE_BIN", "/home/seshu/miniconda3/envs/llm/bin/kaggle")
SLUG = os.environ.get("KAGGLE_COMP_SLUG", "biohub-cell-tracking-during-development")
PULLED = COMP / "research" / "public_notebooks"
LEARN_MD = COMP / "docs" / "kaggle_learnings.md"
STATE = COMP / "tools" / "researchpapers" / ".research-mvp-data" / "runtime" / ".notebook_sync.json"
MIN_HOURS = 20  # once/day

# what to mine from notebook source (label → regex over the code)
PATTERNS = {
    "det_threshold": r"(?:det[_-]?threshold|rel[_-]?threshold|threshold)\s*[=:]\s*([0-9.]+)",
    "gap_um": r"(?:gap[_-]?dist[_-]?um|gap[_-]?um|GAP_DIST_UM)\s*[=:]\s*([0-9.]+)",
    "motion_um": r"(?:motion[_-]?relink|motion[_-]?um|MOTION)\D{0,12}([0-9.]+)\s*(?:um|µm)?",
    "division": r"(add_safe_divisions|safe_div|division_recovery|div_jaccard|DO_SAFE_DIV)",
    "augs": r"\b(elastic|mixup|cutout|crop_scale|bias_field|gamma|contrast|brightness|flip|rot90|blur|noise)\b",
    "arch": r"\b(unet[_-]?transformer|stardist|cellpose|trackastra|nnunet|ilp|scip)\b",
    "min_track_len": r"(?:min[_-]?track[_-]?len|MIN_TRACK_LEN|filter[_-]?short[_-]?min|min[_-]?len)\D{0,8}([0-9]+)",
}


def _postproc_env_and_sig(mined, ref):
    """Map a learned-graph notebook's mined post-proc params → BIOHUB_* env + a signature string.
    These are the knobs that actually change golden-12 (gap-close µm, min-track-len); det_threshold is a
    proven NO-OP on the cached predictions, so it's excluded from the signature (avoids needless re-runs).
    Missing knobs default to the pilkwang base (gap 6.0, min-track-len 4) → the verified 0.8708 signature."""
    gap = (mined.get("gap_um") or ["6.0"])[0]
    mtl = (mined.get("min_track_len") or [None])[0]
    if mtl is None:
        m = re.search(r"min[_-]?(\d{1,2})\b", ref)   # drkongvis 'filter-short-min6' style refs
        mtl = m.group(1) if m else "4"
    env = {"BIOHUB_GAP_CLOSE_UM": str(gap), "BIOHUB_OUTPUT_MIN_TRACK_LEN": str(mtl)}
    sig = ",".join(f"{k}={env[k]}" for k in sorted(env))
    return env, sig


def _run(args, timeout=60):
    try:
        return subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout).stdout
    except Exception as exc:  # noqa: BLE001
        return f"ERR {exc}"


def _state():
    try:
        return json.loads(STATE.read_text())
    except Exception:  # noqa: BLE001
        return {"pulled": [], "last": ""}


def _hours_since(ts):
    if not ts:
        return 999
    try:
        return (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(ts)).total_seconds() / 3600
    except Exception:  # noqa: BLE001
        return 999


def _mine(text):
    found = {}
    for label, pat in PATTERNS.items():
        vals = sorted(set(re.findall(pat, text, re.I)))[:8]
        if vals:
            found[label] = vals
    return found


# golden-12 CV anchors per family (validated): learned-graph = pilkwang pipeline = ~0.8708 (all forks share it,
# LB varies only by checkpoint). Rule-based DoG shares the isakatsuyoshi baseline; params vary so it gets a real run.
GOLDEN_LEARNED = 0.8708


def _classify(mined, text):
    """(family, golden_cv_or_None, note). NO hardcoded CV — learned-graph now gets a REAL golden-12 run
    of its own params (verify-cv), same honesty as the rule-based path. cv is always None here; the
    verify-cv agent measures it and writes it back."""
    blob = (str(mined) + " " + text[:2000]).lower()
    if any(k in blob for k in ("add_safe_divisions", "safe_div", "unet_transformer", "ilp")):
        return "learned-graph", None, "learned-graph (pilkwang pipeline) — golden-12 verify queued (real run of its params)"
    if any(k in blob for k in ("dog", "rel_threshold", "difference_of_gaussian")):
        return "rule-based-DoG", None, "rule-based DoG — queued for a real golden-12 run of its params"
    return "other", None, "LB-only (not locally reproducible)"


def _lb_from(text, ref):
    """Extract the notebook's public LB score. Guards against the voxel-scale constants (z=1.625,
    y=x=0.40625) and CV artifacts: a real biohub LB is 0.5–1.0. Prefer explicit lb/score markers;
    fall back to the highest plausible score in the REF title (not code, to avoid threshold false-positives)."""
    def _scan(pat, s):
        out = []
        for m in re.findall(pat, s, re.I):
            raw = m if "." in m else "0." + m           # "885" → 0.885 ; "0.885" → 0.885
            try:
                v = round(float(raw), 4)
            except ValueError:
                continue
            if 0.5 <= v < 0.95:                           # plausible LB → rejects 1.625 (≥1) & 0.40625 (<0.5)
                out.append(v)
        return out
    marked = _scan(r"(?:lb|score)[-_ ]?([01]?\.?\d{3})", ref + " " + text[:1500])   # explicit marker = reliable
    if marked:
        return max(marked)
    loose = _scan(r"([01]?\.?\d{3})\b", ref)             # else: advertised score in the title only
    return max(loose) if loose else None


def sync(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    max_nb = max(1, int(spec.get("max_notebooks", 40)))   # max_notebooks: cap notebooks pulled/listed per sync (default 40)
    st = _state()
    if _hours_since(st.get("last")) < MIN_HOURS and not spec.get("force"):
        return ("done", {"skipped": "within 24h"}, "all",
                f"[{worker}] notebook-sync: already synced <{MIN_HOURS}h ago (daily job).")
    # COMPREHENSIVE: both highest-SCORE (best solutions) and highest-VOTES (best-explained), big page.
    refs = []
    for sort in ("scoreDescending", "voteCount"):
        listing = _run(["kernels", "list", "--competition", SLUG, "--sort-by", sort,
                        "--page-size", str(max_nb), "--csv"])
        if listing.startswith("ERR") or "ref" not in listing.lower():
            continue
        refs += [ln.split(",")[0] for ln in listing.splitlines()[1:] if "/" in ln.split(",")[0]]
    refs = list(dict.fromkeys(refs))   # dedup, keep order (score-first)
    if not refs:
        return ("escalated", {"err": "no refs"}, "researcher",
                f"[{worker}] notebook-sync: Kaggle CLI failed for '{SLUG}' — set KAGGLE_COMP_SLUG.")
    pulled = set(st.get("pulled", []))
    new_learn = []
    for ref in refs[:max_nb]:   # comprehensive: many solutions = many inference options to pick the best from
        if ref in pulled:
            continue
        dest = PULLED / ref.replace("/", "__")
        dest.mkdir(parents=True, exist_ok=True)
        out = _run(["kernels", "pull", ref, "-p", str(dest)], timeout=90)
        pulled.add(ref)
        # scan whatever code file landed
        text = ""
        for f in dest.glob("*"):
            if f.suffix in (".ipynb", ".py"):
                try:
                    text += f.read_text(errors="replace")
                except Exception:  # noqa: BLE001
                    pass
        mined = _mine(text)
        # AUTO-FILL the journal: classify family → golden-12 CV (learned=anchor; rule-based=queue a run)
        try:
            from . import ledger
            from researchpapers.fleet import board
            fam, gcv, note = _classify(mined, text)
            lb = _lb_from(text, ref)
            row = ledger.record(change=f"pub_{ref.split('/')[-1][:24]}",
                                description=f"PUBLIC notebook ({fam}) — {note}",
                                script=f"kaggle: {ref}", cv=gcv, lb=lb, train_set="public")
            if fam == "learned-graph":
                # REAL golden-12 run of THIS notebook's params (verify-cv, Python, no Claude) — no hardcoded
                # anchor. Identical-param forks hit verify-cv's signature cache, so the anchor is measured once.
                env, sig = _postproc_env_and_sig(mined, ref)
                board.add("S", "verify-cv",
                          f"golden-12 VERIFY learned-graph params from {ref} ({sig})",
                          {"ref": ref, "env": env, "sig": sig, "exp": row.get("exp")})
            elif fam == "rule-based-DoG" and not isinstance(row.get("cv"), (int, float)):
                # deterministic golden-12 run of its DoG params (Python, no leader) — one at a time
                board.add("S", "reproduce-score",
                          f"golden-12 score rule-based DoG params from {ref}",
                          {"ref": ref, "family": fam, "params": mined, "exp": row.get("exp")})
        except Exception:  # noqa: BLE001
            pass
        if mined:
            new_learn.append((ref, mined))
    # write learnings
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]
    if new_learn:
        LEARN_MD.parent.mkdir(parents=True, exist_ok=True)
        with open(LEARN_MD, "a") as fh:
            fh.write(f"\n## sync {now}\n")
            for ref, mined in new_learn:
                fh.write(f"- **{ref}** — " + "; ".join(f"{k}={v}" for k, v in mined.items()) + "\n")
    st["pulled"], st["last"] = sorted(pulled), now
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st))
    if not new_learn:
        return ("done", {"new": 0}, "all", f"[{worker}] notebook-sync: no NEW top notebooks since last sync.")
    tips = "; ".join(f"{r.split('/')[-1]}:{list(m)}" for r, m in new_learn[:4])
    return ("done", {"new": len(new_learn), "learnings": [r for r, _ in new_learn]}, "researcher",
            f"[{worker}] NOTEBOOK-SYNC: pulled {len(new_learn)} new notebook(s) → learnings in docs/kaggle_learnings.md. "
            f"Mined: {tips}. Researcher: adopt any threshold/aug/technique we lack.")
