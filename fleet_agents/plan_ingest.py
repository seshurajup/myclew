"""plan-ingest — the HUMAN's plan file → executed by the Python fleet (works with NO super-leader).

You write experiments/splits/direction into ONE yaml (docs/human_plan.yml); this agent ingests it each
cycle and drives the fleet: builds any splits, generates each experiment's config (config_gen) and
enqueues it (approved → train→score→journal→analysis). Each item runs ONCE (deduped by name), so you can
keep editing + saving the file to add more. This is the manual direction channel when the leader is away.

  File you edit:  docs/human_plan.yml   (a template is written there if it's missing)
"""
from __future__ import annotations

import json
from pathlib import Path

from researchpapers.fleet import board

from . import config_gen, split_build

COMP = Path(__file__).resolve().parent.parent
PLAN = COMP / "docs" / "human_plan.yml"
STATE = COMP / "tools" / "researchpapers" / ".research-mvp-data" / "runtime" / ".plan_ingested.json"

TEMPLATE = """# YOUR plan for the Python fleet — works with NO super-leader.
# Edit + save; the fleet ingests this each cycle and runs everything (train -> score -> journal -> analysis).
# Every item runs ONCE (deduped by 'name'). Add more by appending and saving.

experiments:
  - name: h_contrast_strong                 # unique name -> becomes the method + journal row
    base: config/aug_ablation/00_no_aug.yml  # config to clone (optional; default = no-aug baseline)
    augment:                                 # the ONE change: an aug list ...
      - {name: contrast, p: 0.5, range: 0.3}
    split: splits_screen_matched.json        # which split (optional; default = screen_matched)
    note: contrast at higher strength

  - name: h_recall_tilt
    params: {det_neg_weight: 0.05}           # ... OR a param change instead of an aug
    split: splits_screen_matched.json

splits:                                      # optional: splits to build first (leak-checked)
  - {kind: embryo_disjoint, out: my_embryo_disjoint.json}   # kind: embryo_disjoint | stagebridge | stage_matched

direction: "after these, focus on the division term (div_J=0)"   # free-text -> shown to the super-agents
"""


def _state() -> set:
    try:
        return set(json.loads(STATE.read_text()))
    except Exception:  # noqa: BLE001
        return set()


def _mark(name):
    s = _state()
    s.add(name)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(sorted(s)))


def ingest(q, worker):
    """Read docs/human_plan.yml and enqueue any NEW experiments/splits from it (deduped)."""
    import yaml
    if not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(TEMPLATE)
        return ("done", {"created_template": True}, "all",
                f"[{worker}] PLAN-INGEST: wrote a template → docs/human_plan.yml. Edit it to direct the fleet (no leader needed).")
    try:
        plan = yaml.safe_load(PLAN.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        return ("escalated", {"error": str(exc)}, "researcher",
                f"[{worker}] PLAN-INGEST: docs/human_plan.yml is invalid YAML: {exc}")
    if not isinstance(plan, dict):                          # YAML that isn't a mapping (e.g. a bare list/scalar)
        return ("escalated", {"error": "plan is not a mapping"}, "researcher",
                f"[{worker}] PLAN-INGEST: docs/human_plan.yml must be a YAML mapping (experiments:/splits:), got "
                f"{type(plan).__name__}.")
    done_names = _state()
    acted = []
    # 1) splits first (so experiments can reference them)
    for sp in (plan.get("splits") or []):
        key = "split:" + str(sp.get("out"))
        if key in done_names:
            continue
        st, _r, _to, _m = split_build.build({"spec": sp}, worker)
        _mark(key)
        acted.append(f"split {sp.get('out')} [{st}]")
    # 2) experiments → config-gen → enqueue (approved)
    for e in (plan.get("experiments") or []):
        name = e.get("name")
        if not name or name in done_names:
            continue
        try:
            cfg = config_gen.make(e.get("base", "config/aug_ablation/00_no_aug.yml"), name,
                                  augment=e.get("augment"), params=e.get("params"),
                                  split=e.get("split", "splits_screen_matched.json"),
                                  purpose=f"human plan: {e.get('note', name)}")
            board.add("C", "aug-ablation", f"human-plan: {e.get('note', name)} ({cfg})",
                      {"config": cfg, "description": f"human plan: {e.get('note', name)}", "approved": True})
            _mark(name)
            acted.append(f"exp {name}")
        except Exception as exc:  # noqa: BLE001
            acted.append(f"exp {name} FAILED: {exc}")
    if not acted:
        return ("done", {"new": 0}, "all", f"[{worker}] plan-ingest: no new items in docs/human_plan.yml.")
    return ("done", {"new": len(acted), "items": acted, "direction": plan.get("direction")}, "all",
            f"[{worker}] PLAN-INGEST: enqueued {len(acted)} human-plan item(s) → {', '.join(acted[:6])}. "
            f"Direction: {plan.get('direction', '—')}")
