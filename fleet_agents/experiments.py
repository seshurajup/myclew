"""Experiment adapter — run an EXISTING competition config via the config-driven trainer.

We do NOT build specs from scratch: the competition already declares each experiment as a YAML
(config/aug_ablation/*.yml, model_scratch/config/exp_det_*.yml) that `start_train.sh <cfg>` runs
through scripts/train_from_config.py (embryo-disjoint per its own splits, MLflow-logged). The
deterministic job here is: verify the config exists and hand the trainer the exact ready command.
(Augment mode: the trainer queues it to :7799 and reports the score. Flip AUTO_SUBMIT to have the
fleet POST to :7799 itself once a dry-run gate is wired.)
"""
from __future__ import annotations

from pathlib import Path

from researchpapers.fleet import board, read

from . import dryrun, ledger, runner

COMP = Path(__file__).resolve().parent.parent
_STAGE = {"aug-ablation": 3, "arch-probe": 4}


def _cfg(cfg: str) -> dict:
    try:
        import yaml
        return yaml.safe_load((COMP / cfg).read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def _method_of(cfg: str) -> str:
    """The config's train.method (== MLflow run_name == weights/<method>/ dir). Falls back to name/stem
    so ledger row, MLflow run, and the score step all key on the SAME identity."""
    c = _cfg(cfg)
    return (c.get("train", {}) or {}).get("method") or c.get("name") or Path(cfg).stem


def _trainset(cfg: str) -> str:
    """HONEST data-scope label = the ACTUAL split file the config trains on (no invented names). Scored
    on golden-12 always. Names map 1:1 to the real file so the journal shows exactly what was used:
      splits_screen_matched(_k6) → screen_matched / screen_k6   (stage-matched mini, leak-free vs golden-12)
      fleet_loeo_mini / splits_loeo_* → loeo_mini / loeo_density (embryo-disjoint)
      splits_ft → ft187 (train 187 / val golden-12 — embryo-leaky, CONFOUNDED)."""
    fn = Path(str(((_cfg(cfg).get("paths", {}) or {}).get("splits") or ""))).name.lower()
    if "screen_matched_k6" in fn:
        return "screen_k6"
    if "screen_matched" in fn:
        return "screen_matched"
    if "loeo_mini" in fn or "fleet_loeo" in fn:
        return "loeo_mini"
    if "loeo_density" in fn:
        return "loeo_density"
    if "loeo" in fn:
        return "loeo"
    if "splits_ft" in fn:
        return "ft187"          # train 187 / val golden-12 (embryo-leaky = confounded screen)
    return Path(fn).stem or "?"


def _description(cfg: str, method: str, question: str, spec: dict) -> str:
    """WHAT WE DID — the LEADER supplies it. Precedence: explicit spec.description → the leader's own
    directive in the thread that names this config/method → the config's researcher-written `purpose`
    → the question text. So the DESCRIPTION column is the leader's intent, not machine status noise."""
    if spec.get("description"):
        return str(spec["description"]).strip()
    for m in reversed(read.read_thread(sender="leader", limit=200)):  # newest leader directive first
        d = m.get("data") or {}  # read.py backfills this from the leader's KEY: value template
        # STRUCTURED match only — the template's config/method key, never loose free-text (too greedy)
        if d.get("config") == cfg or d.get("method") == method:
            did = d.get("did") or d.get("description") or ""
            if did.strip():
                return did.strip()[:600]
    purpose = _cfg(cfg).get("purpose")
    if purpose:
        return " ".join(str(purpose).split())[:240]  # first-pass: collapse the config purpose
    return question


def run_config(q, worker):
    """END-TO-END in Python (no Claude trainer): dry-run gate → submit to :7799 → decompose later.

    dryrun (researcher's job) + runner (trainer's job) are both Python now. Claude is only pulled in
    when the dry-run FAILS (config wiring is broken and needs a real fix)."""
    from .base import gpu_train_held
    if gpu_train_held():
        return ("escalated", {"held": True}, "leader",
                f"[{worker}] {q.get('question','train')} HELD — GPU training parked (5090 power-cap gate). "
                f"Remove config/_auto/gpu_train_hold.flag (human GO) before training.")
    cfg = q["spec"].get("config")
    if not cfg or not (COMP / cfg).exists():
        return ("escalated", {"missing_config": cfg}, "researcher",
                f"[{worker}] {q['question']}: config '{cfg}' not found under the competition root.")
    ok, notes = dryrun.validate(cfg)
    if not ok:
        return ("escalated", {"config": cfg, "dryrun": notes}, "researcher",
                f"[{worker}] {q['kind']} DRY-RUN FAILED for {cfg}: {'; '.join(notes)}. "
                f"Researcher: fix the config wiring, then I'll queue it.")
    # don't blindly re-queue a config the monitor already killed for hanging — wait for a real fix
    method = _method_of(cfg)
    for e in ledger.entries():
        if e.get("change") == method and str(e.get("cv")) == "hang":
            return ("escalated", {"config": cfg, "method": method}, "researcher",
                    f"[{worker}] {q['kind']} BLOCKED for {cfg}: it previously HUNG (train-monitor killed it — "
                    f"dataloader worker deadlock). Researcher: set num_workers=0 or fix the aug, clear the 'hang' "
                    f"in the journal, then I'll re-queue. Not wasting the GPU on a known hang.")
    job_id, note = runner.submit(cfg, q["question"], [q["kind"], Path(cfg).stem],
                                 approved=bool(q["spec"].get("approved")))
    script = f"bash start_train.sh {cfg}"  # ALWAYS the config/*.yml train command
    stage = _STAGE.get(q["kind"])
    change = _method_of(cfg)  # == MLflow run_name == weights/<method>/ so the score step + journal align
    description = _description(cfg, change, q["question"], q["spec"])  # WHAT WE DID (leader-supplied)
    if job_id == "DRY":
        return ("done", {"config": cfg, "dry": True}, "all",
                f"[{worker}] {q['kind']} dry-run mode: {cfg} wiring GREEN ({notes[0]}); would queue.")
    if job_id == "READY":
        ledger.record(change=change, description=description, script=script, train_set=_trainset(cfg), stage=stage)
        return ("done", {"config": cfg, "ready": True, "note": note}, "all",
                f"[{worker}] {q['kind']} READY (Python dry-run GREEN): {cfg} — {notes[0]}. Logged to the journal (CV pending). "
                f"Auto-submit is OFF; enable FLEET_AUTO_SUBMIT=1 and I'll queue it to :7799 myself (no Claude).")
    if job_id:
        ledger.record(change=change, description=description, script=script, train_set=_trainset(cfg), stage=stage)
        # STARTED marker (past-tense — NOT 'currently running'; the live one is the highlighted banner row)
        ledger.log("experiments", kind="started", run=change,
                   summary=f"STARTED {change} on {_trainset(cfg)} (leader's decision): {description}"[:220],
                   detail=f"config={cfg}; job={job_id}; screen={_trainset(cfg)}; then predict→golden CV→post-analysis")
        # CLOSE THE LOOP: queue the predict+score follow-up so golden_cv lands after training.
        # Per-method question text so board.add doesn't dedup distinct experiments.
        board.add("A", "score", f"Predict+score '{change}' → golden CV (job {job_id})",
                  {"method": change, "config": cfg, "train_job": job_id, "stage": stage})
        return ("done", {"config": cfg, "job_id": job_id}, "all",
                f"[{worker}] {q['kind']} QUEUED by Python (no Claude): {cfg} dry-run GREEN → "
                f"train-service job {job_id}. Queued the predict+score step → golden_cv to MLflow + journal (no Kaggle).")
    # queue busy → hold and retry next cycle (one experiment at a time)
    return ("holding", {"config": cfg, "note": note}, "all",
            f"[{worker}] {q['kind']} holding: {cfg} dry-run GREEN, waiting for the GPU queue ({note}).")
