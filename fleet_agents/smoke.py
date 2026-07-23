"""smoke — pre-flight: run a TINY real end-to-end of a config (1 epoch, a few iters, real data) under a
HARD timeout, so a broken/hanging config is caught in ~minutes instead of a 32-min silent stall.

`dryrun` only checks wiring (imports/paths) — it did NOT catch the dataloader deadlock because it never
actually trains. This agent DOES: it builds a throwaway tiny copy of the config (keeps num_workers so a
worker deadlock still reproduces), runs `start_train.sh <tiny>` under `timeout -s KILL`, and verifies the
training loop advanced past iteration 0. GREEN → the full run is safe to queue; HANG/FAIL → escalate.

Generic: derives everything from the config; any competition using the config-driven trainer benefits.
"""
from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
TRAIN_SERVICE = "http://127.0.0.1:7799"
CAP_S = 300          # hard wall-clock cap for the smoke
SMOKE_DIR = COMP / "config" / "_smoke"


def _queue_busy() -> bool:
    try:
        with urllib.request.urlopen(f"{TRAIN_SERVICE}/api/board", timeout=4) as r:
            q = json.loads(r.read()).get("queue", {})
        return int(q.get("running_count", 0)) + int(q.get("queued_count", 0)) > 0
    except Exception:  # noqa: BLE001
        return True


def _tiny_config(cfg: str) -> Path | None:
    """Throwaway copy: 1 epoch, a couple of iters, tiny logging — but SAME num_workers (reproduce hangs)."""
    try:
        import yaml
        c = yaml.safe_load((COMP / cfg).read_text()) or {}
    except Exception:  # noqa: BLE001
        return None
    tr = c.setdefault("train", {})
    tr["method"] = f"smoke_{tr.get('method', Path(cfg).stem)}"
    tr["epochs"] = 1
    tr["max_iters"] = 3
    c.setdefault("mlflow", {})["run_name"] = tr["method"]
    c["mlflow"]["experiment"] = "kaggle-biohub-smoke"
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    out = SMOKE_DIR / f"{Path(cfg).stem}.yml"
    import yaml as _y
    out.write_text(_y.safe_dump(c, sort_keys=False))
    return out


def _kill_stragglers():
    try:
        out = subprocess.run(["ps", "-eo", "pid,comm,args"], capture_output=True, text=True, timeout=6).stdout
        for ln in out.splitlines():
            p = ln.split(None, 2)
            if len(p) == 3 and "python" in p[1] and "smoke_" in p[2] and "train" in p[2]:
                subprocess.run(["kill", "-KILL", p[0]], capture_output=True)
    except Exception:  # noqa: BLE001
        pass


def smoke(q, worker):
    """Tiny real run of the config under a hard timeout. GREEN → safe to queue full; else escalate.

    Optional spec: `cap_s` overrides the hard wall-clock cap (default 300s)."""
    from researchpapers.fleet import post as _post
    spec = (q or {}).get("spec", {}) or {}
    cfg = spec.get("config")
    if not cfg or not (COMP / cfg).exists():
        return ("escalated", {"config": cfg}, "researcher", f"[{worker}] SMOKE: config '{cfg}' not found.")
    if getattr(_post, "DRY", False):
        return ("done", {"config": cfg, "dry": True}, "all", f"[{worker}] SMOKE dry-run: would tiny-run {cfg}.")
    if _queue_busy():
        return ("holding", {"config": cfg}, "all", f"[{worker}] SMOKE holding: GPU busy — will smoke {cfg} when free.")
    tiny = _tiny_config(cfg)
    if not tiny:
        return ("escalated", {"config": cfg}, "researcher", f"[{worker}] SMOKE: could not build a tiny config for {cfg}.")
    try:
        cap_s = int(spec.get("cap_s", CAP_S))
    except (TypeError, ValueError):
        cap_s = CAP_S
    cmd = ["timeout", "-s", "KILL", str(cap_s), "bash", "start_train.sh", str(tiny.relative_to(COMP))]
    try:
        proc = subprocess.run(cmd, cwd=str(COMP), capture_output=True, text=True)
    except Exception as e:  # noqa: BLE001 — could not launch smoke run → clean escalate, not a crash
        _kill_stragglers()
        return ("escalated", {"config": cfg, "err": str(e)[:120]}, "researcher",
                f"[{worker}] SMOKE: could not launch tiny run for {cfg} ({type(e).__name__}).")
    _kill_stragglers()
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [""]
    advanced = any(k in proc.stdout for k in ("it/s", "loss=", "Epoch 1", "epoch 1", "1/1")) or "train done" in proc.stdout
    if proc.returncode == 137 or (proc.returncode != 0 and not advanced):  # 137 = KILL (timed out = hang)
        why = "HANG (hard timeout — loop never advanced past iter 0)" if proc.returncode == 137 else f"FAILED (exit {proc.returncode})"
        return ("escalated", {"config": cfg, "rc": proc.returncode, "last": tail[-1][:120]}, "researcher",
                f"[{worker}] 🛑 SMOKE {why} for {cfg} (last: {tail[-1][:100]!r}). Do NOT queue the full run — "
                f"likely num_workers dataloader deadlock; researcher: set num_workers=0 or fix the loader, then re-smoke.")
    return ("done", {"config": cfg, "green": True, "last": tail[-1][:120]}, "all",
            f"[{worker}] ✅ SMOKE GREEN for {cfg}: tiny run advanced (last: {tail[-1][:80]!r}). Safe to queue the full training.")
