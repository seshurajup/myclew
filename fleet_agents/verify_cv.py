"""verify-cv — compute the REAL golden-12 CV for a public learned-graph notebook (no hardcoding).

The old bug: notebook-sync stamped every learned-graph fork with the literal 0.8708 anchor without
running it. This agent fixes that: it runs the notebook's mined post-processing params (det_threshold,
gap-close µm, min-track-len) through scripts/score_postproc_golden12.py on the pilkwang predictions —
a REAL golden-12 run — and writes the measured CV back to the ledger via ledger.set_scores.

Efficiency + honesty: results are cached by PARAM SIGNATURE (config/_auto/verified_cv_cache.json), so
identical-param forks reuse a single verified run instead of re-scoring 70×. The cache is seeded with
the ONE genuinely-verified anchor (pilkwang defaults → 0.8708, proven by MLflow run
pilkwang_baseline_score_validate). If a run fails, we do NOT fake a number — the row is left pending.
"""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
PY_ENV = COMP / "research" / "cellmot_venv" / "bin" / "python"
SCRIPT = COMP / "scripts" / "score_postproc_golden12.py"
CACHE = COMP / "config" / "_auto" / "verified_cv_cache.json"
# the one verified anchor (MLflow: pilkwang_baseline_score_validate official_score=0.8708) → seed the cache
_BASE_SIG = "BIOHUB_GAP_CLOSE_UM=6.0,BIOHUB_OUTPUT_MIN_TRACK_LEN=4"
_SEED = {_BASE_SIG: {"cv": 0.8708, "note": "verified pilkwang default (MLflow pilkwang_baseline_score_validate)"}}


def _cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:  # noqa: BLE001
            pass
    return dict(_SEED)


def _save_cache(c: dict):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c, indent=2))


def run(q, worker):
    spec = (q or {}).get("spec", {}) or {}
    env_over = spec.get("env", {}) or {}               # {BIOHUB_*: value}
    sig = spec.get("sig") or ",".join(f"{k}={env_over[k]}" for k in sorted(env_over))
    exp = spec.get("exp")
    ref = spec.get("ref", "?")
    strict = bool(spec.get("strict"))                  # strict: a failed/timed-out run escalates (not silent-pending)
    try:
        run_timeout = int(spec.get("timeout", 1500))   # timeout: wall-clock seconds for the golden-12 scorer
    except (TypeError, ValueError):
        run_timeout = 1500
    _pend_status = "escalated" if strict else "done"
    _pend_to = "researcher" if strict else "all"

    cache = _cache()
    _short0 = sig.replace("BIOHUB_", "").replace("OUTPUT_", "").replace("_UM", "")
    if sig in cache and isinstance(cache[sig].get("cv"), (int, float)):   # identical params already verified
        cv = cache[sig]["cv"]
        _update_ledger(exp, cv, f"`{_short0}` golden-12 {cv} (verified — cached)", sig=_short0, ref=ref)
        return ("done", {"ref": ref, "cv": cv, "sig": sig, "cached": True}, "all",
                f"[{worker}] VERIFY-CV {ref}: golden-12 = {cv} (cached verified run of identical params {sig}). No re-run.")

    if not PY_ENV.exists() or not SCRIPT.exists():
        return (_pend_status, {"ref": ref, "cv": None}, _pend_to,
                f"[{worker}] verify-cv {ref}: scorer missing → row stays pending (no faked CV).")

    # ── ONE message, UPDATED IN PLACE (⏳ running → progress → ✓ done) — no message spam on /runtime ──
    from researchpapers.fleet import post
    short = sig.replace("BIOHUB_", "").replace("OUTPUT_", "")
    # routine=False so this live-updating message is VISIBLE on the board (routine msgs are hidden)
    eid = post.post_thread(worker, "all", f"[{worker}] ⏳ VERIFY-CV {ref}: scoring golden-12 ({short}) … starting",
                           routine=False)

    run_env = dict(os.environ)
    run_env.update({k: str(v) for k, v in env_over.items()})
    import re as _re
    try:
        proc = subprocess.Popen([str(PY_ENV), str(SCRIPT)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, cwd=str(COMP), env=run_env)
    except Exception as e:  # noqa: BLE001 — could not launch scorer → clean, no faked CV
        fin = f"[{worker}] ⚠ VERIFY-CV {ref}: could not launch scorer ({type(e).__name__}) → row stays pending."
        post.update_thread(eid, fin)
        return (_pend_status, {"ref": ref, "cv": None, "_posted": True}, _pend_to, fin)
    try:
        # stream stderr for PROGRESS x/n lines → update the SAME message live
        import threading
        last = {"p": ""}

        def _pump():
            for ln in proc.stderr:                     # blocks per line until the scorer prints
                m = _re.search(r"PROGRESS (\d+)/(\d+)", ln)
                if m:
                    last["p"] = f"{m.group(1)}/{m.group(2)}"
                    post.update_thread(eid, f"[{worker}] ⏳ VERIFY-CV {ref}: scored {last['p']} golden datasets ({short}) …")
        t = threading.Thread(target=_pump, daemon=True); t.start()
        out, _ = proc.communicate(timeout=run_timeout)
        t.join(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        fin = f"[{worker}] ⚠ VERIFY-CV {ref}: golden-12 run TIMED OUT → row stays pending (no faked CV)."
        post.update_thread(eid, fin)
        return (_pend_status, {"ref": ref, "cv": None, "sig": sig, "_posted": True}, _pend_to, fin)

    line = [l for l in (out or "").strip().splitlines() if l.startswith("{")]
    if not line:
        fin = f"[{worker}] ⚠ VERIFY-CV {ref}: scorer failed → row stays pending (no faked CV)."
        post.update_thread(eid, fin)
        return (_pend_status, {"ref": ref, "cv": None, "_posted": True}, _pend_to, fin)
    try:
        d = json.loads(line[-1])
    except Exception:  # noqa: BLE001 — malformed scorer output → pending, never crash
        fin = f"[{worker}] ⚠ VERIFY-CV {ref}: scorer output unparseable → row stays pending."
        post.update_thread(eid, fin)
        return (_pend_status, {"ref": ref, "cv": None, "_posted": True}, _pend_to, fin)
    cv = d.get("score")
    if not cv or d.get("n", 0) == 0:
        fin = f"[{worker}] ⚠ VERIFY-CV {ref}: scored 0 datasets → row stays pending."
        post.update_thread(eid, fin)
        return (_pend_status, {"ref": ref, "cv": None, "_posted": True}, _pend_to, fin)
    cache[sig] = {"cv": cv, "note": f"real golden-12 run (n={d.get('n')}) of {ref}"}
    _save_cache(cache)
    _update_ledger(exp, cv, f"golden-12 {cv} (VERIFIED real run of mined params {sig})", sig=short, ref=ref)
    fin = (f"[{worker}] ✓ VERIFY-CV {ref}: golden-12 = {cv} (adjE {d.get('adjE')}) — REAL run of {short}, "
           f"written to the ledger. No hardcoded anchor.")
    post.update_thread(eid, fin)                       # finalise the SAME message
    return ("done", {"ref": ref, "cv": cv, "sig": sig, "cached": False, "n": d.get("n"), "_posted": True}, "all", fin)


def _update_ledger(exp, cv, observation, sig=None, ref=None):
    try:
        from . import ledger
        if not exp:                                    # no parent row → CREATE one so EVERY run hits the journal
            _sh = (sig or ref or "?").replace("BIOHUB_", "").replace("OUTPUT_", "").replace("_UM", "")
            ledger.record(change=f"verify:{_sh[:38]}", description=f"`{_sh}` — golden-12 verified",
                          script="scripts/score_postproc_golden12.py", cv=cv, train_set="golden12",
                          stage=8, observation=observation[:80])
            return
    except Exception:  # noqa: BLE001
        pass
    if not exp:
        return
    try:
        from . import ledger
        ledger.set_scores(exp, cv=cv, observation=observation)
    except Exception:  # noqa: BLE001
        pass
