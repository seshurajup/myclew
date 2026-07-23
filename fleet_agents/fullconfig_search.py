"""fullconfig-search — WIDE autonomous golden-12 search over the FULL post-proc config.

combo-search only sweeps 3 knobs (gap / min-track-len / filter) around a tiny pilkwang base and caps
at ~0.8735. This agent starts from the *full* yaroslav-v4 UNet+ILP config (the public 0.897 notebook,
reproduced on our golden-12 at 0.8803) and coordinate-descends over the knobs that actually move the
metric: the ILP edge/division weights, the division-distance gates, motion-relink, gap-close, linefit
smoothing and min-track-len. One combo per tick on the fast embryo-balanced screen; a new best is
confirmed on full golden-12. Writes DrHB "same as EXP_X but <one change>" rows and tracks the best.

This is the fleet's "beat the public best" engine: it searches ABOVE 0.8803 toward 0.897+.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
PY_ENV = COMP / "research" / "cellmot_venv" / "bin" / "python"
SCRIPT = COMP / "scripts" / "score_postproc_golden12.py"
STATE = COMP / "config" / "_auto" / "fullconfig_search_state.json"
BASE_ENV_SH = COMP / "model_scratch" / "results" / "div_probe" / "yaroslav_env.sh"
SCREEN_N = 4          # fast embryo-balanced screen; full-12 confirm on a new best
MIN_GAIN = 0.0015     # a combo must beat the running best by this on the screen to trigger a full-12 confirm

# knobs that plausibly move adj-edge OR the division term, with candidate values around the yaroslav base.
AXES = {
    "BIOHUB_ILP_DIVISION_WEIGHT":        ["0.7", "1.0", "1.3", "1.6"],   # encourage/penalise divisions in the ILP
    "BIOHUB_ILP_EDGE_WEIGHT":            ["-1.2", "-1.0", "-0.8"],       # edge reward strength
    "BIOHUB_OUTPUT_MIN_TRACK_LEN":       ["4", "5", "6", "7", "8"],      # short-track filter
    "BIOHUB_GAP_CLOSE_UM":               ["5.5", "6.0", "6.8"],          # gap-close distance
    "BIOHUB_MOTION_RELINK_LEARNED_BONUS":["0.5", "0.75", "1.0"],         # trust the learned linker more/less
    "BIOHUB_OUTPUT_LINEFIT_WEIGHT":      ["0.6", "0.8", "1.0"],          # trajectory smoothing strength
    "BIOHUB_SAFE_DIV_MAX_UM":            ["4.7", "5.5", "6.5"],          # geometric safe-division parent gate
    "BIOHUB_DIV_PARENT_MAX_UM":          ["9.0", "10.5", "12.0"],        # ILP division parent gate
}


def _base_env() -> dict:
    """Parse the yaroslav full config (the 0.8803 base) from the exported env file."""
    env = {}
    if BASE_ENV_SH.exists():
        for ln in BASE_ENV_SH.read_text().splitlines():
            m = re.match(r"\s*export\s+(BIOHUB_[A-Z0-9_]+)=(.*)", ln)
            if m:
                env[m.group(1)] = m.group(2).strip()
    return env


def _key(env: dict) -> str:
    # lineage key = only the swept axes (base is fixed), so states stay compact/comparable
    env = env or {}
    return ",".join(f"{k}={env[k]}" for k in sorted(env) if k in AXES)


def _load(base):
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {"best": {"env": dict(base), "score": 0.8803, "screen": None, "exp": None},
            "screen_base": None, "evaluated": {}, "queue": [], "round": 0}


def _save(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def _seed_queue(best_env, evaluated):
    q = []
    for axis, vals in AXES.items():
        for v in vals:
            if v == best_env.get(axis):
                continue
            cand = dict(best_env); cand[axis] = v      # vary ONE axis from the current best
            if _key(cand) not in evaluated:
                q.append(cand)
    return q


def _score(env, n, timeout=900):
    """timeout: scorer wall-clock cap (s) — a stuck combo is skipped, not left to hang the fleet."""
    run_env = dict(os.environ)
    run_env.update(env or {})
    try:
        r = subprocess.run([str(PY_ENV), str(SCRIPT), str(n)], capture_output=True, text=True,
                           timeout=max(1, int(timeout)), cwd=str(COMP), env=run_env)
    except subprocess.TimeoutExpired:
        return None, f"scorer timed out after {timeout}s"
    line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    if not line:
        return None, (r.stderr or r.stdout)[-160:]
    try:
        d = json.loads(line[-1])
    except (ValueError, json.JSONDecodeError):
        return None, f"unparseable scorer output: {line[-1][:120]}"
    if d.get("n", 0) == 0 or d.get("score", 0) == 0:
        return None, f"scored 0 datasets (n={d.get('n')})"
    return d, None


def search(q, worker):
    if not PY_ENV.exists() or not SCRIPT.exists() or not BASE_ENV_SH.exists():
        return ("escalated", {}, "researcher",
                f"[{worker}] fullconfig-search: cellmot_venv, score script, or yaroslav base env missing.")
    base = _base_env()
    st = _load(base)
    evaluated, best = st["evaluated"], st["best"]

    # establish the SCREEN baseline (base config on the fast n=SCREEN_N subset) — candidates are compared
    # on the SAME scale (screen≈0.839 ≠ full-12≈0.880), else nothing ever "improves". Full-12 only confirms.
    if st.get("screen_base") is None:
        d, err = _score(base, SCREEN_N)
        if err:
            return ("done", {}, "all", f"[{worker}] fullconfig-search: base screen failed ({err}); retry next tick.")
        st["screen_base"] = d["score"]; best["screen"] = d["score"]
        _save(st)
        return ("done", {"screen_base": d["score"]}, "all",
                f"[{worker}] FULLCONFIG-SEARCH: base (yaroslav-v4 full) golden-{SCREEN_N} screen = "
                f"**{d['score']}** (full-12 = 0.8803). Now descending 8 axes to beat it.")

    # ensure the best env carries the full base (swept axes overlaid)
    full_best = dict(base); full_best.update({k: v for k, v in best["env"].items() if k in AXES})

    if not st["queue"]:
        st["queue"] = _seed_queue(full_best, evaluated)
        st["round"] = st.get("round", 0) + 1

    if not st["queue"]:
        _save(st)
        msg = (f"[{worker}] FULLCONFIG-SEARCH converged (round {st['round']}): best golden-{SCREEN_N} = "
               f"{best['score']} @ {_key(full_best).replace('BIOHUB_','') or 'yaroslav-base'}. "
               f"{len(evaluated)} combos scored over 8 axes.")
        beat = best["score"] and best["score"] > 0.8803 + MIN_GAIN
        return ("escalated" if beat else "done",
                {"best_score": best["score"], "best_env": _key(full_best), "combos": len(evaluated),
                 "beats_base": bool(beat)},
                "leader" if beat else "all", msg)

    env = dict(base); env.update(st["queue"].pop(0))   # full config = base + this candidate's swept axes
    cand_axes = {k: env[k] for k in AXES if k in env}
    d, err = _score(env, SCREEN_N)
    if err:
        evaluated[_key(cand_axes)] = {"error": err}
        _save(st)
        return ("done", {"skipped": _key(cand_axes), "error": err}, "all",
                f"[{worker}] fullconfig-search skipped {_key(cand_axes)} ({err}).")

    evaluated[_key(cand_axes)] = {"score": d["score"], "adjE": d["adjE"]}
    prev_screen = best.get("screen") or st["screen_base"]   # compare on the SCREEN scale
    improved = d["score"] > prev_screen + 1e-9

    from . import ledger
    _sh = lambda k: k.replace("BIOHUB_", "").replace("OUTPUT_", "").replace("ILP_", "ilp_").lower()
    diff = [(_sh(k), env[k]) for k in AXES if env.get(k) != full_best.get(k)]
    desc = ", ".join(f"`{k}`=`{v}`" for k, v in diff) or "yaroslav-v4 base"
    delta = d["score"] - prev_screen
    marker = (f"★ new best (screen Δ+{delta:.4f})" if improved
              else ("`same`" if abs(delta) < 1e-6 else f"`worse` (Δ{delta:+.4f})"))
    row = ledger.record(change=f"fullcfg:{_key(cand_axes).replace('BIOHUB_','')[:44]}", description=desc,
                        script="scripts/score_postproc_golden12.py", cv=d["score"], train_set=f"golden{SCREEN_N}",
                        parent=best.get("exp"), stage=8, observation=marker)
    confirmed = None
    if improved:
        # confirm on FULL golden-12 (the real scale) before crowning; store both
        full, ferr = _score(env, 0)
        confirmed = full["score"] if full else None
        best = st["best"] = {"env": dict(cand_axes), "score": confirmed or 0.8803, "adjE": (full or d)["adjE"],
                             "exp": row.get("exp"), "screen": d["score"]}
        st["queue"] = []                               # reseed a fresh descent from the new best
    _save(st)
    tag = _key(cand_axes).replace("BIOHUB_", "") or "yaroslav-base"
    conf_txt = f" → full-12 **{confirmed}**" if confirmed else ""
    msg = (f"[{worker}] FULLCONFIG-SEARCH {'★ NEW BEST' if improved else 'tried'}: screen = {d['score']} "
           f"(base screen {st['screen_base']}){conf_txt} @ {tag[:60]}. Best full-12 = {best['score']}. "
           f"{len(evaluated)} scored, {len(st['queue'])} queued (8-axis descent from yaroslav-v4 0.8803).")
    return ("done",
            {"combo": _key(cand_axes), "screen": d["score"], "confirmed_full12": confirmed,
             "improved": improved, "running_best": best["score"], "queued": len(st["queue"])},
            "all", msg)
