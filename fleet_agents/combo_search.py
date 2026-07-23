"""combo-search — autonomous golden-12 combination search over PUBLIC-NOTEBOOK post-proc knobs.

Reduces Claude dependency: the fleet (not the leader/researcher) grids the inference/post-processing
parameters the top public notebooks tune — det_threshold, gap-close µm, min-track-len (the drkongvis
`filter-short` lesson), safe-division distance — and scores every combination on the LOCAL golden-12
CV via scripts/score_postproc_golden12.py (fixed pilkwang predictions, NO training). Coordinate
descent: one combo per tick (~3-4 min each), persisted, so repeated fleet ticks cover the grid and it
always reports the running best. Escalates to a human only when the best beats the public LB bar.
"""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
PY_ENV = COMP / "research" / "cellmot_venv" / "bin" / "python"
SCRIPT = COMP / "scripts" / "score_postproc_golden12.py"
STATE = COMP / "config" / "_auto" / "combo_search_state.json"
SCREEN_N = 4         # screen combos on a FAST embryo-balanced mini-golden (full-12 confirm on convergence)
MIN_GAIN = 0.002     # a combo must beat the baseline screen by this to be worth a full-12 confirm

# each axis = the candidate values the top public notebooks actually use (mined into kaggle_learnings.md)
# 2026-07-07 (researcher, leader-approved): dropped the DET_THRESHOLD axis — it is a proven NO-OP on the
# CACHED preds (BIOHUB_DET_THRESHOLD is read at pilk_post.py:11 but used only in prediction-time peak
# detection, never in filter_output_graph; empirically det 0.988/0.99/0.992 → identical 0.8315). True
# det_thresh needs GPU re-detection (deferred spec in docs/baseline_v13_gpu_job_spec.md). Also PARKED the
# SAFE_DIV axis: golden-12 is division-BLIND (~8 GT divs, div-rich not in cache) so a div knob can only
# lose adj_edge here with no visible upside (same trap as add-daughter). Concentrate on min_track_len × gap.
AXES = {
    "BIOHUB_GAP_CLOSE_UM":       ["5.5", "6.0", "6.8"],                  # gap-close distance (µm)
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": ["4", "5", "6", "8", "10", "12", "14"],  # drkongvis/boristown filter-short (min6..min14)
}
# FILTER_SHORT_TRACKS=1 is REQUIRED for the min_track_len axis to fire — pilk_post.filter_short_track_components
# (pilk_post.py:703) early-returns unless this flag is on. Without it the whole min_track_len axis is a silent
# no-op (every value scores identically). Held as a fixed enable-flag in BASE, not a swept axis.
BASE = {"BIOHUB_OUTPUT_FILTER_SHORT_TRACKS": "1", "BIOHUB_GAP_CLOSE_UM": "6.0", "BIOHUB_OUTPUT_MIN_TRACK_LEN": "4"}


def _key(env: dict) -> str:
    env = env or {}
    return ",".join(f"{k}={env[k]}" for k in sorted(env))


def _load():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {"best": {"env": dict(BASE), "score": None}, "evaluated": {}, "queue": [], "round": 0}


def _save(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def _seed_queue(best_env: dict, evaluated: dict) -> list:
    """One-axis-at-a-time variations from the current best that we haven't scored yet."""
    q = []
    if _key(best_env) not in evaluated:
        q.append(dict(best_env))                       # score the anchor itself first
    for axis, vals in AXES.items():
        for v in vals:
            if v == best_env.get(axis):
                continue
            cand = dict(best_env); cand[axis] = v      # vary ONE axis
            if _key(cand) not in evaluated:
                q.append(cand)
    return q


def _score(env: dict, worker: str, screen_n: int = SCREEN_N, timeout: int = 600):
    """screen_n: mini-golden datasets to score on; timeout: per-combo scorer wall-clock cap (s)."""
    run_env = dict(os.environ)
    run_env.update(env or {})
    try:
        r = subprocess.run([str(PY_ENV), str(SCRIPT), str(max(1, int(screen_n)))], capture_output=True,
                           text=True, timeout=max(1, int(timeout)), cwd=str(COMP), env=run_env)
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
    if not PY_ENV.exists() or not SCRIPT.exists():
        return ("escalated", {}, "researcher",
                f"[{worker}] combo-search: cellmot_venv or score_postproc_golden12.py missing.")
    st = _load()
    evaluated = st["evaluated"]
    best = st["best"]

    if not st["queue"]:                                # (re)seed a descent round around the current best
        st["queue"] = _seed_queue(best["env"], evaluated)
        st["round"] = st.get("round", 0) + 1

    if not st["queue"]:                                # converged — no unscored neighbour improves
        _save(st)
        base_screen = st.get("baseline")
        gain = (best["score"] - base_screen) if (base_screen is not None and best["score"] is not None) else 0.0
        better = gain >= MIN_GAIN and best["env"] != BASE
        msg = (f"[{worker}] COMBO-SEARCH converged (round {st['round']}): best mini-golden = "
               f"{best['score']} (baseline {base_screen}, +{gain:.4f}) @ {_key(best['env']).replace('BIOHUB_','')}. "
               f"{len(evaluated)} combos scored. "
               + ("🎯 A better combo than the pilkwang default — confirm on FULL golden-12 + human submission review."
                  if better else "no combo beats the default screen; default stands."))
        return ("escalated" if better else "done",
                {"best_score": best["score"], "best_env": best["env"], "baseline": base_screen,
                 "gain": round(gain, 4), "combos_scored": len(evaluated), "better_than_default": better},
                "leader" if better else "all", msg)

    env = st["queue"].pop(0)                            # evaluate exactly ONE combo this tick
    d, err = _score(env, worker)
    if err:
        evaluated[_key(env)] = {"error": err}
        _save(st)
        return ("done", {"skipped": _key(env), "error": err}, "all",
                f"[{worker}] combo-search skipped {_key(env)} ({err}).")

    evaluated[_key(env)] = {"score": d["score"], "adjE": d["adjE"]}
    if _key(env) == _key(BASE):                        # remember the pilkwang-default screen as baseline
        st["baseline"] = d["score"]
    prev_env = best.get("env", {}); prev_exp = best.get("exp"); prev_score = best.get("score")
    improved = prev_score is None or d["score"] > prev_score

    # ── DrHB grandmaster-journal trick: "same as EXP_X but <ONE change>" + a status marker ──
    from . import ledger
    _sh = lambda k: k.replace("BIOHUB_", "").replace("OUTPUT_", "").lower()
    diff = [(_sh(k), env[k]) for k in sorted(env) if env.get(k) != prev_env.get(k)]
    if _key(env) == _key(BASE):
        desc, parent = "`pilkwang` base post-proc (`gap`=`6.0`, `min_track_len`=`4`)", None
    else:
        desc = ", ".join(f"`{k}`=`{v}`" for k, v in diff) or "post-proc combo"   # ledger prepends "same as EXP_X but"
        parent = prev_exp
    delta = (d["score"] - prev_score) if prev_score is not None else 0.0
    marker = ("first run" if prev_score is None
              else (f"★ new best (Δ+{delta:.4f})" if improved
                    else ("`same`" if abs(delta) < 1e-6 else f"`worse` (Δ{delta:+.4f})")))
    row = ledger.record(change=f"combo:{_key(env).replace('BIOHUB_', '')[:44]}", description=desc,
                        script="scripts/score_postproc_golden12.py", cv=d["score"], train_set=f"golden{SCREEN_N}",
                        parent=parent, stage=8, observation=marker)
    tag = _key(env).replace("BIOHUB_", "").replace("OUTPUT_", "")
    if improved:
        best = st["best"] = {"env": dict(env), "score": d["score"], "adjE": d["adjE"], "exp": row.get("exp")}
        st["queue"] = []                               # new best → reseed a fresh descent round next tick
    _save(st)
    msg = (f"[{worker}] COMBO-SEARCH {'★ NEW BEST' if improved else 'tried'}: mini-golden = {d['score']} "
           f"(adjE {d['adjE']}) @ {tag}. Running best = {best['score']} @ {_key(best['env']).replace('BIOHUB_','')}. "
           f"{len(evaluated)} scored, {len(st['queue'])} queued (screen n={SCREEN_N}; full-12 confirm on convergence).")
    return ("done",
            {"combo": _key(env), "score": d["score"], "adjE": d["adjE"], "improved": improved,
             "running_best": best["score"], "queued": len(st["queue"])},
            "all", msg)
