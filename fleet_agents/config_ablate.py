"""config-ablate — leave-one-BLOCK-out ablation of the yaroslav-v4 full config (0.8803 base).

fullconfig-search tunes knob *values*; this agent asks the orthogonal question: which post-proc
*blocks* actually earn the score? It disables one block at a time (motion-relink, gap-close, gap2,
gap-refine-synthetic, linefit smoothing, safe-divisions, single-parent-repair, short-track filter,
ILP itself) and scores golden-12. A block whose removal DROPS the score is load-bearing (keep + maybe
push); a block whose removal is neutral/positive is dead weight (candidate to prune or re-tune).

Output is a DrHB-style ablation table posted to the board — the "what makes 0.897 work" map that tells
the fleet where the remaining headroom is. One block per tick; persisted.
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
STATE = COMP / "config" / "_auto" / "config_ablate_state.json"
BASE_ENV_SH = COMP / "model_scratch" / "results" / "div_probe" / "yaroslav_env.sh"
SCREEN_N = 4

# block name -> the env override that DISABLES that block (relative to the full base)
BLOCKS = {
    "motion_relink":      {"BIOHUB_OUTPUT_MOTION_RELINK": "0"},
    "gap_close":          {"BIOHUB_OUTPUT_GAP_CLOSE": "0"},
    "gap_refine_synth":   {"BIOHUB_GAP_REFINE_SYNTHETIC": "0"},
    "linefit_smooth":     {"BIOHUB_OUTPUT_LINEFIT_SMOOTH": "0"},
    "safe_divisions":     {"BIOHUB_OUTPUT_SAFE_DIVISIONS": "0"},
    "single_parent_repair": {"BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR": "0"},
    "short_track_filter": {"BIOHUB_OUTPUT_FILTER_SHORT_TRACKS": "0"},
    "enforce_next_frame": {"BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME": "0"},
    "ilp":                {"BIOHUB_USE_ILP": "0"},
    "prune_isolated":     {"BIOHUB_OUTPUT_PRUNE_ISOLATED": "0"},
}


def _base_env():
    env = {}
    if BASE_ENV_SH.exists():
        for ln in BASE_ENV_SH.read_text().splitlines():
            m = re.match(r"\s*export\s+(BIOHUB_[A-Z0-9_]+)=(.*)", ln)
            if m:
                env[m.group(1)] = m.group(2).strip()
    return env


def _load():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {"base_score": None, "done": {}, "queue": list(BLOCKS.keys())}


def _save(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def _score(env, screen_n=SCREEN_N, timeout=900):
    """screen_n: mini-golden datasets to score on; timeout: scorer wall-clock cap (s)."""
    run_env = dict(os.environ); run_env.update(env or {})
    try:
        r = subprocess.run([str(PY_ENV), str(SCRIPT), str(max(1, int(screen_n)))], capture_output=True,
                           text=True, timeout=max(1, int(timeout)), cwd=str(COMP), env=run_env)
    except subprocess.TimeoutExpired:
        return None
    line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    if not line:
        return None
    try:
        d = json.loads(line[-1])
    except (ValueError, json.JSONDecodeError):
        return None
    return d if d.get("n", 0) and d.get("score", 0) else None


def report(q, worker):
    if not BASE_ENV_SH.exists():
        return ("escalated", {}, "researcher", f"[{worker}] config-ablate: yaroslav base env missing.")
    base = _base_env()
    st = _load()

    if st["base_score"] is None:                       # score the intact base first
        d = _score(base)
        if not d:
            return ("done", {}, "all", f"[{worker}] config-ablate: base score failed; retry next tick.")
        st["base_score"] = d["score"]
        _save(st)
        return ("done", {"base_score": d["score"]}, "all",
                f"[{worker}] CONFIG-ABLATE: base (yaroslav-v4 full) golden-{SCREEN_N} = **{d['score']}**. "
                f"Ablating {len(BLOCKS)} blocks one at a time to find the load-bearing ones.")

    if not st["queue"]:                                # converged — post the full ablation table
        rows = sorted(st["done"].items(), key=lambda kv: kv[1].get("delta", 0))
        tbl = "\n".join(f"| {b} | {v.get('score','—')} | {v.get('delta',0):+.4f} | "
                        f"{'🔧 load-bearing' if v.get('delta',0) < -0.001 else ('dead weight' if v.get('delta',0) > 0.0005 else 'neutral')} |"
                        for b, v in rows)
        from . import ledger
        ledger.log("config-ablate",
                   summary=f"yaroslav-v4 block ablation done (base {st['base_score']}) — see load-bearing map",
                   detail="; ".join(f"{b}:{v.get('delta',0):+.4f}" for b, v in rows), kind="finding")
        from researchpapers.fleet import post
        msg = (f"[{worker}] 🧭 **CONFIG-ABLATE map** — which blocks earn the 0.8803 base "
               f"(golden-{SCREEN_N}, Δ = score with block OFF minus base):\n\n"
               f"| block removed | score | Δ | verdict |\n|---|--:|--:|---|\n{tbl}\n\n"
               f"🔧 = removing it hurts (keep/push); dead weight = safe to prune or re-tune for gains.")
        post.post_thread(worker, "all", msg, routine=False, kind="finding")
        return ("done", {"base": st["base_score"], "blocks": len(st["done"])}, "all",
                f"[{worker}] config-ablate converged — {len(st['done'])} blocks mapped (base {st['base_score']}).")

    block = st["queue"].pop(0)
    env = dict(base); env.update(BLOCKS[block])
    d = _score(env)
    if not d:
        _save(st)
        return ("done", {"skipped": block}, "all", f"[{worker}] config-ablate skipped {block} (score failed).")
    delta = d["score"] - st["base_score"]
    st["done"][block] = {"score": d["score"], "delta": round(delta, 4)}
    _save(st)
    verdict = "🔧 load-bearing" if delta < -0.001 else ("dead weight" if delta > 0.0005 else "neutral")
    return ("done", {"block": block, "score": d["score"], "delta": round(delta, 4), "verdict": verdict}, "all",
            f"[{worker}] CONFIG-ABLATE: `{block}` OFF → golden-{SCREEN_N} = {d['score']} "
            f"(Δ{delta:+.4f}, {verdict}). {len(st['queue'])} blocks left.")
