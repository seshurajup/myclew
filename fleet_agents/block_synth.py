"""block-synth — mine the DISTINCT post-proc code blocks across public notebooks and compose NEW ones.

Insight (the user's): every public learned-graph notebook reuses the SAME trained pilkwang model; what
differs is the POST-PROCESSING code blocks each adds — gap-recovery, safe-division, gap-refine, min-
track-len filtering. Each block is a pilk_post feature toggle (a BIOHUB_* env flag). This agent:

  1. scans the pulled notebooks (research/public_notebooks/) and detects which blocks each one uses,
  2. builds a catalog: block → the param values seen across notebooks,
  3. SYNTHESISES novel combinations — blocks toggled together that NO single notebook combined —
  4. enqueues each novel combo as a verify-cv golden-12 run (real score, cached by signature).

So it doesn't just grid known params (combo-search does that) — it composes new code-block recipes from
the differences between notebooks, then lets the fleet score them. Pure Python; no Claude.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
PULLED = COMP / "research" / "public_notebooks"
LEDGER = COMP / "docs" / "experiment_ledger.jsonl"
STATE = COMP / "config" / "_auto" / "block_synth_state.json"

# a post-proc BLOCK → (BIOHUB env flag it maps to, regex that detects the block in notebook code)
BLOCKS = {
    "safe_div":     ("BIOHUB_OUTPUT_SAFE_DIVISIONS", r"add_safe_divisions|safe_div|DO_SAFE_DIV"),
    "gap2_recover": ("BIOHUB_OUTPUT_GAP2_RECOVERY",  r"gap2|gap_?2_?recover|second[_-]?gap"),
    "gap_refine":   ("BIOHUB_GAP_REFINE_SYNTHETIC",  r"gap_?refine|synthetic_?mid|refine_?synth"),
}
MAX_ENQUEUE = 3   # novel combos to score per run (bounded — each is a real golden-12 run)


def _notebook_blocks():
    """Per-notebook: which blocks it uses + its gap/min-track-len values."""
    out = []
    if not PULLED.exists():
        return out
    for d in sorted(PULLED.iterdir()):
        if not d.is_dir():
            continue
        text = ""
        for f in d.glob("*"):
            if f.suffix in (".ipynb", ".py"):
                try:
                    text += f.read_text(errors="replace")
                except Exception:  # noqa: BLE001
                    pass
        if not text:
            continue
        blocks = {name for name, (_, pat) in BLOCKS.items() if re.search(pat, text, re.I)}
        gap = (re.findall(r"gap[_-]?(?:close[_-]?)?um\s*[=:]\s*([0-9.]+)", text, re.I) or ["6.0"])[0]
        mtl = (re.findall(r"min[_-]?track[_-]?len\s*[=:]\s*([0-9]+)", text, re.I)
               or re.findall(r"min[_-]?(\d{1,2})\b", d.name) or ["4"])[0]
        out.append({"ref": d.name.replace("__", "/"), "blocks": blocks, "gap": gap, "mtl": mtl})
    return out


def _seen_sigs():
    sigs = set()
    if STATE.exists():
        try:
            sigs |= set(json.loads(STATE.read_text()).get("enqueued", []))
        except Exception:  # noqa: BLE001
            pass
    return sigs


def synth(q, worker):
    nbs = _notebook_blocks()
    if not nbs:
        return ("done", {"notebooks": 0}, "all",
                f"[{worker}] block-synth: no pulled notebooks yet (run notebook-sync first).")

    # catalog: the block set each notebook uses, and the value menus seen across all notebooks
    max_enqueue = max(1, int((q.get("spec") or {}).get("max_enqueue", MAX_ENQUEUE)))  # max_enqueue: novel combos to queue per run
    seen_block_sets = {frozenset(n["blocks"]) for n in nbs}
    gaps = sorted({n["gap"] for n in nbs}) or ["6.0"]
    def _asint(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 4
    mtls = sorted({n["mtl"] for n in nbs}, key=_asint) or ["4"]
    all_blocks = set(BLOCKS)

    # SYNTHESISE: block-sets NO notebook used (the union of all blocks, and pairwise supersets),
    # crossed with the best gap/min-track-len menus → novel recipes.
    novel = []
    candidate_sets = [all_blocks] + [frozenset(all_blocks - {b}) for b in all_blocks]  # full + leave-one-out
    seen = _seen_sigs()
    for bset in candidate_sets:
        if frozenset(bset) in seen_block_sets:
            continue                                   # a real notebook already used this exact block-set
        for gap in gaps[:2]:
            for mtl in mtls[:3]:
                env = {"BIOHUB_GAP_CLOSE_UM": gap, "BIOHUB_OUTPUT_MIN_TRACK_LEN": mtl}
                for name in all_blocks:
                    flag = BLOCKS[name][0]
                    env[flag] = "1" if name in bset else "0"
                sig = ",".join(f"{k}={env[k]}" for k in sorted(env))
                if sig not in seen:
                    novel.append((sig, env, sorted(bset)))

    if not novel:
        return ("done", {"notebooks": len(nbs), "novel": 0}, "all",
                f"[{worker}] block-synth: scanned {len(nbs)} notebooks — no new block-combos to try "
                f"(all synthesised recipes already queued/scored).")

    try:
        from researchpapers.fleet import board
    except Exception as exc:  # noqa: BLE001 — board unavailable → report synthesis without enqueuing
        return ("done", {"notebooks": len(nbs), "novel": len(novel), "enqueued": 0}, "all",
                f"[{worker}] block-synth: synthesised {len(novel)} novel recipes but board unavailable ({exc}).")
    picked = novel[:max_enqueue]
    for sig, env, bset in picked:
        board.add("S", "verify-cv",
                  f"block-synth NOVEL recipe (blocks {'+'.join(bset) or 'none'}, gap {env['BIOHUB_GAP_CLOSE_UM']}, "
                  f"mtl {env['BIOHUB_OUTPUT_MIN_TRACK_LEN']}) → golden-12",
                  {"ref": f"block-synth:{'+'.join(bset) or 'base'}", "env": env, "sig": sig, "exp": None})
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"enqueued": sorted(seen | {s for s, _, _ in picked})}, indent=2))

    return ("done", {"notebooks": len(nbs), "block_sets_in_notebooks": len(seen_block_sets),
                     "novel_enqueued": len(picked)}, "all",
            f"[{worker}] BLOCK-SYNTH: {len(nbs)} notebooks share the pilkwang model; found "
            f"{len(seen_block_sets)} distinct post-proc block-sets → SYNTHESISED {len(picked)} NEW recipes "
            f"({', '.join('+'.join(b) or 'base' for _, _, b in picked)}) and queued them for golden-12 scoring. No Claude.")
