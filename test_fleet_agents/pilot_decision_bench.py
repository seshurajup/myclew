"""pilot-decision bench — does the fine-tune improve REAL fleet decisions, not just held-out exact-match?

Held-out accuracy measures the training distribution. The question that matters is different: given a real
competition state, does the model choose the agent the fleet actually went to next? This benchmark answers
that with ground truth mined from `docs/experiment_decisions.jsonl` — 2,201 observed (state -> next agent)
transitions across 93 distinct agent pairs, recorded by real runs, not written for this test.

Scored three ways, because "correct" is not binary when several agents are reasonable:
  top1        — chose exactly the agent the fleet chose
  top3        — the fleet's choice was in the model's shortlist (a search-then-pick pilot only needs this)
  dispatchable— named SOMETHING real and runnable (the floor: a pilot that names nothing is useless)

Run it on a FREE GPU. An earlier timing run of this kind was invalid because training held the card and
transformers silently offloaded layers to CPU — contention makes every number meaningless.

    python test_fleet_agents/pilot_decision_bench.py [--adapter models/tool_lora_v6] [--n 60]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "researchpapers"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fleet_agents import agent_routing as AR  # noqa: E402
from fleet_agents import prompt_dataset as PD  # noqa: E402


def gpu_free(min_gb=18):
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA"
        free = torch.cuda.mem_get_info()[0] / 1e9
        return free >= min_gb, f"{free:.1f} GB free"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:60]


def bench(model, adapter, n, comp, seen_prompts=None):
    """Score the model on real observed transitions. Returns per-metric rates plus every wrong pick."""
    data = PD.build_history_tool_dataset({"comp": comp, "history": 3, "kinds": ["next_agent"],
                                          "val_frac": 0.5})
    cases = data["val"] or data["examples"]

    # HELD-OUT ONLY. Measured defect: 48 of the first 60 cases (80.0%) had their EXACT prompt in the
    # adapter's training corpus, and the bench duly reported 80.0% top1 — it was scoring recall of seen
    # prompts, not decision skill. Fleet states repeat, so a case is only informative if the model has
    # never been trained on that prompt. Build the SAME corpus the adapter saw and subtract it.
    if seen_prompts is not None:
        cases = [c for c in cases if c["input"] not in seen_prompts]
    cases = cases[:n]
    if not cases:
        raise SystemExit("no UNSEEN next_agent cases left — every candidate prompt was in training")

    known = set(AR.capability_index())
    hits1 = hits3 = disp = 0
    wrong, t0 = [], time.time()
    for ex in cases:
        gold = ex["capability"]
        try:
            out = AR.capability_loop(ex["input"], model=model, max_steps=2, execute=False,
                                     adapter=adapter, bits=4)
        except Exception as e:  # noqa: BLE001 — a serving failure is a RESULT for this bench
            wrong.append({"gold": gold, "got": None, "why": f"{type(e).__name__}: {str(e)[:70]}"})
            continue
        searched = [s for s in out["steps"] if s.get("tool") == "search_capabilities"]
        shortlist = searched[0]["matches"][:3] if searched else []
        planned = next((s for s in out["steps"] if s.get("tool") == "execute_capability"), None)
        pick = (planned or {}).get("name") or (shortlist[0] if shortlist else None)
        if pick in known:
            disp += 1
        if pick == gold:
            hits1 += 1
        if gold in shortlist or pick == gold:
            hits3 += 1
        else:
            wrong.append({"gold": gold, "got": pick, "shortlist": shortlist})
    n_ = len(cases)
    return {"n": n_, "top1": hits1 / n_, "top3": hits3 / n_, "dispatchable": disp / n_,
            "sec_per_case": round((time.time() - t0) / max(n_, 1), 2), "wrong": wrong[:12],
            "model": model, "adapter": adapter}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="base model path/id (default: cached Gemma-4 E2B)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--compare-base", action="store_true", help="also score the base model, no adapter")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--comp", default="biohub-cell-tracking-during-development")
    ap.add_argument("--force", action="store_true", help="run even if the GPU looks busy")
    ap.add_argument("--include-seen", action="store_true",
                    help="do NOT exclude prompts the adapter trained on (measures recall, not decisions)")
    ap.add_argument("--train-n", type=int, default=2400, help="reconstruct the adapter's training corpus")
    ap.add_argument("--train-comp", default="birdclef-2026")
    ap.add_argument("--direct-per-agent", type=int, default=16)
    a = ap.parse_args()

    ok, why = gpu_free()
    if not ok and not a.force:
        print(f"GPU is busy ({why}) — a contended run silently offloads to CPU and every number becomes "
              f"meaningless. Re-run when free, or pass --force.")
        return 2

    from fleet_agents import llm_tool_train as T
    base = a.model or f"local-hf/{T._resolve_model(None)}"
    runs = [("fine-tuned", a.adapter)] if a.adapter else []
    if a.compare_base or not a.adapter:
        runs.append(("base", None))

    seen = None
    if not a.include_seen:
        from fleet_agents import prompt_dataset as _PD
        tr, _v, _r = T.build_corpus({"n": a.train_n, "comp": a.train_comp,
                                     "direct_per_agent": a.direct_per_agent,
                                     "hist_len": 3, "format": "compact"})
        seen = {e["input"] for e in tr}
        print(f"excluding {len(seen)} prompts the adapter was trained on "
              f"(pass --include-seen to measure recall instead)")

    results = {}
    for label, ad in runs:
        r = bench(base, ad, a.n, a.comp, seen_prompts=seen)
        results[label] = r
        print(f"\n=== {label} ({ad or 'no adapter'}) — {r['n']} UNSEEN real transitions, "
              f"{r['sec_per_case']}s/case")
        print(f"  top1         {r['top1']:.1%}   (chose exactly what the fleet chose)")
        print(f"  top3         {r['top3']:.1%}   (the fleet's choice was in the shortlist)")
        print(f"  dispatchable {r['dispatchable']:.1%}   (named something real)")
        for w in r["wrong"][:5]:
            print(f"    miss: gold={w['gold']} got={w.get('got')} shortlist={w.get('shortlist')}")

    if len(results) == 2:
        f, b = results["fine-tuned"], results["base"]
        print(f"\n=== DELTA (fine-tuned - base) ===")
        for k in ("top1", "top3", "dispatchable"):
            print(f"  {k:13} {b[k]:.1%} -> {f[k]:.1%}  ({f[k] - b[k]:+.1%})")
        print("\nNOTE: these are REAL observed transitions, so a miss is not necessarily wrong — the fleet's"
              "\nchoice is one good answer, not the only one. top3 is the honest headline for a"
              "\nsearch-then-pick pilot; top1 is the strict read.")
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "pilot_decision_bench.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
