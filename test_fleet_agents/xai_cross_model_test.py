"""Data-wise verifier for the CROSS-MODEL layer of the xai agent.

Why this layer exists: every other method in `xai.py` takes ONE model and answers "why did it predict
that?". None could express "where do two views DISAGREE, and is that region worth a second look?" — so
disagreement-gated fusion was structurally unaskable, and the one hand-comparison we did collapsed a
per-dataset table into the verdict "union fails" while that table had union at 1.000 vs .909/.826 on one
dataset. This test pins the three things that make the layer trustworthy:

  1. it finds a lever when one really exists (complementary views),
  2. it says NO LEVER when the views are identical — a tool that always finds something is useless,
  3. it distinguishes ADDING correct contested items from DROPPING wrong ones. Those are opposite
     mechanisms; reporting "gating could add X" when 0% of contested are correct would send someone to
     build precisely the wrong thing (a real bug caught here).

Offline, deterministic, no torch and no network.
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "researchpapers"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fleet_agents import xai  # noqa: E402


def _run():
    checks = {}
    rng = random.Random(1)
    truth = set(range(1000))
    core = set(rng.sample(sorted(truth), 800))
    fp1, fp2 = set(range(2000, 2150)), set(range(3000, 3150))

    # --- 1. identical views → no disagreement → must report NO LEVER
    same = xai.selective_oracle({"a": core, "b": set(core)}, truth)
    checks["identical views report NO LEVER"] = "NO LEVER" in same["_summary"]["verdict"]
    checks["identical views have zero contested items"] = same["_summary"]["contested_items"] == 0

    # --- 2. complementary views (each finds real items the other misses) → must report a LEVER
    c1 = set(rng.sample(sorted(truth), 700))
    c2 = set(rng.sample(sorted(truth), 700))
    comp = xai.selective_oracle({"v14": c1 | fp1, "neural": c2 | fp2}, truth)
    checks["complementary views report a LEVER"] = comp["_summary"]["verdict"].startswith("LEVER:")
    checks["and the gain comes from ADDING correct items"] = (
        comp["_summary"]["gain_from_adding_correct_contested"] > 0.001)
    checks["oracle beats union when views are complementary"] = (
        comp["_summary"]["oracle_headroom_over_union"] > 0)

    # --- 3. contested are ALL false positives → the lever is EXCLUSION, not fusion.
    # The headroom is positive either way, so a verdict based on headroom alone is wrong here.
    allfp = xai.selective_oracle({"v14": core | fp1, "neural": core | fp2}, truth)
    checks["all-FP contested is called EXCLUSION, not fusion"] = (
        "LEVER IS EXCLUSION" in allfp["_summary"]["verdict"])
    checks["and it reports zero gain from adding"] = (
        allfp["_summary"]["gain_from_adding_correct_contested"] <= 0.001)
    checks["while still showing positive headroom (the trap)"] = (
        allfp["_summary"]["oracle_headroom_over_best_single"] > 0)

    # --- 4. partition arithmetic must be exact
    part = xai.disagreement_partition({"a": {1, 2, 3}, "b": {2, 3, 4}})
    checks["partition: core is the intersection"] = part["core"] == {2, 3}
    checks["partition: contested is union minus core"] = part["contested"] == {1, 4}
    checks["partition: counts agree"] = (part["n_union"] == 4 and part["n_core"] == 2
                                         and part["n_contested"] == 2)

    # --- 5. CONDITIONAL breakdown must expose a flipping winner (the failure that hid this lever)
    per_pred, per_truth = {}, {}
    for i, (a_r, b_r) in enumerate([(0.9, 0.6), (0.6, 0.9)]):      # winner deliberately flips
        t = set(range(500))
        per_truth[f"ds{i}"] = t
        per_pred[f"ds{i}"] = {"a": set(rng.sample(sorted(t), int(a_r * 500))),
                              "b": set(rng.sample(sorted(t), int(b_r * 500)))}
    cb = xai.conditional_breakdown(per_pred, per_truth)
    checks["conditional breakdown detects a flipping best-view"] = cb["distinct_winners"] == 2
    checks["and says an aggregate verdict would hide it"] = "FLIPS" in cb["note"]

    # --- 6. it must be reachable AS AN AGENT (spec-driven), not just as a library function
    st, data, to, msg = xai.report({"spec": {"mode": "cross_model",
                                             "preds": {"a": sorted(c1), "b": sorted(c2)},
                                             "truth": sorted(truth)}}, "test")
    checks["agent mode `cross_model` returns done"] = st == "done"
    checks["agent mode returns a verdict in the message"] = "cross-model XAI" in msg
    st2, data2, _, msg2 = xai.report({"spec": {"mode": "cross_model",
                                               "per_unit_preds": {k: {n: sorted(v) for n, v in p.items()}
                                                                  for k, p in per_pred.items()},
                                               "per_unit_truth": {k: sorted(v) for k, v in per_truth.items()}}},
                                     "test")
    checks["agent mode handles per-unit input"] = st2 == "done" and data2["units"] == 2

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  {sum(1 for v in checks.values() if v)}/{len(checks)} passed")
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
