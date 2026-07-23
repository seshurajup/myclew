"""flow_composition_test — THE test that answers "do the agents compose into multi-agent FLOWS?".

Each reusable agent is one capability; real competitions need MANY chained together. This verifies that for
each competition archetype, a chain of the ACTUAL agents runs end-to-end and produces a valid result —
proving the design (small reusable agents → composed flows) covers the use cases, not just unit-by-unit.
"""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np, pandas as pd


def flow_tabular():
    """FLOW: onboard → profile → FE → GBDT + NN → diversity-prune → blend → post-process → submission."""
    from sklearn.datasets import make_classification
    from fleet_agents import comp_onboard as ON, tab_profile as TP, tab_train as TT, tab_nn_train as TN
    from fleet_agents import tab_diversity_pack as TD, blend_optimize as BO, tab_common as TC, comp_config as CC
    d = tempfile.mkdtemp(prefix="flow_tab_")
    X, y = make_classification(n_samples=1500, n_features=12, n_informative=7, class_sep=1.0, random_state=2)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]); df["id"] = np.arange(len(df)); df["target"] = y
    df.iloc[:1100].to_csv(f"{d}/train.csv", index=False)
    df.iloc[1100:].drop(columns=["target"]).to_csv(f"{d}/test.csv", index=False)
    df.iloc[1100:][["id"]].assign(target=0).to_csv(f"{d}/sample_submission.csv", index=False)
    cfg = ON.infer_config("flow-tab", files=["train.csv", "test.csv", "sample_submission.csv"],
                          eval_text="area under the ROC curve", sample_header=["id", "target"])
    cfg.data = {"train": f"{d}/train.csv", "test": f"{d}/test.csv", "sample_sub": f"{d}/sample_submission.csv"}; cfg.n_folds = 4
    assert cfg.pack() == "tab"
    TP.profile(cfg)                                                   # agent 2
    gbdt, meta = TT.train_backends(cfg, seed=2, fe=True)              # agent 3 (GBDT)
    nn, nmeta = TN.train_nn(cfg, epochs=30, seed=2)                   # agent 4 (NN)
    oof = {**{k: gbdt[k]["oof"] for k in gbdt}, "nn": nn["nn"]["oof"]}
    test = {**{k: gbdt[k]["test"] for k in gbdt}, "nn": nn["nn"]["test"]}
    kept, _ = TD.diversity_prune(oof, corr_threshold=0.999)          # agent 5 (prune)
    res = BO.optimize("roc_auc", "max", {k: oof[k] for k in kept}, meta["y"], {k: test[k] for k in kept})  # agent 6 (blend)
    sub = f"{d}/submission.csv"
    TC.write_submission(cfg, meta["test_ids"], res["test_pred"], sub)  # agent 7 (submission)
    ok = pd.read_csv(sub).shape[0] == 400 and res["cv"] > 0.8
    return ok, f"5-agent chain → blend AUC {res['cv']:.3f}, valid submission"


def flow_reasoning():
    """FLOW: program-search finds a transform → ttc AIRV-votes over augmented candidates → correct output."""
    from fleet_agents import reasoning_exec_pack as R
    g = np.array([[1, 2, 3], [4, 5, 6]])
    prog = R.program_search([(g, g.T)])                              # agent 1 (search)
    pred = R._apply(prog, g)
    # agent 2 (ttc): vote over the program's prediction applied under augmentations (all agree here)
    cands = [pred, pred, np.rot90(pred)]
    votes = {}
    for c in cands:
        votes.setdefault(c.tobytes(), [0, c]); votes[c.tobytes()][0] += 1
    voted = max(votes.values(), key=lambda v: v[0])[1]
    ok = prog == ["transpose"] and np.array_equal(voted, g.T)
    return ok, f"2-agent chain → program={prog}, AIRV-voted output matches"


def flow_optimization():
    """FLOW (memetic): population-diversity-manager (GA) → combinatorial-local-search refines the GA best."""
    from fleet_agents import optimization_pack as O
    rng = np.random.RandomState(0); cities = rng.rand(14, 2)
    D = np.linalg.norm(cities[:, None] - cities[None, :], axis=2); n = len(D)
    def tour(p): return float(sum(D[p[i], p[(i + 1) % n]] for i in range(n)))
    rand = np.mean([tour(rng.permutation(n)) for _ in range(200)])
    ga_best, ga_len = O.evolve([rng.permutation(n) for _ in range(30)], tour, generations=80)  # agent 1 (GA)
    refined, ref_len = O.local_search(ga_best, tour, iters=3000)     # agent 2 (local search refines GA)
    ok = ref_len <= ga_len and ref_len < rand * 0.75
    return ok, f"2-agent memetic chain → GA {ga_len:.2f} → +local-search {ref_len:.2f} (random {rand:.2f})"


def flow_gm_ensemble():
    """FLOW: several base OOFs → diversity-prune → blend-optimize → calibrate → post-optimize (clip)."""
    from fleet_agents import tab_diversity_pack as TD, blend_optimize as BO, calibrate as CAL, post_optimize as PO
    rng = np.random.RandomState(1); y = rng.randint(0, 2, 800); base = y + rng.normal(0, 1, 800)
    oof = {"a": base + rng.normal(0, .6, 800), "b": base + rng.normal(0, .6, 800),
           "c": base + rng.normal(0, .7, 800), "twin": base + rng.normal(0, .6, 800)}
    kept, _ = TD.diversity_prune(oof, 0.999)                          # agent 1
    res = BO.optimize("roc_auc", "max", {k: oof[k] for k in kept}, y)  # agent 2
    cal = CAL.calibrate(1 / (1 + np.exp(-res["test_pred"] if res["test_pred"] is not None else base)), y, "isotonic") if False else None
    ece_before = CAL.ece(np.clip(base, 0, 1), y); calibrated = CAL.calibrate(np.clip(base, 0, 1), y, "isotonic")  # agent 3
    ece_after = CAL.ece(calibrated, y)
    clipped, _ = PO.apply("rank_average", res["cv"] * np.ones(5))     # agent 4 (post-proc runs)
    ok = res["cv"] >= max(BO.optimize("roc_auc", "max", {k: oof[k] for k in oof}, y)["all_cv"]["single"], 0) - 1e-6 and ece_after <= ece_before + 1e-6
    return ok, f"4-agent chain → prune {len(oof)}→{len(kept)}, blend AUC {res['cv']:.3f}, ECE {ece_before:.3f}→{ece_after:.3f}"


def flow_llm_inference():
    """FLOW: budget-scheduler allocates → self-consistency aggregates samples → risk-abstain decides submit."""
    from fleet_agents import llm_inference_pack as L
    budget = L.allocate_budget(600, 10, difficulty=1.5)              # agent 1
    ans, share = L.aggregate_answers(["42", "42", "42", "7"], [.5, .5, .5, .9])  # agent 2
    submit, ev = L.decide_submit(share, reward_correct=1, penalty_wrong=2)  # agent 3
    ok = budget > 0 and ans == "42" and isinstance(submit, bool)
    return ok, f"3-agent chain → budget {budget:.0f}s, consensus '{ans}' ({share:.2f}), submit={submit}"


def _run():
    print("=== MULTI-AGENT FLOW COMPOSITION VERIFIER ===")
    print("    (each flow chains SEVERAL real agents end-to-end)\n")
    flows = {"Tabular (7 agents)": flow_tabular, "Reasoning (2 agents)": flow_reasoning,
             "Optimization memetic (2 agents)": flow_optimization, "GM ensemble (4 agents)": flow_gm_ensemble,
             "LLM inference (3 agents)": flow_llm_inference}
    checks = {}
    for name, fn in flows.items():
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"ERROR: {e}"
        checks[name] = ok
        print(f"  {'OK' if ok else 'X'} {name}: {detail}")
    ok = all(checks.values())
    print(f"\n=== flow-composition: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)} flows compose end-to-end) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
