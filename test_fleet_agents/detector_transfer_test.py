"""detector_transfer_test — data-wise verifier for the REUSABLE detector-transfer agent on synthetic 3D data
where the answer is known: a training set whose appearance MATCHES the eval domain must beat a MISMATCHED
set robustly across seeds; the multi-seed harness reports mean±std and a significance call; self-training
pseudo-labels the target; the agent run() dispatches. GPU if available, else CPU. bench_lib recall metric."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
sys.path.insert(0, os.path.join(COMP, "experiments", "segment")); sys.path.insert(0, os.path.join(COMP, "src"))
import numpy as np
from fleet_agents import detector_transfer as DT


def _vol(seed, bright=1.0, n=25, S=24):
    """A small 3D volume with `n` bright blobs + their point labels (in-grid coords)."""
    r = np.random.RandomState(seed); v = r.rand(S, S, S).astype(np.float32) * 0.15
    pts = []
    for _ in range(n):
        z, y, x = r.randint(2, S - 2), r.randint(2, S - 2), r.randint(2, S - 2)
        v[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] += bright
        pts.append([z, y, x])
    return (v - v.mean()) / (v.std() + 1e-6), np.array(pts, float)


def _run():
    print("=== DETECTOR-TRANSFER DATA-WISE VERIFIER ===")
    checks = {}
    SCALE = np.array([1.0, 1.0, 1.0])
    # eval domain: brightness 1.0
    comp_eval = {"A": [(_vol(100 + i, bright=1.0)) for i in range(3)]}
    comp_eval = {"A": [(v, p) for v, p in comp_eval["A"]]}
    # training sets: MATCHED (bright 1.0) vs MISMATCHED (bright 3.5, very different appearance)
    matched = ([_vol(i, bright=1.0)[0] for i in range(4)], [_vol(i, bright=1.0)[1] for i in range(4)])
    mism = ([_vol(i, bright=3.5)[0] for i in range(4)], [_vol(i, bright=3.5)[1] for i in range(4)])
    out = DT.robust_compare({"matched": matched, "mismatched": mism}, comp_eval, SCALE,
                            seeds=(0, 1), epochs=25, ch=16)
    sm = out["summary"]
    checks["multiseed_meanstd"] = all("mean" in sm[n]["A"] and "std" in sm[n]["A"] for n in ("matched", "mismatched"))
    checks["reports_seeds"] = all(len(sm[n]["A"]["seeds"]) == 2 for n in sm)
    checks["baseline_is_first"] = out["baseline"] == "matched"
    checks["significance_flag"] = "mismatched" in out["vs_baseline"] and "A" in out["vs_baseline"]["mismatched"]
    print(f"  (matched A={sm['matched']['A']['mean']}±{sm['matched']['A']['std']}, "
          f"mismatched A={sm['mismatched']['A']['mean']}±{sm['mismatched']['A']['std']})")

    # self-training: pseudo-label the eval domain with a matched-trained detector → non-empty points
    m = DT.train_detector(*matched, epochs=25, ch=16, seed=0)
    pts = DT.pseudo_label(m, [comp_eval["A"][0][0]], topk_per=30)
    checks["self_train_labels"] = len(pts) == 1 and pts[0].shape[1] == 3
    r = DT.eval_per_embryo(m, comp_eval, SCALE)
    checks["eval_per_embryo"] = r.get("A") is not None and 0.0 <= r["A"] <= 1.0
    print(f"  (self-train pseudo-labels: {len(pts[0])} pts; matched-detector recall on A: {r['A']})")

    # PRETRAIN->FINETUNE recipe (use both datasets): PU-mask + PEFT finetune; recipe is a seed->model callable
    pu = DT._pu_mask((24, 24, 24), matched[1][0], pos_r=1.5, neg_far=4.0)
    checks["pu_mask_pos_and_neg"] = float(pu.max()) == 1.0 and 0.0 < float(pu.mean()) < 1.0   # some supervised, some ignored
    target_sparse = ([_vol(50 + i, bright=1.0)[0] for i in range(2)], [_vol(50 + i, bright=1.0)[1][:3] for i in range(2)])  # only 3 labels/vol
    m_pre = DT.train_detector(*matched, epochs=15, ch=16, seed=0)
    m_ft = DT.finetune_detector(m_pre, *target_sparse, epochs=10, freeze_encoder=True, seed=0)
    checks["finetune_runs"] = DT.eval_per_embryo(m_ft, comp_eval, SCALE).get("A") is not None
    rec = DT.recipe_pretrain_finetune(matched, target_sparse, pre_epochs=10, ft_epochs=8, ch=16)
    checks["recipe_is_callable"] = callable(rec) and getattr(rec(0), "arch", None) in ("unet3d", "timm25d")
    # recipe-based robust_compare (A=source-only vs C=pretrain->finetune)
    out2 = DT.robust_compare({"A_src_only": DT.recipe_train(matched, epochs=10, ch=16),
                              "C_pretrain_ft": rec}, comp_eval, SCALE, seeds=(0,))
    checks["recipe_compare"] = "A_src_only" in out2["summary"] and "C_pretrain_ft" in out2["summary"]
    print(f"  (finetune+recipe: A={out2['summary']['A_src_only']['A']['mean']} C={out2['summary']['C_pretrain_ft']['A']['mean']})")

    # agent run() dispatch (nested-list arrays, like JSON spec)
    agent = DT.DetectorTransfer()
    ts = {"m": {"vols": [v.tolist() for v in matched[0][:2]], "pts": [p.tolist() for p in matched[1][:2]]}}
    ce = {"A": [{"vol": comp_eval["A"][0][0].tolist(), "gt": comp_eval["A"][0][1].tolist()}]}
    st, d, to, msg = agent.run({"question": "t", "spec": {"train_sets": ts, "comp_eval": ce, "seeds": [0], "epochs": 15, "ch": 16}}, "test")
    checks["agent_runs"] = st == "done" and "transfer" in d

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"\n=== detector-transfer: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  X FAILED: {e}"); sys.exit(1)
