"""imbalance_sampler_pack_test — DATA-WISE verifier for tempered class-balanced sampling (BirdCLEF 2nd).

Build a long-tailed label set (class 0 x100, class 1 x20, class 2 x5) and assert the sampling knob does what
it claims:
  • power=0  → natural: every sample weight equal → per-class sampling mass ∝ class frequency;
  • power=-1 → fully class-balanced: total sampling mass per class is EQUAL across classes;
  • power=-0.5 (sqrt) → tempered strictly between natural and balanced;
  • a large deterministic resample under power=-1 produces ~uniform class counts (imbalance collapses);
  • effective-number mode gives the RARE class a strictly higher weight than the frequent class.
"""
import os, sys
import numpy as np
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import imbalance_sampler_pack as S


def _mass_per_class(labels, w):
    labels = np.asarray(labels)
    return {c: w[labels == c].sum() for c in np.unique(labels)}


def _run():
    print("=== IMBALANCE-SAMPLER-PACK DATA-WISE VERIFIER ===")
    checks = {}
    labels = np.array([0] * 100 + [1] * 20 + [2] * 5)

    # power=0 → all sample weights equal
    w0 = S.sample_weights(labels, power=0.0)
    checks["power0_uniform_sample_w"] = np.allclose(w0, w0[0])
    mass0 = _mass_per_class(labels, w0)
    checks["power0_mass_follows_freq"] = mass0[0] > mass0[1] > mass0[2]

    # power=-1 → equal total mass per class (fully class-balanced)
    wneg1 = S.sample_weights(labels, power=-1.0)
    m1 = _mass_per_class(labels, wneg1)
    checks["power-1_balanced_mass"] = np.allclose([m1[0], m1[1], m1[2]], m1[0], rtol=1e-9)

    # power=-0.5 strictly between: rare-class mass higher than natural but lower than fully balanced
    whalf = S.sample_weights(labels, power=-0.5)
    mh = _mass_per_class(labels, whalf)
    checks["sqrt_between"] = mass0[2] < mh[2] < m1[2]

    # deterministic resample at power=-1 → class counts ~uniform (imbalance collapses)
    idx = S.resample_indices(labels, n=6000, power=-1.0, seed=1)
    _, cnts = np.unique(labels[idx], return_counts=True)
    frac = cnts / cnts.sum()
    checks["resample_balances"] = np.all(np.abs(frac - 1 / 3) < 0.05)

    # effective-number: rare class weight > frequent class weight
    en = S.effective_number_weight({0: 100, 1: 20, 2: 5}, beta=0.999)
    checks["effnum_rare_gt_frequent"] = en[2] > en[1] > en[0]

    # agent run() contract + resample option
    st, d, to, msg = S.run({"spec": {"labels": labels.tolist(), "power": -0.5, "resample": True, "n": 50}}, "test")
    checks["run_done"] = st == "done" and "weights" in d and len(d.get("resampled_indices", [])) == 50

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    print(f"  -> natural mass={ {int(k): round(v,1) for k,v in mass0.items()} }; "
          f"balanced frac={ [round(f,3) for f in frac] }")
    ok = all(checks.values())
    print(f"=== imbalance-sampler-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
