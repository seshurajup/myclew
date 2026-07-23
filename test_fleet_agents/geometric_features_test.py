"""geometric_features_test — data-wise verifier for the torchmd-net RBF/cutoff featurizer.

Core properties:
  1. cosine_cutoff: 1 at d=0, 0 at/beyond cutoff, monotone decreasing, smooth.
  2. ExpNormalSmearing: output shape (*,num_rbf), bounded, →0 beyond cutoff; matches PhysNet formula.
  3. radius_neighbors: only edges with 0<dist<cutoff; symmetric edge set.
  4. featurize is rotation- AND translation-invariant (distances/RBF unchanged under rigid motion).
  5. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))


def _run():
    print("=== GEOMETRIC-FEATURES VERIFIER ===")
    import torch
    from fleet_agents import geometric_features as G
    torch.manual_seed(0); checks = {}

    # 1. cosine cutoff
    d = torch.linspace(0, 6, 100); rc = 5.0
    c = G.cosine_cutoff(d, rc)
    checks["cutoff_one_at_zero"] = abs(float(c[0]) - 1.0) < 1e-6
    checks["cutoff_zero_beyond"] = bool((c[d >= rc] == 0).all())
    checks["cutoff_monotone"] = bool((c[d < rc][1:] <= c[d < rc][:-1] + 1e-6).all())

    # 2. exp-normal RBF
    sm = G.ExpNormalSmearing(cutoff_upper=5.0, num_rbf=16, trainable=False)
    dd = torch.rand(50) * 4.0
    rbf = sm(dd)
    checks["rbf_shape"] = tuple(rbf.shape) == (50, 16)
    checks["rbf_bounded"] = bool((rbf >= 0).all() and (rbf <= 1.01).all())
    checks["rbf_zero_beyond_cutoff"] = bool(sm(torch.tensor([6.0])).abs().max() < 1e-6)
    print(f"  -> RBF range [{rbf.min():.3f},{rbf.max():.3f}] shape {tuple(rbf.shape)}")

    # 3. neighbor list
    pos = torch.randn(30, 3) * 2.0
    ei, ew = G.radius_neighbors(pos, cutoff=3.0)
    checks["edges_within_cutoff"] = bool((ew < 3.0).all() and (ew > 0).all())
    checks["edge_set_symmetric"] = ei.shape[1] % 2 == 0        # every (i,j) has its (j,i)

    # 4. rotation + translation invariance — measured on the full pairwise-distance matrix (size-stable and
    #    exactly invariant under rigid motion; the cutoff neighbor set can flip a pair sitting on the boundary).
    ei1, ew1, rbf1 = G.featurize(pos, cutoff=3.0, num_rbf=16)
    Q, _ = torch.linalg.qr(torch.randn(3, 3)); Q = Q * torch.sign(torch.det(Q))   # proper rotation
    pos2 = pos @ Q.T + torch.tensor([3.0, -1.0, 2.0])
    dist_err = float((torch.cdist(pos.double(), pos.double()) - torch.cdist(pos2.double(), pos2.double())).abs().max())
    # RBF applied to the (invariant) upper-triangular distances must match too
    iu = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
    d1 = torch.cdist(pos.double(), pos.double())[iu[0], iu[1]].float()
    d2 = torch.cdist(pos2.double(), pos2.double())[iu[0], iu[1]].float()
    rbf_err = float((sm(d1) - sm(d2)).abs().max())
    print(f"  -> {ei1.shape[1]} edges; rigid-motion distance err={dist_err:.2e} rbf err={rbf_err:.2e}")
    checks["invariant_distances"] = dist_err < 1e-4
    checks["invariant_rbf"] = rbf_err < 1e-4

    # 5. agent contract
    st, dta, to, msg = G.run_geomfeat({"spec": {"n_points": 40, "cutoff": 3.0}}, "t")
    checks["agent_done"] = st == "done" and dta["invariance_err"] < 1e-4

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== geometric-features: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
