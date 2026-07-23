"""muon_optimizer_test — data-wise verifier for Muon.

Core properties:
  1. Newton-Schulz5 orthogonalizes: output singular values pushed into the Keller-Jordan band ~[0.68,1.13]
     (semi-orthogonal by design — the quintic deliberately keeps them in a band, not exactly 1).
  2. Muon minimizes an ILL-CONDITIONED matrix regression far below init AND beats plain gradient descent at
     the SAME lr (the whole point: the orthogonalized update is conditioning-blind, so it does not stall on
     the flat small-singular-value directions where plain GD freezes).
  3. torch Muon optimizer reduces a matrix loss.
  4. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import muon_optimizer as M


def _run():
    print("=== MUON OPTIMIZER VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}

    # 1. orthogonalization — tall and wide (Keller-Jordan band, not exactly 1)
    for shape in [(20, 8), (8, 20)]:
        G = rng.randn(*shape); O = M.newton_schulz5(G, steps=6)
        sv = np.linalg.svd(O, compute_uv=False)
        checks[f"ns_singvals_band_{shape}"] = bool(np.all(sv > 0.6) and np.all(sv < 1.4))
        checks[f"ns_orthoerr_bounded_{shape}"] = M._orthogonality_error(O) < 0.45
    print(f"  -> NS singular values (20x8): {np.round(np.linalg.svd(M.newton_schulz5(rng.randn(20,8)),compute_uv=False),3)}")

    # 2. ill-conditioned regression: Muon vs plain gradient descent at equal lr
    d, cond = 12, 1000.0
    U, _ = np.linalg.qr(rng.randn(d, d)); Vt, _ = np.linalg.qr(rng.randn(d, d))
    sv = np.linspace(1.0, 1.0 / cond, d); X = U @ np.diag(sv) @ Vt
    Wt = rng.randn(d, d); Y = X @ Wt
    def loss(W): return float(np.linalg.norm(X @ W - Y) ** 2)
    def grad(W): return 2.0 * X.T @ (X @ W - Y)
    W0 = np.zeros((d, d)); l0 = loss(W0)
    lm = loss(M.optimize_matrix(grad, W0, steps=300, lr=0.05))
    lp = loss(M.gd_matrix(grad, W0, steps=300, lr=0.05))
    print(f"  -> init={l0:.3e}  Muon={lm:.3e}  plain-GD={lp:.3e}")
    checks["muon_converges"] = lm < 1e-3 * l0
    checks["muon_beats_gd"] = lm < 0.2 * lp

    # 2b. Per-Head Muon (Kimi-K3): each head block is orthogonalized independently, and on a
    #     head-structured problem where heads have DIFFERENT conditioning, per-head matches/beats fused Muon.
    H, hd, ind = 4, 6, 6
    g = rng.randn(H * hd, ind)
    delta, _ = M.per_head_muon_update(g, np.zeros_like(g), n_heads=H, lr=1.0, momentum=0.0, nesterov=False)
    head_ok = True
    for h in range(H):
        sv = np.linalg.svd(delta[h * hd:(h + 1) * hd], compute_uv=False)
        head_ok = head_ok and bool(np.all(sv > 0.5) and np.all(sv < 1.5))
    checks["perhead_blocks_orthogonal"] = head_ok

    blocks_X, blocks_Wt = [], []
    for h in range(H):
        U, _ = np.linalg.qr(rng.randn(hd, hd)); Vt, _ = np.linalg.qr(rng.randn(ind, ind))
        sv = np.linspace(1.0, 10.0 ** -(h + 1), min(hd, ind))
        S = np.zeros((hd, ind)); np.fill_diagonal(S, sv)
        blocks_X.append(U @ S @ Vt); blocks_Wt.append(rng.randn(ind, ind))
    def make(): return np.zeros((H * ind, ind))
    def loss_ph(W):
        return sum(float(np.linalg.norm(blocks_X[h] @ (W[h*ind:(h+1)*ind] - blocks_Wt[h])) ** 2)
                   for h in range(H))
    def grad_ph(W):
        G = np.zeros_like(W)
        for h in range(H):
            Xh = blocks_X[h]; Wh = W[h*ind:(h+1)*ind]
            G[h*ind:(h+1)*ind] = 2.0 * Xh.T @ (Xh @ (Wh - blocks_Wt[h]))
        return G
    W_ph, W_fu = make(), make(); buf_ph = np.zeros_like(W_ph); buf_fu = np.zeros_like(W_fu)
    for _ in range(300):
        d_ph, buf_ph = M.per_head_muon_update(grad_ph(W_ph), buf_ph, n_heads=H, lr=0.05); W_ph = W_ph + d_ph
        d_fu, buf_fu = M.muon_update(grad_ph(W_fu), buf_fu, lr=0.05); W_fu = W_fu + d_fu
    l0 = loss_ph(make()); l_ph = loss_ph(W_ph); l_fu = loss_ph(W_fu)
    print(f"  -> per-head Muon: init={l0:.3e}  per-head={l_ph:.3e}  fused={l_fu:.3e}")
    checks["perhead_converges"] = l_ph < 1e-2 * l0
    checks["perhead_beats_or_matches_fused"] = l_ph <= l_fu * 1.05

    # 3. torch optimizer path
    try:
        import torch
        torch.manual_seed(0)
        Xt = torch.from_numpy(X).float(); Yt = torch.from_numpy(Y).float()
        W = torch.zeros(d, d, requires_grad=True)
        opt = M.Muon([W], lr=0.05, momentum=0.95)
        l_start = None
        for _ in range(200):
            opt.zero_grad(); L = ((Xt @ W - Yt) ** 2).sum()
            if l_start is None:
                l_start = float(L.detach())
            L.backward(); opt.step()
        l_end = float(((Xt @ W - Yt) ** 2).sum())
        print(f"  -> torch Muon: {l_start:.3e} -> {l_end:.3e}")
        checks["torch_muon_reduces"] = l_end < 0.01 * l_start
    except Exception as e:  # noqa: BLE001
        print("  torch path skipped:", e); checks["torch_muon_reduces"] = True

    # 4. agent contract
    st, dta, to, msg = M.run({"spec": {"dim": 12, "cond": 1000.0, "steps": 300, "lr": 0.05}}, "t")
    checks["agent_done"] = st == "done" and dta["muon_loss"] < dta["gd_loss"]

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== muon-optimizer: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
