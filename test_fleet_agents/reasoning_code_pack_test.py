"""reasoning_code_pack_test — verifier for code-golf/tuning/meta agents (offline)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import reasoning_code_pack as R


def _run():
    print("=== REASONING/CODE PACK VERIFIER ===")
    checks = {}

    # expression-search: recover y = 2x+1
    xs = [0, 1, 2, 3, 4, 5]; ys = [2 * x + 1 for x in xs]
    expr = R.expression_search(xs, ys)
    checks["expr_recovers"] = expr == "x*2+1"
    # y = x % 3
    checks["expr_mod"] = R.expression_search([0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2]) == "x%3"

    # code-compress: repetitive code compresses; round-trips via the stub
    code = "print('hello')\n" * 50
    o, c, stub = R.compress_code(code)
    checks["compress_smaller"] = c < o
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(stub, {})
    checks["compress_roundtrip"] = buf.getvalue().count("hello") == 50

    # sprt: many wins vs few losses → accept; many losses → reject
    dec_win, llr_w = R.sprt(wins=60, losses=20, elo0=0, elo1=100)
    dec_loss, llr_l = R.sprt(wins=20, losses=60, elo0=0, elo1=100)
    checks["sprt_accept"] = dec_win == "accept"
    checks["sprt_reject"] = dec_loss == "reject"

    # spsa: maximize -(p-3)^2 → moves toward 3
    p0 = np.array([0.0]); p1 = p0
    for i in range(60):
        p1 = R.spsa_step(p1, lambda p: -float((p[0] - 3.0) ** 2), a=0.3, c=0.2, seed=i)
    checks["spsa_converges"] = abs(p1[0] - 3.0) < 1.0
    print(f"  -> spsa p: {p0[0]} → {p1[0]:.3f} (target 3.0)")

    # lb-formula-prober: recover score = 2*f0 - 1*f1 + 0.5
    rng = np.random.RandomState(0); X = rng.rand(20, 2); s = X @ np.array([2.0, -1.0]) + 0.5
    w, b = R.recover_formula(X, s)
    checks["lbformula_recovers"] = np.allclose(w, [2.0, -1.0], atol=1e-6) and abs(b - 0.5) < 1e-6

    # trace-auditor: clean trace learnable; hidden-compute + forward-ref flagged
    clean = [{"refs": []}, {"refs": [0]}, {"refs": [1]}]
    bad = [{"refs": []}, {"refs": [0], "hidden": True}, {"refs": [5]}]
    checks["trace_clean"] = R.audit_trace(clean)["learnable"] is True
    checks["trace_flags"] = R.audit_trace(bad)["learnable"] is False and len(R.audit_trace(bad)["flags"]) >= 2

    # agent contracts
    st, d, to, msg = R.run_expr({"spec": {"inputs": xs, "outputs": ys}}, "t")
    checks["expr_agent"] = st == "done" and d["expression"] == "x*2+1"
    st, d, to, msg = R.run_sprt({"spec": {"wins": 60, "losses": 20, "elo1": 100}}, "t")
    checks["sprt_agent"] = st == "done" and d["decision"] == "accept"

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== reasoning-code-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
