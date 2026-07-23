"""reasoning_code_pack — the code-golf / engine-tuning / Kaggle-meta levers from the one-by-one pass
(google-code-golf, fide-chess, nemotron, llms-you-cant-please). All pure stdlib/numpy, offline-verified:

  • expression-search        — brute-force the shortest arithmetic expression reproducing an int→int table
                              (pysearch-style; code-golf, ARC tiny mappings).
  • code-compress-optimizer  — deflate/zlib byte minimizer for self-decompressing code (+ a self-extract stub);
                              reports the byte win (code-golf compression track).
  • sprt-spsa-tuner          — SPRT sequential accept/reject of a change from win/loss matches + gradient-free
                              SPSA parameter step (chess engine tuning over self-play).
  • lb-formula-prober        — reverse-engineer a hidden LINEAR scoring formula from probe (features, score)
                              pairs by least squares (llms-you-cant-please LB reverse-engineering).
  • trainable-trace-auditor  — audit a reasoning/CoT trace for LEARNABILITY: reference-integrity, operand
                              locality, hidden-compute — the nemotron lever that made low-sample SFT converge.
"""
from __future__ import annotations
import itertools
import numpy as np
import zlib
from .base import BaseAgent


# ---------------------------------------------------------------- expression-search
def expression_search(inputs, outputs, max_const=12, two_op=True):
    """Return the shortest expression string f(x) matching every (x->y), or None. Searches a small op grammar.
    two_op: also search two-operation forms a*x±b (O(max_const^2)); set False to cap the search cost."""
    x = np.asarray(inputs, int); y = np.asarray(outputs, int)
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return None
    mc = max(1, int(max_const))
    cands = [("x", x)]
    for c in range(1, mc + 1):
        cands += [(f"x+{c}", x + c), (f"x-{c}", x - c), (f"x*{c}", x * c),
                  (f"x%{c}", x % c), (f"x//{c}", x // c)]
    if two_op:                                          # two-op: a*x+b, a*x-b
        for a in range(1, mc + 1):
            for b in range(1, mc + 1):
                cands.append((f"x*{a}+{b}", x * a + b)); cands.append((f"x*{a}-{b}", x * a - b))
    matches = [(expr, val) for expr, val in cands if np.array_equal(val, y)]
    if not matches:
        return None
    return min(matches, key=lambda t: len(t[0]))[0]


# ---------------------------------------------------------------- code-compress-optimizer
def compress_code(code, level=9):
    """zlib-deflate a code string; return (orig_bytes, compressed_bytes, self_extracting_python).
    level: zlib compression level 0-9 (default 9 = smallest)."""
    raw = (code or "").encode() if isinstance(code, str) else bytes(code or b"")
    comp = zlib.compress(raw, int(np.clip(level, 0, 9)))
    stub = f"import zlib;exec(zlib.decompress({comp!r}))"
    return len(raw), len(comp), stub


# ---------------------------------------------------------------- sprt-spsa-tuner
def sprt(wins, losses, draws=0, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05):
    """Sequential Probability Ratio Test for 'is the new version elo1 better vs elo0?'. Returns
    'accept' (H1) / 'reject' (H0) / 'continue' from the log-likelihood ratio vs Wald bounds."""
    def p_win(elo):  # logistic Elo→score, clipped away from 0/1 to keep logs finite
        return float(np.clip(1.0 / (1 + 10 ** (-elo / 400)), 1e-6, 1 - 1e-6))
    s = wins + 0.5 * draws; n = wins + losses + draws
    if n == 0:
        return "continue", 0.0
    p0, p1 = p_win(elo0), p_win(elo1)
    # binomial LLR on the score proxy (wins vs losses, draws split)
    llr = wins * np.log(p1 / p0) + losses * np.log((1 - p1) / (1 - p0))
    lower = np.log(beta / (1 - alpha)); upper = np.log((1 - beta) / alpha)
    return ("accept" if llr >= upper else "reject" if llr <= lower else "continue"), float(llr)


def spsa_step(params, eval_fn, a=0.2, c=0.1, seed=0):
    """One gradient-free SPSA update maximizing eval_fn(params). Returns the stepped params."""
    rng = np.random.RandomState(seed); p = np.asarray(params, float)
    delta = rng.choice([-1, 1], size=p.shape)
    fp = eval_fn(p + c * delta); fm = eval_fn(p - c * delta)
    ghat = (fp - fm) / (2 * c) * delta
    return p + a * ghat


# ---------------------------------------------------------------- lb-formula-prober
def recover_formula(probe_features, probe_scores):
    """Recover a hidden LINEAR scoring formula score = X·w (+b) from probe submissions by least squares.
    Returns (weights, intercept). Guards empty probes → (zeros, 0.0)."""
    X = np.asarray(probe_features, float); s = np.asarray(probe_scores, float)
    if X.ndim != 2 or len(X) == 0 or s.size == 0:
        d = X.shape[1] if X.ndim == 2 else 0
        return np.zeros(d), 0.0
    Xa = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(Xa, s, rcond=None)
    return coef[:-1], float(coef[-1])


# ---------------------------------------------------------------- trainable-trace-auditor
def audit_trace(steps, max_distance=8):
    """steps = list of {'refs': [earlier step indices used], 'hidden': bool}. Flags learnability issues:
    reference-integrity (refs must point backward), locality (max operand distance), hidden compute.
    max_distance: operand-locality threshold beyond which a step is flagged as hard to learn."""
    flags = []
    max_dist = 0
    for i, st in enumerate(steps or []):
        st = st or {}
        for r in st.get("refs", []) or []:
            if r >= i or r < 0:
                flags.append(f"step{i}: forward/invalid reference {r}")
            else:
                max_dist = max(max_dist, i - r)
        if st.get("hidden"):
            flags.append(f"step{i}: hidden computation (not derivable from refs) — model can't learn it")
    if max_dist > int(max_distance):
        flags.append(f"long operand distance {max_dist} (>{int(max_distance)}) — hard to learn; move operands closer")
    return {"flags": flags, "max_operand_distance": max_dist, "learnable": len(flags) == 0}


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class ExpressionSearch(_B):
    name = "expression-search"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("inputs", "outputs") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"expression-search needs spec keys {missing} — none provided")
        expr = expression_search(s["inputs"], s["outputs"], int(s.get("max_const", 12)),
                                                   two_op=bool(s.get("two_op", True)))
        msg = f"expression-search: {'found f(x)=' + expr if expr else 'no short expression found'}"
        self.log(msg, kind="finding", recommendation="use the shortest exact mapping (code-golf / tiny formula)")
        return self.done({"expression": expr}, msg)


class CodeCompress(_B):
    name = "code-compress-optimizer"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("code",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"code-compress-optimizer needs spec keys {missing} — none provided")
        o, c, stub = compress_code(s["code"], level=int(s.get("level", 9)))
        msg = f"code-compress-optimizer: {o} → {c} bytes ({100*(1-c/max(o,1)):.0f}% smaller) via deflate self-extract"
        self.log(msg, kind="finding", recommendation="use the self-extracting stub when under a byte limit")
        return self.done({"orig_bytes": o, "compressed_bytes": c, "stub": stub}, msg)


class SprtSpsaTuner(_B):
    name = "sprt-spsa-tuner"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("wins", "losses") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"sprt-spsa-tuner needs spec keys {missing} — none provided")
        dec, llr = sprt(int(s["wins"]), int(s["losses"]), int(s.get("draws", 0)),
                        float(s.get("elo0", 0.0)), float(s.get("elo1", 5.0)))
        msg = f"sprt-spsa-tuner: SPRT verdict={dec} (LLR={llr:.2f}) — {s['wins']}W/{s['losses']}L"
        self.log(msg, kind="finding", recommendation="accept→ship the change; SPSA-tune constants over self-play")
        return self.done({"decision": dec, "llr": llr}, msg)


class LbFormulaProber(_B):
    name = "lb-formula-prober"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("probe_features", "probe_scores") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"lb-formula-prober needs spec keys {missing} — none provided")
        w, b = recover_formula(s["probe_features"], s["probe_scores"])
        msg = f"lb-formula-prober: recovered linear scoring weights (+intercept {b:.4f})"
        self.log(msg, kind="finding", recommendation="exploit the recovered metric / index split (Kaggle-meta)")
        return self.done({"weights": [float(x) for x in w], "intercept": b}, msg)


class TraceAuditor(_B):
    name = "trainable-trace-auditor"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("steps",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"trainable-trace-auditor needs spec keys {missing} — none provided")
        res = audit_trace(s["steps"], max_distance=int(s.get("max_distance", 8)))
        msg = f"trainable-trace-auditor: {'LEARNABLE' if res['learnable'] else str(len(res['flags'])) + ' issue(s)'} (max operand dist {res['max_operand_distance']})"
        self.log(msg, kind="finding", recommendation="fix flagged steps before SFT — trace learnability drives low-sample convergence")
        return self.done(res, msg)


_EX = ExpressionSearch(); _CC = CodeCompress(); _ST = SprtSpsaTuner(); _LB = LbFormulaProber(); _TA = TraceAuditor()


def run_expr(q, worker): return _EX.run(q, worker)
def run_compress(q, worker): return _CC.run(q, worker)
def run_sprt(q, worker): return _ST.run(q, worker)
def run_lbformula(q, worker): return _LB.run(q, worker)
def run_traceaudit(q, worker): return _TA.run(q, worker)
