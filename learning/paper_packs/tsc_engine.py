"""tsc_engine — executable formal-methods engine for *Transformers are Inherently Succinct*
(Bergsträßer, Cotterell, Lin — arXiv:2510.19315, https://arxiv.org/pdf/2510.19315).

Support module for learning/paper_packs/transformers_succinct.py: exact float64 unique-hard-attention
transformers (UHAT), an LTL interpreter, a B-RASP interpreter, DFA tooling, and the paper's own
constructions — Prop. 16's LTL→UHAT compiler, Prop. 13's UHAT→LTL compiler (eqs. 24–33), Lemma 9's
special-form B-RASP→UHA layer, and the App. A.2 counter witness family (Gadgets A+B). Every construction
here is verified in the lessons (learning/annotated/tsc*.learning) by exhaustive enumeration at small
sizes; run this file directly for its self-tests.
"""
import itertools, math
from fractions import Fraction
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

# ---------------- UHAT ----------------
def AFF(W, b=None):
    W = torch.as_tensor(W, dtype=torch.float64)
    b = torch.zeros(W.shape[0], dtype=torch.float64) if b is None else torch.as_tensor(b, dtype=torch.float64)
    return (W, b)

def aff(f, X):
    W, b = f
    return X @ W.T + b

class UHA:
    def __init__(self, A, B, C, mask="none", tie="max"):
        self.A, self.B, self.C, self.mask, self.tie = A, B, C, mask, tie
    def __call__(self, V):
        N = V.shape[0]
        Sc = aff(self.A, V) @ aff(self.B, V).T
        idx = torch.arange(N)
        if self.mask == "none":     Mk = torch.ones(N, N, dtype=torch.bool)
        elif self.mask == "future": Mk = idx.view(1, -1) < idx.view(-1, 1)
        else:                       Mk = idx.view(1, -1) > idx.view(-1, 1)
        neg = torch.finfo(torch.float64).min
        Sm = torch.where(Mk, Sc, torch.full_like(Sc, neg))
        top = Sm.max(1, keepdim=True).values
        Arg = Mk & (Sm == top)
        fill = -1 if self.tie == "max" else N
        pos = torch.where(Arg, idx.view(1, -1).expand(N, N), torch.full((N, N), fill, dtype=torch.long))
        sel = pos.max(1).values if self.tie == "max" else pos.min(1).values
        A_att = torch.zeros_like(V)
        live = Mk.any(1)
        if bool(live.any()):
            A_att[live] = V[sel[live]]
        return aff(self.C, torch.cat([V, A_att], dim=1))

class ReLUL:
    def __init__(self, r): self.r = r
    def __call__(self, V):
        out = V.clone()
        out[:, self.r] = torch.clamp(V[:, self.r], min=0)
        return out

class UHAT:
    def __init__(self, sigma, emb, layers, t, read="last"):
        self.sigma, self.emb, self.layers = list(sigma), emb, layers
        self.t = torch.as_tensor(t, dtype=torch.float64); self.read = read
    def forward(self, w):
        V = torch.tensor([self.emb[c] for c in w], dtype=torch.float64)
        for L in self.layers: V = L(V)
        return V
    def accepts(self, w):
        if len(w) == 0: return False
        V = self.forward(w)
        v = V[-1] if self.read == "last" else V[0]
        return bool((self.t @ v).item() > 0)
    def size(self, nonzero=True):
        cnt = (lambda M: int((M != 0).sum())) if nonzero else (lambda M: M.numel())
        n = sum(sum(1 for x in v if x != 0) if nonzero else len(v) for v in self.emb.values())
        n += cnt(self.t)
        for L in self.layers:
            if isinstance(L, UHA):
                for f in (L.A, L.B, L.C): n += cnt(f[0]) + cnt(f[1])
            else: n += 1
        return int(n)
    def nlayers(self): return len(self.layers)

# ---------------- LTL ----------------
TOP, BOT = ("top",), ("bot",)
def Q(a): return ("Q", a)
def NOT(p): return ("not", p)
def AND(*ps):
    r = ps[0]
    for p in ps[1:]: r = ("and", r, p)
    return r
def OR(*ps):
    r = ps[0]
    for p in ps[1:]: r = ("or", r, p)
    return r
def IMP(p, q): return OR(NOT(p), q)
def SINCE(p, q): return ("S", p, q)
def UNTIL(p, q): return ("U", p, q)
def P_(p): return SINCE(TOP, p)
def F_(p): return UNTIL(TOP, p)
def Y_(p): return SINCE(BOT, p)
def H_(p): return AND(p, NOT(P_(NOT(p))))

def ltl_eval(phi, w, memo=None):
    if memo is None: memo = {}
    key = (phi, w)
    if key in memo: return memo[key]
    N, k = len(w), phi[0]
    if k == "top":   r = [True] * N
    elif k == "bot": r = [False] * N
    elif k == "Q":   r = [c == phi[1] for c in w]
    elif k == "not": r = [not b for b in ltl_eval(phi[1], w, memo)]
    elif k == "and":
        a, b = ltl_eval(phi[1], w, memo), ltl_eval(phi[2], w, memo)
        r = [x and y for x, y in zip(a, b)]
    elif k == "or":
        a, b = ltl_eval(phi[1], w, memo), ltl_eval(phi[2], w, memo)
        r = [x or y for x, y in zip(a, b)]
    elif k == "S":
        f1, f2, r = ltl_eval(phi[1], w, memo), ltl_eval(phi[2], w, memo), []
        for n in range(1, N + 1):
            hit = False
            for j in range(n - 1, 0, -1):
                if f2[j - 1] and all(f1[m - 1] for m in range(j + 1, n)): hit = True; break
                if not f1[j - 1] and j < n - 0: pass
            r.append(hit)
    elif k == "U":
        f1, f2, r = ltl_eval(phi[1], w, memo), ltl_eval(phi[2], w, memo), []
        for n in range(1, N + 1):
            hit = False
            for j in range(n + 1, N + 1):
                if f2[j - 1] and all(f1[m - 1] for m in range(n + 1, j)): hit = True; break
            r.append(hit)
    else: raise ValueError(phi)
    memo[key] = r
    return r

def ltl_size(phi): return 1 + sum(ltl_size(x) for x in phi[1:] if isinstance(x, tuple))
def ltl_ops(phi, acc=None):
    if acc is None: acc = {}
    acc[phi[0]] = acc.get(phi[0], 0) + 1
    for x in phi[1:]:
        if isinstance(x, tuple): ltl_ops(x, acc)
    return acc
def ltl_accepts(phi, w): return bool(w) and ltl_eval(phi, w)[-1]

def words(sigma, lo, hi):
    for L in range(lo, hi + 1):
        for t in itertools.product(sigma, repeat=L): yield "".join(t)

# ---------------- B-RASP ----------------
MNONE = lambda i, j: True
MFUT  = lambda i, j: j < i
MPAST = lambda i, j: j > i

class BRASP:
    def __init__(self, sigma):
        self.sigma, self.ops, self.names = list(sigma), [], []
    def pos(self, name, fn):
        self.ops.append(("pos", name, fn)); self.names.append(name); return name
    def attn(self, name, mask, tie, score, value, default):
        self.ops.append(("attn", name, mask, tie, score, value, default)); self.names.append(name); return name
    def run(self, w):
        N = len(w); P = {}
        for a in self.sigma: P["Q" + a] = [c == a for c in w]
        for op in self.ops:
            kind, name = op[0], op[1]
            if kind == "pos":
                fn = op[2]; P[name] = [bool(fn(P, i, w)) for i in range(1, N + 1)]
            else:
                mask, tie, score, value, default = op[2:]
                out = []
                for i in range(1, N + 1):
                    cs = [j for j in range(1, N + 1) if mask(i, j) and score(P, i, j, w)]
                    if cs:
                        o = min(cs) if tie == "left" else max(cs)
                        out.append(bool(value(P, i, o, w)))
                    else:
                        out.append(bool(default(P, i, w)))
                P[name] = out
        return P
    def accepts(self, w, out=None, read="last"):
        if not w: return False
        P = self.run(w); nm = out or self.names[-1]
        return P[nm][-1] if read == "last" else P[nm][0]
    def size(self): return len(self.ops)

def g(P, nm, i): return P[nm][i - 1]

# ---------------- DFA ----------------
def dfa_accepts(D, w):
    q = D["start"]
    for c in w:
        if (q, c) not in D["delta"]: return False
        q = D["delta"][(q, c)]
    return q in D["finals"]

def dfa_shortest(D):
    from collections import deque
    seen, dq = {D["start"]}, deque([(D["start"], "")])
    while dq:
        q, w = dq.popleft()
        if q in D["finals"]: return w
        for a in D["sigma"]:
            p = D["delta"].get((q, a))
            if p is not None and p not in seen: seen.add(p); dq.append((p, w + a))
    return None

def dfa_reachable(D):
    from collections import deque
    seen, dq = {D["start"]}, deque([D["start"]])
    while dq:
        q = dq.popleft()
        for a in D["sigma"]:
            p = D["delta"].get((q, a))
            if p is not None and p not in seen: seen.add(p); dq.append(p)
    return seen

def dfa_minimize(D):
    R = dfa_reachable(D)
    part = {q: (q in D["finals"]) for q in R}
    while True:
        sig = {q: (part[q],) + tuple(part.get(D["delta"].get((q, a))) for a in D["sigma"]) for q in R}
        groups, new = {}, {}
        for q in R:
            groups.setdefault(sig[q], len(groups)); new[q] = groups[sig[q]]
        if new == part: break
        part = new
    finals = {part[q] for q in R if q in D["finals"]}
    delta = {}
    for q in R:
        for a in D["sigma"]:
            p = D["delta"].get((q, a))
            if p is not None: delta[(part[q], a)] = part[p]
    return dict(sigma=D["sigma"], n=len(set(part.values())), delta=delta,
                start=part[D["start"]], finals=finals)

def dfa_singleton(w, sigma):
    n = len(w); dead = n + 1
    delta = {}
    for i in range(n):
        for a in sigma: delta[(i, a)] = (i + 1) if a == w[i] else dead
    for a in sigma: delta[(n, a)] = dead; delta[(dead, a)] = dead
    return dict(sigma=list(sigma), n=n + 2, delta=delta, start=0, finals={n})

# ---------------- Prop. 16 : LTL -> UHAT ----------------
def ltl_to_uhat(phi, sigma):
    sigma = list(sigma)
    S0 = len(sigma)
    slot = {}                                    # subformula -> coordinate
    layers = []
    width = [S0]
    for k, a in enumerate(sigma): slot[Q(a)] = k

    def eye(n_in, n_out=None):
        n_out = n_out or n_in
        M = torch.zeros(n_out, n_in, dtype=torch.float64)
        for i in range(min(n_in, n_out)): M[i, i] = 1.0
        return M

    def add_coord(rowvec, bias=0.0, relu=False):
        """append a new coordinate = rowvec . v + bias, via a no-attention UHA layer."""
        R = width[0]
        W = torch.zeros(R + 1, 2 * R, dtype=torch.float64)
        W[:R, :R] = eye(R)
        W[R, :R] = torch.as_tensor(rowvec, dtype=torch.float64)
        b = torch.zeros(R + 1, dtype=torch.float64); b[R] = bias
        layers.append(UHA(AFF(eye(R)), AFF(torch.zeros(R, R, dtype=torch.float64)),
                          AFF(W, b), mask="none", tie="max"))
        width[0] = R + 1
        if relu: layers.append(ReLUL(R))
        return R

    def row(**kw):
        v = torch.zeros(width[0], dtype=torch.float64)
        for k, c in kw.items(): v[int(k[1:])] = c
        return v

    def build(f):
        if f in slot: return slot[f]
        k = f[0]
        if k == "top":
            r = torch.zeros(width[0], dtype=torch.float64)
            slot[f] = add_coord(r, 1.0); return slot[f]
        if k == "bot":
            r = torch.zeros(width[0], dtype=torch.float64)
            slot[f] = add_coord(r, 0.0); return slot[f]
        if k == "Q":
            raise ValueError(f"symbol {f[1]} not in alphabet")
        if k == "not":
            c = build(f[1]); r = torch.zeros(width[0], dtype=torch.float64); r[c] = -1.0
            slot[f] = add_coord(r, 1.0); return slot[f]
        if k == "and":
            c1, c2 = build(f[1]), build(f[2])
            r = torch.zeros(width[0], dtype=torch.float64); r[c1] = 1.0; r[c2] = 1.0
            slot[f] = add_coord(r, -1.0, relu=True); return slot[f]
        if k == "or":
            c = build(NOT(AND(NOT(f[1]), NOT(f[2])))); slot[f] = c; return c
        if k in ("S", "U"):
            c1, c2 = build(f[1]), build(f[2])
            cg = build(OR(NOT(f[1]), f[2]))          # g = ~phi1 | phi2, one coordinate
            R = width[0]
            A = torch.zeros(R, R, dtype=torch.float64)          # A(v) = e_0 * 1  (constant)
            bA = torch.zeros(R, dtype=torch.float64); bA[0] = 1.0
            B = torch.zeros(R, R, dtype=torch.float64); B[0, cg] = 1.0
            W = torch.zeros(R + 1, 2 * R, dtype=torch.float64)
            W[:R, :R] = eye(R)
            W[R, R + c2] = 1.0                                   # copy phi2 from the selected position
            b = torch.zeros(R + 1, dtype=torch.float64)
            layers.append(UHA(AFF(A, bA), AFF(B), AFF(W, b),
                              mask="future" if k == "S" else "past",
                              tie="max" if k == "S" else "min"))
            width[0] = R + 1
            slot[f] = R
            return R
        raise ValueError(f)

    top = build(phi)
    t = torch.zeros(width[0], dtype=torch.float64); t[top] = 1.0
    emb = {}
    for k, a in enumerate(sigma):
        v = [0.0] * S0; v[k] = 1.0; emb[a] = v
    return UHAT(sigma, emb, layers, t.tolist())


# ---------------- the counter witness family (Gadget A + Gadget B of App. A.2) ----------------
def counter_word(N):
    """the unique word of C_N : bin_N(0) # bin_N(1) # ... # bin_N(2^N-1) #"""
    return "".join(format(v, f"0{N}b") + "#" for v in range(2 ** N))

def counter_brasp(N, wrap_or=True):
    """B-RASP program recognizing C_N. Gadget-B bits are LSB-first: B_1 = least significant."""
    sig = "01#"
    p = BRASP(sig)
    isbit = lambda P, j, w: g(P, "Q0", j) or g(P, "Q1", j)
    # A_bit,k(i): position i-k is a bit ;  A_#,k(i): position i-k is #
    p.attn("Abit1", MFUT, "right", lambda P, i, j, w: True, lambda P, i, o, w: isbit(P, o, w), lambda P, i, w: False)
    for k in range(2, N + 1):
        pk = f"Abit{k-1}"
        p.attn(f"Abit{k}", MFUT, "right", lambda P, i, j, w: True,
               (lambda pk: lambda P, i, o, w: g(P, pk, o))(pk), lambda P, i, w: False)
    p.attn("Ahash1", MFUT, "right", lambda P, i, j, w: True, lambda P, i, o, w: g(P, "Q#", o), lambda P, i, w: True)
    for k in range(2, N + 2):
        pk = f"Ahash{k-1}"
        p.attn(f"Ahash{k}", MFUT, "right", lambda P, i, j, w: True,
               (lambda pk: lambda P, i, o, w: g(P, pk, o))(pk), lambda P, i, w: True)
    # shape: every # is preceded by N bits and by a # (or the start) N+1 back
    p.pos("Aenc", lambda P, i, w: (not g(P, "Q#", i)) or
          (all(g(P, f"Abit{k}", i) for k in range(1, N + 1)) and g(P, f"Ahash{N+1}", i)))
    p.attn("A", MFUT, "right", lambda P, i, j, w: not g(P, "Aenc", j),
           lambda P, i, o, w: False, lambda P, i, w: g(P, "Aenc", i))
    # Gadget B: B_k(i) = k-th bit leftward from i (B_1 = LSB of the block left of i)
    p.attn("B1", MFUT, "right", lambda P, i, j, w: isbit(P, j, w),
           lambda P, i, o, w: g(P, "Q1", o), lambda P, i, w: False)
    for k in range(2, N + 1):
        pk = f"B{k-1}"
        p.attn(f"B{k}", MFUT, "right", lambda P, i, j, w: isbit(P, j, w),
               (lambda pk: lambda P, i, o, w: g(P, pk, o))(pk), lambda P, i, w: False)
    def inc(P, i, o, w):
        for k in range(1, N + 1):
            if (all((not g(P, f"B{r}", i)) and g(P, f"B{r}", o) for r in range(1, k)) and
                g(P, f"B{k}", i) and not g(P, f"B{k}", o) and
                all(g(P, f"B{r}", i) == g(P, f"B{r}", o) for r in range(k + 1, N + 1))):
                return True
        return False
    p.attn("Binc", MFUT, "right", lambda P, i, j, w: g(P, "Q#", j), inc, lambda P, i, w: False)
    p.attn("Bwrap", MFUT, "right", lambda P, i, j, w: g(P, "Q#", j),
           lambda P, i, o, w: all((not g(P, f"B{k}", i)) and g(P, f"B{k}", o) for k in range(1, N + 1)),
           lambda P, i, w: all(not g(P, f"B{k}", i) for k in range(1, N + 1)))
    join = (lambda x, y: x or y) if wrap_or else (lambda x, y: x and y)
    p.attn("B", MFUT, "right",
           lambda P, i, j, w: g(P, "Q#", j) and not g(P, "Bwrap", j) and not g(P, "Binc", j),
           lambda P, i, o, w: False,
           lambda P, i, w: join(g(P, "Bwrap", i), g(P, "Binc", i)))
    # C: ends with 1^N #   (all bits of the last block are 1)
    p.pos("C", lambda P, i, w: g(P, "Q#", i) and all(g(P, f"B{k}", i) for k in range(1, N + 1)))
    p.pos("Y", lambda P, i, w: g(P, "A", i) and g(P, "B", i) and g(P, "C", i))
    return p

def counter_dfa(N):
    """DFA for C_N : the singleton language {counter_word(N)}."""
    return dfa_singleton(counter_word(N), "01#")


# ---------------- Prop. 13 : UHAT -> LTL (single UHA layer, values in a finite set F) --------
def uhat_to_ltl(T, F, tie="max"):
    """Implements eqs. (24)-(33) for a UHAT of UHA layers whose values all lie in F.
    tie='max' -> eq. (26) (rightmost, uses S); tie='min' -> eq. (28) (leftmost, uses P only)."""
    sigma = T.sigma
    D = len(next(iter(T.emb.values())))
    def vecs(R): return [tuple(float(x) for x in v) for v in itertools.product(F, repeat=R)]
    # base case, eq. (24)
    phis = {}
    for v in vecs(D):
        pre = [a for a in sigma if tuple(float(x) for x in T.emb[a]) == v]
        phis[v] = OR(*[Q(a) for a in pre]) if pre else BOT
    Rin = D
    for L in T.layers:
        assert isinstance(L, UHA) and L.mask in ("future", "past")
        Rout = L.C[0].shape[0]
        Aq = lambda u: aff(L.A, torch.tensor([u], dtype=torch.float64))[0]
        Bk = lambda b: aff(L.B, torch.tensor([b], dtype=torch.float64))[0]
        Cf = lambda u, a: tuple(float(x) for x in aff(L.C, torch.tensor(
            [list(u) + list(a)], dtype=torch.float64))[0])
        U = vecs(Rin)
        sc = {(u, b): float((Aq(u) @ Bk(b)).item()) for u in U for b in U}
        zero = tuple(0.0 for _ in range(Rin))
        Cz = {u: Cf(u, zero) for u in U}
        new = {}
        for v in vecs(Rout):
            terms = []
            for u in U:
                for a in U:
                    if Cf(u, a) != v: continue
                    lower = [phis[b] for b in U if sc[(u, b)] < sc[(u, a)]]
                    higher = [phis[b] for b in U if sc[(u, b)] > sc[(u, a)]]
                    geq = [phis[b] for b in U if sc[(u, b)] >= sc[(u, a)]]
                    LO = OR(*lower) if lower else BOT
                    HI = OR(*higher) if higher else BOT
                    GE = OR(*geq) if geq else BOT
                    # NOTE the engine's SINCE/UNTIL are STRICT (j<i resp. j>i), matching the strict
                    # attention masks — so eqs. (26)/(28) transcribe directly with no extra shift.
                    if L.mask == "future":
                        if tie == "max":                                     # eq. (26)
                            inner = SINCE(LO, AND(phis[a], NOT(P_(HI))))
                        else:                                                # eq. (28)
                            inner = AND(P_(AND(phis[a], NOT(P_(GE)))), NOT(P_(HI)))
                    else:
                        if tie == "min":
                            inner = UNTIL(LO, AND(phis[a], NOT(F_(HI))))
                        else:
                            inner = AND(F_(AND(phis[a], NOT(F_(GE)))), NOT(F_(HI)))
                    terms.append(AND(phis[u], inner))
            # eq. (27): the empty-unmasked case — P_/F_ are strict, so NOT(P_(TOP)) = "I am position 1"
            for u in U:
                if Cz[u] == v:
                    empty = NOT(P_(TOP)) if L.mask == "future" else NOT(F_(TOP))
                    terms.append(AND(empty, phis[u]))
            new[v] = OR(*terms) if terms else BOT
        phis, Rin = new, Rout
    acc = [phis[v] for v in vecs(Rin)
           if float((T.t @ torch.tensor(list(v), dtype=torch.float64)).item()) > 0]
    return OR(*acc) if acc else BOT                                          # eq. (33)


if __name__ == "__main__":
    # LTL eq. (1) for (ab)+
    phi1 = AND(Q("b"), H_(IMP(Q("b"), Y_(Q("a")))), H_(IMP(AND(Q("a"), Y_(TOP)), Y_(Q("b")))))
    tgt = lambda w: len(w) % 2 == 0 and len(w) > 0 and all(w[i] == ("a" if i % 2 == 0 else "b") for i in range(len(w)))
    bad = [w for w in words("ab", 1, 10) if ltl_accepts(phi1, w) != tgt(w)]
    ok("LTL eq.(1) recognizes exactly (ab)+ up to length 10", not bad, f"{len(bad)} mismatches; |phi|={ltl_size(phi1)}")

    T = ltl_to_uhat(phi1, "ab")
    bad2 = [w for w in words("ab", 1, 9) if T.accepts(w) != tgt(w)]
    ok("Prop.16 UHAT from eq.(1) recognizes exactly (ab)+ up to length 9", not bad2,
       f"{len(bad2)} mismatches; |T|={T.size()}, layers={len(T.layers)}")

    # random formula agreement
    import random
    random.seed(0)
    def rnd(d=0):
        if d >= 2: return random.choice([Q("a"), Q("b"), TOP, BOT])
        c = random.choice(["Q", "not", "and", "or", "S", "U"])
        if c == "Q": return Q(random.choice("ab"))
        if c == "not": return NOT(rnd(d + 1))
        if c == "and": return AND(rnd(d + 1), rnd(d + 1))
        if c == "or": return OR(rnd(d + 1), rnd(d + 1))
        if c == "S": return SINCE(rnd(d + 1), rnd(d + 1))
        return UNTIL(rnd(d + 1), rnd(d + 1))
    mism = 0; tot = 0
    for _ in range(12):
        f = rnd()
        Tf = ltl_to_uhat(f, "ab")
        for w in words("ab", 1, 6):
            tot += 1
            if Tf.accepts(w) != ltl_accepts(f, w): mism += 1
    ok("Prop.16 compiler agrees with LTL semantics on random formulas", mism == 0, f"{mism}/{tot} mismatches")

    print("\n--- counter family ---")
    for N in (1, 2, 3, 4):
        p = counter_brasp(N); w = counter_word(N); Dm = dfa_minimize(counter_dfa(N))
        acc = p.accepts(w, out="Y")
        print(f"  N={N} |w|={len(w)} prog_ops={p.size()} dfa_states={Dm['n']} accepts_w={acc}")
    # exhaustive equivalence for N=1,2 up to the word length
    for N in (1, 2):
        p = counter_brasp(N); w = counter_word(N); D = counter_dfa(N)
        bad = [x for x in words("01#", 1, len(w)) if p.accepts(x, out="Y") != dfa_accepts(D, x)]
        print(f"  N={N}: exhaustive up to len {len(w)} -> {len(bad)} disagreements", bad[:5])

    print("\n--- printed AND vs repaired OR in eq. (14e) ---")
    for N in (2, 3):
        pand = counter_brasp(N, wrap_or=False); por = counter_brasp(N, wrap_or=True)
        w = counter_word(N)
        print(f"  N={N}: printed-AND accepts={pand.accepts(w, out='Y')}  repaired-OR accepts={por.accepts(w, out='Y')}")

    print("\n--- Prop.13 UHAT -> LTL ---")
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    R = 2
    Aq = AFF([[1.0, 1.0], [0.0, 0.0]])
    Bq = AFF([[0.0, 1.0], [0.0, 0.0]])
    Cq = AFF([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
    Tt = UHAT("ab", emb, [UHA(Aq, Bq, Cq, mask="future", tie="max")], [1.0, -1.0])
    for tie in ("max", "min"):
        Tv = UHAT("ab", emb, [UHA(Aq, Bq, Cq, mask="future", tie=tie)], [1.0, -1.0])
        # F must contain every REACHABLE value: C sums u0+a1, so 2.0 is reachable — the earlier
        # self-test declared F=[0,1] and silently violated the compiler's own precondition
        ph = uhat_to_ltl(Tv, [0.0, 1.0, 2.0], tie=tie)
        bad = [w for w in words("ab", 1, 7) if ltl_accepts(ph, w) != Tv.accepts(w)]
        print(f"  tie={tie}: |phi|={ltl_size(ph)} mismatches={len(bad)} ops={sorted(ltl_ops(ph))}", bad[:4])
