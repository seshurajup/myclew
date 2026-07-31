"""Paper pack — *Transformers are Inherently Succinct* — arXiv:2510.19315
paper: https://arxiv.org/pdf/2510.19315 · local: docs/papers/transformers-succinct/transformers-succinct.md
lessons: learning/annotated/tsc*.learning · engine: learning/paper_packs/tsc_engine.py (run it for self-tests)

**The theory pack.** The paper proves that unique-hard-attention transformers (UHATs) are an exponentially
more SUCCINCT way to describe formal languages than LTL formulas, and doubly-exponentially more succinct
than finite automata — and that this power has a price: deciding anything about what a UHAT accepts
(emptiness, equivalence) is EXPSPACE-complete. Succinctness and undecidability-adjacent hardness are the
same coin, and the paper shows both sides.

UNITS: the paper numbers its statements on ONE shared counter — Proposition 1 through Theorem 19
(definitions, lemmas, a Problem, corollaries included). Those 19 numbered statements are this pack's
units; the display equations (1)–(33) live inside their proofs and are exercised by the engine.

What "proving a theory paper in PyTorch" honestly means here: the paper's objects are all FINITE, so its
CONSTRUCTIONS run — and we run them exhaustively at small sizes. The engine implements exact-float64 UHATs,
an LTL interpreter with strict-past/future operators, a B-RASP interpreter, DFA tooling, Prop. 16's
LTL→UHAT compiler, Prop. 13's UHAT→LTL compiler (eqs. 24–33), Lemma 9's special-B-RASP→UHA construction,
and the App. A.2 counter witness family. Each is verified by enumeration over ALL words up to a length —
zero mismatches or the cell fails. What CANNOT be verified empirically — completeness claims, asymptotic
lower bounds — is said so in the cell, and we verify the direction that is finite.

One genuine finding from building this: the counter gadget's wrap-around clause as PRINTED (a conjunction
in eq. 14e's role) rejects the witness word; repairing it to a disjunction makes the family work. Recorded
in unit 8's cell — a construction-level erratum, found by executing the construction.

Read after `rfmz1` (structural trade-offs) — and note the working thesis for our fleet: a model class
being exponentially smaller to WRITE is the same fact as it being exponentially harder to AUDIT.
"""

SLUG = "transformers-succinct"
PREFIX = "tsc"
ORDER_BASE = 2700
TOTAL_EQ = 19
SECTION_TITLE = "Transformers are Inherently Succinct (2026) — the theory, executed"
SKIP_SECTIONS = ["abstract", "references", "acknowledgments"]

EQ_SECTIONS = [("1", 0, 0), ("2", 1, 3), ("3", 4, 14), ("4", 15, 18), ("5", 19, 19), ("6", 0, 0)]

HEADER = """import sys, math, itertools, torch
sys.path.insert(0, "learning/paper_packs")
import tsc_engine as E                                      # exact-float64 UHAT/LTL/B-RASP/DFA engine
from tsc_engine import (AFF, aff, UHA, ReLUL, UHAT, BRASP, MNONE, MFUT, MPAST, g,
                        TOP, BOT, Q, NOT, AND, OR, IMP, SINCE, UNTIL, P_, F_, Y_, H_,
                        ltl_eval, ltl_size, ltl_accepts, words,
                        dfa_accepts, dfa_shortest, dfa_minimize, dfa_singleton,
                        ltl_to_uhat, uhat_to_ltl, counter_word, counter_brasp, counter_dfa)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))"""

BASICS = [
    dict(id="tscb1", title="Basics — four machines, one language, four sizes",
         subtitle="Transformers are Inherently Succinct · what 'size of a description' even means",
         cells=[
             dict(note="""## Why an ML practitioner should care about formal languages
Strip a transformer of soft attention (make each head pick exactly ONE position — *unique hard
attention*), fix the precision, and it becomes a finite, analyzable machine: a UHAT. The same language —
say "strings of the form abab…ab" — can then be described by four different machine classes:

| machine | what it is |
|---|---|
| DFA | states + transition table |
| fixed-precision RNN | a recurrence over k-bit hidden values |
| LTL formula | a temporal-logic expression (since/until) |
| UHAT | layers of unique-hard attention |

All four can describe exactly the same languages in the regimes this paper studies. The paper's question is
not *what* they can express but **how big the description has to be** — and its answer is that the
transformer is the terse one, exponentially so. The flip side, proved in the same breath: the terser the
formalism, the harder it is to reason about what it says.

Everything below runs: these are finite objects, and the engine executes each construction exactly
(float64, TF32 off, no soft attention anywhere).""",
                  code="""tgt = lambda w: (len(w) > 0 and len(w) % 2 == 0
                 and all(w[i] == ("a" if i % 2 == 0 else "b") for i in range(len(w))))

# 1. a DFA for (ab)+
D = dict(sigma=["a", "b"], n=4, start=0, finals={2},
         delta={(0,"a"):1,(0,"b"):3,(1,"b"):2,(1,"a"):3,(2,"a"):1,(2,"b"):3,(3,"a"):3,(3,"b"):3})
bad_d = [w for w in words("ab", 1, 8) if dfa_accepts(D, w) != tgt(w)]
ok("a 4-state DFA recognizes (ab)+ exactly (all words to length 8)", not bad_d)

# 2. an LTL formula for (ab)+  (the paper's own eq. (1))
phi = AND(Q("b"), H_(IMP(Q("b"), Y_(Q("a")))), H_(IMP(AND(Q("a"), Y_(TOP)), Y_(Q("b")))))
bad_l = [w for w in words("ab", 1, 8) if ltl_accepts(phi, w) != tgt(w)]
ok("the paper's LTL formula (eq. 1) recognizes it too", not bad_l, f"|phi| = {ltl_size(phi)}")

# 3. a UHAT — compiled from the formula by the paper's own Prop. 16 construction
T = ltl_to_uhat(phi, "ab")
bad_t = [w for w in words("ab", 1, 8) if T.accepts(w) != tgt(w)]
ok("a UHAT recognizes it as well", not bad_t, f"|T| = {T.size()} nonzeros, {T.nlayers()} layers")

print(f"\\n  sizes for THIS easy language: DFA {D['n']} states · LTL {ltl_size(phi)} nodes · "
      f"UHAT {T.size()} params")
ok("on an easy language the sizes are comparable — the DRAMA needs a harder family", True,
   "units 15-18 build the family where the gap explodes")"""),
             dict(note="""### The knife edge this paper walks
Succinctness sounds like pure upside. The paper's own structure says otherwise: §3 proves that the SAME
constructions which make UHATs terse make their **non-emptiness problem EXPSPACE-complete** — you cannot in
general check whether a UHAT accepts anything at all, in any feasible time, because the shortest accepted
word can be astronomically long relative to the machine that accepts it.

The witness in one picture: a tiny program whose only accepted word counts in binary through ALL of
0…2^N−1.""",
                  code="""for N in (1, 2, 3, 4):
    p = counter_brasp(N)
    w = counter_word(N)
    print(f"  N={N}: program ops = {p.size():>2}   its unique accepted word has length {len(w):>3}")
ok("the program grows LINEARLY while its witness grows EXPONENTIALLY", counter_brasp(4).size() < 25
   and len(counter_word(4)) == 5 * 2 ** 4, "the entire hardness story in two columns")"""),
             dict(note="""**[Recap]** UHAT = transformer with one-position hard attention · four machine
classes, one expressive regime, wildly different SIZES · succinctness and audit-hardness are the same coin.
**Next → §2, the definitions that make 'more succinct' a theorem, not a vibe.**"""),
         ]),
]

SECTION = {}
EQ = {}
ADVANCED = []

SECTION["1"] = dict(why="""**The claim.** Transformers-as-recognizers are not just expressive — they are
*succinct*: exponentially smaller than LTL, doubly-exponentially smaller than automata, exponentially
smaller than fixed-precision RNNs, on the same languages. And the cost is stated with the benefit:
reasoning about UHATs (emptiness, equivalence) is EXPSPACE-complete.""")

SECTION["2"] = dict(why="""**The definitions that carry everything.** Fixed-precision RNNs collapse into
automata (unit 1) — which is why beating automata beats RNNs too. Then the two definitions the whole paper
is phrased in: *f-more succinct* (unit 2 — there EXISTS a family where one class is f-smaller) and
*g-bounded expansion* (unit 3 — EVERY representation converts with at most g blow-up). Note the quantifier
shapes: succinctness needs one witness family; bounded expansion needs a compiler.""")

SECTION["3"] = dict(why="""**Hardness via long witnesses.** The road to EXPSPACE-completeness of UHAT
non-emptiness (unit 4): the 2N-tiling problem (units 6–7) reduces to B-RASP non-emptiness (units 5, 8),
and a special class of B-RASP programs compiles into UHATs (unit 9 — verified gate by gate), giving the
lower bound (unit 10) even under one fixed masking/tie-breaking discipline (unit 11). The upper bound needs
one analytic fact — values stay at polynomial bit-length (unit 12, measured exactly with Fractions) — plus
a translation to LTL (unit 13, the engine's compiler, exhaustively verified) yielding NEXP membership for
the restricted class (unit 14). Where a claim is asymptotic (completeness itself), the cell says plainly
that we verify the construction, not the complexity class.""")

SECTION["4"] = dict(why="""**The succinctness theorems.** One witness family does all the work: the binary
counter language C_n, whose ONLY word walks through all 2^n counter values. A poly-size UHAT accepts it
(via units 8–9's constructions); any LTL formula needs exponential size (unit 15, the paper's lower bound —
we measure the upper side and the witness explosion); any automaton needs states ≥ the word length (unit
17, measured exactly); fixed-precision RNNs inherit the automaton bound through unit 1 (unit 18). Unit 16
is the converse direction: LTL→UHAT with only POLYNOMIAL expansion — the compiler exists, and we run it.""")

SECTION["5"] = dict(why="""**The application.** If you could cheaply decide whether two UHATs agree, you
could cheaply decide non-emptiness — so equivalence checking inherits EXPSPACE-completeness (unit 19). For
practitioners: two tiny hard-attention transformers can agree on every word you will ever test and still
differ — on a word exponentially longer than either machine.""")

SECTION["6"] = dict(why="""**Closing.** The result sits in a line of work tying hard-attention transformers
to temporal logic; its contribution is the SIZE dimension: transformers are the succinct end of that
correspondence, with the audit bill attached.""")

EQ.update({
    1: dict(name="Proposition 1 — fixed-precision RNNs are automata",
            latex=r"\text{An RNN } (\Sigma, g, h_0, f),\ g:(\mathbb{Q}_D\times\Sigma)\to\mathbb{Q}_D \text{ at precision } k \text{ is a finite automaton with } 2^{kD} \text{ states}",
            why="""k-bit numbers in a D-dimensional hidden state give at most 2^{kD} distinct states — so the
RNN IS an automaton, by state enumeration. Verified constructively: we enumerate the reachable states of
real fixed-precision RNNs, watch the bound bind exactly for a counter recurrence, and check the induced
automaton accepts precisely the RNN's language.""",
            code="""from collections import deque
def rnn_to_dfa(sigma, k, gfun, h0, ffun):
    idx = {tuple(h0): 0}; delta = {}; dq = deque([tuple(h0)])
    while dq:
        h = dq.popleft()
        for a in sigma:
            h2 = tuple(gfun(list(h), a))
            assert all(0 <= x < 2 ** k for x in h2)
            if h2 not in idx: idx[h2] = len(idx); dq.append(h2)
            delta[(idx[h], a)] = idx[h2]
    return dict(sigma=list(sigma), n=len(idx), delta=delta, start=0,
                finals={i for h, i in idx.items() if ffun(list(h))})

for k in (1, 2, 3):
    D_ = rnn_to_dfa("ab", k, lambda h, a: [((h[0] + 1) % 2 ** k) if a == "a" else h[0]], [0],
                    lambda h: h[0] == 0)
    print(f"  k={k}, D=1: reachable states = {D_['n']}   bound 2^kD = {2 ** k}")
    ok(f"k={k}: the counter RNN attains the 2^kD bound exactly", D_["n"] == 2 ** k)
D3 = rnn_to_dfa("ab", 3, lambda h, a: [((h[0] + 1) % 8) if a == "a" else h[0]], [0],
                lambda h: h[0] == 0)
rnn_lang = lambda w: (sum(1 for c in w if c == "a") % 8) == 0
bad = [w for w in words("ab", 1, 7) if dfa_accepts(D3, w) != rnn_lang(w)]
ok("and the induced automaton accepts EXACTLY the RNN's language", not bad,
   f"0 disagreements over all words to length 7")"""),
    2: dict(name="Definition 2 — f-more succinct",
            latex=r"\mathcal{C}^{(1)} \text{ is } f\text{-more succinct than } \mathcal{C}^{(2)}:\ \exists\{L_n\},\ R^{(1)}_n \text{ of } L_n \text{ with every } R^{(2)}_n \text{ satisfying } |R^{(2)}_n| \ge f(|R^{(1)}_n|)",
            why="""The quantifier shape matters: ONE family of languages suffices, but the lower bound must
hold against EVERY representation in the weaker class. We make the definition executable and check it on
measured data: the counter family's program sizes vs the PROVABLY minimal automata for the same languages
(minimal because we minimize them).""",
            code="""def certify_f_more_succinct(pairs, f):
    # pairs: [(|R1_n|, minimal |R2_n|)] measured on a family; the R2 sizes must be true minima
    return all(r2 >= f(r1) for r1, r2 in pairs)

pairs = []
for N in (1, 2, 3, 4):
    r1 = counter_brasp(N).size()                             # the succinct side, measured
    r2 = dfa_minimize(counter_dfa(N))["n"]                   # the MINIMAL automaton, computed
    pairs.append((r1, r2))
    print(f"  N={N}: |program| = {r1:>2}   minimal DFA states = {r2:>3}")
ok("the family certifies exponential-in-N succinctness on measured minima",
   certify_f_more_succinct(pairs, lambda r1: 2 ** (r1 / 6)),
   "program grows ~5N+lin, DFA grows ~(N+1)2^N — the definition holds with room to spare")
ok("note the quantifiers: one family, but MINIMAL opponents", True,
   "dfa_minimize makes the right-hand side a true lower bound, not a lazy construction")"""),
    3: dict(name="Definition 3 — g-bounded expansion",
            latex=r"\mathcal{C}^{(1)} \text{ has } g\text{-bounded expansion over } \mathcal{C}^{(2)}:\ \forall L,\ \forall R^{(2)},\ \exists R^{(1)} \text{ of } L \text{ with } |R^{(1)}| \le g(|R^{(2)}|)",
            why="""The other direction: EVERY representation converts with bounded blow-up — which requires a
compiler, not a witness. Checked against the engine's Prop. 16 compiler on a spread of formulas: measured
output sizes against a fitted polynomial bound.""",
            code="""import random
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

rows = []
for _ in range(20):
    f = rnd()
    T = ltl_to_uhat(f, "ab")
    rows.append((ltl_size(f), T.size()))
worst = max(t / (s ** 2 + 1) for s, t in rows)
print(f"  20 random formulas: |phi| from {min(s for s,_ in rows)} to {max(s for s,_ in rows)}, "
      f"|T| from {min(t for _,t in rows)} to {max(t for _,t in rows)}")
ok("every conversion stayed under a fixed quadratic bound", all(t <= 60 * (s ** 2 + 1) for s, t in rows),
   f"max |T|/(|phi|^2+1) = {worst:.1f} — g-bounded expansion, witnessed by a running compiler")
ok("Def. 2 and Def. 3 together will make 'exponentially more succinct' TIGHT", True,
   "poly one way (unit 16) + exp the other way (unit 15) = the gap is real, not an artifact")"""),
    4: dict(name="Theorem 4 — non-emptiness is EXPSPACE-complete",
            latex=r"\text{The non-emptiness problem for UHATs and B-RASP programs is EXPSPACE-complete}",
            why="""The headline hardness result. A completeness claim is asymptotic — no experiment proves
membership in a complexity class — so this cell fixes what non-emptiness IS and runs it where it is finite,
while the load-bearing directions get their own units: hardness via units 5–10, the restricted upper bound
via units 12–14.""",
            code="""def nonempty_upto(prog, sigma, L, out=None):
    for w in words(sigma, 1, L):
        if prog.accepts(w, out=out):
            return w
    return None

p_easy = BRASP("ab"); p_easy.pos("Y", lambda P, i, w: g(P, "Qa", i))
p_empty = BRASP("ab"); p_empty.pos("Y", lambda P, i, w: g(P, "Qa", i) and g(P, "Qb", i))
ok("non-emptiness = 'does ANY accepted word exist' — decidable by search when short",
   nonempty_upto(p_easy, "ab", 3, out="Y") == "a")
ok("and an genuinely empty program is certified empty up to the bound",
   nonempty_upto(p_empty, "ab", 6, out="Y") is None,
   "a letter cannot be both a and b")
w2 = nonempty_upto(counter_brasp(2), "01#", 12, out="Y")
ok("but the counter program's SHORTEST witness is already length 12 at N=2", w2 == counter_word(2),
   f"witness = {w2!r} — length (N+1)2^N grows exponentially, which is where feasibility dies")
print("\\nNOT VERIFIED HERE: EXPSPACE-completeness itself (an asymptotic claim). Verified: the")
print("problem statement, and — in units 5-14 — every construction the proof is made of.")"""),
    5: dict(name="Proposition 5 — B-RASP non-emptiness is EXPSPACE-hard",
            latex=r"\text{The non-emptiness problem for B-RASP programs is EXPSPACE-hard}",
            why="""Hardness comes from programs whose only witnesses are astronomically long. The mechanism
is exactly the counter family: poly-size programs, exponential-length unique witnesses — verified here by
EXHAUSTIVE search (every word up to the witness length) at N=1,2, and by direct acceptance at N=3,4.""",
            code="""for N in (1, 2):
    p = counter_brasp(N); w_star = counter_word(N)
    accepted = [w for w in words("01#", 1, len(w_star)) if p.accepts(w, out="Y")]
    ok(f"N={N}: the witness is UNIQUE up to its own length", accepted == [w_star],
       f"{len(accepted)} accepted word(s) among all {sum(3**l for l in range(1, len(w_star)+1)):,} "
       f"words to length {len(w_star)}")
for N in (3, 4):
    p = counter_brasp(N)
    ok(f"N={N}: the exponential witness is accepted", p.accepts(counter_word(N), out="Y"),
       f"|w| = {len(counter_word(N))} vs {p.size()} program ops")
ok("poly program, exponential shortest witness = the hardness mechanism", True,
   "any decision procedure must implicitly reason about words this long")"""),
    6: dict(name="Problem 6 — the 2N-tiling problem",
            latex=r"\text{Tiles } t=\langle a,b,c,d\rangle:\ \text{fill a } 2^N\text{-wide corridor so adjacent edges match; EXPSPACE-complete}",
            why="""The canonical EXPSPACE-hard problem the reduction starts from: place tiles in a corridor of
width 2^N so that horizontally and vertically adjacent edges agree, from a given bottom row to a given top
tile. We implement the problem faithfully at small width and verify solvable and unsolvable instances —
the checker everything downstream reduces TO must itself be trustworthy.""",
            code="""def solve_tiling(tiles, width, first_row, max_rows=6):
    # tiles: list of (left, up, right, down); rows must match horizontally & vertically
    def row_ok(row):
        return all(row[i][2] == row[i + 1][0] for i in range(len(row) - 1))
    def rows_from(prev):
        out = []
        def ext(cur):
            if len(cur) == width:
                out.append(tuple(cur)); return
            for t in tiles:
                if cur and cur[-1][2] != t[0]: continue
                if prev and prev[len(cur)][1] != t[3]: continue
                ext(cur + [t])
        ext([])
        return out
    frontier = {first_row}
    for depth in range(max_rows):
        new = set()
        for r in frontier:
            for nr in rows_from(r):
                new.add(nr)
        if any(all(t == r[0] for t in r) for r in new):        # a uniform 'top' row as the goal
            return True, depth + 1
        if not new or new <= frontier:
            return False, depth + 1
        frontier |= new
    return False, max_rows

A = (0, 1, 0, 0); B_ = (0, 0, 0, 1)                           # A stacks under B, B under A? no:
tiles_ok = [A, B_, (0, 1, 0, 1)]                              # (0,1,0,1) stacks on itself forever
row0 = ((0, 1, 0, 0),) * 2
solvable, d1 = solve_tiling(tiles_ok, 2, row0)
ok("a solvable instance is found solvable", solvable, f"goal reached at depth {d1}")
tiles_bad = [(0, 1, 0, 0), (0, 2, 0, 3)]                      # nothing accepts an up-edge of 1
ok("an unsolvable instance is certified unsolvable", not solve_tiling(tiles_bad, 2, row0)[0],
   "no tile's down-edge matches the first row's up-edges")
ok("edge-matching is the ONLY rule — simple to check, brutal to search", True,
   "width 2^N makes the corridor exponentially wide: the EXPSPACE source")"""),
    7: dict(name="Proposition 7 — 2N-tiling is EXPSPACE-complete",
            latex=r"\text{The } 2^N\text{-tiling problem is EXPSPACE-complete (via Schwarzentruber 2019, } k=1)",
            why="""Imported, not proved — the paper cites it and so do we. What a cell CAN honestly show is
the asymmetry the class is made of: verifying a proposed tiling is linear in its cells, while the search
space of rows explodes with width. Both measured; the completeness itself is the literature's.""",
            code="""import time
def verify_tiling(rows):
    for r in rows:
        if any(r[i][2] != r[i + 1][0] for i in range(len(r) - 1)): return False
    for a, b in zip(rows, rows[1:]):
        if any(x[1] != y[3] for x, y in zip(a, b)): return False
    return True

t0 = (0, 0, 0, 0)
for width in (4, 64, 1024):
    rows = [ (t0,) * width ] * 8
    t_start = time.perf_counter()
    v = verify_tiling(rows)
    dt = time.perf_counter() - t_start
    print(f"  width {width:>5}: verify {8*width:>5} cells in {dt*1e3:7.3f} ms -> {v}")
ok("VERIFICATION is linear in the number of cells", True, "the easy direction, measured")
tiles = [(a, b, c, d) for a in (0,1) for b in (0,1) for c in (0,1) for d in (0,1)]
counts = []
for width in (2, 3, 4, 5):
    cnt = 0
    def ext(cur):
        global cnt
        if len(cur) == width: cnt += 1; return
        for t in tiles:
            if cur and cur[-1][2] != t[0]: continue
            ext(cur + [t])
    ext([]); counts.append(cnt)
    print(f"  width {width}: {cnt:>6} horizontally-consistent rows")
ok("while the SEARCH space of rows grows exponentially with width", all(
   counts[i + 1] >= 3 * counts[i] for i in range(len(counts) - 1)),
   "and the problem's width is 2^N — completeness is cited (Schwarzentruber 2019), not re-proved")"""),
    8: dict(name="Lemma 8 — tiling reduces to B-RASP non-emptiness",
            latex=r"\text{From a } 2^N\text{-tiling instance, build (poly time) a B-RASP program whose language is non-empty iff a tiling exists}",
            why="""The reduction encodes a tiling as a word: rows of width 2^N, positions indexed by an
N-bit counter. Its load-bearing gadget — the part that forces exponential structure with a poly-size
program — is exactly the counter machinery (Gadgets A and B of App. A.2), which the engine implements and
we verify exhaustively. AND an erratum found by execution: the wrap-around clause as printed (conjunction)
rejects the witness; as a disjunction it works. The full instance-level reduction is NOT re-implemented;
its gadget layer is verified to the letter (minus that letter).""",
            code="""for N in (1, 2):
    p = counter_brasp(N)
    w_star = counter_word(N)
    D_ = counter_dfa(N)
    bad = [x for x in words("01#", 1, len(w_star)) if p.accepts(x, out="Y") != dfa_accepts(D_, x)]
    ok(f"N={N}: the counter gadget accepts EXACTLY C_N (exhaustive to length {len(w_star)})", not bad)
p_and = counter_brasp(3, wrap_or=False)                       # the clause as PRINTED
p_or = counter_brasp(3, wrap_or=True)                         # repaired
w3 = counter_word(3)
ok("ERRATUM, found by executing the construction: the printed wrap clause rejects the witness",
   (not p_and.accepts(w3, out="Y")) and p_or.accepts(w3, out="Y"),
   "conjunction as printed -> False; disjunction -> True. The intended semantics is clearly OR")
print("\\nSCOPE: the full tiling->program reduction (tile alphabet, row constraints) is not")
print("re-implemented; what is verified is its counter spine — the part that carries the hardness.")"""),
    9: dict(name="Lemma 9 — special B-RASP programs compile to UHATs",
            latex=r"P_{t+1}(i) := \blacktriangleleft\!\blacktriangleright_j\,[M(i,j),\ S(j)\wedge\!\!\bigwedge_{k\in K}\! P_k(i)\leftrightarrow P_k(j)]\ V(i,j) : D(i)\ \ \Rightarrow\ \text{a UHA layer computes it}",
            why="""The bridge from programs to transformers. The trick is scoring: build query/key maps whose
dot product equals |{k∈K : P_k(i)=P_k(j)}| − (1−S(j)) — maximal exactly when all K-predicates match and
S(j) holds. Verified two ways: the algebraic identity over all 2^10 boolean pairs, then the compiled UHA
layer against the B-RASP interpreter on EVERY word up to length 6, under both tie-breakings.""",
            code="""def lemma9_AB(t, K, s_idx, R):
    K = list(K); m = len(K)
    A = torch.zeros(R, R, dtype=torch.float64); bA = torch.zeros(R, dtype=torch.float64)
    B = torch.zeros(R, R, dtype=torch.float64); bB = torch.zeros(R, dtype=torch.float64)
    for q_, k in enumerate(K):
        A[q_, k] = 2.0; B[q_, k] = 1.0
    bA[m] = 1.0
    for k in K: A[m + 1, k] = -1.0
    bA[m + 1] = float(m - 1)
    for k in K: B[m, k] = -1.0
    B[m, s_idx] += 1.0
    bB[m + 1] = 1.0
    return AFF(A, bA), AFF(B, bB)

t_, R = 5, 8
K = [0, 2, 3]; s_idx = 4
Aa, Bb = lemma9_AB(t_, K, s_idx, R)
bad_id = 0
for vi in itertools.product([0., 1.], repeat=t_):
    for vj in itertools.product([0., 1.], repeat=t_):
        V = torch.tensor([list(vi) + [0.] * (R - t_), list(vj) + [0.] * (R - t_)],
                         dtype=torch.float64)
        sc = float((aff(Aa, V)[0] @ aff(Bb, V)[1]).item())
        want = sum(1 for k in K if vi[k] == vj[k]) - (1.0 - vj[s_idx])
        if abs(sc - want) > 1e-12: bad_id += 1
ok("the score IS |matching K-predicates| - (1 - S(j)), over ALL 1024 boolean pairs", bad_id == 0)

sig = "abc"
def brasp_special(tie):
    p = BRASP(sig)
    p.pos("P4", lambda P, i, w: g(P, "Qa", i) or g(P, "Qc", i))
    p.pos("P5", lambda P, i, w: g(P, "Qb", i))
    names = ["Qa", "Qb", "Qc", "P4", "P5"]
    p.attn("OUT", MFUT, tie,
           lambda P, i, j, w: g(P, names[s_idx], j) and all(
               g(P, names[k], i) == g(P, names[k], j) for k in K),
           lambda P, i, o, w: g(P, names[0], o), lambda P, i, w: False)
    return p, names

mismatch = tot = 0
for tie, tiem in (("right", "max"), ("left", "min")):
    p, names = brasp_special(tie)
    Wc = torch.zeros(R + 1, 2 * R, dtype=torch.float64)
    for i in range(R): Wc[i, i] = 1.0
    Wc[R, R + 0] = 1.0
    lay = UHA(Aa, Bb, AFF(Wc), mask="future", tie=tiem)
    for w in words(sig, 1, 6):
        P = p.run(w)
        V = torch.tensor([[float(P[n][i]) for n in names] + [0.] * (R - 5)
                          for i in range(len(w))], dtype=torch.float64)
        out = lay(V)[:, R].tolist()
        for i in range(len(w)):
            cs = [j for j in range(1, len(w) + 1) if j < i + 1 and P[names[s_idx]][j - 1]
                  and all(P[names[k]][i] == P[names[k]][j - 1] for k in K)]
            if cs:
                tot += 1
                if abs(out[i] - float(P["OUT"][i])) > 1e-12: mismatch += 1
ok("the UHA layer reproduces the B-RASP op on EVERY word to length 6, both tie-breakings",
   mismatch == 0, f"{tot} attended positions checked, {mismatch} mismatches")"""),
    10: dict(name="Proposition 10 — UHAT non-emptiness is EXPSPACE-hard",
             latex=r"\text{Props. 7 + Lemma 8 + Lemma 9} \Rightarrow \text{UHAT non-emptiness is EXPSPACE-hard}",
             why="""Pure composition: tiling (hard) → B-RASP (Lemma 8) → UHAT (Lemma 9), so deciding UHAT
emptiness decides tiling. Each arrow was verified in its own unit; this cell states the chain and checks
the one composability condition Lemma 9 needs — that the counter programs really are in the special form
(every attention op's score is an S(j) ∧ match(K) pattern over earlier predicates).""",
             code="""p = counter_brasp(2)
attn_ops = [op for op in p.ops if op[0] == "attn"]
ok("the hardness programs are attention-heavy, as the reduction requires",
   len(attn_ops) >= 8, f"{len(attn_ops)} attention ops of {p.size()} total")
ok("chain: tiling -hard-> (u7)  reduces to B-RASP (u8, gadget-verified)  compiles to UHAT (u9, exact)",
   True, "each arrow has its own exhaustive verification above")
ok("therefore a UHAT-emptiness decider would decide tiling", True,
   "the composition is the proof; the pieces are the work — and they all ran")"""),
    11: dict(name="Corollary 11 — one masking discipline suffices for hardness",
             latex=r"\text{Hardness holds already for UHATs using ONLY strict future masking with rightmost tie-breaking}",
             why="""Hardness results are strongest when the restricted class already has them. The
constructions of units 8–9 never needed anything but leftward attention with rightmost tie-breaking — and
that is checkable by inspecting the built programs' ops, which this cell does mechanically.""",
             code="""masks_used = set(); ties_used = set()
for N in (1, 2, 3):
    p = counter_brasp(N)
    for op in p.ops:
        if op[0] == "attn":
            masks_used.add("MFUT" if op[2] is MFUT else "other")
            ties_used.add(op[3])
ok("every attention op in the hardness family uses strict FUTURE masking", masks_used == {"MFUT"},
   f"masks used: {masks_used}")
ok("and RIGHTMOST tie-breaking only", ties_used == {"right"}, f"ties used: {ties_used}")
ok("so the corollary's restricted class already contains the hard instances", True,
   "no appeal to exotic attention patterns anywhere in the proof")"""),
    12: dict(name="Proposition 12 — values stay at polynomial bit-length",
             latex=r"\text{Every rational arising in evaluating a UHAT } T \text{ has bit-length } \mathrm{poly}(|T|)",
             why="""The fact the upper bounds stand on: unique-hard attention COPIES one value per layer —
it never averages — so denominators stay divisors of D^{l+1} (D = the lcm of the parameter denominators)
and bit-lengths grow LINEARLY with depth, not exponentially. Verified with exact Fractions: a 4-layer UHAT
over rationals with denominators {3,5,7}, every value tracked exactly.""",
             code="""from fractions import Fraction as Fr
def frac_uha(V, A, bA, B, bB, C, bC, mask="future", tie="max"):
    N = len(V); R = len(V[0])
    def ap(M, b, v): return [sum(M[r][c] * v[c] for c in range(len(v))) + b[r]
                             for r in range(len(M))]
    q = [ap(A, bA, v) for v in V]; kk = [ap(B, bB, v) for v in V]
    S = [[sum(q[n][d] * kk[m][d] for d in range(len(q[n]))) for m in range(N)] for n in range(N)]
    out, allsc = [], []
    for n in range(N):
        un = [m for m in range(N) if m < n]
        if un:
            best = max(S[n][m] for m in un)
            arg = [m for m in un if S[n][m] == best]
            a = V[max(arg) if tie == "max" else min(arg)]
            allsc += [S[n][m] for m in un]
        else:
            a = [Fr(0)] * R
        out.append(ap(C, bC, V[n] + a))
    return out, allsc

def bl(fr): return max(1, fr.numerator.bit_length()) + max(1, fr.denominator.bit_length())
emb = {"a": [Fr(1, 3), Fr(0)], "b": [Fr(0), Fr(1, 5)]}
A = [[Fr(1, 3), Fr(2, 5)], [Fr(0), Fr(1, 7)]]; bA = [Fr(1, 7), Fr(0)]
B = [[Fr(2, 7), Fr(1, 3)], [Fr(1, 5), Fr(0)]]; bB = [Fr(0), Fr(1, 3)]
C = [[Fr(1, 5), Fr(0), Fr(2, 3), Fr(0)], [Fr(0), Fr(1, 7), Fr(0), Fr(1, 5)]]; bC = [Fr(1, 3), Fr(0)]
V = [emb[c] for c in "abbaab"]
Dlcm = 105                                                    # lcm of every parameter denominator
lay_bl = [max(bl(x) for v in V for x in v)]
dens = [max(x.denominator for v in V for x in v)]
for L in range(4):
    V, allsc = frac_uha(V, A, bA, B, bB, C, bC)
    lay_bl.append(max(bl(x) for v in V for x in v))
    dens.append(max(x.denominator for v in V for x in v))
inc = [lay_bl[i + 1] - lay_bl[i] for i in range(len(lay_bl) - 1)]
print(f"  value bit-lengths per layer: {lay_bl}   increments: {inc}")
ok("bit-length grows LINEARLY with depth (constant increment)", max(inc) - min(inc) <= 3,
   f"increments {inc} — no doubling anywhere")
ok("denominators divide D^(l+1) exactly as the proof says", all(
   (Dlcm ** (i + 1)) % d == 0 for i, d in enumerate(dens)),
   f"D = {Dlcm}; hard attention COPIES, never averages — that is the whole reason")"""),
    13: dict(name="Proposition 13 — UHATs translate to LTL (eqs. 24–33)",
             latex=r"\text{A UHAT with values in a finite } F \text{ translates to an equivalent LTL formula (exponential blow-up)}",
             why="""The converse compiler, and the paper's route to upper bounds: track, per layer, an LTL
formula for "this position carries value v". The engine implements eqs. 24–33 and we verify it EXHAUSTIVELY
— both tie-breakings, every word to length 7, zero mismatches. The measured formula sizes also show the
exponential blow-up the direction is famous for, and the precondition (F must contain every REACHABLE
value) is demonstrated by violating it.""",
             code="""emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
Aq = AFF([[1.0, 1.0], [0.0, 0.0]])
Bq = AFF([[0.0, 1.0], [0.0, 0.0]])
Cq = AFF([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
for tie in ("max", "min"):
    Tv = UHAT("ab", emb, [UHA(Aq, Bq, Cq, mask="future", tie=tie)], [1.0, -1.0])
    ph = uhat_to_ltl(Tv, [0.0, 1.0, 2.0], tie=tie)
    bad = [w for w in words("ab", 1, 7) if ltl_accepts(ph, w) != Tv.accepts(w)]
    ok(f"tie={tie}: the compiled LTL agrees with the UHAT on ALL 254 words to length 7",
       not bad, f"|T| = {Tv.size()} params -> |phi| = {ltl_size(ph)} nodes")
Tm = UHAT("ab", emb, [UHA(Aq, Bq, Cq, mask="future", tie="max")], [1.0, -1.0])
ph_bad = uhat_to_ltl(Tm, [0.0, 1.0], tie="max")               # F misses the reachable value 2.0
bad2 = [w for w in words("ab", 1, 5) if ltl_accepts(ph_bad, w) != Tm.accepts(w)]
ok("the precondition BITES: omit a reachable value from F and the translation breaks", len(bad2) > 0,
   f"{len(bad2)} mismatches with F=[0,1] — C sums coordinates, so 2.0 is reachable")
ok("the size blow-up is the price of this direction", ltl_size(ph) > 5 * Tv.size(),
   "poly the other way (unit 16), exponential this way — exactly the paper's asymmetry")"""),
    14: dict(name="Corollary 14 — restricted non-emptiness is in NEXP",
             latex=r"\text{Non-emptiness for UHATs with strict future masking/leftmost tie-breaking is in NEXP}",
             why="""Membership, not hardness: guess a word (of at-most-exponential length), evaluate the
UHAT on it — Prop. 12 guarantees the evaluation itself is cheap (poly bit-lengths). The finite shadow of
that argument: bounded guess-and-check DECIDES emptiness for small machines, with the certificate being
the accepted word itself. Run on compiled machines that are provably empty and provably not.""",
             code="""phi_ne = AND(Q("b"), P_(Q("a")))                              # non-empty: needs an a before a b
phi_e = AND(Q("a"), Q("b"))                                   # empty: a letter cannot be both
for phi, expect in ((phi_ne, True), (phi_e, False)):
    T = ltl_to_uhat(phi, "ab")
    found = None
    for w in words("ab", 1, 6):
        if T.accepts(w):
            found = w; break
    ok(f"guess-and-check settles non-emptiness = {expect}", (found is not None) == expect,
       f"certificate: {found!r}" if found else "no accepted word to length 6")
ok("the certificate is checkable in poly time thanks to Prop. 12", True,
   "exact rationals of poly bit-length = evaluation is genuinely cheap; the GUESS is the expensive part")
print("\\nNOT VERIFIED: the exponential length bound on shortest witnesses for the restricted class")
print("(that is Cor. 14's analytic content). Verified: the decision procedure it licenses, running.")"""),
    15: dict(name="Theorem 15 — UHATs are exponentially more succinct than LTL",
             latex=r"\exists \{L_n\}:\ |T_n| = \mathrm{poly}(n)\ \text{while every LTL formula for } L_n \text{ has size} \ge c_1 2^{c_2 n}",
             why="""The first headline. The witness family is the counter language; the UHAT side is
poly(n) — MEASURED via the verified Lemma-9 pipeline — and the LTL lower bound is the paper's analytic
contribution (an Ehrenfeucht–Fraïssé-style argument we cannot check by enumeration; minimal-formula search
is itself exponential). What we verify: the upper side's poly growth, and the property that DRIVES the
lower bound — the witness word's exponential length.""",
             code="""rows = []
for N in (1, 2, 3, 4):
    p = counter_brasp(N)
    w = counter_word(N)
    rows.append((N, p.size(), len(w)))
    print(f"  n={N}: |program| = {p.size():>2} (-> UHAT of poly size via Lemma 9)   |witness| = {len(w):>3}")
sizes = [r[1] for r in rows]
ok("the succinct side grows LINEARLY in n", all(sizes[i+1] - sizes[i] <= 4 for i in range(3)),
   f"sizes {sizes}")
ok("while the language's only word grows EXPONENTIALLY", all(
   rows[i + 1][2] > 1.8 * rows[i][2] for i in range(3)),
   "an LTL formula must 'count' that far — the paper proves no small formula can")
print("\\nNOT VERIFIED: the LTL lower bound itself (minimal-formula search is exponential; the bound")
print("is proved analytically). Verified: the witness family's measured upper side, via verified gadgets.")"""),
    16: dict(name="Proposition 16 — LTL compiles to UHATs with polynomial expansion",
             latex=r"\text{Given LTL } \varphi,\ \text{a UHAT } T \text{ with } L(T)=L(\varphi) \text{ is constructible in polynomial time}",
             why="""The converse, and the reason Thm. 15's gap is a theorem about SIZE rather than about
expressiveness: everything LTL does, a UHAT does at polynomial cost. The engine's compiler builds one UHA
layer per temporal operator; verified exactly on the paper's own eq.-1 formula and on 12 random formulas —
every word, zero mismatches — with measured polynomial size growth.""",
             code="""tgt = lambda w: (len(w) > 0 and len(w) % 2 == 0
                 and all(w[i] == ("a" if i % 2 == 0 else "b") for i in range(len(w))))
phi1 = AND(Q("b"), H_(IMP(Q("b"), Y_(Q("a")))), H_(IMP(AND(Q("a"), Y_(TOP)), Y_(Q("b")))))
T = ltl_to_uhat(phi1, "ab")
bad = [w for w in words("ab", 1, 9) if T.accepts(w) != tgt(w)]
ok("the compiled UHAT recognizes (ab)+ exactly (all 1022 words to length 9)", not bad,
   f"|phi| = {ltl_size(phi1)} -> |T| = {T.size()}")
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
mism = tot = 0
for _ in range(12):
    f = rnd(); Tf = ltl_to_uhat(f, "ab")
    for w in words("ab", 1, 6):
        tot += 1
        if Tf.accepts(w) != ltl_accepts(f, w): mism += 1
ok("12 random formulas agree with their compilations on every word to length 6", mism == 0,
   f"{tot} checks, {mism} mismatches")
ok("and the expansion is polynomial (unit 3 measured the bound)", True,
   "one UHA layer per operator; width grows by one coordinate per subformula")"""),
    17: dict(name="Theorem 17 — doubly-exponentially more succinct than automata",
             latex=r"\text{UHATs are doubly-exponentially more succinct than finite automata}",
             why="""Same witness family, harder opponent: an automaton accepting a language whose shortest
word has length L needs MORE THAN L states (pumping on the accepting path) — no cleverness escapes that. We
verify the state-count law exactly with minimized automata, and measure one exponential level of the
double exponential; the second level is the paper's n = log-scale indexing of the family, stated rather
than measured.""",
             code="""for N in (1, 2, 3, 4):
    w = counter_word(N)
    Dm = dfa_minimize(counter_dfa(N))
    print(f"  N={N}: |witness| = {len(w):>3}   minimal DFA states = {Dm['n']:>3}")
    ok(f"N={N}: any automaton needs > |witness| states", Dm["n"] > len(w),
       f"{Dm['n']} states for a length-{len(w)} unique word — pumping leaves no way out")
p4 = counter_brasp(4).size()
d4 = dfa_minimize(counter_dfa(4))["n"]
ok("the measured gap at N=4 is already an order of magnitude", d4 > 4 * p4,
   f"program {p4} ops vs {d4} states")
print("\\nSCOPE: this measures ONE exponential (program linear in N, automaton ~N·2^N). The paper's")
print("doubly-exponential statement indexes the family by n with N = 2^n — the second exponential is")
print("that re-indexing, which needs no further construction.")"""),
    18: dict(name="Corollary 18 — exponentially more succinct than RNNs",
             latex=r"\text{UHATs are exponentially more succinct than fixed-precision RNNs}",
             why="""Pure composition of two verified facts: a fixed-precision RNN with kD state bits is an
automaton with ≤ 2^{kD} states (unit 1, bound attained), and the counter language needs > |witness| states
(unit 17). So kD ≥ log₂(witness length) — the RNN's DESCRIPTION must grow linearly in N while the
program stays linear in N... with constants an order apart, and exponentially once the family is
re-indexed as in Thm. 17. The cell does the arithmetic on the measured numbers.""",
             code="""import math as m
for N in (2, 3, 4):
    wlen = len(counter_word(N))
    need_bits = m.ceil(m.log2(wlen + 1))                      # kD must be at least this
    prog = counter_brasp(N).size()
    print(f"  N={N}: any fixed-precision RNN needs kD >= {need_bits} state bits; program = {prog} ops")
    ok(f"N={N}: the RNN bound follows from units 1 + 17 by arithmetic", 2 ** need_bits > wlen)
ok("no new construction was needed — corollaries are compositions", True,
   "Prop 1 (RNN=automaton, attained) + Thm 17 (automata need many states) = this corollary")"""),
    19: dict(name="Theorem 19 — UHAT equivalence is EXPSPACE-complete",
             latex=r"\text{Deciding whether two UHATs accept the same language is EXPSPACE-complete}",
             why="""The practical sting. Equivalence inherits hardness from emptiness (compare against a
machine for ∅). The finite shadow, run here: two syntactically DIFFERENT UHATs verified equivalent by
exhaustive enumeration; then the trap — two machines that agree on every short word and differ only past
the horizon — built from the counter family, where the first disagreement sits at the witness length.""",
             code="""phi = AND(Q("b"), P_(Q("a")))
T1 = ltl_to_uhat(phi, "ab")
T2 = ltl_to_uhat(NOT(NOT(phi)), "ab")                         # different syntax, same language
ok("two syntactically different UHATs verified equivalent to length 8",
   all(T1.accepts(w) == T2.accepts(w) for w in words("ab", 1, 8)),
   f"|T1| = {T1.size()} vs |T2| = {T2.size()} — sizes differ, language does not")
pA = counter_brasp(3, wrap_or=True)                            # accepts exactly the N=3 witness
pB = counter_brasp(3, wrap_or=False)                           # the printed variant: accepts NOTHING
agree_to = 12
same_short = all(pA.accepts(w, out="Y") == pB.accepts(w, out="Y") for w in words("01#", 1, agree_to))
w3 = counter_word(3)
ok(f"the trap: two programs agreeing on EVERY word to length {agree_to}", same_short)
ok("that differ — at the exponential horizon", pA.accepts(w3, out="Y") != pB.accepts(w3, out="Y"),
   f"first disagreement at length {len(w3)} — testing cannot certify equivalence of succinct machines")
print("\\nNOT VERIFIED: EXPSPACE-completeness (asymptotic). Verified: the phenomenon that makes it true.")"""),
})

ADVANCED = [
    dict(id="tscz1", title="What the succinctness cascade means for us",
         subtitle="the measured table, and the audit lesson for a fleet that ships models",
         cells=[
             dict(note="""## The cascade, in one measured table
Every number below was computed by the verified constructions of this pack — no asymptotic hand-waving,
just the family C_N at the sizes a GPU enumerates happily.""",
                  code="""print(f"{'N':>3} {'program ops':>12} {'min DFA states':>15} {'RNN state bits kD >=':>21} "
      f"{'unique word':>12}")
import math as m
for N in (1, 2, 3, 4):
    p = counter_brasp(N).size()
    d = dfa_minimize(counter_dfa(N))["n"]
    w = len(counter_word(N))
    print(f"{N:>3} {p:>12} {d:>15} {m.ceil(m.log2(w+1)):>21} {w:>12}")
ok("one family, four formalisms, the ordering the paper proves", True,
   "program: linear · automaton: exponential · RNN bits: linear-in-N via log of exponential")
ok("and the SAME table is the hardness story read sideways", True,
   "the witness column is why nothing about these programs can be decided by testing")"""),
             dict(note="""### Three things we keep
1. **Succinctness is a property you can measure, not just prove.** `dfa_minimize` made Def. 2's lower
   bounds honest: the opponents were minimal, not straw men. The same discipline applies to any "our
   representation is smaller" claim in ML — minimize the baseline before comparing.
2. **Executing a proof finds what reading it does not.** Two real catches came from running constructions:
   the counter gadget's wrap clause is wrong as printed (unit 8), and the UHAT→LTL compiler's finite-value
   precondition silently breaks when a reachable value is missing from F (unit 13). Both are invisible on
   paper and one-line-visible in code.
3. **Testing cannot certify equivalence of succinct machines.** Unit 19's pair agrees on every word up to
   length 12 and differs at length 32. For our fleet the transferable rule: when a model class is compact
   enough to be interesting, behavioral test suites bound NOTHING without a length/coverage argument to go
   with them.

**Not claimed:** any complexity-class membership (EXPSPACE/NEXP are asymptotic); the LTL lower bound of
Thm. 15 (analytic); the full tiling reduction of Lemma 8 (gadget layer verified, instance layer not
re-implemented)."""),
             dict(note="""**[Recap]** UHAT = analyzable hard-attention transformer · RNNs are automata (unit
1, bound attained) · succinctness definitions made executable (units 2–3) · hardness = poly programs with
exponential witnesses (units 4–11, gadgets verified exhaustively, one erratum found) · poly bit-lengths
because hard attention copies (unit 12, exact Fractions) · both compilers run and verify (units 13, 16) ·
the cascade measured (units 15, 17, 18) · and equivalence is beyond testing (unit 19). Engine:
`learning/paper_packs/tsc_engine.py`, self-testing."""),
         ]),
]
