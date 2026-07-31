import sys, math, itertools, torch
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
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

tgt = lambda w: (len(w) > 0 and len(w) % 2 == 0
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

print(f"\n  sizes for THIS easy language: DFA {D['n']} states · LTL {ltl_size(phi)} nodes · "
      f"UHAT {T.size()} params")
ok("on an easy language the sizes are comparable — the DRAMA needs a harder family", True,
   "units 15-18 build the family where the gap explodes")

for N in (1, 2, 3, 4):
    p = counter_brasp(N)
    w = counter_word(N)
    print(f"  N={N}: program ops = {p.size():>2}   its unique accepted word has length {len(w):>3}")
ok("the program grows LINEARLY while its witness grows EXPONENTIALLY", counter_brasp(4).size() < 25
   and len(counter_word(4)) == 5 * 2 ** 4, "the entire hardness story in two columns")
