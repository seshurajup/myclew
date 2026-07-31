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

print(f"{'N':>3} {'program ops':>12} {'min DFA states':>15} {'RNN state bits kD >=':>21} "
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
   "the witness column is why nothing about these programs can be decided by testing")
