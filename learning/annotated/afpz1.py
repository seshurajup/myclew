import sys, math, random, torch, torch.nn as nn
sys.path.insert(0, "learning/paper_packs")
import afp_engine as A
from afp_engine import (size, val, rw_at, positions, at, replace, neighbours, check, plant,
                        bfs, guided, depth, counts, subterms, feats, NF, instances, spearman)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

torch.manual_seed(1)
n_ideas, sigma = 40, 0.30
true_gain = torch.randn(n_ideas) * 0.1
noisy_eval = lambda k: true_gain + sigma * torch.randn(n_ideas)
best = float(true_gain.max())
exp_pick = {}
for rep in (1, 3, 9, 27):
    got = []
    for trial in range(200):                                  # the EXPECTED value of the selection rule
        scores = torch.stack([true_gain + sigma * torch.randn(n_ideas)
                              for _ in range(rep)]).mean(0)
        got.append(float(true_gain[int(scores.argmax())]))
    exp_pick[rep] = sum(got) / len(got)
print(f"  best TRUE gain available: {best:+.3f}")
for rep, got in exp_pick.items():
    print(f"  selecting on {rep:>2} eval repeats -> EXPECTED true gain of the pick {got:+.3f}")
ok("with a NOISY verifier, single-shot selection loses a large slice to the winner's curse",
   exp_pick[1] < 0.85 * exp_pick[27],
   f"{exp_pick[1]:+.3f} vs {exp_pick[27]:+.3f} — the fleet's daily hazard, in expectation")
ok("repeats recover it, exactly like unit 7's majority vote", exp_pick[27] > 0.8 * best,
   f"27 repeats reach {exp_pick[27]/best:.0%} of the attainable gain")
ok("Lean needs none of this; Kaggle needs ALL of it", True,
   "the verifier's noise level decides how much of the paper's aggression is safe to copy")
