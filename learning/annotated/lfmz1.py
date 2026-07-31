import json, math, time, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

CFG = json.loads(Path("docs/papers/lfm25-encoders/models/LFM2.5-Encoder-350M.config.json").read_text())
CFG_S = json.loads(Path("docs/papers/lfm25-encoders/models/LFM2.5-Encoder-230M.config.json").read_text())

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def hybrid_worth_it(L, d=1024, k=3, n_layers=16, n_attn=6, crossover_len=None):
    """Should you replace (n_layers - n_attn) attention layers with short convs at length L?

    Returns the MAC saving AND a verdict that is only given when a MEASURED crossover length is supplied,
    because the MAC saving alone would say yes at every length (see the note above).
    """
    a1, c1 = 2 * L * L * d, L * d * k
    hybrid = n_attn * a1 + (n_layers - n_attn) * c1
    saving = 1 - hybrid / (n_layers * a1)
    verdict = "measure your crossover first" if crossover_len is None else (
        "ADOPT" if L >= crossover_len else "too short — overhead dominates")
    return {"L": L, "mac_saving": saving, "per_layer_macs": a1 / c1,
            "asymptote": n_attn / n_layers, "verdict": verdict}

print("  MAC saving is nearly LENGTH-INDEPENDENT — which is why it cannot be the gate:")
for L in (128, 512, 2048, 8192):
    r = hybrid_worth_it(L)
    print(f"    L={L:>5}: saving {r['mac_saving']:>6.2%}  per-layer MACs {r['per_layer_macs']:>8.0f}x")
ok("the MAC saving barely moves with length",
   abs(hybrid_worth_it(8192)["mac_saving"] - hybrid_worth_it(128)["mac_saving"]) < 0.02,
   f"{hybrid_worth_it(128)['mac_saving']:.2%} at 128 vs "
   f"{hybrid_worth_it(8192)['mac_saving']:.2%} at 8192")
ok("because it collapses to n_attn / n_layers", abs(
    hybrid_worth_it(8192)["mac_saving"] - (1 - hybrid_worth_it(8192)["asymptote"])) < 0.01,
   f"asymptote = 1 - {hybrid_worth_it(8192)['asymptote']:.3f}")
ok("so with no measurement the helper REFUSES to give a verdict",
   hybrid_worth_it(512)["verdict"] == "measure your crossover first")

print("\n  with a measured crossover (unit 15 measured it on this GPU at L ~ 256):")
for L in (128, 512, 8192):
    print(f"    L={L:>5}: {hybrid_worth_it(L, crossover_len=256)['verdict']}")
ok("below the measured crossover it says no", "too short" in
   hybrid_worth_it(128, crossover_len=256)["verdict"])
ok("above it, adopt", hybrid_worth_it(8192, crossover_len=256)["verdict"] == "ADOPT")
print("\nOur own long-sequence work (volume-time, 199x100 frames) is exactly the regime where this applies")
print("— and exactly the regime where a 2xT4 has too little parallelism to hide a quadratic term.")
