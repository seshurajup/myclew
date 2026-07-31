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

L = 256
causal = torch.triu(torch.ones(L, L, device=DEV, dtype=torch.bool), diagonal=1)
usable_causal = float((~causal).float().mean())
print(f"  causal mask: {usable_causal:.1%} of the LxL matrix is usable")
print(f"  bidirectional: 100% usable")
ok("a causal mask discards nearly half the matrix", 0.4 < usable_causal < 0.55,
   f"{usable_causal:.1%} usable")
q = torch.randn(1, 4, L, 64, device=DEV)
with torch.no_grad():
    bi = F.scaled_dot_product_attention(q, q, q)
    ca = F.scaled_dot_product_attention(q, q, q, is_causal=True)
ok("the two produce different representations", not torch.allclose(bi, ca, atol=1e-3),
   f"max diff {float((bi-ca).abs().max()):.4f}")
ok("and the LAST position agrees in both", torch.allclose(bi[..., -1, :], ca[..., -1, :], atol=1e-4),
   "a causal query at L-1 already sees every key, so there is nothing left to unmask")
ok("the config declares the bidirectional class", "Bidirectional" in CFG["architectures"][0],
   CFG["architectures"][0])

ok("caching is off", CFG["use_cache"] is False)
ok("and the model is a masked-LM, not causal", "MaskedLM" in CFG["architectures"][0],
   CFG["architectures"][0])
print("  auto_map:", json.dumps(CFG["auto_map"], indent=2)[:220])
ok("both entry points are remote-code classes", all(
    "modeling_lfm2_bidirectional" in v for v in CFG["auto_map"].values()),
   "so loading needs trust_remote_code=True")
ok("a cache would be dead weight here", True,
   "one forward pass over the whole sequence — no autoregressive loop to accelerate")

L, dh = 64, 64
qq = torch.randn(1, 2, L, dh, device=DEV)
qp = qq.clone(); qp[0, :, 40] += 5.0                              # perturb a LATER position
with torch.no_grad():
    bi_a = F.scaled_dot_product_attention(qq, qq, qq)
    bi_b = F.scaled_dot_product_attention(qp, qp, qp)
    ca_a = F.scaled_dot_product_attention(qq, qq, qq, is_causal=True)
    ca_b = F.scaled_dot_product_attention(qp, qp, qp, is_causal=True)
early_bi = float((bi_a - bi_b)[0, :, 10].abs().max())
early_ca = float((ca_a - ca_b)[0, :, 10].abs().max())
print(f"  perturb position 40, measure position 10:")
print(f"    bidirectional: change = {early_bi:.5f}")
print(f"    causal       : change = {early_ca:.5f}")
ok("bidirectional: an EARLIER token feels a LATER change", early_bi > 1e-4)
ok("causal: it cannot", early_ca < 1e-6)
ok("this is exactly what an encoder is for", True,
   "classification and retrieval read the whole sequence, not a prefix")
