"""Generates artifacts/ by RUNNING the real attention code."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
HERE = pathlib.Path(__file__).parent
(HERE / "artifacts").mkdir(exist_ok=True)
fig = plt.figure(figsize=(7, 2.2), dpi=160)
fig.patch.set_facecolor("#0a0a0a")
fig.text(0.5, 0.55, r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$",
         ha="center", va="center", fontsize=26, color="#e5e5e5")
fig.savefig(HERE / "artifacts" / "formula.png", facecolor="#0a0a0a", bbox_inches="tight")
torch.manual_seed(0)
ns = {}
exec((HERE / "code.py").read_text().replace("print(", "_ = ("), ns)
w = torch.softmax((ns["Q"] @ ns["K"].transpose(-2, -1)) / ns["Q"].size(-1)**0.5, dim=-1)
fig, ax = plt.subplots(figsize=(4.6, 4), dpi=160)
fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#0a0a0a")
im = ax.imshow(w.detach(), cmap="magma")
ax.set_title("softmax attention weights (6 tokens)", color="#e5e5e5", fontsize=11)
ax.set_xlabel("key", color="#a3a3a3"); ax.set_ylabel("query", color="#a3a3a3")
ax.tick_params(colors="#737373")
cb = fig.colorbar(im); cb.ax.tick_params(colors="#737373")
fig.tight_layout(); fig.savefig(HERE / "artifacts" / "weights.png", facecolor="#0a0a0a")
print("artifacts ok, row0 sum:", float(w[0].sum()))
