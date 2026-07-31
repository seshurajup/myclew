"""Generates artifacts/ by RUNNING the real tutorial code."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).parent
(HERE / "artifacts").mkdir(exist_ok=True)
ns = {}
exec((HERE / "code.py").read_text().replace("print(", "_ = ("), ns)
vals = ns["fibonacci"](10)
fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#0a0a0a")
ax.plot(range(10), vals, "o-", color="#a0caff", lw=2)
ax.set_title("fibonacci(10) growth", color="#e5e5e5")
ax.tick_params(colors="#a3a3a3")
for sp in ax.spines.values(): sp.set_color("#404040")
fig.tight_layout(); fig.savefig(HERE / "artifacts" / "growth.png", facecolor="#0a0a0a")
print("artifacts ok:", vals)
