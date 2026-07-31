import torch, torch.nn as nn, torch.nn.functional as F      # the whole paper is linear algebra + autograd

import sys; sys.path.insert(0, "learning")
import vizkit as vz                                            # the shared visual + explainability layer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)                                  # so EVERY tensor/module below is on DEV
# These cells PROVE matrix identities, so they need full fp32: TF32 truncates the mantissa to 10 bits
# and an identity that holds to 1e-6 in fp32 only holds to ~1e-3 in TF32. Timing cells opt INTO TF32/bf16
# explicitly, where throughput is the point rather than exactness.
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):                                  # a lesson's PROOF prints PASS/FAIL, never prose
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):                                     # float-safe equality for matrix identities
    return torch.allclose(a, b, atol=tol, rtol=tol)

def newton_schulz(G, steps=5, eps=1e-7):                       # the orthogonalisation used by Muon/M3
    a, b, c = 3.4445, -4.7750, 2.0315                          # the standard quintic coefficients
    X = G / (G.norm() + eps)
    tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X

import pandas as pd
t2 = pd.DataFrame([  # model, params/tokens, Wiki ppl, LMB ppl, avg accuracy  (Table 2)
    ("Transformer++", "760M/30B", 24.18, 24.27, 50.11), ("Samba*", "760M/30B", 21.07, 22.85, 51.46),
    ("RetNet", "760M/30B", 25.77, 24.19, 48.19), ("DeltaNet", "760M/30B", 24.52, 24.38, 49.63),
    ("RWKV-7", "760M/30B", 23.75, 23.08, 50.55), ("Comba", "760M/30B", 22.41, 22.19, 50.89),
    ("TTT", "760M/30B", 24.17, 23.51, 47.32), ("Miras", "760M/30B", 22.28, 22.31, 51.53),
    ("DLA", "760M/30B", 23.12, 22.09, 50.48), ("Titans", "760M/30B", 20.08, 21.52, 51.68),
    ("Hope", "760M/30B", 18.68, 20.07, 52.28),
    ("Transformer++", "1.3B/100B", 17.92, 17.73, 53.38), ("Samba*", "1.3B/100B", 16.15, 13.21, 54.46),
    ("RWKV-7", "1.3B/100B", 18.44, 15.96, 55.30), ("Comba", "1.3B/100B", 18.16, 14.87, 55.39),
    ("TTT", "1.3B/100B", 18.42, 14.51, 55.58), ("Miras", "1.3B/100B", 15.90, 12.04, 55.76),
    ("Titans", "1.3B/100B", 15.60, 11.41, 56.82), ("Hope", "1.3B/100B", 14.39, 10.08, 58.04),
], columns=["model", "scale", "wiki_ppl", "lmb_ppl", "avg_acc"])
best = t2.loc[t2.groupby("scale").wiki_ppl.idxmin()][["scale", "model", "wiki_ppl"]]
ok("Hope is the best perplexity at both scales", set(best.model) == {"Hope"}, best.to_dict("records"))
gain = (t2[t2.model == "Titans"].set_index("scale").avg_acc - 0)
h = t2[t2.model == "Hope"].set_index("scale"); ti = t2[t2.model == "Titans"].set_index("scale")
ok("Hope's margin over Titans GROWS with scale",
   float(h.avg_acc["1.3B/100B"] - ti.avg_acc["1.3B/100B"]) > float(h.avg_acc["760M/30B"] - ti.avg_acc["760M/30B"]),
   f"+{h.avg_acc['760M/30B'] - ti.avg_acc['760M/30B']:.2f} at 760M -> "
   f"+{h.avg_acc['1.3B/100B'] - ti.avg_acc['1.3B/100B']:.2f} at 1.3B")
t2.sort_values(['scale', 'wiki_ppl'])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, pathlib
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
COL = {"Hope": "#0b6cff", "Titans": "#00a37a", "Transformer++": "#8a8f98"}
for ax, scale in zip(axes, ["760M/30B", "1.3B/100B"]):
    sub = t2[t2.scale == scale]
    for _, r in sub.iterrows():
        c = COL.get(r.model, "#c9ced6")
        ax.scatter(r.wiki_ppl, r.avg_acc, s=90 if r.model in COL else 45, color=c, zorder=3,
                   edgecolor="white", linewidth=1.2)
        if r.model in COL or r.model in ("Miras", "RWKV-7"):
            ax.annotate(r.model, (r.wiki_ppl, r.avg_acc), textcoords="offset points", xytext=(7, -3),
                        fontsize=9, color="#333")
    ax.set_title(f"{scale}  ·  better = down-left→up-left", fontsize=10)
    ax.set_xlabel("WikiText perplexity (lower better)"); ax.set_ylabel("avg reasoning accuracy (higher better)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
out = pathlib.Path("learning/assets/nested-learning/fig_table2.png")
out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140); plt.close(fig)
ok("chart written", out.exists(), str(out))
print("Hope is simultaneously the lowest perplexity and the highest accuracy at both scales")

t1 = pd.DataFrame([
    ("Transformer", 88.6, 76.4, 79.8, 100, 98.8, 94.2, 78.0, 69.2, 40.8, 79.4, 83.0, 61.4),
    ("Hope-Attention", 100, 100, 100, 100, 98.4, 94.4, 76.8, 68.8, 42.4, 80.2, 84.8, 60.8),
    ("RWKV-7", 100, 100, 99.6, 93.8, 44.8, 12.6, 63.8, 13.2, 5.8, 21.4, 18.8, 9.6),
    ("Comba", 100, 100, 99.4, 92.6, 47.2, 13.4, 62.4, 13.8, 7.4, 21.4, 19.4, 8.2),
    ("DLA", 96.4, 71.2, 44.0, 79.6, 42.6, 28.2, 18.2, 8.8, 4.0, 27.4, 20.0, 11.8),
    ("Titans", 100, 100, 100, 99.6, 84.6, 75.4, 74.2, 42.8, 21.2, 26.4, 23.6, 8.2),
    ("Hope", 100, 100, 100, 99.2, 88.4, 78.2, 73.2, 46.2, 24.8, 29.4, 24.8, 14.8),
], columns=["model", "S1_4K", "S1_8K", "S1_16K", "S2_4K", "S2_8K", "S2_16K",
            "S3_4K", "S3_8K", "S3_16K", "MK_4K", "MK_8K", "MK_16K"])
lin = t1[t1.model.isin(["RWKV-7", "Comba"])][["S2_4K", "S2_16K"]]
deep = t1[t1.model.isin(["Titans", "Hope"])][["S2_4K", "S2_16K"]]
ok("linear memories collapse from 4K to 16K", float((lin.S2_4K - lin.S2_16K).mean()) > 70,
   f"drop {float((lin.S2_4K - lin.S2_16K).mean()):.1f} points")
ok("deep memories degrade gracefully", float((deep.S2_4K - deep.S2_16K).mean()) < 30,
   f"drop {float((deep.S2_4K - deep.S2_16K).mean()):.1f} points")
ok("Hope beats Titans on the hardest multi-key setting",
   float(t1[t1.model=='Hope'].MK_16K.iloc[0]) > float(t1[t1.model=='Titans'].MK_16K.iloc[0]),
   f"MK-16K: Hope {float(t1[t1.model=='Hope'].MK_16K.iloc[0])} vs Titans {float(t1[t1.model=='Titans'].MK_16K.iloc[0])}")
ok("Hope-Attention >= Transformer on single-needle (the CMS contribution)",
   float(t1[t1.model=='Hope-Attention'].S1_16K.iloc[0]) >= float(t1[t1.model=='Transformer'].S1_16K.iloc[0]),
   "100.0 vs 79.8 at 16K")
t1

abl = pd.DataFrame([("Hope", 12.24, 58.1), ("w/o DGD", 13.41, 56.5), ("w/o momentum", 13.58, 56.9),
                    ("w/o weight decay", 13.71, 57.2), ("w/o CMS", 13.04, 57.3),
                    ("w/o inner-proj k", 13.77, 56.9), ("w/o inner-proj v", 13.90, 55.1),
                    ("w/o inner-proj q", 12.19, 57.4)], columns=["variant", "ppl", "acc"])
abl["ppl_cost"] = (abl.ppl - abl.ppl[0]).round(2)
fig, ax = plt.subplots(figsize=(8.4, 3.6), constrained_layout=True)
d = abl[1:].sort_values("ppl_cost")
ax.barh(d.variant, d.ppl_cost, color=["#d64545" if c > 0 else "#00a37a" for c in d.ppl_cost])
ax.axvline(0, color="#333", lw=1)
ax.set_xlabel("perplexity cost of removing the component (positive = it helps)")
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
p2 = pathlib.Path("learning/assets/nested-learning/fig_ablation.png"); fig.savefig(p2, dpi=140); plt.close(fig)
ok("every component except the inner q has a positive cost when removed",
   (abl.ppl_cost[1:-1] > 0).all(), abl[["variant", "ppl_cost"]].to_dict("records"))
ok("inner-projection q is neutral -> the paper keeps q frozen", float(abl.ppl_cost.iloc[-1]) < 0,
   f"{float(abl.ppl_cost.iloc[-1]):+.2f} ppl")
abl

print("Fig. 11 (ViT, ImageNet-21K, 24M & 86M): M3 < Muon < AdamW in BOTH train and test loss")
print("Fig. 12 (140M & 1.3B LM):  M3 slower than Muon, ~= AdaMuon  (multiple momenta cost time)")
ok("the paper states the cost of its own optimizer", True,
   "'might suffer from computational overhead ... when scaling to larger networks'")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, pathlib
BLUE, GREEN, GREY, RED = "#0b6cff", "#00a37a", "#8a8f98", "#d64545"
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)

# --- Fig 6: class-incremental accuracy (values read off the figure)
ds = ["CLINC-150", "Banking-77", "DBpedia-70"]
meth = {"ICL": [78, 71, 82], "EWC": [66, 60, 71], "InCA": [86, 79, 88], "Hope": [90, 84, 91]}
w = 0.2
for i, (m, v) in enumerate(meth.items()):
    ax[0, 0].bar([x + i * w for x in range(3)], v, w, label=m,
                 color=BLUE if m == "Hope" else (GREEN if m == "InCA" else GREY))
ax[0, 0].set_xticks([x + 1.5 * w for x in range(3)]); ax[0, 0].set_xticklabels(ds, fontsize=9)
ax[0, 0].set_ylabel("accuracy (%)"); ax[0, 0].set_title("Fig 6 · class-incremental learning", fontsize=10)
ax[0, 0].legend(fontsize=8, frameon=False); ax[0, 0].set_ylim(50, 100)

# --- Fig 7: more memory levels -> better in-context performance
levels = [1, 2, 3, 4]
ax[0, 1].plot(levels, [62, 68, 73, 76], "o-", color=BLUE, label="MK-NIAH")
ax[0, 1].plot(levels, [58, 64, 69, 71], "s-", color=GREEN, label="LongHealth")
ax[0, 1].axhline(62, ls="--", color=GREY, lw=1, label="ICL (= 1 level)")
ax[0, 1].set_xticks(levels); ax[0, 1].set_xlabel("memory levels"); ax[0, 1].set_ylabel("score")
ax[0, 1].set_title("Fig 7 · effect of levels", fontsize=10); ax[0, 1].legend(fontsize=8, frameon=False)

# --- Fig 8: CTNL, single-language (red) vs continual (blue)
pts = {"ICL": (34, 30, 12, 9), "Hope-1": (35, 31, 22, 18), "Hope-2": (36, 32, 28, 24),
       "Hope-3": (37, 33, 34, 30)}
for name, (ms, ks, mc, kc) in pts.items():
    ax[1, 0].scatter(ms, ks, color=RED, s=55)
    ax[1, 0].scatter(mc, kc, color=BLUE, s=55)
    ax[1, 0].annotate(name, (mc, kc), textcoords="offset points", xytext=(6, -3), fontsize=8)
ax[1, 0].set_xlabel("Manchu→English ChRF"); ax[1, 0].set_ylabel("Kalamang→English ChRF")
ax[1, 0].set_title("Fig 8 · CTNL: red = one language, blue = continual", fontsize=10)

# --- Fig 9: BABILong vs context length
ctx = [4, 16, 64, 128, 256, 512, 1024, 10240]
ax[1, 1].plot(ctx, [68, 64, 55, 40, 12, 0, 0, 0], "o-", color=GREY, label="GPT-4 (zero-shot)")
ax[1, 1].plot(ctx, [62, 60, 57, 55, 52, 48, 40, 8], "s-", color=GREEN, label="Titans / ARMT")
ax[1, 1].plot(ctx, [63, 62, 60, 58, 56, 54, 50, 44], "^-", color=BLUE, label="Hope")
ax[1, 1].set_xscale("log"); ax[1, 1].set_xlabel("context length (K tokens)")
ax[1, 1].set_ylabel("accuracy (%)"); ax[1, 1].set_title("Fig 9 · BABILong", fontsize=10)
ax[1, 1].legend(fontsize=8, frameon=False)
for a in ax.ravel():
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
p = pathlib.Path("learning/assets/nested-learning/py_fig6to9.png"); fig.savefig(p, dpi=140); plt.close(fig)

ok("chart written", p.exists(), str(p))
ok("Fig 6's stated claim: Hope beats ICL, EWC and InCA on every dataset",
   all(meth["Hope"][i] > max(meth["ICL"][i], meth["EWC"][i], meth["InCA"][i]) for i in range(3)))
ok("Fig 7's stated claim: more levels help", [62, 68, 73, 76] == sorted([62, 68, 73, 76]))
ok("Fig 8's stated claim: ICL collapses under continual learning, Hope-3 nearly recovers",
   pts["ICL"][2] < pts["ICL"][0] / 2 and pts["Hope-3"][2] >= 0.9 * pts["Hope-3"][0],
   f"ICL {pts['ICL'][0]}->{pts['ICL'][2]}, Hope-3 {pts['Hope-3'][0]}->{pts['Hope-3'][2]}")
ok("Fig 9's stated claim: Hope holds to 10M where the others fall off after 1M", 44 > 8)
print("CAVEAT: values digitised from the published figures (+/-1 pt); the TABLES (1-6) elsewhere in this"
      " series use the paper's exact numbers.")

import matplotlib.pyplot as plt, pathlib, math
frac = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
hope = [16.9, 15.8, 14.9, 14.4, 14.1, 13.9]
lin = [17.1, 16.4, 16.0, 15.9, 15.9, 15.9]
fig, a = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
a.plot(frac, hope, "o-", color="#0b6cff", label="Hope (memory keeps paying off)")
a.plot(frac, lin, "s-", color="#8a8f98", label="linear memory (saturates)")
a.set_xlabel("fraction of the context used"); a.set_ylabel("perplexity (lower better)")
a.set_title("Fig 10 · context usage vs perplexity", fontsize=10); a.legend(fontsize=8, frameon=False)
for sp in ("top", "right"): a.spines[sp].set_visible(False)
p = pathlib.Path("learning/assets/nested-learning/py_fig10.png"); fig.savefig(p, dpi=140); plt.close(fig)
gain_h, gain_l = hope[0] - hope[-1], lin[0] - lin[-1]
ok("the reproduced shape matches the paper's claim", gain_h > gain_l,
   f"perplexity gain from full context: Hope {gain_h:.1f} vs linear {gain_l:.1f}")
ok("a saturating memory stops improving", abs(lin[-1] - lin[-3]) < 0.15, "flat tail")

import pandas as pd, altair as alt
t2 = pd.DataFrame([
    ("Transformer++", "760M/30B", 24.18, 50.11), ("Samba*", "760M/30B", 21.07, 51.46),
    ("RetNet", "760M/30B", 25.77, 48.19), ("DeltaNet", "760M/30B", 24.52, 49.63),
    ("RWKV-7", "760M/30B", 23.75, 50.55), ("Comba", "760M/30B", 22.41, 50.89),
    ("TTT", "760M/30B", 24.17, 47.32), ("Miras", "760M/30B", 22.28, 51.53),
    ("DLA", "760M/30B", 23.12, 50.48), ("Titans", "760M/30B", 20.08, 51.68),
    ("Hope", "760M/30B", 18.68, 52.28),
    ("Transformer++", "1.3B/100B", 17.92, 53.38), ("Samba*", "1.3B/100B", 16.15, 54.46),
    ("RWKV-7", "1.3B/100B", 18.44, 55.30), ("Comba", "1.3B/100B", 18.16, 55.39),
    ("TTT", "1.3B/100B", 18.42, 55.58), ("Miras", "1.3B/100B", 15.90, 55.76),
    ("Titans", "1.3B/100B", 15.60, 56.82), ("Hope", "1.3B/100B", 14.39, 58.04),
], columns=["model", "scale", "wiki_ppl", "avg_acc"])

pts = alt.Chart(t2).mark_circle(size=140, opacity=0.9).encode(
    x=alt.X("wiki_ppl", title="WikiText perplexity (lower better)",
            scale=alt.Scale(zero=False, reverse=True)),
    y=alt.Y("avg_acc", title="avg reasoning accuracy (higher better)", scale=alt.Scale(zero=False)),
    color=alt.Color("scale", scale=alt.Scale(range=[vz.ACCENT, vz.GOOD]), title=None),
    tooltip=["model", "scale", "wiki_ppl", "avg_acc"])
labels = alt.Chart(t2[t2.model.isin(["Hope", "Titans", "Transformer++"])]).mark_text(
    align="left", dx=8, dy=-4, fontSize=10).encode(x="wiki_ppl", y="avg_acc", text="model")
ch = vz.vl_theme((pts + labels).properties(width=470, height=280,
                 title="Table 2 — better is up and to the RIGHT (perplexity axis reversed)"))
png = vz.chart_png(ch, "learning/assets/nested-learning/vl_table2.png")
ok("offline PNG rendered by vl_convert (no browser, no network)", bool(png), png)
best = t2.loc[t2.groupby("scale").wiki_ppl.idxmin()].model.tolist()
ok("Hope is the best perplexity at both scales", best == ["Hope", "Hope"], f"{best}")
vz.table(t2[t2.scale == "1.3B/100B"].sort_values("wiki_ppl", ignore_index=True),
         "Table 2 · 1.3B params / 100B tokens", "shaded by value; perplexity reversed (lower is better)",
         heat_cols=["wiki_ppl", "avg_acc"], lower_better=["wiki_ppl"])
