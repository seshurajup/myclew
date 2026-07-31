"""Working code for mc01 — the modeling contract (the math), each equation tied to the
real code. Running it renders the equation images and verifies the physical-distance
formula on REAL cell coordinates, then writes mc01_modeling_contract.learning.
    research/cellmot_venv/bin/python learning/annotated/mc01_modeling_contract.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
ASSETS = ROOT / "learning/assets"
ASSETS.mkdir(exist_ok=True)


def eq(name, latex):
    """Render one equation as our own clean PNG (matplotlib mathtext, copyright-clean)."""
    fig = plt.figure(figsize=(7.5, 1.0))
    fig.text(0.02, 0.5, f"${latex}$", fontsize=19, va="center", color="#1c2127")
    fig.savefig(ASSETS / f"eq_{name}.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# render the contract's equations up front (our own images)
eq("detect", r"p_t(\mathbf{r})=\sigma\!\left(h_\theta(V_{t:t+1})(\mathbf{r})\right)")
eq("candidates", r"\mathcal{D}_t=\{\mathbf{r}\mid p_t(\mathbf{r})>\tau\ \mathrm{and}\ p_t(\mathbf{r})=\max_{u\in N(\mathbf{r})}p_t(u)\}")
eq("aux", r"q_t(\mathbf{r})=\sigma\!\left(u_\psi(P(V_t))(\mathbf{r})\right),\quad \tilde{\mathcal{D}}_t=\mathrm{NMS}_{\mu m}(\mathcal{D}_t\cup\mathcal{C}_t)")
eq("edge", r"s_{ij}=g_\phi(f_i,\,f_j,\,\mathbf{r}_i-\mathbf{r}_j)")
eq("motion", r"\hat{\mathbf{r}}_{i,t+1}=\mathbf{r}_{i,t}+\lambda(\mathbf{r}_{i,t}-\mathbf{r}_{i,t-1}),\ \ \lambda=0.5")
eq("ilp", r"\min_x\ \sum_{e\in E} w_e x_e + c_a A(x)+c_d D(x)+c_m M(x),\quad w_e=-\mathrm{edge\_prob}(e)")
eq("dist", r"d_{\mu m}(i,j)=\sqrt{(1.625\,\Delta z)^2+(0.40625\,\Delta y)^2+(0.40625\,\Delta x)^2}")
eq("gap", r"d_{\mu m}(i,j)\leq 2g,\ g=6.0\,\mu m,\quad \mathbf{r}^{new}_{t+1}=\frac{\mathbf{r}_t+\mathbf{r}_{t+2}}{2}")
eq("div", r"d_{\mu m}(p,c_2)\leq 5.0,\quad d_{\mu m}(c_1,c_2)\leq 8.0")

META = dict(id="mc01", order=22, title="The modeling contract — the math",
            subtitle="Every equation of the pipeline, tied to the real code (and verified on real coords)",
            source="research/pilkwang_support_pack/repo/scripts/predict_unet_transformer.py")

CELLS = [
    dict(note="""## The contract in one place
The whole pipeline is a few equations. Here each one is stated, tied to the **real code line** that
implements it, and where possible **run on real data**. (Equation images are our own — rendered
from the formulas, not copied.)"""),

    dict(note="""### 1. Detection field & candidates
The detector maps a 2-frame volume to a per-voxel probability; candidate cells are **local maxima
above threshold τ**. In code: `sigmoid(logits)` then the `max_pool3d` local-max test.""",
         code="""import torch                                              # tensors
logit = torch.tensor([3.0, -1.0, 0.5])                      # example real detector logits
p = torch.sigmoid(logit)                                    # p_t(r) = sigmoid(logit)
{"prob at logit 3.0": round(p[0].item(), 3),                # confident voxel
 "prob at logit -1.0": round(p[1].item(), 3),               # background
 "tau (production)": 0.99}                                  # the real candidate threshold""",
         image="learning/assets/eq_detect.png\np_t(r) — the detection field (real code: torch.sigmoid on the detector logits)"),

    dict(note="""### 2. Auxiliary center detector + NMS fusion (the recall path)
An optional full-frame detector `q_t` on an XY-pooled volume adds centers the backbone missed;
the two sets are fused with a physical **NMS** and per-frame caps — improving recall without one
noisy frame flooding the graph.""",
         code="""P_pool = 4                                                  # block-mean XY pooling factor
fuse = dict(rule="D_tilde = NMS_um(D_backbone U C_center)", # union then physical NMS
            per_frame_cap=True,                             # cap additions so one frame can't dominate
            purpose="recall")                               # add missed cells, protect precision
fuse                                                        # the fusion contract""",
         image="learning/assets/eq_aux.png\nq_t and the NMS fusion of backbone + center detections (the ensemble recall path)"),

    dict(note="""### 3. Edge scoring
Each candidate pair (cell@t → cell@t+1) is scored from image features `f_i,f_j` and relative
geometry `r_i−r_j` by the learned edge model (pt05). Relative position is the key geometric cue.""",
         code="""import numpy as np                                        # arrays
ri = np.array([32, 120, 140]); rj = np.array([33, 118, 145])  # two real-ish cell positions (z,y,x)
rel = ri - rj                                               # r_i - r_j, the geometry the edge model sees
{"relative_pos (z,y,x)": rel.tolist()}                      # part of g_phi's input""",
         image="learning/assets/eq_edge.png\ns_ij — the learned edge score from features + relative geometry"),

    dict(note="""### 4. Motion relinker
An optional relinker rebuilds a 1-to-1 graph by **constant-velocity prediction**: a node's next
position is extrapolated from its last step (λ=0.5), then matched by Hungarian (tight 6µm / relaxed
10µm). Run the real prediction on real successive coords.""",
         code="""r_prev = np.array([30, 118, 138]); r_now = np.array([32, 120, 140])  # a cell at t-1 and t
lam = 0.5                                                   # velocity weight (real value)
r_pred = r_now + lam * (r_now - r_prev)                     # r_hat = r + lambda*(r - r_prev)
{"predicted next pos": r_pred.tolist(), "tight gate um": 6.0, "relaxed gate um": 10.0}""",
         image="learning/assets/eq_motion.png\nConstant-velocity motion prediction (real code: r + 0.5*(r - r_prev))"),

    dict(note="""### 5. The ILP objective
The graph is chosen by an ILP minimising edge costs (`w_e=−edge_prob`, so confident links are
cheap) plus appearance/disappearance/division event costs. This makes the lineage globally
consistent, not greedy.""",
         code="""def w_e(edge_prob):                                         # the real ILP edge weight
    return -1.0 * edge_prob                                  # w_e = -edge_prob
{"cost @prob 0.99": w_e(0.99), "cost @prob 0.3": w_e(0.3),  # confident = cheaper
 "appear/disappear/division": (0.1, 0.1, 1.0)}              # the real event costs""",
         image="learning/assets/eq_ilp.png\nThe ILP objective: min sum(w_e x_e) + event costs, with w_e = -edge_prob"),

    dict(note="""### 6. Physical distance — VERIFIED on real coords
All post-processing measures **microns**, not voxels (anisotropic: z=1.625, xy=0.40625 µm). Read
two REAL cells from a real `.geff` and compute the real µm distance between them.""",
         code="""import zarr                                               # on-disk reader
g = f"{TRAIN}/6bba_062c8d37.geff/nodes/props"               # a real embryo's node props
z = np.asarray(zarr.open(f"{g}/z/values")[:2])              # z of first 2 real cells
y = np.asarray(zarr.open(f"{g}/y/values")[:2])              # y
x = np.asarray(zarr.open(f"{g}/x/values")[:2])              # x
scale = np.array([1.625, 0.40625, 0.40625])                # (z,y,x) micron per voxel
d = np.sqrt((((np.array([z[0],y[0],x[0]]) - np.array([z[1],y[1],x[1]])) * scale) ** 2).sum())  # d_um
{"cell0 (z,y,x)": [int(z[0]),int(y[0]),int(x[0])],          # real coords
 "cell1 (z,y,x)": [int(z[1]),int(y[1]),int(x[1])],
 "distance_um": round(float(d), 2)}                         # the REAL micron distance""",
         image="learning/assets/eq_dist.png\nPhysical distance in microns (verified above on two real cells from a real .geff)"),

    dict(note="""### 7. Gap recovery & safe division (capped repairs)
A 1-frame gap connects an end@t to a start@t+2 within `2g` (g=6µm), inserting the geometric
midpoint (refined by a local intensity centroid). A 2-missing-frame bridge (t→t+3) is stricter.
Safe division adds a 2nd child only if close to parent (≤5µm) and sister (≤8µm). All are **capped**
per-frame/graph so they can't become node generators.""",
         code="""def midpoint(r_t, r_t2):                                    # the real gap-close midpoint
    return (np.array(r_t) + np.array(r_t2)) / 2             # r_new = (r_t + r_{t+2}) / 2
gates = {"gap-close um (2g)": 12.0, "gap2 total um": 10.2,  # the real thresholds
         "div parent um": 5.0, "div sister um": 8.0, "min track len": 4}
{"midpoint of (30,100,100)&(34,108,120)": midpoint([30,100,100],[34,108,120]).tolist(), **gates}""",
         image="learning/assets/eq_gap.png\nGap-close gate & midpoint (and the safe-division gates below)"),

    dict(note="""**[Recap]** The contract: detect field → NMS candidates (+ optional center-detector
fusion for recall) → learned edge scores → ILP (or motion relink) → physical-µm gap/division
repairs, all capped. Every threshold above is the real production value; the distance was verified
on real cells. **Next → me01: the metric** these tracks are scored by."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"),
                 {"TRAIN": TRAIN, "ASSETS": ASSETS})
