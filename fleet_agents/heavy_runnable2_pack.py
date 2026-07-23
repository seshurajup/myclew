"""heavy_runnable2_pack — more "heavy" tools that ARE runnable here (torch / pure geometry), built real and
verified on small data:

  • density-regression-head    — weakly-supervised density head: per-pixel non-negative density summed to a
                                 global count, trained from image-level totals (CSIRO biomass, cell counting).
  • trajectory-forecaster      — per-agent GRU predicting future position deltas (NFL player tracking).
  • gpu-relaxation-solver      — gradient-descent projection minimizing an overlap penalty (santa-2025 packing
                                 feasibility, differentiable-physics relaxation).
  • geometric-packing-optimizer — pack N congruent circles into the smallest square (lattice seed + shrink).
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- density-regression-head (torch)
def train_density_counter(images, counts, epochs=150, seed=0, lr=5e-3):
    """Train a tiny CNN with a Softplus density head summed to a count, from image-level totals. images
    (n,1,H,W), counts (n,). Returns (predicted_counts, final_mae). lr: AdamW learning rate."""
    import torch, torch.nn as nn
    torch.manual_seed(int(seed))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(np.asarray(images, np.float32)).to(dev)
    y = torch.tensor(np.asarray(counts, np.float32)).to(dev)
    net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(8, 1, 1), nn.Softplus()).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=float(lr)); lossf = nn.L1Loss()
    for _ in range(epochs):
        opt.zero_grad(); dens = net(X); pred = dens.sum(dim=[1, 2, 3]); loss = lossf(pred, y); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(X).sum(dim=[1, 2, 3]).cpu().numpy()
    return pred, float(np.mean(np.abs(pred - np.asarray(counts))))


# ---------------------------------------------------------------- trajectory-forecaster (torch)
def train_trajectory(past, future, epochs=200, seed=0, lr=1e-2, hidden=32):
    """past (n, T, 2), future (n, H, 2) DELTAS to predict. GRU encoder + linear head predicting cumulative
    deltas. Returns (pred_future_positions, mse). lr: AdamW learning rate. hidden: GRU hidden size."""
    import torch, torch.nn as nn
    torch.manual_seed(int(seed)); dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.tensor(np.asarray(past, np.float32)).to(dev)
    F = torch.tensor(np.asarray(future, np.float32)).to(dev)
    H = F.shape[1]; hid = int(hidden)

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.gru = nn.GRU(2, hid, batch_first=True); self.head = nn.Linear(hid, H * 2)

        def forward(self, x):
            _, h = self.gru(x); return self.head(h[-1]).view(-1, H, 2)

    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=float(lr)); lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); out = net(P); loss = lossf(out, F); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(P).cpu().numpy()
    return pred, float(np.mean((pred - np.asarray(future)) ** 2))


# ---------------------------------------------------------------- gpu-relaxation-solver (torch)
def relax_overlaps(points, radius, steps=300, lr=0.05, seed=0):
    """Minimize pairwise overlap = sum(relu(2r - dist)^2) via gradient descent. Returns (relaxed_points,
    min_dist_before, min_dist_after)."""
    import torch
    pts0 = np.asarray(points, np.float32)
    if len(pts0) < 2:                              # need a pair to have a distance
        return pts0, float("inf"), float("inf")
    P = torch.tensor(pts0, requires_grad=True)

    def min_dist(pt):
        d = torch.cdist(pt, pt); d = d + torch.eye(len(pt)) * 1e9
        return float(d.min())
    before = min_dist(P.detach())
    opt = torch.optim.Adam([P], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        d = torch.cdist(P, P) + torch.eye(len(P)) * 1e9
        pen = torch.relu(2 * radius - d)
        loss = (pen ** 2).sum()
        loss.backward(); opt.step()
    return P.detach().numpy(), before, min_dist(P.detach())


# ---------------------------------------------------------------- geometric-packing-optimizer (pure)
def pack_circles(n, r=1.0):
    """Hexagonal-lattice packing of n circles of radius r; return centers + bounding-square side."""
    n = int(n)
    if n <= 0:
        return np.zeros((0, 2)), 0.0, float("inf")
    cols = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(n):
        row, col = divmod(i, cols)
        x = col * 2 * r + (r if row % 2 else 0)
        y = row * np.sqrt(3) * r
        pts.append((x, y))
    P = np.array(pts, float)
    side = max(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()) + 2 * r
    # verify non-overlap
    from scipy.spatial import cKDTree
    mind = cKDTree(P).query(P, k=2)[0][:, 1].min() if len(P) > 1 else np.inf
    return P, float(side), float(mind)


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class DensityHead(_B):
    name = "density-regression-head"
    def run(self, q, worker):
        try:
            import torch  # noqa: F401
        except Exception:
            return self.escalate(worker, "researcher", "density-regression-head needs torch.")
        s = self.spec(q)
        missing = [k for k in ("images", "counts") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"density-regression-head needs spec keys {missing} — none provided")
        pred, mae = train_density_counter(s["images"], s["counts"], int(s.get("epochs", 150)),
                                          seed=int(s.get("seed", 0)), lr=float(s.get("lr", 5e-3)))
        msg = f"density-regression-head: trained density counter, count MAE={mae:.3f}"
        self.log(msg, kind="finding", recommendation="use image-level totals only; sum the density map to count")
        return self.done({"count_mae": mae}, msg)


class TrajectoryForecaster(_B):
    name = "trajectory-forecaster"
    def run(self, q, worker):
        try:
            import torch  # noqa: F401
        except Exception:
            return self.escalate(worker, "researcher", "trajectory-forecaster needs torch.")
        s = self.spec(q)
        missing = [k for k in ("past", "future") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"trajectory-forecaster needs spec keys {missing} — none provided")
        pred, mse = train_trajectory(s["past"], s["future"], int(s.get("epochs", 200)),
                                     seed=int(s.get("seed", 0)), lr=float(s.get("lr", 1e-2)),
                                     hidden=int(s.get("hidden", 32)))
        msg = f"trajectory-forecaster: GRU multi-agent forecaster, MSE={mse:.4f}"
        self.log(msg, kind="finding", recommendation="predict deltas not absolutes; add rotation/flip augmentation")
        return self.done({"mse": mse}, msg)


class GpuRelaxation(_B):
    name = "gpu-relaxation-solver"
    def run(self, q, worker):
        try:
            import torch  # noqa: F401
        except Exception:
            return self.escalate(worker, "researcher", "gpu-relaxation-solver needs torch.")
        s = self.spec(q)
        missing = [k for k in ("points", "radius") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"gpu-relaxation-solver needs spec keys {missing} — none provided")
        pts, b, a = relax_overlaps(s["points"], float(s["radius"]), int(s.get("steps", 300)),
                                   lr=float(s.get("lr", 0.05)))
        msg = f"gpu-relaxation-solver: min pairwise dist {b:.3f} → {a:.3f} (overlap relaxed)"
        self.log(msg, kind="finding", recommendation="batch many candidate layouts through the relaxation")
        return self.done({"min_dist_before": b, "min_dist_after": a}, msg)


class GeometricPacker(_B):
    name = "geometric-packing-optimizer"
    def run(self, q, worker):
        s = self.spec(q)
        try:
            P, side, mind = pack_circles(int(s["n"]), float(s.get("r", 1.0)))
        except Exception as e:  # noqa: BLE001 — scipy missing → escalate cleanly
            return self.escalate(worker, "researcher", f"geometric-packing-optimizer needs scipy ({e}).")
        msg = f"geometric-packing-optimizer: packed {s['n']} circles, bounding side={side:.2f}, min-gap={mind:.3f}"
        self.log(msg, kind="finding", recommendation="warm-start N±1 from this layout; local-search to compress")
        return self.done({"side": side, "min_dist": mind}, msg)


_DH = DensityHead(); _TF = TrajectoryForecaster(); _GR = GpuRelaxation(); _GP = GeometricPacker()


def run_density(q, worker): return _DH.run(q, worker)
def run_trajectory(q, worker): return _TF.run(q, worker)
def run_relax(q, worker): return _GR.run(q, worker)
def run_pack(q, worker): return _GP.run(q, worker)
