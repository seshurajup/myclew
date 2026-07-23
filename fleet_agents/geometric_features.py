"""geometric_features — 3D-coordinate → edge-feature primitives from torchmd-net (TensorNet/ET neural
network potentials, torchmdnet/models/utils.py). The reusable core of every equivariant NN-potential is the
featurizer that turns a set of atom/point positions into ROTATION- and TRANSLATION-INVARIANT edge features:
build a neighbor list under a cutoff, take pairwise distances, and expand each distance into a smooth radial
basis (RBF) multiplied by a smooth cutoff so contributions vanish continuously at the boundary. That featurizer
is domain-general — any point cloud (molecules, cells in 3D, particles, geosteering picks) can be fed to a GNN
head with it, and it is what makes the representation equivariant WITHOUT any special layers (distances are
invariants). Ported pure-torch, dropping the compiled torch_cluster neighbor extension for a cdist mask.

Primitives (torch only, CPU-fine, offline-testable):
  • cosine_cutoff(d, upper)              — smooth 0.5(cos(πd/rc)+1) window, →0 at rc (CosineCutoff).
  • ExpNormalSmearing(num_rbf, cutoff)   — PhysNet exp-normal RBF: cutoff(d)·exp(-β(exp(-αd)-μ)²). nn.Module.
  • radius_neighbors(pos, cutoff)        — brute-force (torch.cdist) neighbor edge_index under a cutoff.
  • featurize(pos, cutoff, num_rbf)      — positions → (edge_index, edge_weight, edge_rbf), rotation-invariant.
"""
from __future__ import annotations
import math
from .base import BaseAgent

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


if _HAS_TORCH:
    def cosine_cutoff(d, cutoff_upper: float):
        """Smooth cosine cutoff window (CosineCutoff, cutoff_lower=0): 0.5·(cos(π d/rc)+1) for d<rc, else 0."""
        c = 0.5 * (torch.cos(math.pi * d / cutoff_upper) + 1.0)
        return c * (d < cutoff_upper).to(d.dtype)

    class ExpNormalSmearing(nn.Module):
        """PhysNet exp-normal radial basis (torchmd-net ExpNormalSmearing), byte-faithful to the source:
        means = linspace(exp(-rc), 1, num_rbf), betas = (2/num_rbf·(1-exp(-rc)))^-2, alpha = 5/rc, and
        forward(d) = cutoff(d) · exp(-betas·(exp(-alpha·d) - means)²). Output shape (*d.shape, num_rbf)."""
        def __init__(self, cutoff_upper=5.0, num_rbf=32, cutoff_lower=0.0, trainable=True, dtype=None):
            super().__init__()
            dtype = dtype or torch.float32
            self.cutoff_upper = float(cutoff_upper); self.cutoff_lower = float(cutoff_lower)
            self.num_rbf = int(num_rbf)
            self.alpha = 5.0 / (cutoff_upper - cutoff_lower)
            start = math.exp(-(cutoff_upper - cutoff_lower))
            means = torch.linspace(start, 1, num_rbf, dtype=dtype)
            betas = torch.full((num_rbf,), (2.0 / num_rbf * (1 - start)) ** -2, dtype=dtype)
            if trainable:
                self.means = nn.Parameter(means); self.betas = nn.Parameter(betas)
            else:
                self.register_buffer("means", means); self.register_buffer("betas", betas)

        def forward(self, dist):
            d = dist.unsqueeze(-1)
            return cosine_cutoff(d, self.cutoff_upper) * torch.exp(
                -self.betas * (torch.exp(self.alpha * (-d + self.cutoff_lower)) - self.means) ** 2)

    def radius_neighbors(pos, cutoff: float, self_loops: bool = False):
        """Brute-force neighbor list: all ordered pairs (i,j) with 0<||pos_i-pos_j||<cutoff. pos: (N,3).
        Returns (edge_index (2,E), edge_weight (E,)). Replaces torchmd-net's compiled OptimizedDistance."""
        D = torch.cdist(pos, pos)                                    # (N,N) pairwise distances
        D = 0.5 * (D + D.t())                                        # symmetrize (cdist fp is slightly asymmetric)
        mask = D < cutoff
        if not self_loops:
            mask = mask & ~torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)  # drop diagonal by index
            #   (fp32 self-distances aren't exactly 0, so a `D>0` filter would leak self-loops)
        idx = mask.nonzero(as_tuple=False).t().contiguous()          # (2,E): rows i, cols j
        return idx, D[mask]

    def featurize(pos, cutoff: float = 5.0, num_rbf: int = 32, smearing: "ExpNormalSmearing" = None):
        """Positions (N,3) → (edge_index (2,E), edge_weight (E,), edge_rbf (E,num_rbf)). Rotation- and
        translation-invariant (depends only on pairwise distances). Reusable head input for any point-cloud GNN."""
        edge_index, edge_weight = radius_neighbors(pos, cutoff)
        sm = smearing or ExpNormalSmearing(cutoff_upper=cutoff, num_rbf=num_rbf, trainable=False)
        return edge_index, edge_weight, sm(edge_weight)
else:  # pragma: no cover
    cosine_cutoff = ExpNormalSmearing = radius_neighbors = featurize = None


# ---------------------------------------------------------------- agent
class GeometricFeatures(BaseAgent):
    name = "geometric-features"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        if not _HAS_TORCH:
            return self.escalate(q, "leader", "geometric-features needs torch")
        s = self.spec(q)
        torch.manual_seed(int(s.get("seed", 0)))
        N = int(s.get("n_points", 40)); cutoff = float(s.get("cutoff", 3.0)); nrbf = int(s.get("num_rbf", 32))
        pos = torch.randn(N, 3) * 2.0
        ei, ew, rbf = featurize(pos, cutoff=cutoff, num_rbf=nrbf)
        # rotation/translation invariance: the full pairwise-distance matrix is size-stable and exactly
        # invariant under rigid motion (unlike the cutoff neighbor set, which can flip a boundary pair).
        Q, _ = torch.linalg.qr(torch.randn(3, 3)); Q = Q * torch.sign(torch.det(Q))
        pos2 = pos @ Q.T + torch.tensor([5.0, -2.0, 1.0])
        pd, pd2 = pos.double(), pos2.double()                        # fp64 so we measure the analytic invariance
        inv_err = float((torch.cdist(pd, pd) - torch.cdist(pd2, pd2)).abs().max())
        msg = (f"geometric-features: {N} points, cutoff={cutoff} → {ei.shape[1]} edges, {nrbf}-dim exp-normal RBF "
               f"(range [{rbf.min():.3f},{rbf.max():.3f}]); rotation+translation invariance err={inv_err:.2e}. "
               f"Point-cloud→edge-feature featurizer for any equivariant GNN head (torchmd-net PhysNet RBF)")
        self.log(msg, kind="finding",
                 recommendation="feed featurize(coords) to a message-passing head for 3D point clouds "
                                "(molecules/cells/particles); distances are invariants → equivariance for free")
        return self.done({"n_edges": int(ei.shape[1]), "rbf_dim": nrbf, "invariance_err": inv_err}, msg)


_AGENT = GeometricFeatures()


def run_geomfeat(q, worker):
    return _AGENT.run(q, worker)
