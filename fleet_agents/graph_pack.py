"""graph_pack — the GRAPH/GNN modality pack, GROUNDED in the top-5 solutions of 3 real graph competitions
(predict-ai-model-runtime = GNN over TPU computation graphs, stanford-covid-vaccine = RNA-as-graph,
champs-scalar-coupling = molecular MPNN). Mined via the fleet's `gm-writeup-mine` agent; distilled recurring
techniques + full provenance in docs/graph_pack_grounded.md.

The fleet already had the two GRAPH agents that are LINK-PREDICTION specific (biohub cell tracking) — NOT
rebuilt, only referenced:
  • gnn-link-train  — trains division+flow heads (candidate-edge classify / affinity regress) for the biohub linker
  • gnn-probe       — cheap probe: does neighbourhood context beat pairwise geometry for the biohub linker?

What the fleet was genuinely MISSING (this pack), each recurring across the mined winners:
  • graph-message-passing — a general MPNN forward: aggregator (mean/max/sum/attention) over an edge_index
        with optional edge features, N layers with residual + norm + SAGE root transform, directional +
        DropEdge options; node-classification / graph-regression heads. GraphSAGE was the repeated winner.
  • graph-feature-extractor — deterministic node features (degree/clustering/k-hop counts) + structural/
        positional encodings (Laplacian eigvec PE, random-walk PE) + edge & global features — the FE + PE
        the winners hand-craft (covid distance-matrix PE, champs angles/charges, runtime opcode/degree).
  • graph-readout — graph-level pooling (mean/max/sum/attention/Set2Set) mapping node embeddings + batch
        index → one vector per graph (runtime 1st global-mean-pool, runtime 2nd sum-reduce).

Pure torch/numpy, GPU-FIRST (every tensor op runs on CUDA when available; CPU fallback only if no CUDA).
Message passing is implemented with torch `index_add`/`scatter_reduce` — NO torch_geometric dependency (it is
not installed in our venv). If torch_geometric IS importable it may be used as an optional fast path, but it
is never a hard requirement. No numpy/torch version is touched. Data-wise tests: test_fleet_agents/graph_pack_test.py.
"""
from __future__ import annotations
from .base import BaseAgent

_NODE_CAP = 4000   # dense O(N^2) features (clustering / PE) are guarded to small graphs


def _device(spec):
    import torch
    d = (spec or {}).get("device")
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pyg_available():
    try:
        import torch_geometric  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ════════════════════════════════════════════════════════════ scatter primitives (pure torch, no PyG)
def scatter_sum(src, index, dim_size):
    """Segment-sum of `src` (E, ...) into (dim_size, ...) at rows `index` (E,) — via index_add (no PyG)."""
    import torch
    out = src.new_zeros((int(dim_size),) + tuple(src.shape[1:]))
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_mean(src, index, dim_size):
    """Segment-mean (sum / count). Empty segments → 0."""
    import torch
    s = scatter_sum(src, index, dim_size)
    ones = src.new_ones(src.shape[0])
    cnt = scatter_sum(ones, index, dim_size).clamp_min(1.0)
    return s / cnt.view(-1, *([1] * (s.dim() - 1)))


def scatter_max(src, index, dim_size):
    """Segment-max. Empty segments → 0 (the −inf init is zeroed so downstream stays finite)."""
    import torch
    dim_size = int(dim_size)
    out = src.new_full((dim_size,) + tuple(src.shape[1:]), float("-inf"))
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    out[out == float("-inf")] = 0.0
    return out


def _scatter(src, index, dim_size, aggr):
    if aggr == "sum":
        return scatter_sum(src, index, dim_size)
    if aggr == "max":
        return scatter_max(src, index, dim_size)
    return scatter_mean(src, index, dim_size)


def segment_softmax(scores, index, dim_size):
    """Softmax of `scores` (E,) WITHIN each destination segment `index` (E,) — the GAT attention normaliser."""
    import torch
    m = scatter_max(scores, index, dim_size)          # per-segment max (numerical stability)
    s = scores - m[index]
    e = s.exp()
    denom = scatter_sum(e, index, dim_size)[index] + 1e-16
    return e / denom


def add_self_loops(edge_index, num_nodes):
    """Append i→i self edges (SAGE/GCN self term via message passing). Returns (2, E+N)."""
    import torch
    dev = edge_index.device
    loop = torch.arange(int(num_nodes), device=dev).view(1, -1).repeat(2, 1)
    return torch.cat([edge_index, loop], dim=1)


def propagate(x, edge_index, num_nodes=None, aggr="mean", add_self=False):
    """PARAMETER-FREE one-hop message passing: each node ← aggr of its in-neighbours' features (identity
    messages). The raw propagation operator the MPNN layer builds on; used to prove a planted signal reaches
    neighbours after 1 hop. `edge_index` = (2, E) [src; dst]; message flows src→dst."""
    import torch
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else x.shape[0]
    ei = add_self_loops(edge_index, num_nodes) if add_self else edge_index
    src, dst = ei[0], ei[1]
    return _scatter(x[src], dst, num_nodes, aggr)


# ════════════════════════════════════════════════════════════ graph augmentation (DropEdge / drop-node)
def drop_edge(edge_index, p=0.5, seed=None, edge_attr=None):
    """DropEdge (grounded: runtime-5th) — randomly remove a fraction `p` of edges (out-of-place). Returns the
    kept `edge_index` (and edge_attr if given). Autograd/DataLoader-safe; p=0 is identity."""
    import torch
    if p <= 0 or edge_index.shape[1] == 0:
        return (edge_index, edge_attr) if edge_attr is not None else edge_index
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    keep = torch.rand(edge_index.shape[1], generator=g) >= p
    keep = keep.to(edge_index.device)
    ei = edge_index[:, keep]
    if edge_attr is not None:
        return ei, edge_attr[keep]
    return ei


def drop_node(x, edge_index, p=0.1, seed=None):
    """Node cutout (grounded: champs-1st cutout / champs-2nd knock-out) — zero a fraction `p` of node feature
    rows AND drop edges incident to them (out-of-place). Returns (x_masked, edge_index_masked)."""
    import torch
    if p <= 0:
        return x, edge_index
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    drop = (torch.rand(x.shape[0], generator=g) < p).to(x.device)
    x2 = x.clone()
    x2[drop] = 0.0
    if edge_index.shape[1]:
        src, dst = edge_index[0], edge_index[1]
        keep = ~(drop[src] | drop[dst])
        edge_index = edge_index[:, keep]
    return x2, edge_index


# ════════════════════════════════════════════════════════════ 1. message-passing MPNN
def _mpnn_layer(torch, nn):
    class MPNNLayer(nn.Module):
        """One general message-passing layer (pure torch, scatter-based). GraphSAGE by default:
            out = W_self·x + aggr_{j∈N(i)}( W_neigh·x_j [+ W_edge·e_ij] )
        aggr ∈ {mean,max,sum,attention}. attention = GAT-style segment-softmax over incoming edges.
        `directional=True` runs src→dst and dst→src with separate neighbour weights and concatenates
        (grounded: runtime-5th bidirectional conv)."""
        def __init__(self, in_dim, out_dim, edge_dim=0, aggr="mean", directional=False, heads=1):
            super().__init__()
            self.aggr = aggr
            self.directional = directional
            self.heads = heads
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_neigh = nn.Linear(in_dim, out_dim)
            self.lin_neigh_rev = nn.Linear(in_dim, out_dim) if directional else None
            self.lin_edge = nn.Linear(edge_dim, out_dim) if edge_dim else None
            if aggr == "attention":
                self.att = nn.Linear(2 * out_dim + (out_dim if edge_dim else 0), 1)
                self.leaky = nn.LeakyReLU(0.2)

        def _agg(self, x, edge_index, edge_attr, lin_neigh, N):
            src, dst = edge_index[0], edge_index[1]
            m = lin_neigh(x)[src]
            if self.lin_edge is not None and edge_attr is not None:
                m = m + self.lin_edge(edge_attr)
            if self.aggr == "attention":
                h_dst = lin_neigh(x)[dst]
                cat = [h_dst, m] + ([self.lin_edge(edge_attr)] if (self.lin_edge is not None and edge_attr is not None) else [])
                a = self.leaky(self.att(torch.cat(cat, dim=-1))).squeeze(-1)
                alpha = segment_softmax(a, dst, N).unsqueeze(-1)
                return scatter_sum(m * alpha, dst, N)
            return _scatter(m, dst, N, self.aggr)

        def forward(self, x, edge_index, edge_attr=None):
            N = x.shape[0]
            out = self.lin_self(x) + self._agg(x, edge_index, edge_attr, self.lin_neigh, N)
            if self.directional:
                rev = edge_index.flip(0)
                out = out + self._agg(x, rev, edge_attr, self.lin_neigh_rev, N)
            return out
    return MPNNLayer


def build_mpnn(in_dim, hidden=64, n_layers=3, out_dim=1, edge_dim=0, aggr="mean", task="graph",
               readout="mean", norm=True, dropout=0.0, directional=False, device=None):
    """Build a general MPNN (pure torch). `task`='node' → per-node output (N, out_dim); 'graph' → per-graph
    output (G, out_dim) via `readout` pooling. N layers each: MPNNLayer → norm → GELU → residual. Grounded:
    GraphSAGE stack + residual (runtime 1st/2nd/5th), global-mean readout (runtime 1st). Returns an nn.Module.
    """
    import torch
    from torch import nn
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    MPNNLayer = _mpnn_layer(torch, nn)

    class MPNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.task = task
            self.readout = readout
            self.inp = nn.Linear(in_dim, hidden)
            self.layers = nn.ModuleList(
                [MPNNLayer(hidden, hidden, edge_dim=edge_dim, aggr=aggr, directional=directional)
                 for _ in range(n_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) if norm else nn.Identity() for _ in range(n_layers)])
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, out_dim)

        def encode(self, x, edge_index, edge_attr=None):
            h = self.inp(x)
            for lyr, nrm in zip(self.layers, self.norms):
                h = h + self.drop(self.act(nrm(lyr(h, edge_index, edge_attr))))   # residual
            return h

        def forward(self, x, edge_index, edge_attr=None, batch=None):
            h = self.encode(x, edge_index, edge_attr)
            if self.task == "node":
                return self.head(h)
            if batch is None:
                batch = h.new_zeros(h.shape[0], dtype=torch.long)
            g = graph_readout(h, batch, mode=self.readout)
            return self.head(g)

    return MPNN().to(dev)


class GraphMessagePassing(BaseAgent):
    name = "graph-message-passing"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            torch.manual_seed(int(spec.get("seed", 0)))
            N = int(spec.get("num_nodes", 12))
            in_dim = int(spec.get("in_dim", 5))
            hidden = int(spec.get("hidden", 32))
            n_layers = int(spec.get("n_layers", 3))
            aggr = str(spec.get("aggr", "mean"))
            task = str(spec.get("task", "graph"))
            out_dim = int(spec.get("out_dim", 3))
            # self-synthesize a small random graph for a health-check when none is supplied
            x = spec.get("x")
            ei = spec.get("edge_index")
            if x is None:
                x = torch.randn(N, in_dim, device=dev)
            else:
                x = torch.as_tensor(x, dtype=torch.float32, device=dev); N, in_dim = x.shape
            if ei is None:
                E = max(N, 2 * N)
                ei = torch.randint(0, N, (2, E), device=dev)
            else:
                ei = torch.as_tensor(ei, dtype=torch.long, device=dev)
            model = build_mpnn(in_dim, hidden=hidden, n_layers=n_layers, out_dim=out_dim, aggr=aggr,
                               task=task, device=dev)
            model.eval()
            with torch.no_grad():
                out = model(x, ei)
            exp = (N, out_dim) if task == "node" else (1, out_dim)
            ok = tuple(out.shape) == exp and bool(torch.isfinite(out).all())
            # propagation sanity: a planted signal on node 0 reaches its neighbours after 1 hop
            xp = torch.zeros(N, 1, device=dev); xp[0] = 1.0
            prop = propagate(xp, ei, num_nodes=N, aggr="mean")
            nbrs = ei[1][ei[0] == 0]
            reached = bool((prop[nbrs] > 0).any()) if nbrs.numel() else True
            n_params = sum(p.numel() for p in model.parameters())
            msg = (f"graph-message-passing: MPNN(aggr={aggr}, layers={n_layers}, task={task}) out={tuple(out.shape)} "
                   f"finite={bool(torch.isfinite(out).all())} params={n_params/1e3:.1f}k propagates={reached} "
                   f"device={dev}; pure scatter/index_add (PyG_available={_pyg_available()}, not required).")
            self.log(msg, kind="finding",
                     recommendation="build_mpnn(in_dim, aggr='mean'|'attention', task='node'|'graph'); GraphSAGE "
                                    "root+neighbour with residual+LayerNorm was the runtime/CHAMPS winner; edge_dim>0 "
                                    "folds edge features into the message; drop_edge/drop_node for graph augmentation")
            return self.done({"out_shape": list(out.shape), "finite": bool(torch.isfinite(out).all()),
                              "propagates": reached, "params": int(n_params), "pyg": _pyg_available(),
                              "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher", f"[{worker}] graph-message-passing bad output {tuple(out.shape)}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] graph-message-passing FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ 2. graph feature extractor + PE
def degrees(edge_index, num_nodes):
    """In / out / total degree per node (grounded: implicit in every SAGE aggregation). Returns (N, 3)."""
    import torch
    src, dst = edge_index[0], edge_index[1]
    ones = torch.ones(edge_index.shape[1], device=edge_index.device)
    out_deg = scatter_sum(ones, src, num_nodes)
    in_deg = scatter_sum(ones, dst, num_nodes)
    return torch.stack([in_deg, out_deg, in_deg + out_deg], dim=1)


def _dense_adj(edge_index, num_nodes, symmetric=True):
    import torch
    A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    if edge_index.shape[1]:
        A[edge_index[0], edge_index[1]] = 1.0
    if symmetric:
        A = ((A + A.t()) > 0).float()
    A.fill_diagonal_(0.0)
    return A


def clustering_coeff(edge_index, num_nodes):
    """Local clustering coefficient per node C_i = (A^3)_ii / (deg_i·(deg_i−1)) (undirected). Dense; guarded to
    small graphs. Grounded: the neighbours-of-pair adjacency structure covid-3rd found worth 20 bps. Returns (N,)."""
    import torch
    A = _dense_adj(edge_index, num_nodes)
    deg = A.sum(1)
    tri = torch.diagonal(A @ A @ A)
    denom = (deg * (deg - 1.0)).clamp_min(1.0)
    return (tri / denom).clamp(0.0, 1.0)


def khop_counts(edge_index, num_nodes, k=2):
    """Number of DISTINCT nodes reachable within 1..k hops per node (structural locality). Dense boolean
    reachability powers; guarded. Returns (N, k)."""
    import torch
    A = (_dense_adj(edge_index, num_nodes) > 0)
    reach = A.clone()
    cols = []
    frontier = A.clone()
    for _ in range(int(k)):
        cnt = reach.sum(1).float()
        cols.append(cnt)
        frontier = (frontier.float() @ A.float()) > 0
        reach = reach | frontier
    return torch.stack(cols, dim=1)


def laplacian_pe(edge_index, num_nodes, k=4):
    """Laplacian eigenvector positional encoding: k smallest non-trivial eigenvectors of the symmetric
    normalized Laplacian L = I − D^-1/2 A D^-1/2 (grounded: covid structural/graph-distance PE; the Fiedler
    vector separates communities). Sign made deterministic (first-nonzero-entry positive). Returns (N, k)."""
    import torch
    A = _dense_adj(edge_index, num_nodes)
    deg = A.sum(1)
    dinv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    L = torch.eye(num_nodes, device=A.device) - (dinv.view(-1, 1) * A * dinv.view(1, -1))
    L = 0.5 * (L + L.t())
    evals, evecs = torch.linalg.eigh(L)
    k = min(int(k), max(num_nodes - 1, 1))
    pe = evecs[:, 1:1 + k]                                 # skip the trivial constant eigenvector
    # deterministic sign: make the first non-negligible entry of each column positive
    for j in range(pe.shape[1]):
        col = pe[:, j]
        nz = torch.nonzero(col.abs() > 1e-6)
        if nz.numel() and col[nz[0]] < 0:
            pe[:, j] = -col
    if pe.shape[1] < int(k):                               # pad if graph smaller than k
        pe = torch.cat([pe, pe.new_zeros(num_nodes, int(k) - pe.shape[1])], dim=1)
    return pe


def random_walk_pe(edge_index, num_nodes, k=4):
    """Random-walk positional encoding: return probability landing vector [ (P)_ii, (P^2)_ii, …, (P^k)_ii ]
    with P = D^-1 A (grounded: structural locality the winners encode). Returns (N, k)."""
    import torch
    A = _dense_adj(edge_index, num_nodes)
    deg = A.sum(1).clamp_min(1.0)
    P = A / deg.view(-1, 1)
    cols, Pk = [], P.clone()
    for _ in range(int(k)):
        cols.append(torch.diagonal(Pk))
        Pk = Pk @ P
    return torch.stack(cols, dim=1)


def global_features(edge_index, num_nodes):
    """Graph-level descriptors: n_nodes, n_edges, density, mean/max degree. Returns a (5,) tensor."""
    import torch
    E = edge_index.shape[1]
    deg = degrees(edge_index, num_nodes)[:, 2]
    density = E / max(num_nodes * (num_nodes - 1), 1)
    return torch.tensor([float(num_nodes), float(E), float(density),
                         float(deg.mean()), float(deg.max() if num_nodes else 0.0)], device=edge_index.device)


def graph_features(edge_index, num_nodes, k_hop=2, pe_dim=4, want_pe=True):
    """Assemble the deterministic node feature matrix (degree | clustering | k-hop | Laplacian-PE | RW-PE) plus
    global features. Dense pieces guarded to graphs under _NODE_CAP nodes. Returns dict with 'node' (N, F)."""
    import torch
    feats = [degrees(edge_index, num_nodes)]
    small = num_nodes <= _NODE_CAP
    if small:
        feats.append(clustering_coeff(edge_index, num_nodes).view(-1, 1))
        feats.append(khop_counts(edge_index, num_nodes, k=k_hop).float())
        if want_pe:
            feats.append(laplacian_pe(edge_index, num_nodes, k=pe_dim))
            feats.append(random_walk_pe(edge_index, num_nodes, k=pe_dim))
    node = torch.cat(feats, dim=1)
    return {"node": node, "global": global_features(edge_index, num_nodes),
            "dense_used": bool(small), "n_features": int(node.shape[1])}


class GraphFeatureExtractor(BaseAgent):
    name = "graph-feature-extractor"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            torch.manual_seed(int(spec.get("seed", 0)))
            N = int(spec.get("num_nodes", 10))
            ei = spec.get("edge_index")
            if ei is None:                                    # self-synthesize a 2-community toy graph
                ei = _two_community_edges(N, dev)
            else:
                ei = torch.as_tensor(ei, dtype=torch.long, device=dev)
                N = int(ei.max().item()) + 1 if ei.numel() else N
            feats = graph_features(ei, N, k_hop=int(spec.get("k_hop", 2)), pe_dim=int(spec.get("pe_dim", 4)))
            node = feats["node"]
            ok = bool(torch.isfinite(node).all()) and node.shape[0] == N and node.shape[1] >= 3
            msg = (f"graph-feature-extractor: node feats {tuple(node.shape)} (degree|clustering|k-hop|Lap-PE|RW-PE) "
                   f"+ {feats['global'].shape[0]} global; dense_used={feats['dense_used']} finite={ok} device={dev}. "
                   f"The FE+PE winners hand-craft (covid distance-PE, champs charges/angles, runtime degree/opcode).")
            self.log(msg, kind="finding",
                     recommendation="graph_features(edge_index, num_nodes) for node FE; laplacian_pe (Fiedler "
                                    "separates communities) / random_walk_pe for structural positional encoding")
            return self.done({"n_features": feats["n_features"], "num_nodes": N, "dense_used": feats["dense_used"],
                              "finite": ok, "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher", f"[{worker}] graph-feature-extractor bad output {tuple(node.shape)}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] graph-feature-extractor FAILED ({str(e)[:180]})")


def _two_community_edges(n, device):
    """Two equal cliques joined by ONE bridge edge — the canonical community-detection toy graph."""
    import torch
    half = max(2, n // 2)
    edges = []
    for a in range(half):
        for b in range(a + 1, half):
            edges += [(a, b), (b, a)]
    for a in range(half, 2 * half):
        for b in range(a + 1, 2 * half):
            edges += [(a, b), (b, a)]
    edges += [(0, half), (half, 0)]                          # the single bridge
    ei = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return ei


# ════════════════════════════════════════════════════════════ 3. graph readout / pooling
def graph_readout(x, batch, mode="mean", gate=None):
    """Pool per-node embeddings `x` (N, D) into one vector per graph `(G, D)` using `batch` (N,) graph ids.
    mode ∈ {mean, sum, max, attention}. attention = segment-softmax over a per-node score (`gate` (N,) if
    given, else the node-feature L2 norm — parameter-free). Permutation-invariant, variable-size safe.
    Grounded: runtime-1st global-mean-pool, runtime-2nd sum-reduce."""
    import torch
    G = int(batch.max().item()) + 1 if batch.numel() else 1
    if mode == "sum":
        return scatter_sum(x, batch, G)
    if mode == "max":
        return scatter_max(x, batch, G)
    if mode == "attention":
        score = gate if gate is not None else x.norm(dim=1)
        alpha = segment_softmax(score, batch, G).unsqueeze(-1)
        return scatter_sum(x * alpha, batch, G)
    return scatter_mean(x, batch, G)


def build_set2set(in_dim, processing_steps=3, device=None):
    """Set2Set readout (Vinyals et al.) — an LSTM iteratively attends over the node set to produce an
    order-invariant graph vector of size 2·in_dim (grounded: the canonical learnable molecular-GNN pool).
    Pure torch. Returns an nn.Module: forward(x, batch) → (G, 2·in_dim)."""
    import torch
    from torch import nn
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    class Set2Set(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_dim = in_dim
            self.out_dim = 2 * in_dim
            self.steps = processing_steps
            self.lstm = nn.LSTMCell(self.out_dim, in_dim)

        def forward(self, x, batch):
            G = int(batch.max().item()) + 1 if batch.numel() else 1
            h = x.new_zeros(G, self.in_dim)
            c = x.new_zeros(G, self.in_dim)
            q_star = x.new_zeros(G, self.out_dim)
            for _ in range(self.steps):
                h, c = self.lstm(q_star, (h, c))
                e = (x * h[batch]).sum(dim=1)                # attention logits per node
                a = segment_softmax(e, batch, G).unsqueeze(-1)
                r = scatter_sum(a * x, batch, G)             # read vector
                q_star = torch.cat([h, r], dim=1)
            return q_star

    return Set2Set().to(dev)


class GraphReadout(BaseAgent):
    name = "graph-readout"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            torch.manual_seed(int(spec.get("seed", 0)))
            D = int(spec.get("dim", 8))
            sizes = spec.get("sizes") or [4, 7, 3]           # 3 graphs of different sizes → one vector each
            xs, bs = [], []
            for gi, s in enumerate(sizes):
                xs.append(torch.randn(int(s), D, device=dev))
                bs.append(torch.full((int(s),), gi, dtype=torch.long, device=dev))
            x = torch.cat(xs, dim=0); batch = torch.cat(bs, dim=0)
            G = len(sizes)
            checks = {}
            for m in ("mean", "sum", "max", "attention"):
                r = graph_readout(x, batch, mode=m)
                checks[m] = tuple(r.shape) == (G, D) and bool(torch.isfinite(r).all())
            s2s = build_set2set(D, device=dev)
            with torch.no_grad():
                rs = s2s(x, batch)
            checks["set2set"] = tuple(rs.shape) == (G, 2 * D) and bool(torch.isfinite(rs).all())
            ok = all(checks.values())
            msg = (f"graph-readout: {sum(checks.values())}/{len(checks)} ok ({checks}); pooled {G} variable-size "
                   f"graphs {sizes} → fixed vectors (mean/sum/max/attention → (G,{D}); set2set → (G,{2*D})). device={dev}.")
            self.log(msg, kind="finding",
                     recommendation="graph_readout(node_emb, batch, mode='mean'|'attention') for the graph-regression "
                                    "head (runtime 1st global-mean-pool); build_set2set for the learnable molecular pool")
            return self.done({"checks": {k: bool(v) for k, v in checks.items()}, "n_graphs": G, "dim": D,
                              "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher", f"[{worker}] graph-readout checks failed: {checks}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] graph-readout FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ handlers
_MP = GraphMessagePassing()
_FE = GraphFeatureExtractor()
_RO = GraphReadout()


def run_message_passing(q, worker):
    return _MP.run(q, worker)


def run_feature_extractor(q, worker):
    return _FE.run(q, worker)


def run_readout(q, worker):
    return _RO.run(q, worker)
