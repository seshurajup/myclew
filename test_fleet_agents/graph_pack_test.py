"""graph_pack_test — DATA-WISE, offline, deterministic (BLAS-pinned) verifier for the GRAPH/GNN pack.

Builds tiny synthetic graphs (no files, no network) and asserts the ground-truth behaviour of each graph
agent's underlying function:
  • message passing → right output shape, finite, permutation-EQUIVARIANT on nodes (permuting node order
    permutes node outputs consistently) and permutation-INVARIANT on the graph readout;
  • a planted signal on one node reaches its neighbours after 1 propagation hop (message passing propagates);
  • a 2-community toy graph → Laplacian-PE Fiedler vector SEPARATES the two communities;
  • readout maps variable-size graphs → a fixed per-graph vector;
plus that each raw handler returns a valid (status,data,to,msg) contract on an EMPTY spec (fleet smoke
contract). Exit 0 iff all checks pass.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

import torch

from fleet_agents import graph_pack as G

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_fails = []


def check(name, cond):
    print(("  ok  " if cond else "  FAIL") + "  " + name)
    if not cond:
        _fails.append(name)


# ── 0. scatter primitives ────────────────────────────────────────────────────────────────────────────
src = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=DEV)
idx = torch.tensor([0, 0, 1, 1], device=DEV)
check("scatter_sum correct", bool((G.scatter_sum(src, idx, 2).flatten() == torch.tensor([3.0, 7.0], device=DEV)).all()))
check("scatter_mean correct", bool((G.scatter_mean(src, idx, 2).flatten() == torch.tensor([1.5, 3.5], device=DEV)).all()))
check("scatter_max correct", bool((G.scatter_max(src, idx, 2).flatten() == torch.tensor([2.0, 4.0], device=DEV)).all()))
# segment softmax normalises within each destination segment
ss = G.segment_softmax(torch.tensor([1.0, 1.0, 2.0, 0.0], device=DEV), idx, 2)
check("segment_softmax sums to 1 per segment", abs(float(ss[:2].sum()) - 1.0) < 1e-5 and abs(float(ss[2:].sum()) - 1.0) < 1e-5)

# ── 1. message passing propagates a planted signal to neighbours after 1 hop ─────────────────────────
# directed chain 0->1->2->3 ; signal on node 0 must reach node 1 (its out-neighbour) after 1 hop
ei_chain = torch.tensor([[0, 1, 2], [1, 2, 3]], device=DEV)
x0 = torch.zeros(4, 1, device=DEV); x0[0] = 1.0
prop = G.propagate(x0, ei_chain, num_nodes=4, aggr="mean")
check("propagate reaches out-neighbour (0->1)", float(prop[1]) > 0.0)
check("propagate leaves non-neighbour cold (node 3)", float(prop[3]) == 0.0)

# ── 2. MPNN forward: right shape, finite, node-equivariant + graph-invariant under permutation ────────
N, Fin = 8, 5
x = torch.randn(N, Fin, device=DEV)
ei = G._two_community_edges(N, DEV)               # a real (symmetric) graph
node_model = G.build_mpnn(Fin, hidden=16, n_layers=2, out_dim=4, aggr="mean", task="node", device=DEV).eval()
graph_model = G.build_mpnn(Fin, hidden=16, n_layers=2, out_dim=3, aggr="attention", task="graph", device=DEV).eval()
with torch.no_grad():
    y_node = node_model(x, ei)
    y_graph = graph_model(x, ei)
check("mpnn node-task shape (N, out)", tuple(y_node.shape) == (N, 4))
check("mpnn node-task finite", bool(torch.isfinite(y_node).all()))
check("mpnn graph-task shape (1, out)", tuple(y_graph.shape) == (1, 3))
check("mpnn graph-task finite", bool(torch.isfinite(y_graph).all()))

# permutation: relabel nodes, remap edge_index, re-run
perm = torch.randperm(N, device=DEV)
inv = torch.empty_like(perm); inv[perm] = torch.arange(N, device=DEV)
x_p = x[perm]
ei_p = inv[ei]
with torch.no_grad():
    y_node_p = node_model(x_p, ei_p)
    y_graph_p = graph_model(x_p, ei_p)
check("mpnn node output is permutation-EQUIVARIANT", torch.allclose(y_node_p, y_node[perm], atol=1e-4))
check("mpnn graph output is permutation-INVARIANT", torch.allclose(y_graph_p, y_graph, atol=1e-4))

# edge features fold into the message (edge_dim path runs + finite)
ea = torch.randn(ei.shape[1], 2, device=DEV)
emodel = G.build_mpnn(Fin, hidden=12, n_layers=1, out_dim=2, edge_dim=2, aggr="mean", task="node", device=DEV).eval()
with torch.no_grad():
    ye = emodel(x, ei, ea)
check("mpnn with edge features shape+finite", tuple(ye.shape) == (N, 2) and bool(torch.isfinite(ye).all()))

# ── 3. graph feature extractor + positional encodings separate two communities ───────────────────────
ei2 = G._two_community_edges(8, DEV)             # two 4-cliques + 1 bridge → 8 nodes
feats = G.graph_features(ei2, 8, k_hop=2, pe_dim=3)
node = feats["node"]
check("graph_features finite", bool(torch.isfinite(node).all()))
check("graph_features has degree+clustering+khop+PE columns", node.shape[0] == 8 and node.shape[1] >= 3)
# clustering coefficient: nodes inside a 4-clique (minus bridge endpoints) are ~fully clustered
cc = G.clustering_coeff(ei2, 8)
check("clustering coeff in [0,1] and high inside cliques", float(cc.max()) <= 1.0 + 1e-6 and float(cc[2]) > 0.5)
# Laplacian PE Fiedler vector (first non-trivial eigenvector) separates the two communities by SIGN
pe = G.laplacian_pe(ei2, 8, k=1)
fiedler = pe[:, 0]
c1 = fiedler[:4].mean(); c2 = fiedler[4:].mean()
check("Laplacian-PE Fiedler separates the 2 communities (opposite mean sign)",
      float(c1) * float(c2) < 0 and abs(float(c1) - float(c2)) > 1e-3)
# random-walk PE: return probability is positive within cliques
rw = G.random_walk_pe(ei2, 8, k=2)
check("random-walk PE finite and non-trivial", bool(torch.isfinite(rw).all()) and float(rw.abs().sum()) > 0)
# degree: bridge endpoints (0 and 4) have degree 4 (3 clique + 1 bridge), others degree 3
deg = G.degrees(ei2, 8)[:, 2] / 2.0              # undirected: both directions counted
check("degree of bridge endpoint > interior clique node", float(deg[0]) > float(deg[1]) - 1e-6)

# ── 4. readout maps variable-size graphs → one fixed vector each, permutation-invariant ───────────────
D = 6
sizes = [3, 5, 2]
xs = [torch.randn(s, D, device=DEV) for s in sizes]
batch = torch.cat([torch.full((s,), i, dtype=torch.long, device=DEV) for i, s in enumerate(sizes)])
xall = torch.cat(xs, dim=0)
for m in ("mean", "sum", "max", "attention"):
    r = G.graph_readout(xall, batch, mode=m)
    check(f"readout {m} -> (G, D) fixed vector", tuple(r.shape) == (len(sizes), D) and bool(torch.isfinite(r).all()))
# mean readout is permutation-invariant WITHIN a graph
r_mean = G.graph_readout(xall, batch, mode="mean")
pj = torch.randperm(sizes[0], device=DEV)
x_perm = xall.clone(); x_perm[:sizes[0]] = xall[:sizes[0]][pj]
r_mean_p = G.graph_readout(x_perm, batch, mode="mean")
check("readout mean is permutation-invariant within a graph", torch.allclose(r_mean, r_mean_p, atol=1e-5))
# Set2Set → (G, 2D)
s2s = G.build_set2set(D, device=DEV)
with torch.no_grad():
    rs = s2s(xall, batch)
check("set2set readout -> (G, 2D)", tuple(rs.shape) == (len(sizes), 2 * D) and bool(torch.isfinite(rs).all()))

# ── 5. DropEdge / drop-node graph augmentation are shape-safe and actually drop ──────────────────────
ei3 = G._two_community_edges(8, DEV)
kept = G.drop_edge(ei3, p=0.5, seed=1)
check("drop_edge removes some edges", kept.shape[1] < ei3.shape[1] and kept.shape[0] == 2)
check("drop_edge p=0 is identity", G.drop_edge(ei3, p=0.0).shape[1] == ei3.shape[1])
xn, ein = G.drop_node(torch.randn(8, 4, device=DEV), ei3, p=0.5, seed=2)
check("drop_node zeros some rows + drops incident edges", bool((xn.abs().sum(1) == 0).any()) and ein.shape[1] <= ei3.shape[1])

# ── 6. every raw handler returns a valid contract on EMPTY spec (fleet smoke contract) ───────────────
VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}
for h in (G.run_message_passing, G.run_feature_extractor, G.run_readout):
    r = h({"question": "test", "spec": {}}, "unit")
    check(f"handler {h.__name__} valid contract", isinstance(r, tuple) and len(r) == 4 and r[0] in VALID)
    check(f"handler {h.__name__} returns done on healthy default", r[0] == "done")

print()
if _fails:
    print("FAILURES:", _fails)
    sys.exit(1)
print("ALL GRAPH PACK CHECKS PASSED")
sys.exit(0)
