# Graph / GNN modality pack — grounded in top solutions of real graph competitions

**Mined** (2026-07-16) via the fleet's `gm-writeup-mine` agent → nvidia-kaggle bearer API
(`fetch_leaderboard_writeups` → `fetch_writeup`, KGAT token in `.env` + `~/.kaggle/access_token`,
`PROJECT_ROOT=<comp>`), top-5 solutions each saved under `docs/gm_writeups/<slug>/rank{1..5}_*.md`.
Source: **bearer API (real writeups)**, not WebSearch — every technique below traces to a specific
placed writeup file. (The token worked; WebSearch/WebFetch fallback was NOT needed.)

Same self-improving loop as the audio pack: mine → extract recurring GNN techniques → DEDUP against the
existing fleet → build only what is genuinely-missing and graph-general → register + data-wise test.

## Competitions mined (most recent first)

| slug | task | graph | metric | top solns saved |
| --- | --- | --- | --- | --- |
| predict-ai-model-runtime | TPU/XLA op-graph runtime ranking (Fast-or-Slow) | GNN over computation graphs | Kendall-tau / slowdown | rank1..5 |
| stanford-covid-vaccine (OpenVaccine) | per-base RNA degradation regression | RNA sequence-as-graph (base-pairing adjacency) | column-wise MCRMSE | rank1..5 |
| champs-scalar-coupling | molecular scalar-coupling regression | molecular graph / MPNN | log-MAE | rank1..5 |

## Recurring GNN techniques (with provenance)

### 1. Message passing / graph convolution — GraphSAGE is the workhorse; GAT for noisy graphs — UNIVERSAL
The core operation: each node aggregates transformed messages from its neighbours, adds a self/root
transform, residual + norm.
- runtime **rank1** (viet/BR-connection): block = `InstanceNorm → SAGEConv → SelfChannelAttention → CrossConfigAttention → +residual → GELU`; 2 conv blocks. "We tried many types [of graph conv] but none was better than `SAGEConv`… GAT variants… none worked well… for TPU graphs all connections are real and important so graph attention was not that helpful."
- runtime **rank2** (latenciaga): "All GNN layers are SageConv layers with **residual connections** whenever in/out channels match." 2x64+2x128+2x256 (XLA) / 4x256+4x512 (NLP/Tile). GATv2Conv/GINEConv **did not work**.
- runtime **rank5** (knshnb): "3-layer GraphSage… in each layer I operate graph convolution in **both directions of edges by different weights and concatenate** the outputs" (directional message passing). GAT/GATv2/GIN did not beat it.
- champs **rank4** (Hyperspatial): a **graph attention network** (GAT) where "attention updates atom embeddings by aggregating over **edge embeddings**, not neighbouring node embeddings"; gated residual connections (arXiv:1805.10988); directional (bidirectional) edges.
- champs **rank1** (Bosch, 1st): a "soft graph transformer" — self-attention biased by the graph:
  `Z' = W1 Z softmax(Zᵀ W2ᵀ W3 Z − γ·D)`, D = squared distance matrix, γ learnable (γ→0 = plain
  transformer, γ→∞ = hard graph mask). This is attention-based message passing with a distance/edge bias.
→ The pack's `graph-message-passing`: pure-torch MPNN with configurable aggregator **mean/max/sum/attention**
(GAT-style segment-softmax), optional **edge features** folded into the message, N layers with **residual +
norm**, root/self transform (SAGE), optional **directional** (both-direction) messages and **DropEdge**.
Runs via `index_add`/`scatter_reduce` — **no torch_geometric dependency** (PyG is only an optional fast path,
never a hard dep; and it is not installed in our venv). GraphSAGE `= W_self·x + W_neigh·mean(neighbours)` is
the default because it was the repeated winner.

### 2. Graph readout / pooling — global mean/sum pool + attention — UNIVERSAL
Map per-node embeddings → one vector (or scalar) per graph.
- runtime **rank1**: "Global (graph) **mean pooling**" fuses the (possibly disconnected) pruned sub-graphs into
  a single graph prediction before the final linear.
- runtime **rank2**: "Features produced by the GNN layer stack are transformed to one value per node and then
  **sum-reduced** to form a single graph-wise prediction."
- runtime **rank1** (Cross-Config Attention) + champs attention heads → attention-weighted pooling is a
  recurring reduction; Set2Set is the canonical learnable order-invariant pool for molecular GNNs.
→ The pack's `graph-readout`: `mean | sum | max | attention | set2set` pooling of `(node embeddings, batch
index) → (G, D)`, permutation-invariant, variable-size-safe. Attention pool = segment-softmax gate; Set2Set =
LSTM-driven iterative content attention.

### 3. Node / edge / global feature engineering + structural/positional encodings — recurring
Winners hand-craft node/edge features AND structural encodings because the raw graph alone underperforms.
- covid **rank1** (Jiayang Gao): **distance embedding** + a graph-**distance matrix** computed as position
  difference *adjusting for primary pairs* ("if (5,20) are a pair, then 5 has distance 1 to 4,6,20…, computed
  iteratively") — i.e. shortest-path / structural distance as a positional encoding; plus "distance to the
  closest paired / unpaired position" (a strong structural feature).
- covid **rank3** (striderl) / **rank5** (tito): the **base-pairing-probability (bpp) adjacency matrix** IS the
  graph; edge/node features `bpps_max, bpps_sum, bpps_sum−max, pair type (CG/GU/AU), entropy`, plus "two
  matrices to specify the neighbours of each node's pair" (multi-hop adjacency) — "this feature alone
  increased 20 bps".
- champs **rank4**: node features `atom-type embed, electronegativity, ionization energy, electron affinity,
  Mulliken charge`; edge features `distance, edge-type embed, bond-angle (2J), dihedral (3J)`; artificial edges
  at graph-distance 2 and 3.
- champs **rank1**: **Fourier / positional encoding** of scalar node/edge constants (partial charge, distance,
  angle) "much like positional encoding in a standard Transformer".
- runtime **rank1/rank2/rank5**: learned **opcode / node-type embeddings** (12–16 ch), `StandardScaler` on
  numeric node feats, `sign(x)·log(|x|)` compression, log-transform of input features, node **degree** implicit
  in the aggregation.
→ The pack's `graph-feature-extractor`: deterministic **node** features (in/out **degree**, local **clustering
  coefficient**, **k-hop reachable counts**), **structural/positional encodings** (**Laplacian eigenvector PE**
  — Fiedler vector separates communities; **random-walk PE** — return-probability landing vector), plus
  **edge** features (graph/adjacency-derived) and **global** graph descriptors (n_nodes, n_edges, density, mean
  degree). Pure numpy/torch dense linear algebra for small graphs (guarded by a node cap).

### 4. Graph augmentation (edge/node dropout, cutout) — recurring
- runtime **rank5**: **DropEdge** (arXiv:1907.10903) as the graph augmentation.
- champs **rank1**: "a kind of **cutout** procedure (randomly drop two atoms plus all bonds/triplets containing
  them) worked as a very effective regulariser"; champs **rank2** "knock-out" (delete 10% of input atoms).
→ Exposed as `drop_edge` / `drop_node` helpers inside `graph-message-passing` (DataLoader/autograd-safe,
  out-of-place), not a separate agent.

### 5. Ranking losses over per-graph slates — recurring (runtime-specific, already fleet-adjacent)
- runtime rank1 `PairwiseHingeLoss`; rank2 `ListMLE / MarginRankingLoss / DiffMat`; rank5 pairwise hinge.
"Models trained with a ranking loss heavily outperformed element-wise losses." This is a **loss**, applicable
to any learning-to-rank comp; it is a training-objective, not a graph primitive, so it is left to the training
packs (not rebuilt here). Noted for provenance.

### 6. Set / point-cloud transformers that AVOID graphs — counterpoint (not a graph primitive)
- champs **rank2** (Quantum Uncertainty, 2nd): a pure **Atomic Transformer** on the atom *set* (x,y,z +
  atom-type + coupling-type embeddings), **no graph, no positional encoding** — permutation invariance of a
  bare transformer encoder does the work. covid GRU/LSTM/CNN heads similarly treat the sequence directly.
These are covered by existing transformer/sequence machinery (`masked-sequence-pool`, LLM/attention packs);
they argue that a strong **readout + attention** can substitute for explicit message passing, which is why the
pack ships `graph-readout` with attention/Set2Set alongside the MPNN.

## Dedup verdict vs the existing fleet

- `gnn-link-train` (fleet) — **link-prediction specific**: trains division + flow heads for biohub cell
  tracking (candidate edge classification / affinity regression on a spatiotemporal graph). NOT a general
  message-passing model. Referenced, not duplicated.
- `gnn-probe` (fleet) — **link-prediction specific**: a cheap sklearn probe asking whether neighbourhood
  context beats pairwise geometry for the biohub *linker*. NOT a graph classifier/regressor, no readout.
- The fleet had **no general message-passing model, no graph feature/PE extractor, no graph readout/pooling** —
  exactly the three genuinely-missing, graph-general primitives the mined winners all rely on. That is this
  pack.

## Agents built (this pack) — pure torch, GPU-first, ABI-safe (no torch_geometric hard dep)

- `graph-message-passing` — general MPNN forward: aggregator ∈ {mean, max, sum, attention}, optional edge
  features in the message, N layers with residual + norm + SAGE root transform, directional option, DropEdge;
  node-classification / graph-regression heads. Implemented with `index_add` / `scatter_reduce` (PyG optional
  fast path only).
- `graph-feature-extractor` — deterministic node features (degree, clustering, k-hop counts) + structural/
  positional encodings (Laplacian eigvec PE, random-walk PE) + edge & global features from an `edge_index`.
- `graph-readout` — graph-level pooling (mean/max/sum/attention/Set2Set) mapping node embeddings + batch index
  → one vector per graph.

New coverage pack **"Graph"**; routed to Kaggle modalities **graph + multimodal** (graphs also appear inside
multimodal comps). Data-wise verifier: `test_fleet_agents/graph_pack_test.py`.
