# #4 Solution [Hyperspatial Engineers]

Hi everyone,

First I'd like to thank the organizers for the well organized competition, but I'd also like to thank other competitors for making things fun, your great scores pushed us to do much more and get much better scores than we have originally though we could achieve.

Here we'd like to share our solution which is also based on a Graph Transformer network, though with a few twists.

### Input data
We have used OpenBabel to infer bonds from atom coordinates, and have used some custom code to fix what we identified as mistakes made by OpenBabel.

We construct a graph where each node is an atom, each bond becomes an edge, then we add artificial edges between all nodes at distance 2 and 3 in the graph (2J and 3J edges respectively).
All edges are directional, so we have one edge for each direction to make the graph fully bidirectional.

**Attributes**:
- Nodes have atom type embedding, electronegativity, first ionization energy, electron affinity, Mulliken charge taken from the QM9 data (originally predicted using the same model architecture trained on the provided Mulliken charges)
- All edges have distance and edge type embedding (single bond, double bond, triple bond, 2J, 3J)
- 2J edges have bond angle on the atom they are skipping over.
- 3J edges have dihedral angle between the atoms they are connecting

No explicit XYZ data is used, as we wanted to make the model rotation and translation invariant.

All input data was normalized to zero mean unit variance.

### Model
Core of the architecture is the graph attention network with multiple heads, with a few twists:
- Attention heads do not attend to all data from previous layer, but only the output of the same head from previous layer
- All edge embeddings are first updated from triplets (src, edge, dst), the attention then updates atom embeddings by aggregating over edge embeddings, not neighboring node embeddings
- We use gated residual connections between attention blocks similar to: https://arxiv.org/abs/1805.10988
- We output scalar coupling constants directly on the edges, this makes it two predictions, one for each edge direction and each is treated independently in the loss function. These two predictions are averaged to get a final prediction making this a kind of micro-ensemble.

Loss function is a mean of MAEs per coupling type. We have tried mean log MAE but it was giving us worse results. We have used both scaled targets to zero mean unit variance and zero mean targets with no variance scaling, as different types benefited from different setups.

Quickest way to outline the network would be using code:

```python
emb = 48
heads = 24
bias = False

def AttnBlock(in_emb, out_emb):
    return nn.Sequential(
        EdgeLinear(in_emb, out_emb),
        NodeLinear(in_emb, out_emb),
        GraphLambda(lambda x: x.view(x.shape[0], heads, -1)),
        TripletCat(out='triplet'),
        MagicAttn(emb, 3 * emb, heads, attn_key='triplet'),
        TripletMultiLinear(emb, emb, emb, heads, bias=bias),
        GraphLambda(torch.nn.LayerNorm(heads * emb))
    )

net = nn.Sequential(
    Embed(emb, emb),
    AttnBlock(emb, emb * heads), GraphLambda(nn.PReLU()),
    GatedResidual(AttnBlock(emb * heads, emb * heads), emb * heads, emb * heads), GraphLambda(nn.PReLU()),
    GatedResidual(AttnBlock(emb * heads, emb * heads), emb * heads, emb * heads), GraphLambda(nn.PReLU()),
    GatedResidual(AttnBlock(emb * heads, emb * heads), emb * heads, emb * heads), GraphLambda(nn.PReLU()),
    GatedResidual(AttnBlock(emb * heads, emb * heads), emb * heads, emb * heads), GraphLambda(nn.PReLU()),
    GatedResidual(AttnBlock(emb * heads, emb * heads), emb * heads, emb * heads), GraphLambda(nn.PReLU()),
    GatedResidual(AttnBlock(emb * heads, emb * heads), emb * heads, emb * heads), GraphLambda(nn.PReLU()),
    GatedResidual(AttnBlock(emb * heads, emb * heads), emb * heads, emb * heads), GraphLambda(nn.PReLU()),
    EdgeLinear(emb * heads, 512, bias=True), GraphLambda(nn.PReLU(), node_key=None),
    EdgeLinear(512, 8, bias=True)
)
```

### Optimizer
We have used LAMB optimizer (https://arxiv.org/abs/1904.00962), again with a small twist, we have noticed that the weight decay is included in the step norm on which the trust ratio is calculated, this didn't make sense to us as weight decay should be independent of the update based on the batch gradient, so we moved the application of weight decay after the application of LAMB update, and this gave us better results. We call this LAMBW

### Training regime
We have split the data into 90/10 train/eval, two times with two different. We have used one cycle learning rate for 30 epochs with high weight decay, then dropped weight decay and continued training until eval saturation (~70 more epochs). Then we fine tune 100 epochs for each type to get further improvement (except for *JHN types, as they didn't improve with fine tuning).

We have used Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407) of last 25 epoch to get to a better minimum.

We then average the predictions of the two training runs for the two splits we had.

### What didn't work
Many, many things:
- RAdam
- combining RAdam and LAMBW
- many different forms of attention mechanisms
- mean log MAE loss, mean MSE loss
- GradNorm for multitask learning for different coupling types https://arxiv.org/abs/1711.02257
- Multi-Task Learning as Multi-Objective Optimization https://arxiv.org/abs/1810.04650
And many many other things

The full package with transformed data, additional data (and all code, which is also included in champs_code.tgz) is 7GB, so we are providing a link https://zenodo.org/record/3406154#.XXpX_ygzabg instead of uploading directly to Kaggle forum.